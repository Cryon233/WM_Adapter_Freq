from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor, nn

from wm_adapter.adapters.base import PEFTMethod
from wm_adapter.utils.checkpoints import sha256_file, verify_upstream_commits
from wm_adapter.utils.reproducibility import resolve_path


@dataclass(frozen=True)
class TokenLayout:
    total_tokens: int
    prefix_tokens: int
    patch_tokens: int
    grid_height: int
    grid_width: int
    token_dim: int


def _prepend_upstream_path(path: Path) -> None:
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


class JEPAWMDroidBackend(nn.Module):
    """Official DINOv3-L/16 encoder and DROID JEPA-WM predictor with a split final block."""

    def __init__(
        self,
        *,
        third_party_root: str | Path,
        jepa_checkpoint: str | Path,
        dinov3_checkpoint: str | Path,
        official_planning_config: str | Path,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.third_party_root = resolve_path(third_party_root)
        self.upstream_commits = verify_upstream_commits(self.third_party_root)
        self.jepa_repo = self.third_party_root / "jepa-wms"
        self.dinov3_repo = self.third_party_root / "dinov3"
        _prepend_upstream_path(self.jepa_repo)

        self.jepa_checkpoint = resolve_path(jepa_checkpoint)
        self.dinov3_checkpoint = resolve_path(dinov3_checkpoint)
        self.official_planning_config = resolve_path(official_planning_config)
        for label, path in (
            ("JEPA-WM checkpoint", self.jepa_checkpoint),
            ("DINOv3 checkpoint", self.dinov3_checkpoint),
            ("official planning config", self.official_planning_config),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")

        expected_dino_name = "dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth"
        if self.dinov3_checkpoint.name != expected_dino_name:
            raise ValueError(
                f"The official JEPA-WM DinoEncoder requires {expected_dino_name}, "
                f"received {self.dinov3_checkpoint}"
            )
        expected_dino_repo = self.third_party_root / "dinov3"
        if self.dinov3_repo != expected_dino_repo or not (self.dinov3_repo / "hubconf.py").is_file():
            raise FileNotFoundError(f"Official DINOv3 source checkout is incomplete: {self.dinov3_repo}")

        os.environ["JEPAWM_HOME"] = str(self.third_party_root)
        os.environ["JEPAWM_OSSCKPT"] = str(self.dinov3_checkpoint.parent.parent)

        planning = OmegaConf.load(self.official_planning_config)
        self.official_planning_template = planning
        data_config = OmegaConf.to_container(planning.model_kwargs.data, resolve=True)
        augmentation_config = OmegaConf.to_container(planning.model_kwargs.data_aug, resolve=True)
        model_config = OmegaConf.to_container(planning.model_kwargs.pretrain_kwargs, resolve=True)
        wrapper_config = OmegaConf.to_container(planning.model_kwargs.wrapper_kwargs, resolve=True)
        if not all(isinstance(value, dict) for value in (data_config, augmentation_config, model_config, wrapper_config)):
            raise TypeError(f"Official planning config has invalid nested mappings: {self.official_planning_config}")
        self.model_config = model_config
        self.data_config = data_config
        self.augmentation_config = augmentation_config

        from app.plan_common.datasets.preprocessor import Preprocessor
        from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module

        image_size = int(data_config["img_size"])
        normalize = augmentation_config["normalize"]
        transform = make_transforms(
            img_size=image_size,
            normalize=normalize,
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
        )
        inverse_transform = make_inverse_transforms(img_size=image_size, normalize=normalize)
        self.preprocessor = Preprocessor(
            action_mean=torch.zeros(7),
            action_std=torch.ones(7),
            state_mean=torch.zeros(7),
            state_std=torch.ones(7),
            proprio_mean=torch.zeros(7),
            proprio_std=torch.ones(7),
            transform=transform,
            inverse_transform=inverse_transform,
        )

        heads_config = model_config.get("heads_cfg", {})
        heads_config["architectures"] = {}
        heads_config["pretrain_dec_path"] = None
        model_config["heads_cfg"] = heads_config
        self.official_model = init_module(
            folder=self.jepa_checkpoint.parent,
            checkpoint=self.jepa_checkpoint.name,
            model_kwargs=model_config,
            device=self.device,
            action_dim=7,
            proprio_dim=7,
            preprocessor=self.preprocessor,
            cfgs_data=data_config,
            wrapper_kwargs=wrapper_config,
        )
        self.video_model = self.official_model.model
        self.encoder_wrapper = self.video_model.encoder
        if not hasattr(self.encoder_wrapper, "base_model"):
            raise TypeError(
                f"Official encoder wrapper is not a DINOv3 DinoEncoder: {type(self.encoder_wrapper).__name__}"
            )
        self.encoder = self.encoder_wrapper.base_model
        if not hasattr(self.encoder, "prepare_tokens_with_masks") or not hasattr(self.encoder, "blocks"):
            raise TypeError(f"Unsupported DINOv3 encoder implementation: {type(self.encoder).__name__}")
        if len(self.encoder.blocks) < 2:
            raise ValueError(f"DINOv3 encoder must have at least two blocks, found {len(self.encoder.blocks)}")
        self.last_block = self.encoder.blocks[-1]
        self.final_norm = self.encoder.norm
        self.image_size = image_size
        self.patch_size = int(self.encoder.patch_size)
        self.grid_height = image_size // self.patch_size
        self.grid_width = image_size // self.patch_size
        self.token_dim = int(self.encoder.num_features)
        self.num_patch_tokens = self.grid_height * self.grid_width
        self._token_layout: TokenLayout | None = None
        self.base_checkpoint_sha256 = sha256_file(self.jepa_checkpoint)
        self.dinov3_checkpoint_sha256 = sha256_file(self.dinov3_checkpoint)
        self._validate_checkpoint_parameters()
        self.requires_grad_(False)
        self.eval()

    @property
    def token_layout(self) -> TokenLayout:
        if self._token_layout is None:
            raise RuntimeError("Token layout is established by the first encode_prefix call")
        return self._token_layout

    def train(self, mode: bool = True) -> JEPAWMDroidBackend:
        del mode
        super().train(False)
        return self

    def _validate_checkpoint_parameters(self) -> None:
        checkpoint = torch.load(self.jepa_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("predictor"), dict):
            raise RuntimeError(f"JEPA-WM checkpoint has no predictor state dict: {self.jepa_checkpoint}")
        predictor_state = {
            key.removeprefix("module."): value for key, value in checkpoint["predictor"].items()
        }
        key_mapping = {
            "state_encoder.weight": "proprio_encoder.weight",
            "state_encoder.bias": "proprio_encoder.bias",
        }
        predictor_state = {key_mapping.get(key, key): value for key, value in predictor_state.items()}
        self._validate_module_state("predictor", self.video_model.predictor, predictor_state)
        for checkpoint_key, module in (
            ("action_encoder", self.video_model.action_encoder),
            ("proprio_encoder", self.video_model.proprio_encoder),
        ):
            state = checkpoint.get(checkpoint_key)
            if module is None:
                if state not in (None, {}):
                    raise RuntimeError(
                        f"Checkpoint contains {checkpoint_key}, but the official configured model does not"
                    )
            else:
                if not isinstance(state, dict):
                    raise RuntimeError(f"Checkpoint is missing required {checkpoint_key} state")
                cleaned = {key.removeprefix("module."): value for key, value in state.items()}
                self._validate_module_state(checkpoint_key, module, cleaned)
        del checkpoint

    @staticmethod
    def _validate_module_state(name: str, module: nn.Module, state: dict[str, Tensor]) -> None:
        expected_parameters = dict(module.named_parameters())
        missing = sorted(set(expected_parameters).difference(state))
        unexpected = sorted(set(state).difference(module.state_dict()))
        mismatched = [
            f"{key}: checkpoint={tuple(state[key].shape)}, model={tuple(parameter.shape)}"
            for key, parameter in expected_parameters.items()
            if key in state and tuple(state[key].shape) != tuple(parameter.shape)
        ]
        if missing or unexpected or mismatched:
            raise RuntimeError(
                f"JEPA-WM {name} checkpoint does not completely match the official model: "
                f"missing={missing}, unexpected={unexpected}, shape_mismatches={mismatched}"
            )

    def _normalize_images(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Images must have shape [B,T,3,H,W], received {tuple(images.shape)}")
        values = images.to(device=self.device, non_blocking=True)
        if values.dtype == torch.uint8:
            values = values.float().div(255.0)
        else:
            values = values.float()
            minimum = float(values.detach().amin().cpu())
            maximum = float(values.detach().amax().cpu())
            if minimum < 0.0 or maximum > 1.0:
                raise ValueError(
                    f"Floating RGB input must be in [0,1] before official preprocessing; "
                    f"found range [{minimum}, {maximum}]"
                )
        normalized = self.preprocessor.transform(values)
        expected = (images.shape[0], images.shape[1], 3, self.image_size, self.image_size)
        if tuple(normalized.shape) != expected:
            raise RuntimeError(
                f"Official JEPA-WM preprocessing returned {tuple(normalized.shape)}, expected {expected}"
            )
        return normalized

    def encode_prefix(self, images: Tensor) -> Tensor:
        normalized = self._normalize_images(images)
        batch, sequence_length = normalized.shape[:2]
        flattened = rearrange(normalized, "b t c h w -> (b t) c h w")
        tokens, grid = self.encoder.prepare_tokens_with_masks(flattened)
        grid_height, grid_width = (int(grid[0]), int(grid[1]))
        if (grid_height, grid_width) != (self.grid_height, self.grid_width):
            raise RuntimeError(
                f"DINOv3 patch grid mismatch: expected {(self.grid_height, self.grid_width)}, "
                f"found {(grid_height, grid_width)} for images {tuple(images.shape)}"
            )
        patch_tokens = grid_height * grid_width
        prefix_tokens = tokens.shape[1] - patch_tokens
        if prefix_tokens < 1:
            raise RuntimeError(
                f"DINOv3 token layout has no prefix tokens: total={tokens.shape[1]}, patches={patch_tokens}"
            )
        layout = TokenLayout(
            total_tokens=int(tokens.shape[1]),
            prefix_tokens=int(prefix_tokens),
            patch_tokens=patch_tokens,
            grid_height=grid_height,
            grid_width=grid_width,
            token_dim=int(tokens.shape[2]),
        )
        if self._token_layout is not None and layout != self._token_layout:
            raise RuntimeError(f"DINOv3 token layout changed from {self._token_layout} to {layout}")
        self._token_layout = layout
        rope = self.encoder.rope_embed(H=grid_height, W=grid_width)
        for block in self.encoder.blocks[:-1]:
            tokens = block(tokens, rope)
        return tokens.reshape(batch, sequence_length, layout.total_tokens, layout.token_dim)

    def encode_from_prefix(
        self,
        prefix_tokens: Tensor,
        method: PEFTMethod,
        batch_size: int,
        sequence_length: int,
    ) -> Tensor:
        if prefix_tokens.ndim != 4:
            raise ValueError(
                f"Prefix tokens must have shape [B,T,N,D], received {tuple(prefix_tokens.shape)}"
            )
        if tuple(prefix_tokens.shape[:2]) != (batch_size, sequence_length):
            raise ValueError(
                f"Prefix batch/time mismatch: declared {(batch_size, sequence_length)}, "
                f"received {tuple(prefix_tokens.shape[:2])}"
            )
        total_tokens = int(prefix_tokens.shape[2])
        prefix_count = total_tokens - self.num_patch_tokens
        if prefix_count < 1 or prefix_tokens.shape[3] != self.token_dim:
            raise ValueError(
                f"Prefix layout mismatch: expected D={self.token_dim}, P={self.num_patch_tokens}; "
                f"received {tuple(prefix_tokens.shape)}"
            )
        prefix = prefix_tokens[:, :, :prefix_count]
        patches = prefix_tokens[:, :, prefix_count:]
        adapted_patches = method.apply_patch_tokens(patches)
        if adapted_patches.shape != patches.shape:
            raise RuntimeError(
                f"Method {method.method_name} changed patch shape from {tuple(patches.shape)} "
                f"to {tuple(adapted_patches.shape)}"
            )
        tokens = torch.cat((prefix, adapted_patches), dim=2)
        flattened = tokens.reshape(batch_size * sequence_length, total_tokens, self.token_dim)
        rope = self.encoder.rope_embed(H=self.grid_height, W=self.grid_width)
        final = self.last_block(flattened, rope)
        if self.encoder.untie_cls_and_patch_norms or self.encoder.untie_global_and_local_cls_norm:
            prefix_norm = self.encoder.cls_norm if self.encoder.untie_cls_and_patch_norms else self.final_norm
            normalized_prefix = prefix_norm(final[:, :prefix_count])
            normalized_patches = self.final_norm(final[:, prefix_count:])
            final = torch.cat((normalized_prefix, normalized_patches), dim=1)
        else:
            final = self.final_norm(final)
        patches = final[:, prefix_count:]
        expected = (batch_size, sequence_length, self.num_patch_tokens, self.token_dim)
        patches = patches.reshape(expected)
        if tuple(patches.shape) != expected:
            raise RuntimeError(f"Final visual latent shape {tuple(patches.shape)} does not match {expected}")
        return patches

    def encode_images(
        self,
        images: Tensor,
        method: PEFTMethod,
        batch_size: int,
        sequence_length: int,
    ) -> Tensor:
        if tuple(images.shape[:2]) != (batch_size, sequence_length):
            raise ValueError(
                f"Image batch/time mismatch: declared {(batch_size, sequence_length)}, "
                f"received {tuple(images.shape[:2])}"
            )
        prefix = self.encode_prefix(images)
        return self.encode_from_prefix(prefix, method, batch_size, sequence_length)

    def predict(self, context_latents: Tensor, actions: Tensor) -> Tensor:
        if context_latents.ndim != 4 or context_latents.shape[2:] != (
            self.num_patch_tokens,
            self.token_dim,
        ):
            raise ValueError(
                f"Context latents must be [B,T,{self.num_patch_tokens},{self.token_dim}], "
                f"received {tuple(context_latents.shape)}"
            )
        if actions.ndim != 3 or actions.shape[:2] != context_latents.shape[:2] or actions.shape[-1] != 7:
            raise ValueError(
                f"Actions must be [B,T,7] matching context {tuple(context_latents.shape[:2])}, "
                f"received {tuple(actions.shape)}"
            )
        visual = rearrange(
            context_latents,
            "b t (h w) d -> b t 1 h w d",
            h=self.grid_height,
            w=self.grid_width,
        )
        encoded_actions = self.video_model.encode_act(actions)
        predicted, _, _ = self.video_model.forward_pred(visual, encoded_actions, None)
        return rearrange(predicted, "b t 1 h w d -> b t (h w) d")

    def planning_latents(self, patch_latents: Tensor) -> Tensor:
        if patch_latents.ndim != 4:
            raise ValueError(f"Planning latents require [B,T,P,D], received {tuple(patch_latents.shape)}")
        return rearrange(
            patch_latents,
            "b t (h w) d -> b t 1 h w d",
            h=self.grid_height,
            w=self.grid_width,
        )
