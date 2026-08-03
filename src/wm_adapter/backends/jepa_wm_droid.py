from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor, nn

from wm_adapter.adapters.base import PEFTMethod
from wm_adapter.backends.frozen_projection import frozen_base_projection
from wm_adapter.utils.checkpoints import sha256_file, verify_upstream_commits
from wm_adapter.utils.reproducibility import resolve_path


LOGGER = logging.getLogger(__name__)


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
        planning_tag: str | None = None,
        planning_subtask: str | None = None,
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
        if planning_tag is not None:
            planning.tag = planning_tag
        if planning_subtask is not None:
            planning.task_specification.env.subtask = planning_subtask
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
        self.num_encoder_blocks = len(self.encoder.blocks)
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
        self._predictor_compiled = False

    @property
    def token_layout(self) -> TokenLayout:
        if self._token_layout is None:
            raise RuntimeError("Token layout is established by the first encode_prefix call")
        return self._token_layout

    def train(self, mode: bool = True) -> JEPAWMDroidBackend:
        del mode
        super().train(False)
        return self

    def configure_planning_inference(
        self,
        *,
        inference_precision: str,
        allow_tf32: bool,
        compile_predictor: bool,
    ) -> None:
        if self.device.type == "cuda" and allow_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        LOGGER.info(
            "Planning CUDA settings: inference_precision=%s, allow_tf32=%s",
            inference_precision,
            str(allow_tf32).lower(),
        )

        if compile_predictor and not self._predictor_compiled:
            LOGGER.info(
                "Compiling JEPA-WM predictor with "
                "torch.compile(mode=reduce-overhead)"
            )
            try:
                compiled_predictor = torch.compile(
                    self.video_model.predictor,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=False,
                )
            except Exception as error:
                raise RuntimeError(
                    "Failed to compile the JEPA-WM predictor with "
                    "torch.compile(mode=reduce-overhead)"
                ) from error
            self.video_model.predictor = compiled_predictor
            if self.official_model.model.predictor is not compiled_predictor:
                raise RuntimeError(
                    "Compiled predictor is not shared by backend.video_model and "
                    "backend.official_model.model"
                )
            self._predictor_compiled = True

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
        return self._encode_normalized_prefix(normalized)

    def _encode_normalized_prefix(self, normalized: Tensor) -> Tensor:
        return self._encode_normalized_until_site(
            normalized, self.num_encoder_blocks - 1
        )

    def _prepare_normalized_tokens(
        self, normalized: Tensor
    ) -> tuple[Tensor, int, int, TokenLayout, Tensor]:
        if normalized.ndim != 5 or normalized.shape[2:] != (
            3,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "Preprocessed images must have shape "
                f"[B,T,3,{self.image_size},{self.image_size}], received {tuple(normalized.shape)}"
            )
        batch, sequence_length = normalized.shape[:2]
        flattened = rearrange(normalized, "b t c h w -> (b t) c h w")
        tokens, grid = self.encoder.prepare_tokens_with_masks(flattened)
        grid_height, grid_width = (int(grid[0]), int(grid[1]))
        if (grid_height, grid_width) != (self.grid_height, self.grid_width):
            raise RuntimeError(
                f"DINOv3 patch grid mismatch: expected {(self.grid_height, self.grid_width)}, "
                f"found {(grid_height, grid_width)} for preprocessed images {tuple(normalized.shape)}"
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
        return tokens, batch, sequence_length, layout, rope

    def _encode_normalized_until_site(
        self, normalized: Tensor, site_index: int
    ) -> Tensor:
        if not 0 <= site_index <= self.num_encoder_blocks:
            raise ValueError(
                f"site_index must be in [0,{self.num_encoder_blocks}], "
                f"received {site_index}"
            )
        tokens, batch, sequence_length, layout, rope = (
            self._prepare_normalized_tokens(normalized)
        )
        for block_index in range(site_index):
            tokens = self.encoder.blocks[block_index](tokens, rope)
        return tokens.reshape(
            batch, sequence_length, layout.total_tokens, layout.token_dim
        )

    def encode_until_site(self, images: Tensor, site_index: int) -> Tensor:
        return self._encode_normalized_until_site(
            self._normalize_images(images), site_index
        )

    def _apply_method_site(
        self,
        tokens: Tensor,
        method: PEFTMethod,
        site_index: int,
        batch_size: int,
        sequence_length: int,
        prefix_count: int,
    ) -> Tensor:
        total_tokens = int(tokens.shape[1])
        shaped = tokens.reshape(
            batch_size, sequence_length, total_tokens, self.token_dim
        )
        prefix = shaped[:, :, :prefix_count]
        patches = shaped[:, :, prefix_count:]
        adapted = method.apply_at_site(site_index, patches)
        if adapted.shape != patches.shape:
            raise RuntimeError(
                f"Method {method.method_name} changed patch shape at site "
                f"{site_index}: input={tuple(patches.shape)}, "
                f"output={tuple(adapted.shape)}"
            )
        return torch.cat((prefix, adapted), dim=2).reshape(
            batch_size * sequence_length, total_tokens, self.token_dim
        )

    def _final_patch_norm(self, tokens: Tensor, prefix_count: int) -> Tensor:
        if self.encoder.untie_cls_and_patch_norms or self.encoder.untie_global_and_local_cls_norm:
            prefix_norm = (
                self.encoder.cls_norm
                if self.encoder.untie_cls_and_patch_norms
                else self.final_norm
            )
            return torch.cat(
                (
                    prefix_norm(tokens[:, :prefix_count]),
                    self.final_norm(tokens[:, prefix_count:]),
                ),
                dim=1,
            )
        return self.final_norm(tokens)

    def encode_from_site(
        self,
        tokens: Tensor,
        start_site_index: int,
        method: PEFTMethod,
    ) -> Tensor:
        if tokens.ndim != 4:
            raise ValueError(
                "Site tokens must have shape [B,T,N,D], "
                f"received {tuple(tokens.shape)}"
            )
        if not 0 <= start_site_index < self.num_encoder_blocks:
            raise ValueError(
                f"start_site_index must be in [0,{self.num_encoder_blocks - 1}], "
                f"received {start_site_index}"
            )
        batch_size, sequence_length, total_tokens, dimension = tokens.shape
        prefix_count = total_tokens - self.num_patch_tokens
        if prefix_count < 1 or dimension != self.token_dim:
            raise ValueError(
                f"Site token layout mismatch: expected D={self.token_dim}, "
                f"P={self.num_patch_tokens}, received {tuple(tokens.shape)}"
            )
        sites = method.adapter_site_indices(self.num_encoder_blocks)
        if tuple(sorted(set(sites))) != sites:
            raise RuntimeError(
                f"Method {method.method_name} returned invalid adapter sites {sites}"
            )
        missed = [site for site in sites if site < start_site_index]
        if missed:
            raise RuntimeError(
                f"Cannot encode method {method.method_name} from site "
                f"{start_site_index}; earlier adapter sites would be skipped: {missed}"
            )
        flattened = tokens.reshape(
            batch_size * sequence_length, total_tokens, self.token_dim
        )
        rope = self.encoder.rope_embed(H=self.grid_height, W=self.grid_width)
        site_set = set(sites)
        for block_index in range(start_site_index, self.num_encoder_blocks):
            if block_index in site_set:
                flattened = self._apply_method_site(
                    flattened,
                    method,
                    block_index,
                    batch_size,
                    sequence_length,
                    prefix_count,
                )
            flattened = self.encoder.blocks[block_index](flattened, rope)
        final = self._final_patch_norm(flattened, prefix_count)
        patches = final[:, prefix_count:].reshape(
            batch_size,
            sequence_length,
            self.num_patch_tokens,
            self.token_dim,
        )
        return patches

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
        return self.encode_from_site(
            prefix_tokens,
            self.num_encoder_blocks - 1,
            method,
        )

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
        return self.encode_images_with_method(images, method)

    def encode_images_with_method(
        self, images: Tensor, method: PEFTMethod
    ) -> Tensor:
        if images.ndim != 5:
            raise ValueError(
                f"Images must have shape [B,T,3,H,W], received {tuple(images.shape)}"
            )
        tokens = self.encode_until_site(images, 0)
        return self.encode_from_site(tokens, 0, method)

    def encode_images_frozen_base(self, images: Tensor) -> Tensor:
        from wm_adapter.adapters.base import BaseMethod

        with frozen_base_projection(self):
            return self.encode_images_with_method(
                images, BaseMethod().to(self.device)
            )

    def differentiable_unroll(
        self, context_latents: Tensor, actions: Tensor
    ) -> Tensor:
        if context_latents.ndim != 4 or context_latents.shape[1:] != (
            3,
            self.num_patch_tokens,
            self.token_dim,
        ):
            raise ValueError(
                "Differentiable rollout context must be "
                f"[B,3,{self.num_patch_tokens},{self.token_dim}], "
                f"received {tuple(context_latents.shape)}"
            )
        if actions.ndim != 3 or tuple(actions.shape[1:]) != (3, 7):
            raise ValueError(
                "Differentiable rollout actions must be [B,3,7], "
                f"received {tuple(actions.shape)}"
            )
        visual = rearrange(
            context_latents,
            "b t (h w) d -> b t 1 h w d",
            h=self.grid_height,
            w=self.grid_width,
        )
        context_window = int(self.official_model.ctxt_window)
        if context_window <= 0:
            raise ValueError(
                f"JEPA-WM ctxt_window must be positive, received {context_window}"
            )
        history_steps = max(min(context_window, visual.shape[1]) - 1, 0)
        raw_timeline = actions
        if history_steps:
            raw_timeline = torch.cat(
                (torch.zeros_like(actions[:, :1]).repeat(1, history_steps, 1), actions),
                dim=1,
            )
        action_features = self.video_model.encode_act(raw_timeline)
        predictions: list[Tensor] = []
        for step in range(3):
            action_end = history_steps + step + 1
            visual_window = visual[:, -context_window:]
            action_window = action_features[
                :, max(0, action_end - context_window) : action_end
            ]
            if visual_window.shape[1] != action_window.shape[1]:
                raise RuntimeError(
                    "Differentiable JEPA-WM rollout context mismatch: "
                    f"step={step}, visual={tuple(visual_window.shape)}, "
                    f"actions={tuple(action_window.shape)}"
                )
            predicted, _, _ = self.video_model.forward_pred(
                visual_window, action_window, None
            )
            next_visual = predicted[:, -1:]
            predictions.append(next_visual)
            visual = torch.cat((visual, next_visual), dim=1)
        future = torch.cat(predictions, dim=1)
        return rearrange(future, "b t 1 h w d -> b t (h w) d")

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
