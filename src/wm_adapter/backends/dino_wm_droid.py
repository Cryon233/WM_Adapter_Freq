from __future__ import annotations

import os
from pathlib import Path

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch import Tensor

from wm_adapter.adapters.base import PEFTMethod
from wm_adapter.backends.jepa_wm_droid import (
    JEPAWMDroidBackend,
    TokenLayout,
    _prepend_upstream_path,
)
from wm_adapter.utils.checkpoints import (
    git_commit,
    sha256_file,
    verify_upstream_commits,
)
from wm_adapter.utils.reproducibility import resolve_path


DINOV2_UPSTREAM_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


class DinoWMDroidBackend(JEPAWMDroidBackend):
    """Official JEPA-WMs DINO-WM DROID model with a local DINOv2 encoder."""

    def __init__(
        self,
        *,
        third_party_root: str | Path,
        dino_wm_checkpoint: str | Path,
        dinov2_checkpoint: str | Path,
        dinov2_root: str | Path,
        official_planning_config: str | Path,
        device: torch.device | str,
        planning_tag: str | None = None,
        planning_subtask: str | None = None,
    ) -> None:
        # This initializer deliberately does not call JEPAWMDroidBackend.__init__:
        # the official DINO-WM uses DINOv2-S/14 rather than DINOv3-L/16.
        torch.nn.Module.__init__(self)
        self.backend_name = "dino_wm_droid"
        self.device = torch.device(device)
        self.third_party_root = resolve_path(third_party_root)
        pinned_environment_commits = verify_upstream_commits(
            self.third_party_root
        )
        self.jepa_repo = self.third_party_root / "jepa-wms"
        self.dinov2_repo = resolve_path(dinov2_root)
        self.dinov3_repo = self.third_party_root / "dinov3"
        _prepend_upstream_path(self.jepa_repo)

        self.jepa_checkpoint = resolve_path(dino_wm_checkpoint)
        self.dino_wm_checkpoint = self.jepa_checkpoint
        self.dinov2_checkpoint = resolve_path(dinov2_checkpoint)
        self.official_planning_config = resolve_path(official_planning_config)
        for label, path in (
            ("DINO-WM DROID checkpoint", self.dino_wm_checkpoint),
            ("DINOv2 ViT-S/14 checkpoint", self.dinov2_checkpoint),
            ("official DINO-WM planning config", self.official_planning_config),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if not (self.dinov2_repo / "hubconf.py").is_file():
            raise FileNotFoundError(
                f"Official DINOv2 source checkout is incomplete: {self.dinov2_repo}"
            )
        actual_dinov2_commit = git_commit(self.dinov2_repo)
        if actual_dinov2_commit != DINOV2_UPSTREAM_COMMIT:
            raise RuntimeError(
                "DINOv2 upstream commit mismatch: "
                f"expected={DINOV2_UPSTREAM_COMMIT}, actual={actual_dinov2_commit}, "
                f"path={self.dinov2_repo}"
            )

        os.environ["JEPAWM_HOME"] = str(self.third_party_root)
        os.environ["JEPAWM_DINOV2_HOME"] = str(self.dinov2_repo)
        os.environ["JEPAWM_DINOV2_WEIGHTS"] = str(self.dinov2_checkpoint)

        planning = OmegaConf.load(self.official_planning_config)
        if planning_tag is not None:
            planning.tag = planning_tag
        if planning_subtask is not None:
            planning.task_specification.env.subtask = planning_subtask
        self.official_planning_template = planning
        data_config = OmegaConf.to_container(
            planning.model_kwargs.data, resolve=True
        )
        augmentation_config = OmegaConf.to_container(
            planning.model_kwargs.data_aug, resolve=True
        )
        model_config = OmegaConf.to_container(
            planning.model_kwargs.pretrain_kwargs, resolve=True
        )
        wrapper_config = OmegaConf.to_container(
            planning.model_kwargs.wrapper_kwargs, resolve=True
        )
        if not all(
            isinstance(value, dict)
            for value in (
                data_config,
                augmentation_config,
                model_config,
                wrapper_config,
            )
        ):
            raise TypeError(
                f"Official DINO-WM config has invalid mappings: {self.official_planning_config}"
            )
        self.model_config = model_config
        self.data_config = data_config
        self.augmentation_config = augmentation_config
        visual_config = model_config["visual_encoder"]
        predictor_config = model_config["predictor"]
        if visual_config.get("enc_type") != "dino" or visual_config.get(
            "enc_version"
        ) != "dinov2_vits14":
            raise RuntimeError(
                "Official DINO-WM DROID config must use dinov2_vits14: "
                f"visual_encoder={visual_config}"
            )
        if predictor_config.get("pred_type") != "dino_wm":
            raise RuntimeError(
                "Official DINO-WM DROID config must use pred_type=dino_wm: "
                f"predictor={predictor_config}"
            )

        from app.plan_common.datasets.preprocessor import Preprocessor
        from app.plan_common.datasets.transforms import (
            make_inverse_transforms,
            make_transforms,
        )
        from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import (
            init_module,
        )

        self.image_size = int(data_config["img_size"])
        normalize = augmentation_config["normalize"]
        transform = make_transforms(
            img_size=self.image_size,
            normalize=normalize,
            random_horizontal_flip=False,
            random_resize_aspect_ratio=(1.0, 1.0),
            random_resize_scale=(1.0, 1.0),
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
        )
        inverse_transform = make_inverse_transforms(
            img_size=self.image_size, normalize=normalize
        )
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
            folder=self.dino_wm_checkpoint.parent,
            checkpoint=self.dino_wm_checkpoint.name,
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
                "Official DINO-WM encoder wrapper has no DINOv2 base model: "
                f"{type(self.encoder_wrapper).__name__}"
            )
        self.encoder = self.encoder_wrapper.base_model
        if not hasattr(self.encoder, "prepare_tokens_with_masks") or not hasattr(
            self.encoder, "blocks"
        ):
            raise TypeError(
                f"Unsupported DINOv2 encoder: {type(self.encoder).__name__}"
            )
        if len(self.encoder.blocks) < 2:
            raise ValueError(
                f"DINOv2 encoder needs at least two blocks, found {len(self.encoder.blocks)}"
            )
        self.num_encoder_blocks = len(self.encoder.blocks)
        self.last_block = self.encoder.blocks[-1]
        self.final_norm = self.encoder.norm
        patch_size = self.encoder.patch_size
        if isinstance(patch_size, (tuple, list)):
            if len(patch_size) != 2 or patch_size[0] != patch_size[1]:
                raise RuntimeError(f"Unsupported DINOv2 patch size: {patch_size}")
            patch_size = patch_size[0]
        self.patch_size = int(patch_size)
        if self.image_size % self.patch_size:
            raise RuntimeError(
                f"DINO-WM image size {self.image_size} is not divisible by patch size {self.patch_size}"
            )
        self.grid_height = self.image_size // self.patch_size
        self.grid_width = self.image_size // self.patch_size
        self.token_dim = int(self.encoder.num_features)
        self.num_patch_tokens = self.grid_height * self.grid_width
        self._token_layout: TokenLayout | None = None
        self.base_checkpoint_sha256 = sha256_file(self.dino_wm_checkpoint)
        self.dinov2_checkpoint_sha256 = sha256_file(self.dinov2_checkpoint)
        # Kept as a schema alias for existing v2 readers; new metadata also records
        # the correctly named encoder_checkpoint_sha256 field.
        self.dinov3_checkpoint_sha256 = self.dinov2_checkpoint_sha256
        self.encoder_checkpoint_sha256 = self.dinov2_checkpoint_sha256
        self.encoder_name = "dinov2_vits14"
        self.predictor_depth = int(predictor_config["pred_depth"])
        self.upstream_commits = {
            **pinned_environment_commits,
            "dinov2": actual_dinov2_commit,
        }
        self._validate_checkpoint_parameters()
        self.requires_grad_(False)
        self.eval()
        self._predictor_compiled = False

    def train(self, mode: bool = True) -> DinoWMDroidBackend:
        del mode
        torch.nn.Module.train(self, False)
        return self

    def _prepare_normalized_tokens(
        self, normalized: Tensor
    ) -> tuple[Tensor, int, int, TokenLayout, None]:
        if normalized.ndim != 5 or normalized.shape[2:] != (
            3,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "Preprocessed DINO-WM images must have shape "
                f"[B,T,3,{self.image_size},{self.image_size}], received {tuple(normalized.shape)}"
            )
        batch, sequence_length = normalized.shape[:2]
        flattened = rearrange(normalized, "b t c h w -> (b t) c h w")
        tokens = self.encoder.prepare_tokens_with_masks(flattened, masks=None)
        patch_tokens = self.grid_height * self.grid_width
        prefix_tokens = int(tokens.shape[1]) - patch_tokens
        if prefix_tokens < 1:
            raise RuntimeError(
                "DINOv2 token layout lacks the CLS/prefix token: "
                f"total={tokens.shape[1]}, patches={patch_tokens}"
            )
        layout = TokenLayout(
            total_tokens=int(tokens.shape[1]),
            prefix_tokens=prefix_tokens,
            patch_tokens=patch_tokens,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            token_dim=int(tokens.shape[2]),
        )
        if self._token_layout is not None and layout != self._token_layout:
            raise RuntimeError(
                f"DINOv2 token layout changed from {self._token_layout} to {layout}"
            )
        self._token_layout = layout
        return tokens, batch, sequence_length, layout, None

    def _encode_normalized_until_site(
        self, normalized: Tensor, site_index: int
    ) -> Tensor:
        if not 0 <= site_index <= self.num_encoder_blocks:
            raise ValueError(
                f"site_index must be in [0,{self.num_encoder_blocks}], received {site_index}"
            )
        tokens, batch, sequence_length, layout, _ = self._prepare_normalized_tokens(
            normalized
        )
        for block_index in range(site_index):
            tokens = self.encoder.blocks[block_index](tokens)
        return tokens.reshape(
            batch, sequence_length, layout.total_tokens, layout.token_dim
        )

    def _final_patch_norm(self, tokens: Tensor, prefix_count: int) -> Tensor:
        del prefix_count
        return self.final_norm(tokens)

    def encode_from_site(
        self,
        tokens: Tensor,
        start_site_index: int,
        method: PEFTMethod,
    ) -> Tensor:
        if tokens.ndim != 4:
            raise ValueError(
                f"Site tokens must be [B,T,N,D], received {tuple(tokens.shape)}"
            )
        if not 0 <= start_site_index < self.num_encoder_blocks:
            raise ValueError(
                f"start_site_index must be in [0,{self.num_encoder_blocks - 1}], received {start_site_index}"
            )
        batch_size, sequence_length, total_tokens, dimension = tokens.shape
        prefix_count = total_tokens - self.num_patch_tokens
        if prefix_count < 1 or dimension != self.token_dim:
            raise ValueError(
                "DINOv2 site-token layout mismatch: "
                f"expected D={self.token_dim}, P={self.num_patch_tokens}, received={tuple(tokens.shape)}"
            )
        sites = method.adapter_site_indices(self.num_encoder_blocks)
        if tuple(sorted(set(sites))) != sites:
            raise RuntimeError(
                f"Method {method.method_name} returned invalid adapter sites {sites}"
            )
        missed = [site for site in sites if site < start_site_index]
        if missed:
            raise RuntimeError(
                f"Cannot start at block {start_site_index}; adapter sites {missed} would be skipped"
            )
        flattened = tokens.reshape(
            batch_size * sequence_length, total_tokens, self.token_dim
        )
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
            flattened = self.encoder.blocks[block_index](flattened)
        final = self.final_norm(flattened)
        return final[:, prefix_count:].reshape(
            batch_size,
            sequence_length,
            self.num_patch_tokens,
            self.token_dim,
        )
