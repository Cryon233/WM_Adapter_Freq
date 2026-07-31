# Sequence-Stable Adaptive DCT Adapter

本项目为 `stable-worldmodel` 的 TwoRoom 世界模型提供序列稳定的频率域视觉适配，支持：

- PreJEPA / DINO-WM：DINOv2-Small 的 256 个 384 维 patch latent。
- LeWM：Tiny ViT 的 CLS token 经原 projector 得到的 192 维 global latent。

两种后端分别训练 Adapter 权重，但共用同一个 `SequenceStableAdaptiveDCTAdapter` 类、数学结构和训练目标。Adapter 位于视觉 ViT 最后一个 Transformer block 之前：

```text
pixels
→ patch embedding
→ 前 L-1 个 Transformer blocks
→ Sequence-Stable Adaptive DCT Adapter
→ 最后一个 Transformer block
→ final norm
→ 世界模型原 latent readout
```

Adapter 只修改 16×16 patch tokens，CLS token 不直接进入 DCT。前三帧 context 使用一个序列共享频率 mask，单帧 goal 独立编码；predictor 产生的 future latent 不再次经过 Adapter。

## 目录

```text
configs/
  adapter/   # 共用 Adapter 结构
  cache/     # 两种后端的配对 feature cache 配置
  train/     # 两种后端的 Adapter 训练配置
  plan/      # TwoRoom clean/OOD MPC 配置
src/wm_adapter_freq/
  adapters/  # 正交二维 DCT 与频率 Adapter
  encoders/  # DINOv2、LeWM ViT 的显式拆分
  backends/  # 两种基础世界模型接口
  data/      # 序列外观扰动、paired windows、HDF5 cache
  envs/      # 原始 render 阶段的在线 OOD wrapper
  models/    # 上游规划兼容的 adapted models
  objectives/
  training/
  planning/
  io/        # Adapter checkpoint 与基础模型 fingerprint
scripts/
  build_feature_cache.py
  train_adapter.py
  plan.py
```

## 环境与安装

项目固定使用 `stable-worldmodel` commit
`73dade035ff789e007194971ca5a59b3c3f77e6b`。重型训练依赖由上游的 `all` extra 提供，本项目本身不会要求 pip 更换 PyTorch、torchvision、transformers 或 CUDA wheel。

```bash
conda activate wm
cd ~/control-frequency-wm
python -m pip install -e "./third_party/stable-worldmodel[all]"
python -m pip install -e .
```

## 基础 checkpoint

本地 checkpoint 目录必须包含：

```text
checkpoint_dir/
├── weights.pt
└── config.json
```

也可把明确的 `.pt` 文件作为 `base_model_ref`，此时同目录仍须包含 `config.json`。PreJEPA checkpoint 必须包含 DINOv2-Small backbone、predictor 以及 action/proprio extra encoders；LeWM checkpoint 必须包含 Tiny ViT、原 projector、action encoder、predictor 和 pred projection。

默认引用为 `tworoom_prejepa` 和 `tworoom_lewm`。可在相应命令后用 `base_model_ref=/path/to/checkpoint_dir` 覆盖。构建 cache、训练和规划必须引用同一基础 checkpoint：cache 和 Adapter checkpoint 会记录权重、配置及固定上游 commit 的组合 SHA256，加载时不一致会直接拒绝组合。

## 构建配对 feature cache

```bash
python scripts/build_feature_cache.py \
    --config-name prejepa_tworoom

python scripts/build_feature_cache.py \
    --config-name lewm_tworoom
```

默认输出：

- `data/features/prejepa_tworoom.h5`
- `data/features/lewm_tworoom.h5`

每个 clean 四帧物理窗口只读取一次，并从同一轨迹生成 photometric、background texture、palette shift 和 composed 四个序列级外观视图。cache 使用 float16 token/latent、chunked HDF5 和 LZF 压缩。

默认规模及未压缩 feature 体积估算：

| 后端 | 窗口数 | encoder batch | writer chunk | feature 原始体积 |
|---|---:|---:|---:|---:|
| PreJEPA | 2,000 | 8 | 8 | 约 8.8 GiB |
| LeWM | 5,000 | 32 | 16 | 约 9.2 GiB |

实际文件大小取决于 LZF 压缩率，并另有少量 action、proprio 和 HDF5 元数据开销。

## 训练 Adapter

```bash
python scripts/train_adapter.py \
    --config-name prejepa_tworoom

python scripts/train_adapter.py \
    --config-name lewm_tworoom
```

默认只保存最终 Adapter：

- `checkpoints/adapters/prejepa_tworoom.pt`
- `checkpoints/adapters/lewm_tworoom.pt`

基础模型保持冻结，loss 只包含 clean canonical latent 对齐和原 dynamics predictor 对齐。cache 使用的 action/proprio z-score 统计随 Adapter checkpoint 保存，规划阶段直接恢复同一份统计，不从评估数据重新拟合。

16GB 显卡默认配置：

| 后端 | batch size | gradient accumulation | precision |
|---|---:|---:|---|
| PreJEPA | 4 | 8 | fp16 |
| LeWM | 16 | 4 | bf16 |

## TwoRoom clean 与在线 OOD MPC

规划继续使用上游 `CEMSolver`、`WorldModelPolicy` 和 `PlanConfig`。默认 `history_len=3`、`action_block=5`、`horizon=5`、`receding_horizon=1`；PreJEPA history keys 为 pixels/proprio，LeWM 为 pixels。

clean 规划：

```bash
python scripts/plan.py \
    --config-name prejepa_tworoom \
    appearance.enabled=false

python scripts/plan.py \
    --config-name lewm_tworoom \
    appearance.enabled=false
```

OOD 规划：

```bash
python scripts/plan.py \
    --config-name prejepa_tworoom \
    appearance.enabled=true \
    appearance.shift_type=composed \
    appearance.severity=1.0

python scripts/plan.py \
    --config-name lewm_tworoom \
    appearance.enabled=true \
    appearance.shift_type=composed \
    appearance.severity=1.0
```

OOD 评估采用 clean-goal 协议：在线 current observation 在 TwoRoom 原始 HWC `uint8` render 阶段施加一个整回合固定的 appearance shift，之后才进入上游 resize 和 ImageNet normalization；dataset 提供的 goal image 保持 clean。shifted current 与 clean goal 都通过同一个训练后 Adapter，映射到 clean canonical latent 空间。wrapper 不改变 state、proprio、goal state、action、reward、碰撞或终止条件。

默认规划结果和视频位于：

- `outputs/plan/prejepa_tworoom/`
- `outputs/plan/lewm_tworoom/`

结果 JSON 同时记录基础模型 fingerprint、Adapter checkpoint、appearance 配置、planning 配置和 CEM 配置。数据、feature cache、基础权重、Adapter checkpoint、规划输出及视频均由 `.gitignore` 排除。
