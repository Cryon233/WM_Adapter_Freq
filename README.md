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
  models/    # 上游规划兼容的 adapted models
  objectives/
  training/
  planning/  # current-only OOD transform 与 MPC policy 构建
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

每个 clean 四帧物理窗口只读取一次，并从同一轨迹生成 photometric、background texture、palette shift 和 composed 四个序列级外观视图。cache 配置中的 `seed: 42` 用于生成这些训练 appearance views。cache 使用 float16 token/latent、chunked HDF5 和 LZF 压缩。分块方式与训练访问一致：clean feature 和 clean target 按单个 window 分块，shifted feature 按单个 window、单个 view 分块；action、proprio、shift type 和 shift seed 使用最多 64 个 window 的 chunk。

默认规模及未压缩 feature 体积估算：

| 后端 | 窗口数 | encoder batch | writer chunk | feature 原始体积 |
|---|---:|---:|---:|---:|
| PreJEPA | 2,000 | 8 | 64 | 约 8.8 GiB |
| LeWM | 5,000 | 32 | 64 | 约 9.2 GiB |

`writer_chunk_size` 只控制这些小字段；大 feature tensor 始终使用上述访问对齐布局。实际文件大小取决于 LZF 压缩率，并另有少量 action、proprio 和 HDF5 元数据开销。cache 根元数据会记录 appearance severity、shift names 和 shift pipeline version。

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

基础模型保持冻结，loss 只包含 clean canonical latent 对齐和原 dynamics predictor 对齐。cache 使用的 action/proprio z-score 统计随 Adapter checkpoint 保存，规划阶段直接恢复同一份统计，不从评估数据重新拟合。Adapter checkpoint 同时记录训练 cache 的 appearance severity、shift names 和 shift pipeline version。

16GB 显卡默认配置：

| 后端 | batch size | gradient accumulation | precision |
|---|---:|---:|---|
| PreJEPA | 4 | 8 | fp16 |
| LeWM | 16 | 4 | bf16 |

## TwoRoom clean 与在线 OOD MPC

规划继续使用上游 `CEMSolver`、`WorldModelPolicy` 和 `PlanConfig`。默认 `history_len=3`、`action_block=5`、`horizon=5`、`receding_horizon=1`；PreJEPA history keys 为 pixels/proprio，LeWM 为 pixels。base 对照使用完全未挂载 Adapter 的原始基础模型，adapter 条件使用训练后的 Adapter；两者都从同一 Adapter checkpoint 恢复 fingerprint、normalization 和训练 appearance 元数据。

最终 OOD 协议为 fixed appearance domain，版本为 `2.0`。planning 默认 `appearance.seed: 2026`，它与训练 cache 的 seed 42 分离，用来定义默认未见测试域。修改 `appearance.seed` 可以生成其他固定 OOD 域；比较不同 backend、base 和 adapter 时必须使用相同的 `appearance.seed`。

同一次规划运行中的所有环境、episode、history 帧和 current observation 复用同一个 `AppearanceShiftSpec`。appearance shift 在 policy 的 current pixels transform 中、标准 resize 和 ImageNet normalization 之前执行，因此 dataset-driven evaluation 注入的第一帧和后续环境帧都会且只会被处理一次。goal 使用独立的 clean 标准预处理器，不经过 appearance shift。

### PreJEPA 四个条件

Base Clean：

```bash
python scripts/plan.py \
    --config-name prejepa_tworoom \
    model.use_adapter=false \
    appearance.enabled=false
```

Base OOD：

```bash
python scripts/plan.py \
    --config-name prejepa_tworoom \
    model.use_adapter=false \
    appearance.enabled=true \
    appearance.shift_type=composed \
    appearance.severity=1.0 \
    appearance.seed=2026
```

Adapter Clean：

```bash
python scripts/plan.py \
    --config-name prejepa_tworoom \
    model.use_adapter=true \
    appearance.enabled=false
```

Adapter OOD：

```bash
python scripts/plan.py \
    --config-name prejepa_tworoom \
    model.use_adapter=true \
    appearance.enabled=true \
    appearance.shift_type=composed \
    appearance.severity=1.0 \
    appearance.seed=2026
```

### LeWM 四个条件

Base Clean：

```bash
python scripts/plan.py \
    --config-name lewm_tworoom \
    model.use_adapter=false \
    appearance.enabled=false
```

Base OOD：

```bash
python scripts/plan.py \
    --config-name lewm_tworoom \
    model.use_adapter=false \
    appearance.enabled=true \
    appearance.shift_type=composed \
    appearance.severity=1.0 \
    appearance.seed=2026
```

Adapter Clean：

```bash
python scripts/plan.py \
    --config-name lewm_tworoom \
    model.use_adapter=true \
    appearance.enabled=false
```

Adapter OOD：

```bash
python scripts/plan.py \
    --config-name lewm_tworoom \
    model.use_adapter=true \
    appearance.enabled=true \
    appearance.shift_type=composed \
    appearance.severity=1.0 \
    appearance.seed=2026
```

最小实验矩阵：

| backend | model | domain |
|---|---|---|
| PreJEPA | base | clean |
| PreJEPA | base | OOD |
| PreJEPA | adapter | clean |
| PreJEPA | adapter | OOD |
| LeWM | base | clean |
| LeWM | base | OOD |
| LeWM | adapter | clean |
| LeWM | adapter | OOD |

输出按模型 variant 和 domain 自动隔离：

```text
outputs/plan/prejepa_tworoom/
├── base/
│   ├── clean/results.json
│   └── composed_severity1p0_seed2026/results.json
└── adapter/
    ├── clean/results.json
    └── composed_severity1p0_seed2026/results.json

outputs/plan/lewm_tworoom/
├── base/
│   ├── clean/results.json
│   └── composed_severity1p0_seed2026/results.json
└── adapter/
    ├── clean/results.json
    └── composed_severity1p0_seed2026/results.json
```

默认 `output.video=false`，因为上游 panel video 显示环境 clean render，而 appearance shift 发生在 policy input 中；`results.json` 的评价使用真实 shifted policy input，当前版本不把模型看到的 OOD 图像写入 panel video。结果还记录 model variant、基础模型 fingerprint、Adapter checkpoint、训练/评估 appearance 域、evaluation protocol version、planning 配置和 CEM 配置。数据、checkpoint 与规划输出均由 `.gitignore` 排除。
