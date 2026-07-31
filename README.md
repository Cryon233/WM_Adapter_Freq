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

本地 checkpoint 目录必须包含 `config.json` 和恰好一个 `.pt` 权重文件：

```text
checkpoint_dir/
├── weights_epoch_10.pt
└── config.json
```

推荐把具体 `.pt` 绝对路径直接作为 `base_model_ref`；权重后缀必须严格使用小写 `.pt`，同目录必须存在 `config.json`。目录引用必须恰好包含一个 `.pt` 文件，如果目录中有多个 `.pt`，必须指定具体文件。本地相对路径必须以 `./`、`../` 或 `~` 开头；裸名称 `tworoom_prejepa` 按上游 checkpoint cache 名称解析，裸名称 `tworoom_prejepa.pt` 按上游 checkpoint cache 文件解析，只有 `./tworoom_prejepa.pt` 才表示当前目录文件。`owner/repo` 形式表示 Hugging Face 引用。

| 引用 | 解析位置 |
|---|---|
| `/home/user/model/weights.pt` | 本地绝对路径 |
| `./model/weights.pt` | 当前目录相对路径 |
| `~/model/weights.pt` | 用户目录路径 |
| `tworoom_prejepa` | 上游 checkpoint cache |
| `tworoom_prejepa.pt` | 上游 checkpoint cache 文件 |
| `owner/repo` | Hugging Face |

以下环境变量仍然只是格式示例，必须替换为本机真实存在的文件，并在使用前确认变量非空：

```bash
export PREJEPA_CKPT="$HOME/checkpoints/tworoom_prejepa/weights_epoch_10.pt"
export LEWM_CKPT="$HOME/checkpoints/tworoom_lewm/weights_epoch_10.pt"
```

替换为真实路径后，先检查变量、权重文件和同目录配置，再构建 cache：

```bash
test -n "$PREJEPA_CKPT" || {
    echo "PREJEPA_CKPT is not set"
    exit 1
}

test -f "$PREJEPA_CKPT" || {
    echo "Checkpoint does not exist: $PREJEPA_CKPT"
    exit 1
}

test -f "$(dirname "$PREJEPA_CKPT")/config.json" || {
    echo "config.json is missing beside checkpoint"
    exit 1
}

python scripts/build_feature_cache.py \
    --config-name prejepa_tworoom \
    base_model_ref="$PREJEPA_CKPT"
```

环境变量和 `~` 会在解析前展开。空引用、未定义的环境变量、不存在的明确本地路径、非 `.pt` 本地文件、缺少或包含多个 `.pt` 的目录，以及缺少 `config.json` 的本地 checkpoint 都会在任何 Hugging Face 联网前报错。PreJEPA checkpoint 必须包含 DINOv2-Small backbone、predictor 以及 action/proprio extra encoders；LeWM checkpoint 必须包含 Tiny ViT、原 projector、action encoder、predictor 和 pred projection。

默认引用为 `tworoom_prejepa` 和 `tworoom_lewm`，它们作为上游 checkpoint cache 短名称解析；`owner/repo` 形式仍按上游 Hugging Face 规则解析。可在相应命令后用 `base_model_ref="$PREJEPA_CKPT"` 或 `base_model_ref="$LEWM_CKPT"` 覆盖。构建 cache、训练和规划必须引用同一基础 checkpoint：cache 和 Adapter checkpoint 会记录权重、配置及固定上游 commit 的组合 SHA256，加载时不一致会直接拒绝组合。

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

每个 clean 四帧物理窗口只读取一次，并从同一轨迹生成 photometric、background texture、palette shift 和 composed 四个序列级外观视图。cache 配置中的 `seed: 42` 用于生成这些训练 appearance views。

Adapter cache 与最终规划评估采用 episode-disjoint 协议。`deterministic_episode_partition_v1` 使用 split seed 42 确定性打乱由 `dataset.lengths` 定义的 episode 索引，默认前 `floor(80%)` 为 Adapter train partition，其余约 20% 只用于最终规划评估。窗口选择只在 train episode 内执行 `episode_balanced_round_robin_v1`：确定性打乱允许的 episode 和各 episode 内候选窗口，再逐轮从每个 episode 最多取一个窗口，直至达到配置数量。

PreJEPA 和 LeWM 使用完全相同的 split seed、train episode、window seed 和 2,000 个物理窗口。在同一数据集上构建的两个 cache，其 `train_episode_indices_sha256`、`eval_episode_indices_sha256` 和正式物理窗口标识 `selected_window_pairs_sha256` 应相同；`selected_window_indices_sha256` 作为列表位置摘要继续保留。cache 使用 float16 token/latent、chunked HDF5 和 LZF 压缩。分块方式与训练访问一致：clean feature 和 clean target 按单个 window 分块，shifted feature 按单个 window、单个 view 分块；action、proprio、shift type 和 shift seed 使用最多 64 个 window 的 chunk。

默认规模及未压缩 feature 体积估算：

| 后端 | 窗口数 | encoder batch | writer chunk | feature 原始体积 |
|---|---:|---:|---:|---:|
| PreJEPA | 2,000 | 8 | 64 | 约 8.8 GiB |
| LeWM | 2,000 | 32 | 64 | 约 3.7 GiB |

`writer_chunk_size` 只控制这些小字段；大 feature tensor 始终使用上述访问对齐布局。实际文件大小取决于 LZF 压缩率，并另有少量 action、proprio 和 HDF5 元数据开销。cache 根元数据会记录 appearance 信息、episode split 策略与 partition 摘要、窗口选择策略，以及列表位置和实际 `(episode_index, start_step)` 对的 SHA256。action/proprio normalization 继续在完整 source dataset 上拟合，以保持基础世界模型输入契约；episode split 只约束 Adapter 配对训练样本。

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

基础模型保持冻结，loss 只包含 clean canonical latent 对齐和原 dynamics predictor 对齐。基础世界模型本身仍是 clean-domain 模型；80/20 episode split 约束的是 Adapter 配对训练与最终规划评估。cache 使用的 action/proprio z-score 统计随 Adapter checkpoint 保存，规划阶段直接恢复同一份统计，不从评估数据重新拟合。Adapter checkpoint 同时记录训练 cache 的 appearance、episode partition 和窗口选择摘要，不保存完整 episode 或窗口列表。

16GB 显卡默认配置：

| 后端 | batch size | gradient accumulation | precision |
|---|---:|---:|---|
| PreJEPA | 4 | 8 | fp16 |
| LeWM | 16 | 4 | bf16 |

## TwoRoom clean 与在线 OOD MPC

规划继续使用上游 `CEMSolver`、`WorldModelPolicy` 和 `PlanConfig`。默认 `history_len=3`、`action_block=5`、`horizon=5`、`receding_horizon=1`；PreJEPA history keys 为 pixels/proprio，LeWM 为 pixels。base 对照不挂载 Adapter，adapter 条件使用训练后的 Adapter；两者都从同一 Adapter checkpoint 恢复 fingerprint、normalization 和训练 appearance 元数据。

两个 backend 统一采用 visual-only terminal latent MSE。PreJEPA cost 只比较最后预测时刻的 patch-level pixel latent 与 clean goal pixel latent；current proprio 和 action 仍作为冻结 predictor 的动力学条件。`goal_proprio` 只用于 `_set_goal_state` 设置 TwoRoom 真实环境目标并满足上游接口，不进入 world-model planning cost。LeWM 继续通过 `GoalMSE` 比较 global visual latent。

最终 evaluation protocol 版本为 `4.0`：current-only fixed appearance shift、clean image goal、visual-only terminal latent MSE、goal proprio 不进入 cost、PreJEPA current proprio 仅作 dynamics conditioning、base/adapter 直接对照、episode-disjoint Adapter 训练与评估、held-out episode-balanced evaluation，以及 `low_compute_16gb_v1` CEM 预算。planning 默认 `appearance.seed: 2026`，它与训练 cache 的 seed 42 分离，用来定义默认未见测试域。修改 `appearance.seed` 可以生成其他固定 OOD 域；比较不同 backend、base 和 adapter 时必须使用相同的 `appearance.seed`。

同一次规划运行中的所有环境、episode、history 帧和 current observation 复用同一个 `AppearanceShiftSpec`。appearance shift 在 policy 的 current pixels transform 中、标准 resize 和 ImageNet normalization 之前执行，因此 dataset-driven evaluation 注入的第一帧和后续环境帧都会且只会被处理一次。goal 使用独立的 clean 标准预处理器，不经过 appearance shift。

规划从 Adapter checkpoint 恢复 episode split 配置并重新得到 held-out eval partition，只在其中筛选长度满足 goal offset 的 episode。随后无放回选择不同 eval episode，并从每个 episode 的合法范围随机选择一个 start step；传给上游 `World.evaluate` 的是 dataset episode 索引，而不是数据列中的 ID。`results.json` 会记录完整 episode split 摘要、实际 eval episode 索引、start steps、goal offset、评估预算和 evaluation seed。

面向 RTX 5070 Ti 16GB 的默认 CEM 配置：

| 后端 | batch size | num samples | CEM steps | top-k |
|---|---:|---:|---:|---:|
| PreJEPA | 1 | 128 | 5 | 16 |
| LeWM | 4 | 128 | 5 | 16 |

两个 backend 的每环境搜索预算完全相同。`batch_size` 只表示 CEM 同时处理的环境数量；PreJEPA 每个候选包含 3×256 个 patch tokens，因此使用 1，LeWM 的 global latent 较小，因此保留 4。本次规划协议修改不改变 Adapter checkpoint，不需要重新构建 feature cache 或重新训练 Adapter，已有 checkpoint 可直接用于 protocol v4。

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

输出按模型 variant、appearance domain、protocol 和 evaluation seed 自动隔离：

```text
outputs/plan/prejepa_tworoom/
├── base/
│   ├── clean/protocol_v4/eval_seed42/results.json
│   └── composed_severity1p0_seed2026/protocol_v4/eval_seed42/results.json
└── adapter/
    ├── clean/protocol_v4/eval_seed42/results.json
    └── composed_severity1p0_seed2026/protocol_v4/eval_seed42/results.json

outputs/plan/lewm_tworoom/
├── base/
│   ├── clean/protocol_v4/eval_seed42/results.json
│   └── composed_severity1p0_seed2026/protocol_v4/eval_seed42/results.json
└── adapter/
    ├── clean/protocol_v4/eval_seed42/results.json
    └── composed_severity1p0_seed2026/protocol_v4/eval_seed42/results.json
```

配置中的 `seed` 是 evaluation seed，同时控制 episode/start 选择、CEM 和 Torch 随机数。不同 seed 写入不同目录，不会互相覆盖，例如：

```bash
python scripts/plan.py \
    --config-name prejepa_tworoom \
    model.use_adapter=true \
    appearance.enabled=true \
    seed=42

python scripts/plan.py \
    --config-name prejepa_tworoom \
    model.use_adapter=true \
    appearance.enabled=true \
    seed=43
```

默认 `output.video=false`，因为上游 panel video 显示环境 clean render，而 appearance shift 发生在 policy input 中；`results.json` 的评价使用真实 shifted policy input，当前版本不把模型看到的 OOD 图像写入 panel video。结果还记录 planning objective、planner profile、model variant、基础模型 fingerprint、Adapter checkpoint、训练数据选择摘要、episode split、实际 held-out 评估样本、训练/评估 appearance 域、evaluation protocol version、planning 配置和 CEM 配置。数据、checkpoint 与规划输出均由 `.gitignore` 排除。
