# Sequence-Stable Adaptive DCT Adapter

本项目为 `stable-worldmodel` 的 TwoRoom 世界模型提供序列稳定的频率域视觉适配。支持两种基础架构：

- PreJEPA / DINO-WM：DINOv2-Small 输出 256 个 patch latent，每个 latent 为 384 维。
- LeWM：Tiny ViT 的 CLS token 经原 projector 得到 192 维 global latent。

两种架构使用同一个 `SequenceStableAdaptiveDCTAdapter` 模块类和同一训练目标，但分别训练、分别保存 Adapter 权重。Adapter 位于视觉 ViT 的最后一个 Transformer block 之前：

```text
pixels
→ patch embedding
→ 前 L-1 个 Transformer blocks
→ Sequence-Stable Adaptive DCT Adapter
→ 最后一个 Transformer block
→ final norm
→ 世界模型原 latent readout
```

Adapter 只处理 16×16 patch tokens，CLS token 不变。前三帧 context 共享一个频率 mask，单帧 goal 独立生成 mask；预测出的 future latent 不会再次进入 Adapter。

## 目录

```text
configs/
  adapter/   # 共用 Adapter 结构
  cache/     # 两种后端的 feature cache 配置
  train/     # 两种后端的 Adapter 训练配置
  plan/      # 两种后端的 TwoRoom MPC 配置
src/wm_adapter_freq/
  adapters/  # 正交二维 DCT 与最终 Adapter
  encoders/  # DINOv2、LeWM Tiny ViT 的显式拆分
  backends/  # 两种世界模型的 latent/predictor 接口
  data/      # 外观扰动、paired windows、HDF5 cache
  models/    # 上游规划兼容的 adapted models
  objectives/
  training/
  planning/
  io/
scripts/
  build_feature_cache.py
  train_adapter.py
  plan.py
```

## 环境与安装

```bash
cd ~/control-frequency-wm
conda activate wm
pip install -e third_party/stable-worldmodel
pip install -e .
```

当前代码针对仓库固定的 `stable-worldmodel` 提交和环境中安装的 `stable-pretraining 0.1.8` 实现。项目不会复制或修改上游源码。

## 基础 checkpoint

基础 checkpoint 必须采用 `stable_worldmodel.wm.utils.load_pretrained` 支持的格式：checkpoint 目录中包含一个 `.pt` 权重文件和 `config.json`。PreJEPA checkpoint 必须使用 `facebook/dinov2-small`/`dinov2_small` 视觉骨干并包含 action、proprio extra encoders；LeWM checkpoint 必须使用 `stable_pretraining.backbone.utils.vit_hf(size=tiny, patch_size=14, image_size=224)`。

默认引用为：

- PreJEPA：`base_model_ref: tworoom_prejepa`
- LeWM：`base_model_ref: tworoom_lewm`

按实际 checkpoint 名称或路径修改对应的 cache、train、plan 配置，或者在命令末尾使用 Hydra 覆盖，例如 `base_model_ref=/path/to/checkpoint_dir`。TwoRoom 数据集默认引用为 `tworoom.h5`。

## 1. 构建 feature cache

```bash
python scripts/build_feature_cache.py \
    --config-name prejepa_tworoom

python scripts/build_feature_cache.py \
    --config-name lewm_tworoom
```

默认输出：

- `data/features/prejepa_tworoom.h5`
- `data/features/lewm_tworoom.h5`

cache 使用 float16、chunked HDF5 和 LZF 压缩。每个 clean 四帧物理窗口只读取一次，再生成 photometric、background texture、palette shift、composed 四个序列级外观视图。

## 2. 训练 Adapter

```bash
python scripts/train_adapter.py \
    --config-name prejepa_tworoom

python scripts/train_adapter.py \
    --config-name lewm_tworoom
```

默认只保存最终 Adapter：

- `checkpoints/adapters/prejepa_tworoom.pt`
- `checkpoints/adapters/lewm_tworoom.pt`

基础模型保持冻结。训练使用 clean canonical latent 对齐和原 dynamics predictor 对齐，不使用 SIGReg 或额外辅助损失。

16GB 显卡默认设置：

| 后端 | batch size | gradient accumulation | precision |
|---|---:|---:|---|
| PreJEPA | 4 | 8 | fp16 |
| LeWM | 16 | 4 | bf16 |

## 3. TwoRoom MPC

```bash
python scripts/plan.py \
    --config-name prejepa_tworoom

python scripts/plan.py \
    --config-name lewm_tworoom
```

规划直接使用上游 `CEMSolver`、`WorldModelPolicy` 和 `PlanConfig`，默认 `history_len=3`、`action_block=5`。PreJEPA 的 history 包含 pixels 和 proprio，LeWM 的 history 只包含 pixels；action/proprio normalizer 从配置的数据集重新恢复。

默认规划输出位于：

- `outputs/plan/prejepa_tworoom/`
- `outputs/plan/lewm_tworoom/`

所有数据、feature cache、基础权重、Adapter checkpoint、规划结果和视频均由 `.gitignore` 排除。
