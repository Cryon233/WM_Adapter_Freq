# WM Adapter Freq：ICRA / Transactions 实验设计（简化执行版）

## 最终用户入口

实现完成后，用户只需要三个命令：

1. 全流程自检：

   `cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/test_full_pipeline.sh`

2. 一键运行全部正式训练和评测：

   `cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/run_all_paper_experiments.sh`

3. 查看进度：

   `cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/watch_all_paper_experiments.sh`

不要求用户操作 dry-run、manifest 或单独拼接实验命令。

## ICRA 推荐实验规模

### 主闭环规划

子任务：

- reach
- reach-pick
- place
- reach-pick-place

方法：

- base
- dct_adapter
- token_mlp
- lora

域：

- clean
- ood

每组 32 episodes，seed 42。现有 place 50-episode 结果复用，统一主表取前 32。

### 多 seed

任务：

- place
- reach-pick-place

方法：

- base
- dct_adapter

域：

- clean
- ood

seeds：

- 7
- 42
- 2026

每组 32 episodes。

### OOD severity

任务：

- place
- reach-pick-place

方法：

- base
- dct_adapter

severity：

- 0.5
- 1.0
- 1.5

每组 20 episodes。

### DCT 消融

离线完整消融：

- full
- framewise_mask
- static_low_rank_mask
- no_rms_norm
- canonical_only
- dynamics_only
- rank2
- rank4
- rank8
- rank16

闭环关键消融：

- full
- framewise_mask
- static_low_rank_mask
- canonical_only

任务：

- place
- reach-pick-place

域：

- ood

每组 20 episodes。

### 世界模型直接指标

- clean / OOD canonical latent MSE
- one-step rollout MSE
- two-step rollout MSE
- three-step rollout MSE
- shuffled-action MSE
- zero-action MSE
- action-shuffle gap
- zero-action gap
- OOD / Clean degradation ratio

### 统计

- Wilson 95% CI
- paired bootstrap 95% CI
- McNemar exact test
- Holm correction
- seed mean/std
- paired episode identity validation

## Transactions 扩展

ICRA 套件之外，再增加：

- 真实机器人 3 tasks × Base/DCT × 3 visual conditions × 20 trials；或
- 第二模拟环境至少 3 tasks × 4 methods × Clean/OOD × 30 episodes；
- 5 种独立视觉 corruption family；
- 3 个训练 seed；
- 500 / 1000 / 2000 windows 数据规模曲线；
- full fine-tuning 或 last-block fine-tuning baseline。
