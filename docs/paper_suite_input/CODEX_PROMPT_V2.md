# Codex Prompt V2：实现无 dry-run 的论文实验全流程

仓库：

`/home/zhaoyang/control-frequency-wm`

服务器：

`/data/users/zhaoyanghe/control-frequency-wm`

目标：实现一个面向 ICRA / Transactions 的完整实验系统。用户不需要操作 dry-run、manifest 或逐项命令。最终必须提供：

1. `scripts/test_full_pipeline.sh`
2. `scripts/run_all_paper_experiments.sh`
3. `scripts/watch_all_paper_experiments.sh`

其中前两个是强制最终交付。

---

## 一、不可破坏的正式协议

保留：

- 已完成 `protocol_v2/place/seed_42` 结果；
- methods：base、dct_adapter、token_mlp、lora；
- iterations=15；
- num_samples=300；
- num_elites=10；
- horizon=3；
- num_act_stepped=1；
- history_len=3；
- wrapper ctxt_window=2；
- checkpoint、数据划分、动作限制、success 判定不变；
- 当前 RoboCasa / robosuite / external XML / gripper 补丁；
- 不下载或替换 assets、checkpoint、HDF5；
- 不覆盖旧结果；
- 所有 shell 命令必须支持单物理行执行。

正式实验不得降低 CEM 预算。只有全流程自检允许使用独立的轻量 self-test planning 配置。

---

## 二、公开用户接口必须极简

用户只运行：

### 全流程自检

`cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/test_full_pipeline.sh`

### 全部正式实验

`cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/run_all_paper_experiments.sh`

### 监控

`cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/watch_all_paper_experiments.sh`

不要要求用户使用：

- `--dry-run`
- 手工 manifest
- 手工 phase
- 手工 GPU 分配
- 手工逐项运行

内部可以使用 Python job registry 和状态文件，但这些属于实现细节。

---

## 三、实现全流程自检

新增：

- `scripts/test_full_pipeline.py`
- `scripts/test_full_pipeline.sh`

### 目的

这不是形式上的 import test，而是实际运行一个隔离的最小端到端 pipeline，尽可能在正式长实验之前发现：

- 配置错误；
- 路径错误；
- cache schema 错误；
- adapter shape 错误；
- checkpoint 保存/加载错误；
- clean/OOD domain 错误；
- planning model 接口错误；
- RoboCasa reset/XML 错误；
- results.json 写入错误；
- 分析脚本读取错误。

不能声称数学上保证“绝对无 Bug”。脚本成功的含义是：所有当前覆盖的关键路径都完成了一次真实执行，并通过结构与数值检查。

### 隔离输出

所有自检产物必须写到：

- `storage/self_test/`
- `checkpoints/self_test/`
- `outputs/self_test/`
- `logs/self_test/`

不得使用或覆盖正式 cache、checkpoint、results。

每次开始前可以清理 `self_test` 目录，但不能删除其他目录。

### 自检步骤

1. 检查依赖和资源：
   - CUDA 可用；
   - 4 张 GPU不是硬要求，自检可在1张 GPU顺序运行；
   - JEPA-WM checkpoint；
   - DINOv3 checkpoint；
   - RoboCasa HDF5；
   - RoboCasa assets；
   - 上游源码路径；
   - 当前固定 commit / fingerprint 可读取。

2. 构建微型 feature cache：
   - 16 个 train windows；
   - encoder batch size 2；
   - 独立 cache 路径；
   - 检查 finalized、schema、shape、无 NaN/Inf。

3. 训练三个 adapter：
   - dct_adapter；
   - token_mlp；
   - lora；
   - 每个 1 epoch；
   - batch size 2；
   - gradient accumulation 1；
   - 独立 checkpoint；
   - 检查 checkpoint 可重新加载；
   - 检查 method name、参数量、fingerprint、state dict；
   - 检查训练 loss 为有限值。

4. 离线评测：
   - base；
   - dct_adapter；
   - token_mlp；
   - lora；
   - 至少 4 个 held-out windows；
   - clean 和 OOD；
   - one-step / multi-step 路径至少执行一次；
   - 检查 metrics.json 和 per-window 文件。

5. planning 自检：
   - 使用独立 self-test planning YAML；
   - 只用于自检，允许：
     - iterations=1
     - num_samples=8
     - num_elites=2
     - horizon=1
     - max_episode_steps=5
     - evaluation.num_episodes=1
   - task 使用 place；
   - base、dct_adapter、token_mlp、lora；
   - clean、ood；
   - 共 8 个最小 job；
   - 可以在可用 GPU 上调度；
   - 每个 job 必须真正完成 environment reset、goal encoding、CEM、agent execution 和 results.json 写入。

6. 分析自检：
   - 读取上述最小 results；
   - 生成最小 CSV / Markdown / statistics JSON；
   - 检查字段完整、成功率范围 [0,1]、episode 数匹配。

7. 最终报告：
   - `outputs/self_test/self_test_report.json`
   - `outputs/self_test/self_test_report.md`
   - 每个阶段 PASS / FAIL；
   - 运行时间；
   - artifact 路径；
   - 失败异常和 traceback；
   - 成功时退出码 0；
   - 任意失败退出码非 0。

### shell wrapper

`scripts/test_full_pipeline.sh` 必须：

- `set -euo pipefail`
- 激活 `wm-a100`
- `cd` 到项目根目录
- 设置 `PYTHONUNBUFFERED=1`
- 调用 `python scripts/test_full_pipeline.py`
- 不使用反斜杠多行命令

---

## 四、一键运行全部正式训练与评测

新增：

- `scripts/run_all_paper_experiments.py`
- `scripts/run_all_paper_experiments.sh`

### 运行顺序

自动执行：

1. preflight；
2. 正式 cache 检查或构建；
3. 正式三个 adapter checkpoint 检查或训练；
4. DCT ablation checkpoint 训练；
5. offline world-model metrics；
6. 四任务主闭环 planning；
7. multi-seed planning；
8. severity planning；
9. key ablation planning；
10. 统计分析；
11. 生成论文表格和总结。

### 自动恢复

- 已有且通过完整性检查的 artifact 自动跳过；
- 中断后重新运行同一 shell 即可继续；
- 不要求用户指定 resume；
- 不覆盖旧结果；
- 结果复用必须记录 source path 和 SHA256；
- place seed42 正式 50 episodes 结果复用；
- 主表使用前32个 episode；
- severity=1.0 使用前20个；
- full DCT ablation使用正式 DCT结果；
- 某个 planning job 未完整写 results.json 时，该 job 从头重跑；
- 不需要实现单 job episode 中间恢复。

### GPU 调度

- 自动检测 GPU；
- 默认使用 0,1,2,3；
- 环境变量 `GPUS=0,1,2,3` 可覆盖；
- 每张 GPU 同时最多一个重任务；
- cache 只运行一个；
- 三个正式 adapter 可并行；
- planning job按空闲 GPU 自动提交；
- 失败时停止提交新 job，等待运行中 job 退出，然后整体返回非0；
- 状态写入 `logs/paper_suite/state.json`；
- 所有 JSON 原子写入。

### 正式实验矩阵

#### 主任务

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

每组 32 episodes，seed42。

#### multi-seed

- tasks: place, reach-pick-place
- methods: base, dct_adapter
- domains: clean, ood
- seeds: 7,42,2026
- 32 episodes

#### severity

- tasks: place, reach-pick-place
- methods: base, dct_adapter
- domain: ood
- severity: 0.5,1.0,1.5
- 20 episodes

#### DCT ablation

离线：

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

闭环：

- full
- framewise_mask
- static_low_rank_mask
- canonical_only

tasks：

- place
- reach-pick-place

domain：

- ood

20 episodes。

### shell wrapper

`scripts/run_all_paper_experiments.sh` 必须：

- `set -euo pipefail`
- 激活 `wm-a100`
- `cd` 到项目根目录
- 默认 `GPUS=0,1,2,3`
- 设置：
  - DOWNLOAD_ASSETS=0
  - FORCE_ASSETS=0
  - PYTHONUNBUFFERED=1
- 调用 `python scripts/run_all_paper_experiments.py`
- 不使用反斜杠多行命令

---

## 五、监控

新增：

- `scripts/monitor_all_paper_experiments.py`
- `scripts/watch_all_paper_experiments.sh`

显示：

- 总 phase；
- completed / running / pending / failed；
- GPU util / memory / temperature；
- 当前 job、PID、运行时间；
- planning episode / step / success / last CEM / ETA；
- train epoch / batch / loss；
- offline evaluated windows；
- 最后日志更新时间；
- 总体预计剩余时间。

调用：

`cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/watch_all_paper_experiments.sh`

---

## 六、多任务配置

基于当前完整 `place` model-specific planning config生成：

- reach
- reach-pick
- place
- reach-pick-place

只改变：

- tag
- task_specification.env.subtask

正式 CEM参数完全不变。

对应生成 model config 与 experiment config，避免任何输出覆盖。

---

## 七、DCT 消融实现

在 `SequenceStableAdaptiveDCTAdapter` 中增加：

- temporal_pool: mean | none
- mask_type: adaptive | static_low_rank
- use_rms_norm: bool

默认行为必须与当前 checkpoint兼容。

static mask使用低秩分解：

- channel_factor [D,rank]
- frequency_factor [rank,H,W]

至少一个 factor零初始化，初始严格 identity。

训练器增加：

- canonical_weight
- dynamics_weight

默认均为1.0，禁止同时为0。

---

## 八、离线评测和统计

新增：

- `scripts/evaluate_offline_dynamics.py`
- `scripts/analyze_paper_suite.py`

输出：

- main_results.csv / md / tex
- multiseed.csv
- severity.csv
- ablations.csv
- offline_metrics.csv
- efficiency.csv
- statistics.json
- paper_summary.md

统计包括：

- Wilson CI
- paired bootstrap 10000次
- McNemar exact
- Holm correction
- seed mean/std

配对 identity不一致时禁止做paired test。

---

## 九、最终验收条件

Codex完成后必须实际执行：

1. `bash scripts/test_full_pipeline.sh`
2. 确认退出码0；
3. 展示 `self_test_report.md`；
4. 不启动正式长实验；
5. 展示 `scripts/run_all_paper_experiments.sh` 的最终内容；
6. 给出服务器上唯一正式启动命令：

   `cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/run_all_paper_experiments.sh`

不要再要求用户执行 dry-run 或手工拼接命令。
