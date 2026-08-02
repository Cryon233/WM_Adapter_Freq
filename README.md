# JEPA-WM Parameter-Efficient Adaptation

本项目的当前主实验后端是官方 `facebookresearch/jepa-wms` 发布的 DROID/RoboCasa JEPA-WM：DINOv3 ViT-L/16 编码器、256×256 输入、depth-12 AdaLN predictor。基础视觉编码器、动作条件模块和 predictor 全部冻结，只比较以下四种方法：

- `base`：不训练参数；
- `dct_adapter`：Sequence-Stable Adaptive DCT Adapter；
- `token_mlp`：同插入位置的 rank-8 token bottleneck；
- `lora`：DINOv3 最后一个 block fused QKV 中 Q、V 的 rank-4 LoRA。

旧的 `third_party/stable-worldmodel` 与 `wm_adapter_freq` 源码仍被保留，但不是当前配置或三个核心 CLI 的依赖。

## 固定上游

| 上游 | commit |
| --- | --- |
| `facebookresearch/jepa-wms` | `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` |
| `facebookresearch/dinov3` | `6876159a11b4df116f30f667f8c9888617df0751` |
| `Basile-Terv/robosuite` (`robocasa-dev`) | `9548a5a35bde8eabf47f760802045cca447e9c0c` |
| `Basile-Terv/robocasa` | `2544dc2e38bb44f5ced80fbc91114a2f7934016a` |

代码在加载后端时核对这些 commit，并把 SHA 写入 cache、方法 checkpoint 和 planning `results.json`。期望目录为：

```text
third_party/
├── jepa-wms/
├── dinov3/
├── robosuite/
├── robocasa/
└── stable-worldmodel/   # 仅保留，不参与新主实验
```

上游源码应直接检出到表中 commit（不要复制进 `src/wm_adapter`）：

```bash
git clone https://github.com/facebookresearch/jepa-wms third_party/jepa-wms
git -C third_party/jepa-wms checkout 13cf1d9c7e476f53c17714d2e0f1dc239a883ce0
git clone https://github.com/facebookresearch/dinov3 third_party/dinov3
git -C third_party/dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
git clone --branch robocasa-dev https://github.com/Basile-Terv/robosuite third_party/robosuite
git -C third_party/robosuite checkout 9548a5a35bde8eabf47f760802045cca447e9c0c
git clone https://github.com/Basile-Terv/robocasa third_party/robocasa
git -C third_party/robocasa checkout 2544dc2e38bb44f5ced80fbc91114a2f7934016a
```

按官方依赖顺序安装；RoboCasa kitchen assets 约 20 GB，需要用户按官方许可和说明手动下载，本项目不会自动下载大文件：

```bash
conda activate wm
cd ~/control-frequency-wm
python -m pip install -e third_party/jepa-wms
python -m pip install -e third_party/robosuite
python -m pip install -e third_party/robocasa
python third_party/robocasa/robocasa/scripts/setup_macros.py
python -m pip install -e .
```

## 必需资产与环境变量

必须先自行取得官方 JEPA-WM checkpoint、DINOv3 ViT-L/16 权重、RoboCasa 离线 HDF5 和 simulator assets。推荐全部使用真实绝对路径：

```bash
export JEPA_WM_DROID_CKPT="/absolute/path/to/jepa_wm_droid.pth.tar"
export DINOV3_VITL16_CKPT="/absolute/path/to/opensource-checkpoints/dinov3/dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth"
export JEPAWM_DSET="/absolute/path/to/jepa-wms-datasets"
export JEPAWM_ROBOCASA_HDF5="$JEPAWM_DSET/robocasa/combine_all_im256.hdf5"
```

`DINOV3_VITL16_CKPT` 必须保持官方 `JEPAWM_OSSCKPT/dinov3/<filename>` 布局；加载器直接复用官方 `DinoEncoder`。RoboCasa custom loader 的 `$JEPAWM_DSET/robocasa/` 下应只有本次指定的 `*im256.hdf5`，避免官方递归扫描混入其他数据。

官方来源：

- JEPA-WM DROID/RoboCasa：Hugging Face `facebook/jepa-wms` 中的 `jepa_wm_droid.pth.tar`；
- DINOv3：官方 DINOv3 ViT-L/16 LVD-1689M 权重；
- RoboCasa 数据：Hugging Face dataset `facebook/jepa-wms`，官方脚本为 `third_party/jepa-wms/src/scripts/download_data.py --dataset robocasa`；
- RoboCasa assets：官方 `third_party/robocasa/robocasa/scripts/download_kitchen_assets.py`。

以上下载均不由本项目 CLI 自动触发。

## 真实模型接入

后端复用以下官方实现：

- `app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds.init_module` 和 `EncPredWM`；
- `app.vjepa_wm.video_wm.VideoWM.forward_pred` / `EncPredWM.unroll`；
- `app.plan_common.models.dino.DinoEncoder`；
- `dinov3.models.vision_transformer.DinoVisionTransformer.prepare_tokens_with_masks`、原始 blocks 和 final norm；
- `app.plan_common.datasets.robocasa_dset.RoboCasaDataset`；
- `evals.simu_env_planning` 的 `make_env`、`GC_Agent`、`CEMPlanner`、`ReprTargetDistMPCObjective` 与 `PlanEvaluator`。

256×256 输入的实际 token layout 为：

```text
1 CLS + 4 storage/register + 16×16 patch = 261 tokens, D=1024
```

prefix 数量不是写死的：运行时以实际 token 总数减去 patch grid 计算，并检查所有 shape。DCT 和 Token MLP 只修改 256 个 patch token；CLS 与四个 register/storage token保持不变。LoRA 只包装最后一层 attention 的 fused QKV，并只对 Q、V 输出切片增加低秩 delta。

可训练参数量由模型接口运行时报告；在当前官方 D=1024 配置下为：

| 方法 | 可训练参数 |
| --- | ---: |
| base | 0 |
| dct_adapter, rank 8 | 17,496 |
| token_mlp, rank 8 | 17,416 |
| lora, Q/V rank 4 | 16,384 |

## 数据协议与 feature cache

默认从 episode 级确定性 80/20 划分中只取 train partition，按 episode-balanced round-robin 选择同一组 2,000 个物理窗口。每个窗口为四帧、frameskip 5；没有 validation split 或 validation loader。所有学习方法共享同一 HDF5 cache。

`composed_photometric` 只改变未 normalization 的 RGB：brightness、contrast、gamma、RGB channel gain 和 smooth low-frequency illumination field。同一窗口四帧共享同一个完整 spec，随后才执行官方 resize/normalization。规划时只对 current observation 应用固定 seed 2026 的 spec，goal image保持 clean。

Cache 的 final latent 使用 fp16；pre-final prefix token 保持 float32，因为官方 DINOv3 prefix 的实际数值可能超过 fp16 有限范围。所有 dataset 均按 window 分块，并使用 LZF compression、临时文件原子替换与 finalized marker；论文套件直接复用这一既有 schema：

| dataset | shape |
| --- | --- |
| `clean_prefix_tokens` | `[N,4,261,1024]` |
| `ood_prefix_tokens` | `[N,4,261,1024]` |
| `clean_context_final_latent` | `[N,3,256,1024]` |
| `clean_future_latent` | `[N,1,256,1024]` |
| `actions` | `[N,4,7]` |
| `episode_id` | `[N]` |
| `window_id` | `[N]` |
| `appearance_seed` | `[N]` |

根 metadata 记录 appearance、基础 JEPA-WM/DINOv3 SHA256、全部上游 commit、官方 preprocessing、episode/window 摘要、tensor shape、cache fingerprint 和 finalized 状态。读取时会核对这些加载契约。

构建唯一共享 cache：

```bash
python scripts/build_feature_cache.py \
    --config configs/experiment/robocasa_pilot.yaml
```

## 训练

所有方法使用同一 `L_canonical + L_dynamics`：OOD prefix 经方法与冻结 final block 后对齐 clean final visual patch latent；冻结官方 predictor 接收 adapted context 和原始动作后对齐 clean future visual patch latent。默认 20 epochs、batch 4、gradient accumulation 8、AdamW、bf16，只保存 final PEFT checkpoint。若设备不支持 bf16，必须显式覆盖 `training.precision=fp16`，不会静默切换。

训练启动时会在首个 batch 内执行一次零初始化 identity invariant，然后立即进入正式训练；不创建 validation。

```bash
python scripts/train_adapter.py \
    --config configs/experiment/robocasa_pilot.yaml \
    method=dct_adapter

python scripts/train_adapter.py \
    --config configs/experiment/robocasa_pilot.yaml \
    method=token_mlp

python scripts/train_adapter.py \
    --config configs/experiment/robocasa_pilot.yaml \
    method=lora
```

输出为：

```text
checkpoints/jepa_wm_droid/robocasa/
├── dct_adapter_final.pt
├── token_mlp_final.pt
└── lora_final.pt
```

每个文件仅含 method 名称、PEFT state、method config、参数量、基础权重 fingerprint、上游 commits、cache fingerprint、appearance metadata 和 training config，不复制基础模型。

## RoboCasa planning

第一阶段固定采用官方 RoboCasa planning 配置中首个 JEPA-WM 任务：`robocasa-PnPCounterTop` 的 `place` subtask。默认 50 episodes，evaluation seed 42。环境、观测、动作、predictor rollout、L2 goal cost、CEM、success 判定和 episode termination 均走官方实现。

Protocol 2.0 中，current observation 使用三帧左填充 history：第一步为 `[x0,x0,x0]`，第二步为 `[x0,x0,x1]`，随后始终使用 `[x_{t-2},x_{t-1},x_t]`。goal 使用单张 clean image 独立编码，不进入 current history；Base、Token MLP、LoRA 和 DCT Adapter 共享完全相同的 history。

官方 CEM 预算保持 `horizon=3`、`num_samples=300`、`iterations=15`、`top-k/num_elites=10`。正式配置使用 `planning.candidate_chunk_size=300` 一次处理完整候选集；该字段只控制 rollout cost 的执行分块，不改变候选顺序、全局 top-k、horizon 或 iterations。四种方法使用完全相同的候选规模和随机种子。

八组运行命令：

```bash
# Base Clean
python scripts/plan.py --config configs/experiment/robocasa_pilot.yaml method=base domain=clean

# Base OOD
python scripts/plan.py --config configs/experiment/robocasa_pilot.yaml method=base domain=ood

# Token MLP Clean
python scripts/plan.py --config configs/experiment/robocasa_pilot.yaml method=token_mlp domain=clean

# Token MLP OOD
python scripts/plan.py --config configs/experiment/robocasa_pilot.yaml method=token_mlp domain=ood

# LoRA Clean
python scripts/plan.py --config configs/experiment/robocasa_pilot.yaml method=lora domain=clean

# LoRA OOD
python scripts/plan.py --config configs/experiment/robocasa_pilot.yaml method=lora domain=ood

# DCT Adapter Clean
python scripts/plan.py --config configs/experiment/robocasa_pilot.yaml method=dct_adapter domain=clean

# DCT Adapter OOD
python scripts/plan.py --config configs/experiment/robocasa_pilot.yaml method=dct_adapter domain=ood
```

结果隔离为：

```text
outputs/jepa_wm_droid/robocasa/protocol_v2/place/seed_42/<method>/<clean_or_ood>/results.json
```

`results.json` 记录 success count、episode 总数、success rate、逐 episode success、environment/CEM seed、appearance spec、耗时、peak CUDA memory、方法参数量、完整配置以及基础/PEFT/cache fingerprint。是否成功只能由实际完成的 RoboCasa 运行结果确定。

## 本地开发与服务器部署

主仓库和 GitHub 是项目代码的唯一 source of truth。`/third_party/` 继续由 `.gitignore` 排除，四个上游仓库保持在“固定 upstream commit + 主仓库内 patch”的状态；不会把嵌套仓库、RoboCasa assets 或本地模型文件直接加入主仓库。

| 嵌套仓库 | 固定 base commit | 主仓库中的 patch |
| --- | --- | --- |
| `third_party/jepa-wms` | `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` | `patches/third_party/jepa-wms.patch` |
| `third_party/dinov3` | `6876159a11b4df116f30f667f8c9888617df0751` | `patches/third_party/dinov3.patch` |
| `third_party/robosuite` | `9548a5a35bde8eabf47f760802045cca447e9c0c` | `patches/third_party/robosuite.patch` |
| `third_party/robocasa` | `2544dc2e38bb44f5ced80fbc91114a2f7934016a` | `patches/third_party/robocasa.patch` |

`scripts/export_third_party_patches.sh` 始终相对于表中的固定 commit 导出 binary-safe diff，因此能够同时包含嵌套仓库中已经 commit 的本地变化、staged 修改和 unstaged tracked-file 修改。导出前会拒绝任何非 ignored 的 untracked 文件，避免源码被静默遗漏；ignored assets 不会被扫描、复制或提交。空 diff 不保留空 patch，`patches/third_party/MANIFEST.txt` 记录 base commit、当前 HEAD、patch SHA256 和生成时间。

首次克隆后启用仓库自带的 pre-commit hook：

```bash
git config core.hooksPath .githooks
```

之后正常在本地修改主仓库或嵌套仓库源码。每次 commit 前，hook 会自动运行导出脚本并把最新 patch 与 manifest 加入该次 commit；它不运行测试、训练或 planning。也可以随时手动刷新并检查 patch：

```bash
bash scripts/export_third_party_patches.sh
git diff --cached -- patches/third_party
```

新机器或服务器先把四个嵌套仓库准备到上表所列对象可用的状态，再执行：

```bash
bash scripts/apply_third_party_patches.sh
```

应用脚本会依次把每个嵌套仓库的 tracked 文件 `reset --hard` 到固定 base commit，然后执行 `git apply --check` 和 `git apply`。它绝不执行 `git clean`，因此不会删除 ignored 的 RoboCasa assets、本地模型或其他未跟踪数据；重复执行会先恢复固定 commit 再应用同一 patch，结果幂等。运行前应先导出需要保留的嵌套仓库 tracked 修改。

完成本地修改后，可以一次性 commit、push 并同步服务器：

```bash
bash scripts/publish_and_sync.sh "描述本次修改的 commit message"
```

默认部署目标为 `zhaoyanghe@172.28.11.129:/data/users/zhaoyanghe/control-frequency-wm`，可在单次命令中覆盖：

```bash
DEPLOY_HOST="user@server" \
DEPLOY_ROOT="/absolute/server/path/control-frequency-wm" \
bash scripts/publish_and_sync.sh "描述本次修改的 commit message"
```

部署端只执行当前 branch 的 `git fetch`、`git reset --hard origin/<branch>` 和第三方 patch 应用，不会重建 feature cache、重新训练或启动 planning。`storage/`、RoboCasa assets、`outputs/`、`checkpoints/`、feature cache 与训练 checkpoint 均不上传 GitHub，也不会被部署流程清理。当前依赖使用 editable install，纯 Python 源码更新后无需重新执行 `pip install`。

## ICRA 2027 完整实验套件

论文套件固定使用 `protocol_v2`，保留并复用已完成的 place/seed-42 正式结果，不会覆盖它们。四任务配置由同一份官方 place planning YAML 派生，只改变 `tag` 与 `task_specification.env.subtask`：`reach`、`reach-pick`、`place` 和 `reach-pick-place`。正式运行前会硬性核对 CEM 为 15 iterations、300 samples、10 elites、horizon 3、`num_act_stepped=1`，以及 60 个环境步骤；只有隔离 self-test 配置允许 1/8/2、horizon 1 和 5 个环境步骤。

用户只运行三个入口：

```bash
cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/test_full_pipeline.sh
cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/run_all_paper_experiments.sh
cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/watch_all_paper_experiments.sh
```

三个 wrapper 默认激活 `wm-a100`。自检不是 mock 或 import-only 检查：它使用真实 JEPA-WM、DINOv3、RoboCasa HDF5、assets 和 CUDA，在 `storage/self_test/`、`checkpoints/self_test/`、`outputs/self_test/`、`logs/self_test/` 中依次构建 16-window cache，训练三个方法各一 epoch，运行四方法 clean/OOD 离线评测与八个最小 place planning job，最后调用正式分析器。任一资源缺失或任一路径失败都会写入 `outputs/self_test/self_test_report.{json,md}` 并返回非零；它不会使用假数据或静默 fallback。开始自检时只会清理上述四个 `self_test` 目录。

正式 runner 自动完成资源/commit preflight、共享 2,000-window cache、三种正式 PEFT checkpoint、十项 DCT 离线消融、四任务主表、多 seed、severity、关键闭环消融与统计分析。它从 `GPUS`（默认 `0,1,2,3`）分配空闲 GPU，每张卡最多一个重任务；失败后停止提交新任务、等待已启动任务结束并返回非零。重新运行同一命令会验证并复用完整 artifact；复用结果在分析表中记录源路径和 SHA256。未完成或损坏的 planning/offline JSON 会先按 SHA 保留为 `*.incomplete-<sha>.json`，再从该 job 起点重跑，有效结果从不覆盖。

DCT 消融由同一个 `SequenceStableAdaptiveDCTAdapter` 提供：`temporal_pool=mean|none`、`mask_type=adaptive|static_low_rank` 和 `use_rms_norm`。默认参数不会改变既有 checkpoint 的 method metadata；static low-rank mask 使用 `[D,rank]` 与 `[rank,H,W]` 因子，频率因子零初始化以保持严格 identity。训练目标权重 `canonical_weight`、`dynamics_weight` 默认均为 1，禁止同时为 0。

OOD severity 中 `appearance.training_severity=1.0` 始终描述 checkpoint/cache 的训练域，`appearance.severity` 独立描述评估域，因此 0.5、1.0、1.5 不会改变 checkpoint 绑定。正式输出位于 `outputs/jepa_wm_droid/robocasa/protocol_v2/` 与 `outputs/paper_suite/protocol_v2/`；论文产物为：

```text
outputs/paper_suite/analysis/
├── main_results.csv
├── main_results.md
├── main_results.tex
├── multiseed.csv
├── severity.csv
├── ablations.csv
├── offline_metrics.csv
├── efficiency.csv
├── statistics.json
└── paper_summary.md
```

统计器生成 Wilson 95% CI、10,000 次 paired bootstrap、exact McNemar、Holm 校正和多 seed mean/std。任何需要配对的记录若 environment/episode identity 不一致会立即拒绝统计。调度状态原子写入 `logs/paper_suite/state.json`；监控器显示 phase、job/GPU/PID、训练 loss、离线窗口进度、planning episode/step/success/CEM 和日志心跳。

本机若没有 `wm-a100`、CUDA、checkpoint、HDF5 或 RoboCasa assets，只应提交代码到服务器，不应运行 `test_full_pipeline.sh`。上传服务器、设置前述四个资源环境变量并应用固定上游 patch 后，先运行真实自检；只有它退出码为 0 后再启动正式 runner。
# Cross-benchmark ICRA suite (`cross_benchmark_v1`)

The current compact paper protocol evaluates the frozen JEPA-WM/DINOv3 backend on two RoboCasa tasks and two official LIBERO tasks. The main matrix is exactly four tasks × four methods (`base`, `dct_adapter`, `token_mlp`, `lora`) × clean/OOD × 20 paired closed-loop rollouts. Existing `icra2027_suite`, `paper_suite`, and `protocol_v2` artifacts remain supported and are never rewritten by this suite.

The task set is fixed as RoboCasa `PnPCounterTop/place`, the first resource-compatible articulated task in the immutable order `OpenDrawer`, `CloseDrawer`, `OpenCabinet`, `CloseCabinet`, official `libero_spatial` task ID 0, and official `libero_goal` task ID 0. Resolution happens before method evaluation and is saved under `outputs/cross_benchmark_v1/manifests/`. A missing fixed LIBERO task or an articulated candidate pool with no valid task is a preflight error; the runner never substitutes a task based on success rate.

LIBERO is accessed through its official benchmark registry (`get_benchmark_dict`, `get_task`, `get_task_init_states`) and `OffScreenRenderEnv`. The agent-view HDF5 key is resolved from supported agent-view-only keys and recorded in the task manifest. Preflight restores a recorded simulator state, checks RGB orientation/channel parity, executes a recorded action, checks state and gripper response, and confirms the official success interface. Its canonical 7-D action is explicitly recorded as normalized OSC-Pose translation, axis-angle rotation, and the LIBERO `-1=open, +1=close` gripper convention; shape alone is not accepted as evidence of compatibility.

Set the existing JEPA-WM variables in `env_jepa.sh`, then create a private `env_libero.sh` from [env_libero.example.sh](env_libero.example.sh). The suite reads `LIBERO_ROOT`, `LIBERO_DATA_ROOT`, and optional `LIBERO_SPATIAL_DATA_ROOT` / `LIBERO_GOAL_DATA_ROOT`. It never downloads checkpoints, demonstrations, BDDL files, init states, or assets.

The data protocol uses 2,000 physical four-frame windows per task, deterministic episode-balanced selection, trajectory-level 80/20 splitting, and split/window/training seed 42. Every task gets one immutable 20-instance evaluation manifest shared across every method and domain. LIBERO instances use distinct held-out successful demonstrations; RoboCasa segments retain their source trajectory and are analyzed with source-trajectory cluster bootstrap. Current frames use `composed_photometric_v1`, while the goal is always a single clean image.

Artifacts are isolated at:

- caches: `storage/feature_cache/cross_benchmark_v1/<benchmark>/<task>.h5`
- main checkpoints: `checkpoints/cross_benchmark_v1/<benchmark>/<task>/<method>_final.pt`
- main planning: `outputs/cross_benchmark_v1/main/<benchmark>/<task>/seed_42/<method>/<clean_or_ood>/results.json`
- analysis: `outputs/cross_benchmark_v1/analysis/`

The runner dynamically constructs the job graph, atomically updates its state, schedules at most one heavyweight subprocess per GPU, validates artifacts before reuse, and SHA-archives an incomplete planning/offline result before rerunning it. A reused source remains in place and is referenced by path and SHA256. When a reused source has more than 20 episodes, success uses the manifest-aligned first 20 and efficiency is normalized by the source file's full available episode count.

Dry-run (no model load and no formal artifact writes): `python scripts/launch_cross_benchmark_suite.py --dry-run`

Isolated resource-backed self-test: `bash scripts/run_cross_benchmark_suite.sh --self-test`

Formal launch with the single-screen curses Dashboard: `bash scripts/run_cross_benchmark_suite.sh`

Attach without starting another runner: `bash scripts/run_cross_benchmark_suite.sh --attach`

Status snapshot: `bash scripts/run_cross_benchmark_suite.sh --status`

Explicit stop: `bash scripts/run_cross_benchmark_suite.sh --stop`

Leaving the Dashboard with `q` or `Ctrl+C` only detaches it; the runner remains in its independent process group. The formal CEM settings remain 15 iterations, 300 candidates, 10 elites, horizon 3, and one executed action step for every method within a task.
