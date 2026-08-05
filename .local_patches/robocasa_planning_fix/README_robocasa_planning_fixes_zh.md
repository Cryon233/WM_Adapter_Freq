# RoboCasa Planning 修复包

基准提交：`2ca0a5d5a6beb885354ccb01ca429e75b5e3811d`

该修复包采用**锚点校验 + 原文件备份 + 原子写入**，不会直接用旧提交的完整文件覆盖服务器上的其他未提交改动。脚本只有在五个目标文件中的预期代码片段全部匹配时才会写入；任一片段不匹配会在写入前终止。

## 修改的文件

1. `third_party/jepa-wms/evals/simu_env_planning/planning/plan_evaluator.py`
   - 将未经标准化的dataset RGB按`uint8`或`[0,1] float`处理，不再错误调用`inverse_transform`。
   - 第一帧使用`env.prepare()`返回的实时simulator render，消除dataset首帧与后续render的来源/尺寸切换。
   - success按episode内任意低层step累计。
   - Reach使用最小`hand_obj_dist`，Place使用最小`obj_goal_dist`作为诊断距离。
   - `optional_plots=false`时不再无条件生成视频/PDF和执行额外latent-distance分析。

2. `third_party/jepa-wms/evals/simu_env_planning/planning/utils.py`
   - 删除视频逐帧min-max归一化。
   - MP4、GIF、PDF统一使用固定RGB数值合同。

3. `src/wm_adapter/planning/jepa_wm_planner.py`
   - Planning protocol从`2.0`提升到`2.1`。
   - 增加历史帧尺寸变化的显式检查。
   - Place不再接受可变长度legacy segment。
   - goal latent fingerprint使用与Planner一致的量化RGB。

4. `src/wm_adapter/benchmarks/robocasa.py`
   - Place manifest真正执行`goal_span_steps: 25`，选择Place segment最后26帧。
   - manifest写入`fixed_goal_span_steps`并禁用legacy复用标记。

5. `scripts/run_cross_backend_adapter_suite.py`
   - 复用验证要求Planning protocol `2.1`。
   - Place manifest必须是25-step固定跨度。
   - 验证`success_count/total_episodes/success_rate`与`per_episode_success`一致。

6. `scripts/monitor_all_paper_experiments.py`
   - Dashboard优先读取cross-backend state中的`artifact_validation`。
   - 已完成Planning使用`episodes`作为分母，修复`completed 0/0 | success 1/0`。

## 最安全的执行方法

先把以下三个文件放到服务器同一目录：

- `apply_robocasa_planning_fixes.py`
- `apply_and_prepare_robocasa_rerun.sh`
- 本README

停止当前suite。Dashboard中按大写`X`，然后确认：

```bash
cd /data/users/zhaoyanghe/control-frequency-wm
pgrep -af 'run_cross_backend_adapter_suite.py|scripts/plan.py'
```

先只检查是否可应用，不写文件：

```bash
python3 /path/to/apply_robocasa_planning_fixes.py \
  --repo /data/users/zhaoyanghe/control-frequency-wm \
  --check
```

输出`Applicability check passed`后，一键修复并归档旧Planning结果：

```bash
bash /path/to/apply_and_prepare_robocasa_rerun.sh \
  /data/users/zhaoyanghe/control-frequency-wm \
  /path/to/apply_robocasa_planning_fixes.py
```

脚本不会移动或删除：

- `storage/feature_cache/cross_backend_adapter_v1`
- `checkpoints/cross_backend_adapter_v1`
- `outputs/cross_backend_adapter_v1/offline`
- LIBERO evaluation manifests

它会归档：

- `outputs/cross_backend_adapter_v1/main`
- `outputs/cross_backend_adapter_v1/ablations`
- RoboCasa Reach/Place evaluation manifests
- 旧`state.json`和失效runner PID文件

每个被修改文件的**原始完整版本**与**修改后完整版本**会保存到：

```text
.planning_fix_backups/<timestamp>/original/...
.planning_fix_backups/<timestamp>/patched/...
```

同时保存：

```text
applied.patch
git-status-before.txt
git-diff-before.patch
git-diff-cached-before.patch
restore.sh
```

## 回滚

应用后终端会打印具体备份目录。执行：

```bash
bash .planning_fix_backups/<timestamp>/restore.sh \
  /data/users/zhaoyanghe/control-frequency-wm
```

归档的旧结果不会自动搬回；需要时手动从`archive/robocasa_planning_protocol_v2_buggy_<timestamp>/`恢复。

## 不要立即恢复198个Planning job

先运行四个Base Clean smoke test，每个3个episode：

```bash
cd /data/users/zhaoyanghe/control-frequency-wm

CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 \
python scripts/plan.py \
  --config configs/experiment/cross_backend_adapter/robocasa_reach.yaml \
  model_config=configs/model/jepa_wm_droid_reach.yaml \
  method=base domain=clean \
  evaluation.num_episodes=3 \
  planning.compile_predictor=false \
  output.run_directory=outputs/debug/protocol_v2_1/jepa_reach_base_clean

CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 \
python scripts/plan.py \
  --config configs/experiment/cross_backend_adapter/robocasa_place.yaml \
  model_config=configs/model/jepa_wm_droid_place.yaml \
  method=base domain=clean \
  evaluation.num_episodes=3 \
  planning.compile_predictor=false \
  output.run_directory=outputs/debug/protocol_v2_1/jepa_place_base_clean

CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 \
python scripts/plan.py \
  --config configs/experiment/cross_backend_adapter/robocasa_reach.yaml \
  model_config=configs/model/dino_wm_droid_reach.yaml \
  method=base domain=clean \
  evaluation.num_episodes=3 \
  planning.compile_predictor=false \
  output.run_directory=outputs/debug/protocol_v2_1/dino_reach_base_clean

CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 \
python scripts/plan.py \
  --config configs/experiment/cross_backend_adapter/robocasa_place.yaml \
  model_config=configs/model/dino_wm_droid_place.yaml \
  method=base domain=clean \
  evaluation.num_episodes=3 \
  planning.compile_predictor=false \
  output.run_directory=outputs/debug/protocol_v2_1/dino_place_base_clean
```

上述命令依赖seed 42的RoboCasa evaluation manifest。一键归档后，先直接重建这两个manifest，不需要启动完整suite：

```bash
cd /data/users/zhaoyanghe/control-frequency-wm

python scripts/build_evaluation_manifest.py \
  --config configs/experiment/cross_backend_adapter/robocasa_reach.yaml \
  model_config=configs/model/jepa_wm_droid_reach.yaml \
  paths.evaluation_manifest=outputs/cross_backend_adapter_v1/manifests/evaluation/robocasa_reach/seed_42.json \
  evaluation.num_episodes=20 \
  evaluation.eval_seed=42

python scripts/build_evaluation_manifest.py \
  --config configs/experiment/cross_backend_adapter/robocasa_place.yaml \
  model_config=configs/model/jepa_wm_droid_place.yaml \
  paths.evaluation_manifest=outputs/cross_backend_adapter_v1/manifests/evaluation/robocasa_place/seed_42.json \
  evaluation.num_episodes=20 \
  evaluation.eval_seed=42
```

确认Place跨度已经固定为25：

```bash
jq '[.instances[] | (.segment_end - .segment_start)] | unique' \
  outputs/cross_backend_adapter_v1/manifests/evaluation/robocasa_place/seed_42.json
```

预期输出：

```json
[25]
```

然后运行上述4个smoke test。不要先恢复完整Planning矩阵。

## 提交前检查

```bash
git diff --check
git diff --stat
git diff -- \
  third_party/jepa-wms/evals/simu_env_planning/planning/plan_evaluator.py \
  third_party/jepa-wms/evals/simu_env_planning/planning/utils.py \
  src/wm_adapter/planning/jepa_wm_planner.py \
  src/wm_adapter/benchmarks/robocasa.py \
  scripts/run_cross_backend_adapter_suite.py \
  scripts/monitor_all_paper_experiments.py
```
