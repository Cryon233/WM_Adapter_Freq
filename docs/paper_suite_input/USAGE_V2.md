# 使用说明 V2

## 1. 让 Codex 实现

将 `CODEX_PROMPT_V2.md` 放入仓库，例如：

`docs/paper_suite_input/CODEX_PROMPT_V2.md`

然后在 Codex 中输入：

> 请读取 docs/paper_suite_input/CODEX_PROMPT_V2.md，并严格实现。不要启动正式长实验。实现完成后必须实际运行 bash scripts/test_full_pipeline.sh，修复所有失败，直到该脚本退出码为0。最后给出修改文件清单、自检报告，以及服务器上一键正式启动命令。

## 2. Codex 完成后

本地或服务器首先运行：

`cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/test_full_pipeline.sh`

成功后正式运行：

`cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/run_all_paper_experiments.sh`

监控：

`cd /data/users/zhaoyanghe/control-frequency-wm && bash scripts/watch_all_paper_experiments.sh`

中断后重新执行同一个正式脚本即可继续已完成 artifact 之后的工作。
