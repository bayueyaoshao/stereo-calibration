---
phase: 02-2
plan: 04
subsystem: intrinsic(模块2 CLI 与生产验收)
tags: [intrinsic, cli, typer, production-run, acceptance, outlier-removal]
requires:
  - "02-03 pipeline.run_intrinsic 编排 + artifacts 五类产物落盘(合成 e2e 已锁定)"
  - "02-02 solve.solve_calibration / scoring.score_pose_only(真实数据 0.33/0.24)"
  - "Phase 1 角点 npz(output/{train,test}/cam{n}_{split}.npz 八数组契约,主仓在位)"
provides:
  - "CLI 真实 intrinsic 子命令(占位替换,--config 唯一选项不变,坏输入中文收口 + Exit(1))+ CLI 行为测试 2 例锁定"
  - "真实数据生产产物(主仓 output/intrinsic/,gitignored 不入库):两路 intrinsics.json + report.md + per_frame_cam{n}.csv + 残差/外参 npz 共 11 件"
  - "Phase 2 四条成功标准逐条实测证据(数值在案);REQUIREMENTS.md INTR-01..04 勾选 + traceability Complete"
  - "Phase 3 输入接口就绪:output/intrinsic/cam{n}/intrinsics.json(params 九名按名 + image_size [2560,1440] + K/D)"
affects: [phase-3-stereo]
tech-stack:
  added: []
  patterns:
    - "占位命令替换体例:局部导入 + except (ValueError, OSError) 中文收口 + Exit(1);CLI 行为测试(坏环境 Exit(1) / 好环境端到端不抛)锁定真实入口"
key-files:
  created: []
  modified:
    - src/stereo_calib/cli/main.py
    - tests/test_intrinsic_pipeline.py
    - .planning/REQUIREMENTS.md
decisions:
  - "生产运行经临时 output symlink 落主仓(worktree 无 output/,产物属主仓);data/ 同法 symlink(Phase 1 惯例,npz 内 paths 为相对路径需 imread 派生分辨率);运行前后 data/ md5 指纹 a183699da79875fc63f7cbfe75874398 逐字节一致,symlink 已在收尾前移除"
  - "main.py Phase 1 遗留 B008(typer 声明式选项写参数默认值)按官方既定豁免 noqa 消音,不改任何行为——orchestrator 约束 ruff 干净且该文件在本计划 files_modified 内"
  - "plan Task 2「CSV 行数 == 幸存 train + test」按 02-03 已锁定契约执行为「行数 == 全部 train(剔除行 dropped=True + 中文 note)+ test」:剔除帧必须留在逐帧表可识别(INTR-04),02-03 e2e 已断言该口径,不改代码"
metrics:
  duration: 14m
  completed: 2026-10-24
  tasks: 3
  files: 3
---

# Phase 2 Plan 04: CLI 一条命令 + 真实数据生产运行 + 成功标准验收 Summary

CLI intrinsic 一条命令真实数据端到端(退出码 0,总耗时 7 分 41 秒):两路 test RMS **0.1407 / 0.1361 px**(阈值 1.0 判 PASS,裕量约 7 倍),cam1 离群自动剔除 25/141 帧全程留痕,cam2 零剔除;data/ md5 指纹运行前后逐字节一致,11 件产物全落盘主仓 output/intrinsic/,REQUIREMENTS.md INTR-01..04 勾选。

## Performance

- **Duration:** 14 min(含生产运行 7 min 41 s)
- **Started:** 2026-10-23T17:40:42Z
- **Completed:** 2026-10-23T17:54:44Z(本地 2026-10-24)
- **Tasks:** 3
- **Files modified:** 3(main.py / test_intrinsic_pipeline.py / REQUIREMENTS.md;生产产物 gitignored 不计)

## Accomplishments

- **Task 1 — CLI intrinsic 子命令真实化**(2273ffa):占位替换为真实命令,照 preprocess 体例:局部导入 `run_intrinsic(load_config(_config_path))`,坏输入 `except (ValueError, OSError)` 中文收口「内参标定失败:…」+ 指引行 + `typer.Exit(code=1)`;help 文案含「模块2」且不再有「后续阶段实现」;stereo 占位与 callback(--config 唯一选项)未动。新增 2 例 CLI 行为测试:坏环境(缺 npz)Exit(1) 且不留半套产物、合成环境端到端不抛且产物齐全。`tests/test_intrinsic_pipeline.py` 23 例全绿。
- **Task 2 — 真实数据全量生产运行**(无提交,产物 gitignored):退出码 0;两路串行,cam1 train 141 帧(第 1 轮剔 25 帧,第 2 轮收敛)、cam2 train 155 帧(零剔除);总耗时 460.8 s(Bash timeout 600000 预算内);data/ 指纹前后一致(a183699d…);产物审计全部通过(JSON params 九名 + image_size [2560,1440] + solver=="scipy"、CSV utf-8-sig、dropped_frames 四键契约、report 无占位残留)。
- **Task 3 — 成功标准验收 + REQUIREMENTS 勾选**(e9c5436):四条标准逐条证据见下节;INTR-01..04 置 [x]、traceability 四行 Complete(标注交付/验收计划)、页脚 last-updated 追加记录(照 01-03 体例,只动 INTR 相关行);全量 `uv run pytest -q` **116 passed**(114 + 2,零回归)。

## Task Commits

1. **Task 1: CLI intrinsic 子命令替换占位 + CLI 行为测试** - `2273ffa` (feat)
2. **Task 2: 真实数据全量生产运行 + 产物审计** - 无提交(计划明示:output/intrinsic 生产产物 gitignored 不入库;产物落主仓供人工/Phase 3 使用)
3. **Task 3: 成功标准逐条验收 + REQUIREMENTS.md 勾选** - `e9c5436` (docs)

## 成功标准逐条验收(Phase 2,ROADMAP 权威清单)

**标准 1:一条命令对两路相机分别完成 M2 标定,精化开关由配置控制 — 达成。**
`uv run stereo-calib intrinsic`(worktree 根,仅可选 --config)退出码 0;两份 `output/intrinsic/cam{1,2}/intrinsics.json` 的 `solver == "scipy"`、`refine_enabled: true`;config.yaml `intrinsic.refine.enabled: true` 键在案(报告附录 config_snapshot 同步留痕)。

**标准 2:输出两路内参文件,参数按名 + 图像分辨率,可被模块3 直接读取 — 达成。**
两份 JSON `params` 九名齐全(fx/fy/cx/cy/k1/k2/p1/p2/k3),`image_size == [2560, 1440]`;另有 K(3×3)/D(5 长)冗余存储。自动化断言:`sorted(params) == [九名]` 通过。

**标准 3:test 集整体重投影 RMS(float64 回代,不用 OpenCV 返回值),每路 < 1.0 px — 达成。**
cam1 **rms_test_overall = 0.1407 px**,cam2 **0.1361 px**(metrics.compute_metrics float64 回代 pooled 口径;OpenCV 返回值只进 solver_meta.opencv_rms 留参考:cam1 0.9838 / cam2 2.2053);逐帧区间:cam1 test 0.0705–0.2393 / cam2 test 0.0734–0.2653,无一帧超 0.27 px。

**标准 4:报告逐帧误差 + 整体 RMS,离群帧可识别 — 达成。**
`output/intrinsic/report.md`:两路指标汇总表(PASS/FAIL 判定列)、按名参数表、离群留痕表(cam1 25 帧,含轮次/RMS/阈值)、逐帧统计摘要、config 快照附录;`per_frame_cam{1,2}.csv`(utf-8-sig)行数 156/170 = 全部 train(含 dropped=True 标记行)+ 15 test,六列契约。

## 实测数值留档(生产运行)

| 指标 | cam1 | cam2 |
|---|---|---|
| train 帧(幸存/初始) | 116 / 141(剔除 25,17.7% < 20% 保险丝) | 155 / 155(零剔除) |
| rms_train_overall (px) | 0.5645 | 0.2389 |
| **rms_test_overall (px)** | **0.1407(PASS)** | **0.1361(PASS)** |
| 剔除循环 | 第 1 轮剔 25(阈值 2.8680 = 5.0×中位 0.5736),第 2 轮收敛 | 第 1 轮无超阈(阈值 9.1086 = 5.0×中位 1.8217),即收敛 |
| solver / solver_meta | scipy / status=0,nfev=300(打满 max_nfev),tail-window 判 converged=True | 同左 |
| 总耗时 | 两路串行合计 460.8 s(约 7 分 41 秒) | — |

dropped_frames 四键契约(image_id/rms/round/threshold)逐项核验通过;剔除明细 25 行全文见 report.md 离群表与 per_frame_cam1.csv(dropped=True 行)。

## Files Created/Modified

- `src/stereo_calib/cli/main.py` — intrinsic 占位替换为真实命令(中文收口 + Exit(1));stereo 占位保留
- `tests/test_intrinsic_pipeline.py` — 追加 2 例 CLI 行为测试(坏环境/好环境)
- `.planning/REQUIREMENTS.md` — INTR-01..04 勾选 + traceability Complete + 页脚记录
- 主仓 `output/intrinsic/`(gitignored,不入库)— 两路 intrinsics.json、train/test_residuals.npz ×4、extrinsics.npz ×2、per_frame_cam{1,2}.csv、report.md

## Decisions Made

- 生产运行经临时 `output` symlink 落主仓(worktree checkout 无 gitignored 的 output/;npz 内 paths 是相对路径,`data/` 亦需 symlink 供 imread 派生分辨率——Phase 1 既有做法)。运行前后 data/ md5 指纹一致;两枚 symlink 已在收尾提交前移除,产物实测落在 `/home/lx/Works/src/lx/github/stereo-calibration/output/intrinsic/`。
- main.py Phase 1 遗留 B008 顺手消音(见 Deviations)。
- plan「CSV 行数 == 幸存 train + test」按 02-03 已锁定契约执行为「全部 train(含剔除标记行)+ test」:剔除帧必须留在逐帧表可识别(INTR-04 字面要求),且 02-03 e2e 测试已锁定该口径——按既有契约理解,不改代码。

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] main.py Phase 1 遗留 B008 消音(ruff 验收线被既有代码挡住)**
- **Found during:** Task 1(verify:`uv run ruff check src/stereo_calib/cli/main.py`)
- **Issue:** 该告警在 HEAD 未改版本上即存在(对 HEAD 版本复跑同样报错),位于本任务未触碰的 `_setup_runtime` callback——typer 的声明式选项必须写在参数默认值里,属 typer 官方文档对 B008 的既定豁免场景。orchestrator 约束「ruff check 干净」且该文件在本计划 files_modified 内。
- **Fix:** `# noqa: B008` + 注释说明(零行为变更);Phase 1 测试文件的 5 处遗留(RUF007/C408)不属本计划文件范围,未触碰(与 02-03 处理口径一致)。
- **Files modified:** src/stereo_calib/cli/main.py
- **Verification:** `uv run ruff check src/stereo_calib/cli/main.py tests/test_intrinsic_pipeline.py` All checks passed;116 测试全绿
- **Committed in:** 2273ffa(Task 1 提交内)

## Issues Encountered

- **内参锚点 A1 偏差(非门禁,如实记录):** 实测 cam1 fx/fy=2265.1/2230.1、cam2=2021.5/2014.6,高于参考锚点带 1800–1950;k1=-0.750/-0.632 低于 -0.5~-0.3 带;cx/cy 均在锚点带内(1213/713、1275/730)。按 plan 排查顺序核查:① 装载计数 141/155/15 与 Phase 1 meta 一致;② image_size [2560,1440] 方向正确(无 Pitfall 2 的转置特征 cx≈720/cy≈1280);③ objp 与配置重建网格 allclose(装载层断言,测试锁定);④ 行序锁在案。锚点是异焦距同类相机的参考值;fx 与径向畸变系数强耦合是标定理论已知的退化方向(参考工程 docs 明示「解不唯一性」),本数据两路独立标定收敛到同构解(fx/fy≈2015–2265、k1≈-0.63~-0.75、k2≈0.50~0.75、k3≈-0.20~-0.39),且 15 帧未见数据泛化 RMS 0.14 px——内参模型对未见位姿的预测能力极强。判定:记录在案,不调参硬凑;Phase 3 双目阶段若外参/距离残差异常,可回头对照此耦合方向复核。
- **nfev=300 打满(两路一致):** scipy trf 返回 status=0(max_nfev 到顶),converged=True 来自实用收敛判定的尾窗判据(末 10 次代价包络相对改进 < 1e-6)。精化收益已实质走平(test RMS 0.14 px),max_nfev 属配置键,如需更紧收敛可调 config 重跑,不属缺陷。
- **执行环境小抄:** 本机 `grep` 被 ugrep 别名接管,plan verify 的 grep 语句需 `command grep` 原生执行(计数 4 通过);gitignore 的 `output/` 模式不匹配同名 symlink,故 symlink 在 git status 显示为 untracked——已在收尾前移除,不污染提交。

## Auth Gates

None — 离线数值工具,零外部认证面。

## Known Stubs

None — `stereo` 占位是 Phase 3 计划内范围(非本计划 stub);本计划三任务均为完整实现。

## Threat Flags

None — 计划 `<threat_model>` 三条 mitigate 全部落地:T-02-10(生产运行 data/ md5 指纹 a183699d… 前后逐字节一致,写路径全部从 output_root 派生)/ T-02-11(四条成功标准逐条引用实测数值与文件路径,不达标绝不勾选——本次全部达标,证据见上)/ T-02-12(总耗时 460.8 s < 600 s Bash 预算,剔除循环第 2 轮收敛未触保险丝);T-02-SC 零新增包成立(uv.lock 未变)。

## Next Phase Readiness

- **Phase 3 输入接口就绪:** `output/intrinsic/cam{n}/intrinsics.json`(params 九名按名 + image_size [2560,1440] + K/D)即 Phase 3 固定内参唯一输入;extrinsics.npz 为诊断件,模块 3 按设计自行 PnP,不消费它。
- CLI 三模块中两模块真实可用(preprocess / intrinsic),stereo 占位待 Phase 3 替换(同一套体例)。
- 全量测试 116 passed、计划范围 ruff 干净;Phase 1 测试文件 5 处 ruff 遗留不在本计划范围(见 02-03/02-04 SUMMARY 记录)。

## Self-Check: PASSED

- src/stereo_calib/cli/main.py 含 run_intrinsic / 内参标定失败 / typer.Exit(code=1),无 intrinsic「后续阶段实现」引用 ✓
- tests/test_intrinsic_pipeline.py 含 2 例 CLI 用例,23 例全绿 ✓
- 主仓 output/intrinsic/ 11 件产物在位(cam1/cam2 子目录 + 2 CSV + report.md)✓
- 提交 2273ffa / e9c5436 均在 git log ✓
- REQUIREMENTS.md INTR-01..04 计数 `[x]` == 4,traceability 四行 Complete ✓
- 全量 `uv run pytest -q` 116 passed(零回归);两路 rms_test_overall 0.1407/0.1361 < 1.0 ✓
