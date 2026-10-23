---
phase: 02-2
verified: 2026-10-23T18:16:26Z
status: human_needed
score: 4/4 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: none
  previous_score: n/a
  gaps_closed: []
  gaps_remaining: []
  regressions: []
deferred:
  - truth: "intrinsics.json 可直接被模块3 读取为固定内参(端到端消费证明)"
    addressed_in: "Phase 3"
    evidence: "Phase 3 goal:『消费 Phase 1 公共视场角点 + Phase 2 固定内参』;本阶段已交付契约完整的按名 JSON(params 九名 + image_size + K/D,Phase 3 按名读取的唯一接口),实际消费链路在 Phase 3 兑现"
human_verification:
  - test: "剔除帧质量人工抽查:打开 output/train/visualization/ 下 cam1 被剔 25 帧对应的角点标注图(清单 = intrinsics.json dropped_frames 的 image_id),并另看幸存帧中 RMS > 1.0px 的 4 帧(cam1,最大 20261023_165247_020677_cam1 达 4.87px)与 cam2 的 1 帧(1.68px)"
    expected: "被剔 25 帧的角点图应有可见劣化(模糊/遮挡/出画/角点漂移);幸存高 RMS 帧应可判断是轻度劣化(可接受)还是应加入 config.yaml intrinsic.exclude_images 手动剔除"
    why_human: "『这帧角点是不是真的坏』是视觉语义判断;数值审计只能证明剔除流程合规留痕,不能证明剔除恰如其分"
  - test: "内参锚点偏差专家评审:对照参考锚点带(fx/fy 1800–1950、k1 -0.5~-0.3)复核 report.md 参数表——cam1 fx/fy=2265/2230、cam2=2021/2015,k1=-0.750/-0.632 均在带外"
    expected: "确认接受当前解(判定依据:两路独立收敛到同构解、test 未见帧泛化 RMS 0.14px、fx-k1 耦合是标定理论已知退化方向)或决定在 Phase 3 前采取行动(如补充不同深度位姿数据重标)"
    why_human: "参数绝对正确性是领域专家判断;自动化验证只能证明内部一致性与 test 集泛化,无法证明物理分解唯一正确;该组内参将冻结为 Phase 3 固定输入"
---

# Phase 2: 模块2 内参标定 Verification Report

**Phase Goal:** 内参标定模块可独立运行:消费 Phase 1 角点数据,对两路相机分别标定 M2 模型内参,并用 test 集给出量化验证指标
**Verified:** 2026-10-23T18:16:26Z
**Status:** human_needed(全部技术 must-haves 自动验证通过;2 项领域判断项路由人工)
**Re-verification:** No — initial verification

## MVP Mode Notes

Phase 2 在 ROADMAP.md 标注 `Mode: mvp`。ROADMAP 目标文本非规范 User Story 格式(`As a …, I want to …, so that …`)——与 Phase 1 验证相同的形式偏差。本验证沿用 Phase 1 先例:以目标结果子句 + 02-04-PLAN objective 的用户故事实质(«标定产线操作员用一条命令得到两路相机 M2 内参文件与量化验证报告»)做 User Flow Coverage。**建议:** 运行 `/gsd mvp-phase 2` 重排格式,或维持现状(人工决策;不影响验证实质)。注:本环境 `gsd-tools` 不可用,`user-story.validate` 未执行,格式判断为人工比对。

## User Flow Coverage

| Step | Expected | Evidence | Status |
|------|----------|----------|--------|
| 运行一条命令 | `uv run stereo-calib intrinsic` 仅凭 config.yaml(可选 --config)端到端成功 | cli/main.py:77-91 真实子命令(占位已替换,局部导入 run_intrinsic,坏输入中文收口 Exit(1));`intrinsic --help` 输出「模块2」且无「后续阶段实现」,退出码 0;生产运行产物时间戳(cam1 01:47 / cam2 01:51 / report 01:51:43)与 02-04 SUMMARY 记录的生产运行(460.8s)吻合 | ✓ |
| 消费 Phase 1 角点数据 | 装载 output/{train,test}/cam{n}_{split}.npz | observation.py:36/94 经 preprocess.artifacts.load_corners(allow_pickle=False);per_frame CSV 行数 141+15(cam1)/155+15(cam2)与 Phase 1 可用帧数逐一相等;data/ 指纹 a183699da79875fc63f7cbfe75874398 与 Phase 1 基线一致(只读硬证明) | ✓ |
| 得到两路 M2 内参文件 | 参数按名 + 图像分辨率 | 两份 intrinsics.json:params 九名齐全(fx/fy/cx/cy/k1/k2/p1/p2/k3)、image_size=[2560,1440]、K(3×3)/D(5 长)冗余存储;K/D 与 params 数值一致(抽核 cam1) | ✓ |
| 得到量化验证指标 | test 集 RMS 亚像素(< 1.0px) | rms_test_overall:cam1 **0.1407** / cam2 **0.1361**(预期带 0.1–0.5);本验证从残差 npz 独立重算 pooled RMS,与 JSON **精确一致(diff 0.00e+00)**,float64 回代链路完整 | ✓ |
| 离群帧自动剔除且可识别 | train 剔除全程留痕,test 永不剔除 | cam1 剔 25/141 帧(dropped_frames 四键契约逐项核验通过;CSV 25 行 dropped=True 带中文 note;report 离群表 25 行含轮次/RMS/阈值);cam2 零剔除;两路 test 帧 dropped 恒 False | ✓ |
| Outcome:内参标定模块可独立运行 | 三模块之一独立可用,输出即模块3 输入 | CLI 三子命令中 preprocess/intrinsic 真实可用(stereo 占位属 Phase 3 范围);intrinsics.json 契约锁定(Phase 3 按名读取) | ✓ |

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria — the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 一条命令对两路相机分别完成 M2 标定(fx,fy,cx,cy,k1,k2,p1,p2,k3):calibrateCameraExtended 初值 + scipy least_squares 精化(精化开关由配置控制) | ✓ VERIFIED | solve.py:128-129 `cv2.calibrateCameraExtended(objs32, imgs32, size, None, None, flags=0)`(float32 显式装包 L122-123);solve.py:396-405 `least_squares(method=refine_cfg.method, tr_solver="lsmr", x_scale="jac", max_nfev=refine_cfg.max_nfev…)`;config.yaml:53-59 intrinsic.refine 段(enabled: true);两份 JSON solver=="scipy"、refine_enabled==true;精化开关关闭行为由 test_refine_disabled_returns_initial_solution 测试锁定(116 测试全绿);`stereo-calib intrinsic --help` 退出码 0 |
| 2 | 输出两路相机内参文件:参数按名保存 + 图像分辨率,可直接被模块3 读取为固定内参 | ✓ VERIFIED | output/intrinsic/cam{1,2}/intrinsics.json 在位;params 九个命名键(sorted == ['cx','cy','fx','fy','k1','k2','k3','p1','p2'])、image_size [2560,1440]、K/D 一致;Pattern 7 十五字段全集无缺失(程序核验);write_intrinsics_json 写入时契约断言 + roundtrip 测试锁定;Phase 3 实际消费证明 → deferred |
| 3 | test 集整体重投影 RMS 以 float64 回代计算(不使用 OpenCV 返回值),每路 < 1.0 px(预期 ~0.1–0.5) | ✓ VERIFIED | metrics.py:68-77 残差只由 models.transform_to_camera + models.project 构成、全 float64;OpenCV rms 只进 solver_meta.opencv_rms(cam1 0.9838/cam2 2.2053,与 rms_test_overall 明显不同值,非同一来源);**独立重算**:从 train/test_residuals.npz 按 metrics 同口径重算 pooled RMS 与 JSON 四值全部 diff 0.00e+00;cam1 0.1407 / cam2 0.1361,均 < 1.0 且落在预期 0.1–0.5 带;源码防线:metrics/solve/scoring/pipeline 四文件 `_distort` 计数全 0 |
| 4 | 输出内参标定报告:逐帧重投影误差 + 整体 RMS,可识别离群帧 | ✓ VERIFIED | output/intrinsic/report.md:两路指标汇总表(PASS/FAIL 判定列)、九参数按名表、离群帧表 25 行(image_id/轮次/RMS/阈值)、逐帧摘要(每路 train/test 最高/最低/中位,明细指向 CSV)、配置快照附录;per_frame_cam{1,2}.csv(utf-8-sig)六列契约,156/170 行 = 全部 train(含剔除标记行)+ 15 test;JSON dropped_frames 四键(image_id/rms/round/threshold)逐项核验通过 |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `src/stereo_calib/intrinsic/models.py` | M2 唯一数学源(投影/雅可比/反投影/Rodrigues) | ✓ VERIFIED | 290 行;INTRINSIC_NAMES/intrinsics_matrix/transform_to_camera(cv2.Rodrigues 现成雅可比 L134)/project/project_jacobian/project_point_jacobian/unproject 全部真实实现;parity/数值差分/往返测试锁定 |
| `src/stereo_calib/intrinsic/observation.py` | npz 装载(exclude/detected/finite 三重防线 + objp 交叉核对) | ✓ VERIFIED | 171 行;复用 load_corners/build_object_points(L36),exclude→detected→finite 过滤链,objp allclose 交叉核对,min_views 中文报错 |
| `src/stereo_calib/intrinsic/metrics.py` | float64 回代 RMS 唯一指标尺 | ✓ VERIFIED | 77 行;compute_metrics pooled 口径;残差只经 models;本验证独立重算证实其输出与落盘数值精确一致 |
| `src/stereo_calib/intrinsic/solve.py` | OpenCV 初值 + scipy 精化 + 编排入口 | ✓ VERIFIED | 432 行;solve_opencv/refine_scipy/solve_calibration 齐全;calibrateCameraExtended flags=0、theta 装箱、row_index 行序契约、解析稀疏雅可比、实用收敛判定全在 |
| `src/stereo_calib/intrinsic/scoring.py` | test 只估位姿评分 + 失败契约 | ✓ VERIFIED | 117 行;unproject 反投影初值 → solvePnP → 6 参数 trf;_failure 契约(NaN/None/中文备注) |
| `src/stereo_calib/intrinsic/artifacts.py` | 五类产物落盘 | ✓ VERIFIED | 271 行;write_intrinsics_json/write_residuals_npz/write_extrinsics_npz/write_per_frame_csv/write_report_md 全部真实;roundtrip 测试锁定 |
| `src/stereo_calib/intrinsic/pipeline.py` | run_intrinsic 编排(零数学) | ✓ VERIFIED | 284 行;两路循环 + 剔除循环(中位数阈值 + min_views/max_drop_fraction/max_iterations 三保险丝)+ test 评分 + 中文汇总;无 `_distort`/`least_squares`/`calibrateCameraExtended`(零数学防线成立) |
| `src/stereo_calib/cli/main.py` | intrinsic 真实子命令 | ✓ VERIFIED | 含 run_intrinsic 局部导入、中文收口、typer.Exit(code=1);--help 实测退出码 0 |
| `config.yaml` | intrinsic 段全键 + 中文注释 | ✓ VERIFIED | L52-67:refine(6 键)/outlier(5 键)/rms_test_max/exclude_images 全键在案,逐键中文注释 |
| `output/intrinsic/`(生产产物 ×11) | 两路 JSON + 报告 + CSV ×2 + npz ×6 | ✓ VERIFIED | cam{1,2}/ 各 4 件(intrinsics.json/train_residuals.npz/test_residuals.npz/extrinsics.npz)+ per_frame_cam{1,2}.csv + report.md;全部可机读审计 |
| `.planning/REQUIREMENTS.md` | INTR-01..04 勾选 | ✓ VERIFIED | L21-24 四行全部 [x];Traceability L67-70 四行 Complete(标注交付/验收计划);页脚记录本次验收数值 |
| tests/test_intrinsic_{config,models,pipeline,solve,scoring}.py | 数值锁定 + e2e + 源码防线 | ✓ VERIFIED | 55 例收集实测;parity/雅可比×3/往返/真值恢复/精化开关/行序/评分失败契约/writer roundtrip/e2e 全契约/离群行为/CLI 行为/源码防线全覆盖 |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | --- | --- | ------ | ------- |
| cli/main.py | intrinsic/pipeline.py | 局部导入 run_intrinsic + except (ValueError, OSError) + Exit(1) | ✓ WIRED | main.py:84-91 实测 |
| pipeline.py | observation/solve/scoring/metrics | 编排只调模块入口 | ✓ WIRED | import L41-53;load_views ×4/solve_calibration ×4/score_pose_only ×3/compute_metrics ×3 调用点实测 |
| observation.py | preprocess/artifacts.py | import load_corners, build_object_points | ✓ WIRED | L36;绝不直接 np.load |
| metrics.py | models.py | 残差只由 models 构成 | ✓ WIRED | L70-71;`_distort` 计数 0 |
| solve.py | models.py + cv2.Rodrigues | 残差/雅可比经 models;domega 用 Rodrigues rot_jac | ✓ WIRED | `_distort` 计数 0;Rodrigues 调用在 models.transform_to_camera L134 |
| scoring.py | models.py | unproject 反投影初值 | ✓ WIRED | L90 models.unproject → L95 solvePnP |
| artifacts.py → intrinsics.json | Phase 3 固定内参接口 | params 按名 + image_size | ✓ WIRED | 契约写入时断言 + 落盘实测九名齐全;消费证明 deferred 到 Phase 3 |
| config.yaml intrinsic 段 | 求解/剔除/阈值行为 | RefineConfig/OutlierConfig/rms_test_max | ✓ WIRED | io/config.py _parse_intrinsic(9 例配置测试含 7 例非法值);solver_meta 记录 method/tr_solver 与配置一致 |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| intrinsics.json (×2) | params / rms_train_overall / rms_test_overall / dropped_frames | Phase 1 真实 npz(141/155/15/15 帧)→ load_views → solve_calibration → compute_metrics | 是——残差 npz 独立重算 pooled RMS 与 JSON 四值 diff 0.00e+00 | ✓ FLOWING |
| per_frame_cam{1,2}.csv | rms / dropped / note | 同一 image_ids 行序锁定的逐帧指标 + 剔除留痕 | 是——行数与 Phase 1 可用帧数一致,剔除行带中文 note | ✓ FLOWING |
| report.md | 汇总表/参数表/离群表 | pipeline summaries(与 JSON 同源) | 是——数值与 JSON 逐项一致(抽核 cam1 全部 9 参数 + 两路 RMS) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| 全量测试套件 | `uv run pytest -q` | 116 passed in 21.53s | ✓ PASS |
| CLI 真实命令 | `uv run stereo-calib intrinsic --help` | 「模块2」帮助文案,退出码 0,无占位残留 | ✓ PASS |
| JSON 契约审计 | 程序核验 15 字段全集 + params 九名 + 四键剔除契约 | 两路全部通过,无缺失字段 | ✓ PASS |
| float64 回代链路 | 从 residuals npz 按 metrics 口径独立重算 pooled RMS | cam1/cam2 train/test 四值 diff 全部 0.00e+00 | ✓ PASS |
| test RMS 达标 | JSON rms_test_overall 判定 | 0.1407 / 0.1361 均 < 1.0,且落在预期 0.1–0.5 带 | ✓ PASS |
| CSV 结构 | utf-8-sig 读取六列 | 156/170 行,dropped 25/0,test 帧零剔除 | ✓ PASS |
| data/ 只读 | `find data/ -type f … \| md5sum` | a183699da79875fc63f7cbfe75874398(与 Phase 1 基线一致) | ✓ PASS |
| ruff 计划范围 | `uv run ruff check src/stereo_calib/intrinsic/ src/stereo_calib/cli/main.py` | All checks passed | ✓ PASS |
| 完整生产命令重跑 | `uv run stereo-calib intrinsic` | SKIP——会改写 output/ 产物(约 8 分钟 + 状态变更);以产物时间戳、数值一致性、e2e 测试(test_cli_intrinsic_happy_path_end_to_end)作间接证据 | ? SKIP |
| probe 脚本 | find scripts tests -name "probe-*.sh" | 无(本项目以 pytest 为准) | ? SKIP |

### Probe Execution

无 `scripts/*/tests/probe-*.sh` 约定探针;PLAN/SUMMARY 亦未声明 probe 路径。行为验证由 pytest 116 例与上述 spot-checks 承担。

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ---------- | ----------- | ------ | -------- |
| INTR-01 | 02-01, 02-02, 02-03, 02-04 | train 集标定 M2 模型九参数:calibrateCameraExtended 初值 + scipy least_squares 精化 | ✓ SATISFIED | solve.py 双路径实现 + solver=="scipy" 生产证据 + 合成真值 fx 相对误差 < 1e-3 测试锁定 |
| INTR-02 | 02-03, 02-04 | 输出两路内参文件(参数按名 + 图像分辨率) | ✓ SATISFIED | 两份 intrinsics.json params 九名 + image_size [2560,1440](落盘实测) |
| INTR-03 | 02-01, 02-02, 02-03, 02-04 | test 集整体重投影 RMS(float64 回代,不用 OpenCV 返回值) | ✓ SATISFIED | metrics.py 唯一指标尺 + solver_meta.opencv_rms 隔离 + npz 独立重算精确一致 |
| INTR-04 | 02-03, 02-04 | 输出内参标定报告(逐帧误差 + 整体指标) | ✓ SATISFIED | report.md + per_frame CSV + dropped_frames 留痕 |

**Orphaned requirements:** 无——REQUIREMENTS.md 映射到 Phase 2 的 INTR-01..04 与四个 PLAN 的 requirements 字段完全对齐,无遗漏。

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| src/stereo_calib/intrinsic/scoring.py | 114 | except 元组缺 AssertionError(models.project Z>0 防线可逃逸,02-REVIEW CR-01) | ⚠️ Warning | 潜在健壮性缺口:病态 test 帧可致整批中断(生产运行未触发);修复一行,建议 Phase 3 前处理 |
| src/stereo_calib/io/config.py | 232-235, 308-312 | 容器型 yaml 值致 TypeError 绕过中文收口(02-REVIEW WR-01) | ⚠️ Warning | 手写配置笔误时吐堆栈而非中文一句;不影响本阶段目标 |
| src/stereo_calib/intrinsic/solve.py | 396-405 | jac_sparsity 在 callable jac 下被 scipy 静默忽略(02-REVIEW WR-02) | ⚠️ Warning | 死计算 + 文档口径不准,无正确性影响 |
| output/intrinsic(per_frame_cam1.csv) | — | cam1 幸存 train 帧仍有 4 帧 RMS > 1.0px(最高 4.87px,贡献该路 train 平方误差约 64%) | ℹ️ Info | 剔除循环按 RESEARCH Pattern 6 在初值解尺度运行所致,设计如此且有 CSV/report 如实留痕;不影响 test 门禁;可通过 exclude_images 手动叠加 |
| output/intrinsic(report.md) | — | 内参锚点偏差:fx/fy、k1 在参考锚点带外 | ℹ️ Info | SUMMARY 已如实记录并给出耦合方向分析;test 泛化 0.14px;非门禁项 → 路由人工评审 |

其余扫描零命中:全阶段 9 个源文件无 TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER 债务标记,无占位返回、无空实现、无 console 残留。

### Human Verification Required

### 1. 剔除帧质量人工抽查

**Test:** 打开 output/train/visualization/ 下 cam1 被剔 25 帧的角点标注图(intrinsics.json dropped_frames 列出 image_id),另看幸存帧 RMS > 1.0px 的 4 帧(cam1,最大 20261023_165247_020677_cam1 = 4.87px)与 cam2 的 1 帧(1.68px)。
**Expected:** 被剔帧应有可见劣化(模糊/遮挡/出画/角点漂移);幸存高 RMS 帧应可判断为轻度劣化或应加入 intrinsic.exclude_images。
**Why human:** 「这帧角点是不是真的坏」是视觉语义判断;数值审计只能证明剔除流程合规留痕,不能证明剔除恰如其分。

### 2. 内参锚点偏差专家评审

**Test:** 对照参考锚点带(fx/fy 1800–1950、k1 -0.5~-0.3)复核 report.md 参数表:cam1 fx/fy=2265/2230、cam2=2021/2015,k1=-0.750/-0.632 均在带外。
**Expected:** 确认接受当前解(test 泛化 0.14px + 两路同构收敛 + fx-k1 耦合已知退化方向),或决定 Phase 3 前补充数据重标。
**Why human:** 参数绝对正确性是领域专家判断;自动化验证只能证明内部一致性与泛化能力;该组内参将冻结为 Phase 3 固定输入。

### Gaps Summary

无技术性缺口。四条成功标准全部有代码库与生产产物双重实证,SCORE 4/4:

1. **SC1** — CLI 一条命令真实存在(help 实测退出码 0),solve.py 完整实现 calibrateCameraExtended(flags=0,float32 装包)与 scipy least_squares 精化(配置控制,开关关闭行为测试锁定),两路生产产物 solver=="scipy"。
2. **SC2** — 两份 intrinsics.json params 九名按名 + image_size [2560,1440] + K/D,Pattern 7 十五字段无缺失(程序核验),roundtrip 测试锁定;Phase 3 实际消费按设计留待下一阶段(deferred)。
3. **SC3** — RMS 唯一出口是 metrics.py float64 回代;OpenCV 返回值被隔离在 solver_meta(数值明显不同,非同源);本验证从残差 npz 独立重算与 JSON 精确一致(diff 0.00e+00),cam1 0.1407 / cam2 0.1361 < 1.0。
4. **SC4** — report.md 两路对比 + 逐帧摘要 + 25 行离群留痕表;CSV 156/170 行六列契约;dropped_frames 四键审计链完整。

过程证据:116 测试全绿、ruff 计划范围干净、data/ 指纹跨阶段不变、SUMMARY 声称的全部数值(含 25/141 剔除明细、阈值 2.8680、solver_meta status=0/nfev=300)与落盘产物逐项吻合,无一夸大。

状态为 **human_needed** 仅因两项领域判断(剔除帧视觉抽查、锚点偏差评审)无法程序化;按 OVERNIGHT AUTONOMY 授权,orchestrator 可在记录这两项后继续推进。02-REVIEW 的 CR-01(scoring 异常逃逸)不推翻目标达成——生产运行已成功、全部 must-haves 为真——但属应修的真实健壮性缺口,建议随 Phase 3 计划一并修复。

---

_Verified: 2026-10-23T18:16:26Z_
_Verifier: Claude (gsd-verifier)_
