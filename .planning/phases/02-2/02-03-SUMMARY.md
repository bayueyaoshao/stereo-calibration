---
phase: 02-2
plan: 03
subsystem: intrinsic(模块2 编排与产物契约)
tags: [intrinsic, pipeline, outlier-removal, artifacts-contract, e2e]
requires:
  - "02-01 observation.load_views / metrics.compute_metrics / models.INTRINSIC_NAMES"
  - "02-02 solve.solve_calibration(RefineConfig 开关)/ scoring.score_pose_only"
  - "Phase 1 角点 npz 八数组契约 + load_corners(allow_pickle=False)"
provides:
  - "intrinsic/pipeline.py:run_intrinsic 模块唯一编排入口(两路循环 + OVERNIGHT AUTONOMY 剔除机制 + test 评分 + 中文汇总)"
  - "intrinsic/artifacts.py:write_intrinsics_json(Pattern 7 契约,Phase 3 按名读取的唯一固定内参接口)/ write_residuals_npz / write_extrinsics_npz / write_per_frame_csv / write_report_md"
  - "output/intrinsic/ 产物布局契约:cam{n}/intrinsics.json + 三件套 npz + per_frame_cam{n}.csv + report.md"
  - "e2e 证据:合成 npz 全契约 + 离群行为 + 清理/只读防线(测试锁定,INTR-02/INTR-04 落盘面全有自动化断言)"
affects: [02-04-cli, phase-3-stereo]
tech-stack:
  added: []
  patterns:
    - "编排零数学:pipeline 只调模块入口,禁符号源码防线测试锁定"
    - "剔除全程留痕:image_id/rms/轮次/阈值进 JSON dropped_frames 与报告离群表(T-02-08)"
    - "writer roundtrip 契约锁定:JSON 字段全集断言 + npz allow_pickle=False 读回 + CSV BOM/列序"
key-files:
  created:
    - src/stereo_calib/intrinsic/artifacts.py
    - src/stereo_calib/intrinsic/pipeline.py
  modified:
    - tests/test_intrinsic_pipeline.py
decisions:
  - "params 真值恢复判据按噪声限分参设定(5 种子实测):fx/fy/cx/cy 相对 < 1e-3;k1/k2/k3 绝对 < 1e-3/2e-3/2e-3;p1/p2 绝对 < 1e-4——plan 原文'逐项相对 < 1e-3'对高阶畸变项在 12 视图/0.05px 噪声下物理不可达(k2/k3 实测 2e-3~1.6e-2),沿用 02-02 切向对判据修正的同一逻辑"
  - "离群坏帧注入模型由'恒定 +15px 偏移'修正为同量级逐角点散布(N(0,15)px):恒定偏移等价于棋盘小平移,会被该帧自己的位姿解几乎完美吸收(实测注入后该帧逐帧 RMS 仅 0.2px,永远到不了剔除阈值)"
  - "e2e 配置 min_views 显式取 2:plan 的 4 帧 test 划分过不了 min_views=10 装载下限;剔除保险丝余量由 max_drop_fraction 兜底(12 帧下限 9.6,可剔 2 帧)"
metrics:
  duration: 17m
  completed: 2026-10-24
  tasks: 3
  files: 3
---

# Phase 2 Plan 03: 模块2 编排与产物契约(run_intrinsic + Pattern 7 落盘)Summary

run_intrinsic 一次调用从 Phase 1 角点 npz 走到五类产物:两路串行标定 + train 离群自动剔除(中位数阈值 × 5,三保险丝,全程留痕)+ test 只估位姿评分;intrinsics.json Pattern 7 契约(Phase 3 固定内参接口)与逐帧 CSV/报告全经 roundtrip 与 e2e 测试锁定,合成数据上 train/test RMS ≈ 0.07px、fx 相对误差 3.6e-5。

## What Was Built

- **Task 1 — artifacts.py 落盘层**(bde98bf):`write_intrinsics_json` 逐字段锁定 Pattern 7 十五项契约(cam/image_size/params 按名/K/D/solver/refine_enabled/rms_train_overall/rms_test_overall/n_views_train/n_views_test/dropped_frames/solver_meta/board/config_snapshot),字段集漂移当场断言;`write_residuals_npz`(image_ids 定长 unicode 对账列 + (V,88,2) float64 + per_view_rms,NaN 失败留痕语义)/`write_extrinsics_npz`(docstring 注明模块 3 不消费)allow_pickle=False 可读;`write_per_frame_csv` utf-8-sig 六列契约;`write_report_md` f-string markdown(两路指标对比 PASS/FAIL、按名参数表、离群留痕表、逐帧摘要、配置快照附录)。writer 单元 roundtrip 测试 5 例。
- **Task 2 — pipeline.py run_intrinsic 编排**(1aa586e):运行前 rmtree output_root/intrinsic(安全前提注释:load_config 互斥校验兜底,T-02-07);每路串行:装载 → 剔除循环(仅 train;全程 OpenCV 初值解,float64 回代逐帧 RMS 唯一判据尺(Pitfall 12,OpenCV per-view 不读),阈值 = rms_multiplier × 幸存中位数;min_views/max_drop_fraction/max_iterations 三保险丝,触发原因如实进报告绝不静默,T-02-08;outlier.enabled 开关生效)→ 幸存集定稿一次精化 → test 逐帧 score_pose_only(单帧失败 NaN + 中文备注不中断;全部失败中文 ValueError,T-02-09)→ 五类产物落盘 → 两路完成后单份 report.md + 中文汇总与产物绝对路径清单。零数学源码防线测试锁定。
- **Task 3 — 合成 npz e2e**(81d3c10):合成 npz 工厂(真值投影 + 0.05px 噪声,八数组契约,渲染真实小图进 data 树供 image_size 派生);e2e 全契约(Pattern 7 字段全集、params 真值恢复、image_size=[2560,1440]、RMS 有限正、dropped_frames 空、CSV 16 行、report 两路 PASS、residuals npz 行序对账、solver_meta 收敛留痕);离群行为(train 坏帧自动剔除留痕 + test 坏帧永不剔除只标记,12 视图保险丝余量断言);重跑 rmtree 清 stale;data 树快照逐字节不变。共 6 例 e2e + 1 例源码防线。

## Verification Results

- `uv run pytest -q`:**114 passed**(02-02 后 102 + 新增 12,零回归;e2e 全文件 3.97s)
- `uv run ruff check src/stereo_calib/intrinsic/pipeline.py src/stereo_calib/intrinsic/artifacts.py tests/test_intrinsic_pipeline.py`:**干净**(test_corners.py RUF007、test_preprocess_pipeline.py C408 等 5 处为 Phase 1 遗留,不在本计划范围,未触碰)
- 合成数据 e2e 实测(种子 7/3,两路):train RMS 0.0702/0.0708px,test RMS 0.0726/0.0707px,fx 相对误差 3.6e-5/3.1e-4,dropped_frames 空,判定 PASS
- 离群注入实测:train 坏帧逐帧 RMS 20.1px vs 阈值 0.75px → 恰剔 1 帧,轮次/阈值留痕;test 坏帧 RMS ~20px 如实标记、dropped 恒 False
- 源码防线:pipeline.py 无 `_distort`/`least_squares`/`calibrateCameraExtended`,含 rmtree/load_views/solve_calibration/score_pose_only/compute_metrics/median/rms_multiplier 全部编排入口

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] params 真值恢复判据按噪声限分参设定(plan 的逐项相对 1e-3 对高阶畸变项物理不可达)**
- **Found during:** Task 3(写 e2e 断言前的 5 种子数值探测)
- **Issue:** plan 要求"params 对真值逐项相对误差 < 1e-3"。5 个种子(3/7/42/101/202)实测:fx/fy/cx/cy 全部 < 8.2e-4 ✓,但 k1 达 1.3e-3、k2 达 4e-3、k3 达 1.6e-2——高阶畸变项在 12 视图/0.05px 噪声下弱约束,相对误差没有判别力(k3 误差 1.3e-3 的像素效应 ~1e-4px 量级,低于检测噪声);p1/p2 沿 02-02 同款结论(绝对判别下限)。
- **Fix:** fx/fy/cx/cy 保持相对 < 1e-3;k1/k2/k3 改绝对 < 1e-3/2e-3/2e-3;p1/p2 绝对 < 1e-4。口径与 5 种子证据记录在测试常量注释,种子选实测居中的 7/3(两路)。
- **Files modified:** tests/test_intrinsic_pipeline.py
- **Commit:** 81d3c10

**2. [Rule 1 - Bug] 离群坏帧注入模型修正(恒定偏移会被位姿解吸收,剔除机制无法触发)**
- **Found during:** Task 3(e2e 首跑 dropped_frames 为空)
- **Issue:** plan 指定"注入 +15px 恒定偏移"的坏帧。实测恒定偏移等价于棋盘小平移(t 位移 ≈ 15px×Z/fx ≈ 5mm),被该帧自己的位姿解几乎完美吸收——注入后该帧逐帧 RMS 仅 0.207px,中位数 0.072、阈值 0.36,永远不会被剔,OVERNIGHT AUTONOMY 剔除机制在 e2e 里成了无法验证的死代码。
- **Fix:** 注入改为同量级逐角点散布(N(0,15)px,种子 99 固定)——真实"该帧角点整体劣化"异常模型;实测该帧逐帧 RMS 20.1px vs 阈值 0.75px,恰剔 1 帧、不误伤好帧。口径记录在 _synth_npz 注释。
- **Files modified:** tests/test_intrinsic_pipeline.py
- **Commit:** 81d3c10

**3. [Rule 3 - Blocking] e2e 配置 min_views 显式取 2(4 帧 test 划分过不了默认装载下限)**
- **Found during:** Task 3(夹具设计期)
- **Issue:** plan Task 2 规定 test 与 train 同法装载(共用 outlier.min_views,默认 10),而 plan Task 3 的 e2e 只有 4 帧 test——min_views=10 会在装载层直接中文报错,两条任务规格互相矛盾。
- **Fix:** 生产代码不改(照 plan 传同一配置值);e2e 配置 intrinsic 段全键显式且 min_views: 2(配置驱动,不硬编码)。剔除保险丝余量相应改由 max_drop_fraction 兜底:12 帧下限 max(2, 9.6)=9.6,可剔 2 帧,剔 1 帧的离群用例不受影响。plan 原文"可剔上限 = 12 - max(10, 0.8×12) = 2"的算术按此口径平移。
- **Files modified:** tests/test_intrinsic_pipeline.py(_E2E_CONFIG_YAML)
- **Commit:** 81d3c10

## Auth Gates

None — 离线数值工具,零外部认证面。

## Known Stubs

None — 编排与落盘层均为完整实现;02-04(CLI 子命令接入)是计划内的后续切片,非本计划 stub。

## Threat Flags

None — 计划 `<threat_model>` 三条 mitigate 全部落地并有测试锁定:T-02-07(rmtree 目标仅 output_root/intrinsic,安全前提注释,全部写路径从 cfg.output_root 派生)/ T-02-08(剔除帧 image_id/rms/轮次/阈值进 JSON dropped_frames 与报告离群表,保险丝触发如实留痕,e2e 断言)/ T-02-09(test 全部失败中文 ValueError,不静默产出 NaN 报告);T-02-SC 零新增包成立(uv.lock 未变)。

## Self-Check: PASSED

- src/stereo_calib/intrinsic/{artifacts,pipeline}.py 存在 ✓
- tests/test_intrinsic_pipeline.py 存在且含 writer 单元 + e2e 全套 ✓
- 提交 bde98bf / 1aa586e / 81d3c10 均在 git log ✓
- 全量 114 测试通过、ruff(计划范围文件)干净、零数学源码防线锁定(见上方 Verification Results)✓
