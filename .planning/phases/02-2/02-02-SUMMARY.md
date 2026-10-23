---
phase: 02-2
plan: 02
subsystem: intrinsic(模块2 标定求解引擎)
tags: [intrinsic, solve, scipy-refine, pose-scoring, real-data-acceptance]
requires:
  - "02-01 models.py(M2 投影/解析雅可比/unproject/transform_to_camera 唯一数学源)"
  - "02-01 metrics.py(compute_metrics float64 回代唯一指标尺)"
  - "02-01 RefineConfig(精化开关/容差配置载体)+ observation.load_views"
  - "Phase 1 角点 npz 八数组契约(output/{train,test}/cam{n}_{split}.npz)"
provides:
  - "intrinsic/solve.py:SolveResult + solve_opencv(初值)/ refine_scipy(theta 装箱+解析稀疏雅可比+实用收敛)/ solve_calibration(编排入口,02-03 唯一消费)"
  - "intrinsic/scoring.py:score_pose_only(test 帧固定内参只估位姿,unproject 反投影初值,PoseScore 失败契约)"
  - "make_residual_fn/make_jacobian_fn/make_sparsity/row_index/pack_theta/unpack_theta(残差与雅可比供给侧)"
  - "真实数据引擎验收记录:cam1 141 帧 RMS 0.3316px / cam2 155 帧 RMS 0.2389px(< 1.0 px)"
affects: [02-03-pipeline, 02-04-artifacts-report]
tech-stack:
  added: []
  patterns:
    - "移植 + 不变量测试锁定(参考工程只读移植;θ 全维数值差分逐项校验)"
    - "行序契约单一出处:残差与雅可比共用 row_index(静默错序防线)"
    - "实用收敛判定:status/nfev/final_cost/tail_improvement/converged 全进 solver_meta 不静默"
key-files:
  created:
    - src/stereo_calib/intrinsic/solve.py
    - src/stereo_calib/intrinsic/scoring.py
    - tests/test_intrinsic_solve.py
    - tests/test_intrinsic_scoring.py
  modified: []
decisions:
  - "初值口径的切向对判据:p1/p2(真值 ~1e-4,像素效应 ~1e-3px,低于 0.05px 检测噪声)的相对误差无判别力(3 个种子探测 24–43%),改用绝对判别下限 5e-5;其余 7 参保持 plan 的 5e-2 相对阈值"
  - "精化关闭语义不与另一次 OpenCV 调用做逐位比对(实测其内部并行归约有 1e-12 非确定性),改锁'未发生精化'solver/元数据形态"
  - "refine 非收敛不抛异常:converged 标志进 solver_meta 由编排层消费(引擎层不吞数值异常,如实上抛交 02-03 中文收口)"
metrics:
  duration: 18m9s
  completed: 2026-10-24
  tasks: 3
  files: 4
---

# Phase 2 Plan 02: 标定求解引擎(OpenCV 初值 + scipy 精化 + 只估位姿评分)Summary

M2 标定引擎全链路落地:合成真值 12 视图从初值 5e-2 相对精度精化到 fx < 1e-3 / RMS < 0.2px;真实数据两路 train 验收 cam1 141 帧 0.3316px、cam2 155 帧 0.2389px(阈值 1.0px,与参考工程同类相机实测 0.09–0.16px 同量级),大倾角错误盆地回归锁定在 1px 内。

## What Was Built

- **Task 1 — solve.py OpenCV 初值路径**(RED 6c3fbde / GREEN f86af81):`SolveResult` frozen dataclass(无 rms 指标字段,OpenCV rms 只进 solver_meta 对账)+ `solve_opencv`:float32 显式装包(绑定层 32 位进、64 位出)、(宽,高) 口径注释(Pitfall 2)、flags=0 全 5 畸变、D 解包拉平后恰 5 长断言、第 8 个返回值 per_view 下划线不读(Pitfall 12);入口形状/视图数/image_size 中文防线,cv2.error 包装为中文 ValueError。合成真值 12 视图初值:非切向 7 参相对误差 < 5e-2(实测最大 4.6e-4),切向对按绝对判别下限。
- **Task 2 — solve.py scipy 精化路径**(RED 0831653 / GREEN 447def0):`pack_theta/unpack_theta`([9 内参 | 逐视图 6 外参],9+6V 维)、`row_index` 行序契约(残差与雅可比共用,静默错序防线)、`make_residual_fn/make_jacobian_fn/make_sparsity`(残差只经 models.project;domega 链式块由 models.transform_to_camera 内部经 cv2.Rodrigues 第二返回值给出,严禁手写;COO→csr,每残差行恰 15 非零);trf + lsmr + x_scale="jac" + jac_sparsity,loss 写死 linear(离群由 02-03 剔除机制处理);`_is_practically_converged` 纯函数(status>0 收敛 / <=-1 失败 / ==0 看尾部 10 次评估包络相对改进 < 1e-6);`solve_calibration` 编排入口由 RefineConfig.enabled 控制开关。合成真值:fx 相对误差 < 1e-3、回代 RMS < 0.2px、精化不劣化(<= 初值解 + 1e-6)、解析雅可比对 θ 全维数值差分逐项一致 < 1e-6(2 视图小场)。
- **Task 3 — scoring.py 只估位姿评分 + 真实数据验收**(RED 4e2c916 / GREEN c60ce37):`PoseScore(rms, residuals, note)` 失败契约(任何失败路径返回 NaN/None/中文备注,绝不抛异常中断整批)+ `score_pose_only`:models.unproject 反投影初值(天然去畸变)→ 归一化平面坐标 cv2.solvePnP(单位阵 K + 零畸变)→ 6 参数 trf 精化(内参加锁,残差只经 models)。大倾角错误盆地回归:40° 倾角 + 强畸变 + 板面偏离主点 ≥ 500px 合成帧评分 < 1.0px(参考工程老路线同场景 4.82px 真实案例的合成层锁定)。

## Verification Results

- `uv run pytest -q`:**102 passed**(既有 88 + 新增 14,零回归)
- `uv run ruff check src/stereo_calib/intrinsic/ tests/test_intrinsic_solve.py tests/test_intrinsic_scoring.py`:**干净**
- 源码防线:solve.py / scoring.py 源文 count 的畸变核符号为 0(残差/评分只经 models);solve.py 含 Rodrigues 雅可比路线声明与 trf+lsmr 配置;scoring.py 含反投影初值路线
- **真实数据引擎验收**(临时 symlink 主仓 output/ 后运行,已拆除):
  - cam1:141 视图,solver=scipy,回代 **RMS 0.3316 px**(OpenCV 内部 rms 2.5782 留对账),status=0 走平收敛
  - cam2:155 视图,solver=scipy,回代 **RMS 0.2389 px**(OpenCV 内部 rms 2.2053 留对账),status=0 走平收敛
  - 两路均 < 1.0 px 阈值(INTR-01 引擎达成;预期带 0.1–0.5px 与参考实测锚点 0.09px 同量级,留有余量——OpenCV 初值 RMS 偏高说明真实数据存在离群帧,正是 02-03 剔除机制的用武之地)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 初值口径的切向对判据修正(plan 阈值在该参数上物理不可达)**
- **Found during:** Task 1(RED 前数值探测)
- **Issue:** plan 要求"params 逐项相对误差 < 5e-2"。3 个种子探测实测:非切向 7 参全部 < 0.6%,但 p1(真值 1.1e-4)/p2(真值 -3.8e-5)相对误差 24–43%——它们对像素的全部影响在千分之几像素,比 0.05px 检测噪声低一个量级以上,12 视图下相对误差对其没有判别力(即使无噪估计器也定不到 5%)。
- **Fix:** 非切向 7 参保持 plan 的 5e-2 相对阈值;p1/p2 改用绝对判别下限 5e-5(对应像素效应 ~0.005px,仍比噪声低一个量级),口径与理由记录在测试常量注释。
- **Files modified:** tests/test_intrinsic_solve.py
- **Commit:** 6c3fbde

**2. [Rule 1 - Bug] 精化关闭用例的逐位比对修正(OpenCV 内部非确定性)**
- **Found during:** Task 2(GREEN 期)
- **Issue:** 用例原写法把 solve_calibration(enabled=False) 的结果与另一次独立 solve_opencv 调用做 np.array_equal 逐位比对——实测 OpenCV 内部并行归约有 ~1e-12 非确定性(两次调用 opencv_rms 在 1e-12 位不同),逐位相等不是契约语义。
- **Fix:** 改锁"未发生精化"语义:solver == "opencv"、solver_meta 无精化收敛键、初值口径真值恢复;理由记录在测试 docstring。
- **Files modified:** tests/test_intrinsic_solve.py
- **Commit:** 447def0

**3. [Rule 3 - Blocking] OpenCV 的 D 以 (1,5) 矩阵返回**
- **Found during:** Task 1(GREEN 期)
- **Issue:** 解包断言 D.shape == (5,) 直接失败——OpenCV 实际返回 (1,5) 矩阵形状。
- **Fix:** 先 ravel 再断言恰 5 长(断言文字保留 plan 指定的字面形式),注释写明。
- **Files modified:** src/stereo_calib/intrinsic/solve.py
- **Commit:** f86af81

## Auth Gates

None — 离线数值工具,零外部认证面。

## Known Stubs

None — 求解引擎与评分器均为完整实现,无占位/无 mock 数据源。02-03(剔除编排/CLI)是计划内的后续切片,非本计划 stub。

## Threat Flags

None — 计划 `<threat_model>` 三条 mitigate 全部落地并有测试锁定:T-02-04(solve 入口形状/视图数/image_size 再断言,中文防线)/ T-02-05(status/nfev/final_cost/tail_improvement/converged/opencv_rms 全进 solver_meta,不静默)/ T-02-06(unproject 反投影初值路线 + 大倾角错误盆地回归测试);T-02-SC 零新增包成立(uv.lock 未变)。

## Self-Check: PASSED

- src/stereo_calib/intrinsic/{solve,scoring}.py 存在 ✓
- tests/test_intrinsic_{solve,scoring}.py 存在 ✓
- 提交 6c3fbde / f86af81 / 0831653 / 447def0 / 4e2c916 / c60ce37 均在 git log ✓
- 全量 102 测试通过、ruff 干净、真实数据两路 RMS < 1.0px(见上方 Verification Results)✓
