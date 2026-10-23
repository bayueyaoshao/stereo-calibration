---
phase: 02-2
plan: 01
subsystem: intrinsic(模块2 内参标定地基)
tags: [intrinsic, math-source, observation, metrics, config]
requires:
  - "Phase 1 角点 npz 八数组契约(output/{train,test}/cam{n}_{split}.npz)"
  - "io/config.py frozen dataclass + _fail 中文校验模式"
  - "preprocess/artifacts.py 的 load_corners/build_object_points"
provides:
  - "intrinsic/models.py:M2 投影/解析雅可比/牛顿反投影/transform_to_camera(唯一数学源)"
  - "intrinsic/observation.py:npz 装载 Views(exclude/detected/finite 三重防线 + objp 交叉核对 + 行序锁 + image_size 派生)"
  - "intrinsic/metrics.py:compute_metrics float64 回代 RMS(唯一指标尺,pooled 口径)"
  - "IntrinsicConfig/RefineConfig/OutlierConfig + _parse_intrinsic 接入 load_config"
  - "config.yaml intrinsic 段(Phase 2 全部算法参数/开关)"
affects: [02-02-solve, 02-03-pipeline, 02-04-artifacts-report]
tech-stack:
  added: []
  patterns:
    - "移植 + 不变量测试锁定(参考工程只读移植,parity/数值差分/真值噪声三层锁)"
    - "单一数学源:残差/精化/评分只准 import models(源码防线测试锁定)"
key-files:
  created:
    - src/stereo_calib/intrinsic/__init__.py
    - src/stereo_calib/intrinsic/models.py
    - src/stereo_calib/intrinsic/observation.py
    - src/stereo_calib/intrinsic/metrics.py
    - tests/test_intrinsic_config.py
    - tests/test_intrinsic_models.py
    - tests/test_intrinsic_pipeline.py
  modified:
    - src/stereo_calib/io/config.py
    - config.yaml
decisions:
  - "强畸变边缘点测试选点避开 k1=-0.40 折返边界:极端图像角像素落在畸变映射近奇异区(物理不可逆),测试改用 r²≈0.3 量级的强畸变边缘点(机器精度收敛),口径记录在测试 docstring"
  - "装载层错误统一 ValueError(非参考工程的 AssertionError):进入 CLI except (ValueError, OSError) 中文收口,操作员可读"
  - "metrics 源码防线锁定 models.project( 恰 1 次、metrics 无畸变核符号、observation 无直接 np.load 调用(计数形式避开 docstring 误伤)"
metrics:
  duration: 15m21s
  completed: 2026-10-24
  tasks: 3
  files: 9
---

# Phase 2 Plan 01: 内参标定地基(配置 + 数学源 + 装载 + 指标尺)Summary

M2 相机模型唯一数学源(models.py)与 npz 装载三重防线落地:cv2.projectPoints parity < 1e-9px、三组解析雅可比对数值差分 < 1e-6、往返投影 < 1e-9px 全部测试锁定;真实 Phase 1 数据装载冒烟 141/155/15/15 帧 @ (2560,1440)。

## What Was Built

- **Task 1 — 配置层 intrinsic 段**(46d8fe1):`RefineConfig`/`OutlierConfig`/`IntrinsicConfig` 三只 frozen dataclass(默认值逐字取 RESEARCH Pattern 8),`_parse_intrinsic` 全键中文校验(bool 冒充 int 拦截、method 白名单只收 "trf" 拒收 'lm'、rms_multiplier > 1.0、max_drop_fraction ∈ (0,1)、min_views ≥ 2),接入 `load_config`;config.yaml 追加 intrinsic 段逐键中文注释;缺段向后兼容走默认。9 例测试全绿(合法全字段断言 + 7 例非法值参数化 + 缺段兼容)。
- **Task 2 — models.py 移植**(30dc0dc):参考工程 pinhole_brown.py(M2 投影/`project_jacobian`/`project_point_jacobian`/`unproject` 牛顿迭代/`intrinsics_matrix`)+ base.py `transform_to_camera`(cv2.Rodrigues 现成雅可比,严禁手写 −[Xc]×)移植为纯函数模块;裁剪多模型分支与注册机制;`INTRINSIC_NAMES` 按名布局。9 例数值锁定测试(parity/雅可比×3/往返/Z 防线/强畸变收敛/源码防线)。
- **Task 3 — 装载层 + 指标尺**(3cc62df):`observation.py` 复用 `load_corners`(allow_pickle=False 绝不绕过)+ 三重防线(exclude → detected → finite 断言)+ objp allclose 交叉核对(不一致提示重跑 preprocess)+ 行序锁 + image_size (宽,高) 派生含首/中/末 3 帧抽查;`metrics.py` `compute_metrics` pooled RMS 唯一指标尺(残差只由 models 构成)。9 例测试 + 真实数据冒烟。

## Verification Results

- `uv run pytest -q`:**88 passed**(既有 61 + 新增 27,零回归)
- `uv run ruff check`(全部新/改文件):**干净**(既有 cli/main.py B008、test_corners.py RUF007 等 6 处为 Phase 1 遗留,不在本计划范围,未触碰)
- 真实数据冒烟:四份 npz 幸存 **141/155/15/15**,image_size 全部 **(2560, 1440)**,objp[87] == [250, 175, 0]
- 真值 + 0.05px 噪声:pooled RMS ≈ 0.0707 ∈ (0.03, 0.08) 落带(test 锁定)
- 源码防线:metrics.py 无畸变公式;observation.py 无直接 np.load;models.py 无多模型分支/注册痕迹

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 强畸变边缘点测试选点修正**
- **Found during:** Task 2
- **Issue:** 计划的"强畸变边缘点"最初取图像极端角像素 (0,0)/(2559,1439)——在 k1=-0.40 强畸变下这些像素落在畸变映射可逆区的折返边界(r²≈1.42,雅可比近奇异),牛顿迭代 50 步不收敛(往返差 ~10px);fsolve 同样停滞,解析雅可比经有限差分逐项验证无误——是模型物理边界而非实现缺陷。
- **Fix:** 测试选点改为强畸变但物理可逆的边缘点(归一化 |xd|≈0.5+,r²≈0.3 量级,远超真实棋盘角点工作范围),机器精度收敛;口径与"为何不用极端角像素"记录在测试 docstring。实现保持与参考工程逐字一致,未改。
- **Files modified:** tests/test_intrinsic_models.py
- **Commit:** 30dc0dc

**2. [Rule 3 - Blocking] 源码防线计数被 docstring 提及误伤**
- **Found during:** Task 2
- **Issue:** 防线断言 `source.count("cv2.Rodrigues") == 1` 被 transform_to_camera docstring 里的方法名提及击中(计数 2)。
- **Fix:** docstring 改写措辞避开调用串原文(照 test_corners.py 防线自伤规避体例),断言改锁调用形式 `cv2.Rodrigues(` 恰 1 次;Task 3 的 np.load 防线同样用 `np.load(` 调用形式规避 docstring 误伤。
- **Files modified:** src/stereo_calib/intrinsic/models.py, tests/test_intrinsic_models.py, tests/test_intrinsic_pipeline.py
- **Commit:** 30dc0dc / 3cc62df

## Auth Gates

None — 离线数值工具,零外部认证面。

## Known Stubs

None — 本计划三个模块全部为完整实现,无占位/无 mock 数据源。02-03(e2e)与 02-02(solve)是计划内的后续切片,非本计划 stub。

## Threat Flags

None — 未引入计划 `<threat_model>` 之外的安全面:T-02-01(npz 只经 load_corners)/ T-02-02(yaml.safe_load + _parse_intrinsic 全键校验)/ T-02-03(三重防线 + allclose 交叉核对)全部按缓解计划落地并有测试锁定;T-02-SC 零新增包成立(uv.lock 未变)。

## Self-Check: PASSED

- src/stereo_calib/intrinsic/{__init__,models,observation,metrics}.py 存在 ✓
- tests/test_intrinsic_{config,models,pipeline}.py 存在 ✓
- 提交 46d8fe1 / 30dc0dc / 3cc62df 均在 git log ✓
- 全量 88 测试通过、ruff 干净(见上方 Verification Results)✓
