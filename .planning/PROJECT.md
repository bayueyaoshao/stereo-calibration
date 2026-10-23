# 双目相机标定生产工具 (stereo-calibration)

## What This Is

基于已采集的同步双目图像对(data/train 188 组、data/test 15 组,cam1.jpg/cam2.jpg),快速计算可靠的双目相机标定结果的生产工具。分三个相互独立、可单独调试的模块:数据预处理(角点检测与筛选)、单相机内参标定与验证、双目外参标定与验证。使用者是标定产线操作人员/工程师,输入图像数据,输出标定参数与量化指标。

## Core Value

快速、可靠地产出经过测试集指标验证的双目标定结果(内参 + 双目外参 R/T)。

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] 模块1 数据预处理:对每路相机独立做棋盘格角点检测(SB 主路线 + classic 兜底,窗口自适应推导),筛选出棋盘格完整的图像
- [ ] 模块1 输出:每路相机可用图像清单、两路相机公共视场图像对清单、所有图像对应的角点数据
- [ ] 模块2 内参标定:对两路相机分别标定 M2 模型(fx,fy,cx,cy,k1,k2,p1,p2,k3),OpenCV calibrateCameraExtended 初值 + scipy least_squares 精化
- [ ] 模块2 输出:两路相机内参文件 + test 数据集整体重投影 RMS 指标
- [ ] 模块3 双目标定:用固定内参 + 公共视场角点求外参 R/T(PnP-median 初值 + 外参精化)
- [ ] 模块3 输出:双目外参 R/T + test 数据集棋盘格最远两角点物理距离残差 RMS 指标
- [ ] 三个模块各自可独立运行调试,输入输出明确,通过配置文件串联

### Out of Scope

- 相机类型判断/鱼眼模型 — 用户明确指定不管相机类型,统一 M2 模型
- 图像采集功能 — 数据已采集完成,本工具只做离线计算
- 在线/实时标定 — 生产离线用途,只算这一次刚性支架的数据
- 修改参考工程 — 两个参考工程只读,绝不修改

## Context

- **数据结构**:`data/train/`(188 组)与 `data/test/`(15 组);每组目录 `<timestamp>/<timestamp>_<us>/` 下含 `cam1.jpg` + `cam2.jpg` 同步图像对
- **数据特点**:很多图像对两路相机不能同时看到棋盘格——这些图像仍可用于各自内参标定,但只能用公共视场对做双目外参
- **标定板**:chessboard,11×8 内角点(88 个),方格 25.0mm;对角两角点真实距离 = sqrt((11-1)² + (8-1)²) × 25mm
- **参考工程(只读,绝不修改)**:
  - 内参标定:`/home/lx/Works/src/lx/github/intrinsic-calibration`
    - Python ≥3.12, opencv 4.x, numpy, scipy, typer CLI,uv 构建
    - 角点:`findChessboardCornersSB`(输出已亚像素,严禁二次精化)+ classic 兜底(`cornerSubPix` 窗口从本帧实测方格像素边长推导,缺陷闸门+窗口阶梯重提)
    - M2 模型 = OpenCV 默认五系数 dist `[k1,k2,p1,p2,k3]`,flags=0(非 RATIONAL);`calibrateCameraExtended` + scipy `least_squares(trf)` 精化
    - RMS 用 float64 回代计算,不用 OpenCV 返回值;输出 result.json + extrinsics.npz + residuals.npz
  - 多目标定:`/home/lx/Works/src/bolight_robot/vvd_daq/multi_cameras/data_acquire/calibration`
    - **不用 `cv2.stereoCalibrate`**(该机型有 bug):每共视图两路各 solvePnP → `R_rel = R_b@R_a.T, t_rel = t_b − R_rel@t_a` → 全帧取中位数(R 转 Rodrigues 向量取中位)
    - 最远角点距离残差已验证实现(`cli/06_evaluate.py`):角点序 180° 消歧 → per-frame 板位姿 → `undistortPoints` 去畸变 → 多视图 DLT 三角化 → 板系反变换 → 对角距离残差
- **同类相机已在参考工程中研究透彻**,本项目复用其方案,不重新发明

## Constraints

- **Tech stack**: Python + OpenCV 4.x + numpy + scipy,参考工程1风格(配置文件驱动,命令行零算法参数)— 与已验证工程保持一致
- **配置文件**: 所有传参必须用配置文件控制(棋盘格规格、路径、算法参数、开关),禁止硬编码
- **模块分离**: 三个模块相互独立,每个可单独运行调试,输入输出文件明确;上游输出即下游输入
- **依赖**: 只读引用两个参考工程的方案与代码,绝不修改其任何内容
- **性能**: 生产用途,快速计算——角点检测可并行(两路相机、多图像)
- **数据只读**: data/ 目录图像不可修改,所有中间产物和结果写到独立输出目录

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 畸变模型固定 M2(p1,p2,k1,k2,k3) | 参考工程1已验证的同类别相机最优模型 | — Pending |
| 内参求解:OpenCV 初值 + scipy 精化 | 工程1完整方案,精度更高,配置可开关 | — Pending |
| 双目外参:PnP-median 初值 + 外参精化(不用 stereoCalibrate) | 工程2验证 stereoCalibrate 该机型有 bug | — Pending |
| 角点检测:SB 主路线 + classic 兜底 | 工程1验证的双检测器策略 | — Pending |
| 双目验证指标:最远两角点物理距离残差 RMS | 直接反映测量精度,工程2已验证实现 | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-10-23 after initialization*
