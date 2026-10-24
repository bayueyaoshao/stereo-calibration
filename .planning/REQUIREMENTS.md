# Requirements: 双目相机标定生产工具

**Defined:** 2026-10-23
**Core Value:** 快速、可靠地产出经过测试集指标验证的双目标定结果(内参 + 双目外参 R/T)

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### 模块1 — 数据预处理 (PREP)

- [x] **PREP-01**: 读取 YAML 配置文件(棋盘格规格、数据路径、输出路径、角点检测参数),命令行零算法参数
- [x] **PREP-02**: 对每路相机全部图像做棋盘格角点检测:SB 主路线(输出已亚像素,严禁二次精化)+ classic 兜底(cornerSubPix 窗口从本帧实测方格像素边长推导),支持多进程并行加速
- [x] **PREP-03**: 输出每路相机"棋盘格完整可用"图像清单
- [x] **PREP-04**: 输出两路相机公共视场图像对清单(同组目录两路均检测成功的图像对)
- [x] **PREP-05**: 输出所有成功图像的角点数据文件(2D 角点坐标 + 对应 3D 板系坐标 + 图像索引)
- [x] **PREP-06**: 输出角点可视化抽查图(绘制检测角点,便于人工核验)

### 模块2 — 内参标定 (INTR)

- [x] **INTR-01**: 用 train 集每路相机全部可用图像标定 M2 模型(fx,fy,cx,cy,k1,k2,p1,p2,k3):`cv2.calibrateCameraExtended` 初值 + scipy `least_squares` 精化
- [x] **INTR-02**: 输出两路相机内参文件(参数按名保存 + 图像分辨率)
- [x] **INTR-03**: 用 test 集角点计算每路相机整体重投影 RMS(float64 回代计算,不用 OpenCV 返回值)
- [x] **INTR-04**: 输出内参标定报告(逐帧误差 + 整体指标)
- [x] **INTR-05**(quick 261024 增补): 内参模型充分性体检闸门——A0 残差能量比 / A1 残差径向结构 / A2 去畸变直线度(仅报告不判红)/ A3 切片 fx 稳定性 / A4 升阶 rational8 探针;红检数 ≥ min_red_count 判「内参不可靠」,fail-closed 拦截其进入双目标定(产物留痕后中文报错、退出码 1),阈值全部进 config.yaml

### 模块3 — 双目标定 (STER)

- [ ] **STER-01**: 用固定内参 + train 集公共视场角点求双目外参:PnP-median 初值(每共视图两路 solvePnP → 相对位姿 → 全帧中位数,不用 `cv2.stereoCalibrate`)+ scipy 外参精化
- [ ] **STER-02**: 输出双目外参 R/T(cam1 为参考系)及 baseline 距离
- [ ] **STER-03**: test 集验证:角点序 180° 消歧 → 去畸变 → DLT 三角化 → 板系最远两角点物理距离 vs 真值(√(10²+7²)×25mm),输出残差 RMS
- [ ] **STER-04**: 输出双目标定报告(每帧距离残差 + 整体 RMS)

### 工程基础 (INFRA)

- [x] **INFRA-01**: 三个模块独立 CLI 入口,可单独运行调试,上游输出即下游输入,文件接口明确
- [x] **INFRA-02**: 全部传参由配置文件控制,禁止硬编码算法参数
- [x] **INFRA-03**: data/ 目录只读,所有中间产物/结果写入独立输出目录

## v2 Requirements

(None — 一次性生产标定工具,无预留功能)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| 相机类型自动判断/鱼眼模型 | 用户明确指定统一 M2 模型(p1p2k1k2k3) |
| 图像采集功能 | 数据已采集完成,工具只做离线计算 |
| 在线/实时标定 | 生产离线用途,只算这一次刚性支架数据 |
| 修改参考工程 | 两个参考工程只读,绝不修改任何内容 |
| cv2.stereoCalibrate | 参考工程2验证该机型此函数有 bug,改用 PnP-median + 精化 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PREP-01 | Phase 1 | Complete |
| PREP-02 | Phase 1 | Complete (01-02) |
| PREP-03 | Phase 1 | Complete |
| PREP-04 | Phase 1 | Complete (01-03) |
| PREP-05 | Phase 1 | Complete |
| PREP-06 | Phase 1 | Complete (01-03) |
| INTR-01 | Phase 2 | Complete (02-02 引擎, 02-04 验收) |
| INTR-02 | Phase 2 | Complete (02-03 契约, 02-04 验收) |
| INTR-03 | Phase 2 | Complete (02-02 引擎, 02-04 验收) |
| INTR-04 | Phase 2 | Complete (02-03 报告, 02-04 验收) |
| INTR-05 | Phase 2 (quick 261024 增补) | Complete (体检闸门 fail-closed 接入, 生产双机判不可靠) |
| STER-01 | Phase 3 | Pending |
| STER-02 | Phase 3 | Pending |
| STER-03 | Phase 3 | Pending |
| STER-04 | Phase 3 | Pending |
| INFRA-01 | Phase 1 | Complete |
| INFRA-02 | Phase 1 | Complete |
| INFRA-03 | Phase 1 | Complete (01-03) |

**Coverage:**

- v1 requirements: 18 total
- Mapped to phases: 18 ✓
- Unmapped: 0

---
*Requirements defined: 2026-10-23*
*Last updated: 2026-10-24 after quick 261024-intrinsic-adequacy-battery (INTR-05 增补:内参模型充分性体检闸门 fail-closed 接入内参管线,生产重跑双机判「不可靠」退出码 1;前次 2026-10-24 after 02-04:INTR-01/02/03/04 complete,两路 test RMS 0.1407/0.1361px)*
