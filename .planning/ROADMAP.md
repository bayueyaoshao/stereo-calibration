# ROADMAP: 双目相机标定生产工具

**Core Value:** 快速、可靠地产出经过测试集指标验证的双目标定结果(内参 + 双目外参 R/T)

**Granularity:** coarse(3 阶段,每阶段交付一个可端到端独立运行的模块)

## Phases

- [x] **Phase 1: 工程骨架 + 模块1 数据预处理** — 配置驱动工程骨架,角点检测/筛选/清单输出端到端可运行 (completed 2026-10-23)
- [ ] **Phase 2: 模块2 内参标定** — 两路相机 M2 内参标定 + test 集 RMS 验证
- [ ] **Phase 3: 模块3 双目标定** — PnP-median + 精化求外参 R/T,test 集物理距离残差验证

## Phase Details

### Phase 1: 工程骨架 + 模块1 数据预处理

**Goal:** 建立配置文件驱动的工程骨架,数据预处理模块可独立运行:对 data/train 与 data/test 全部图像完成角点检测与筛选,产出两路相机可用清单、公共视场对清单与角点数据文件
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: PREP-01, PREP-02, PREP-03, PREP-04, PREP-05, PREP-06, INFRA-01, INFRA-02, INFRA-03
**Success Criteria** (what must be TRUE):

  1. 一条命令(仅传配置文件路径)即可对 data/train(188 组)与 data/test(15 组)全部 cam1/cam2 图像完成角点检测,全程多进程并行,不向 data/ 写入任何文件
  2. 输出目录中存在:每路相机"棋盘格完整可用"图像清单、两路相机公共视场图像对清单(同组目录两路均检测成功)、角点数据文件(2D 角点坐标 + 3D 板系坐标 + 图像索引)
  3. 角点检测采用 SB 主路线(不做二次亚像素)+ classic 兜底(窗口从本帧实测方格像素边长推导),策略由配置开关控制
  4. 输出角点可视化抽查图,人工核验角点位置与顺序正确(角点序一致,无错序/漏检)
  5. 棋盘格规格(11×8 内角点、25mm)、路径、算法参数均来自 YAML 配置文件,命令行无算法参数

**Plans**: 3 plans
Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Walking Skeleton:uv 骨架 + 配置层 + SB 检测核心 + 串行流水线,npz/CSV 落盘(PREP-01/02/03/05, INFRA-01/02/03)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — classic 兜底移植(缺陷闸门+阶梯重提)+ joblib loky 并行 + 异常信封(PREP-02)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-03-PLAN.md — 角点可视化核验图 + 失败帧原图拷贝 + 公共视场对清单 + 全量生产运行与 data/ 只读锁定(PREP-04/06, INFRA-03)

### Phase 2: 模块2 内参标定

**Goal:** 内参标定模块可独立运行:消费 Phase 1 角点数据,对两路相机分别标定 M2 模型内参,并用 test 集给出量化验证指标
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: INTR-01, INTR-02, INTR-03, INTR-04
**Success Criteria** (what must be TRUE):

  1. 一条命令对两路相机分别完成 M2 模型标定(fx,fy,cx,cy,k1,k2,p1,p2,k3):`calibrateCameraExtended` 初值 + scipy `least_squares` 精化(精化开关由配置控制)
  2. 输出两路相机内参文件:参数按名保存 + 图像分辨率,可直接被模块3 读取为固定内参
  3. test 集整体重投影 RMS 以 float64 回代计算(不使用 OpenCV 返回值),每路相机 RMS 为亚像素量级(< 1.0 px,预期 ~0.1–0.5 px)
  4. 输出内参标定报告:逐帧重投影误差 + 整体 RMS,可识别离群帧

**Plans**: TBD

### Phase 3: 模块3 双目标定

**Goal:** 双目标定模块可独立运行:消费 Phase 1 公共视场角点 + Phase 2 固定内参,产出双目外参 R/T 与 baseline,并用 test 集最远两角点物理距离残差完成端到端验证
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: STER-01, STER-02, STER-03, STER-04
**Success Criteria** (what must be TRUE):

  1. 一条命令完成双目外参求解:每共视图两路 solvePnP → 相对位姿(Rodrigues 向量取中位)→ scipy 外参精化,全程不调用 `cv2.stereoCalibrate`
  2. 输出双目外参 R/T(cam1 为参考系)及 baseline 距离,与支架物理结构量级相符
  3. test 集验证链路端到端跑通:角点序 180° 消歧 → undistortPoints 去畸变 → DLT 三角化 → 板系反变换 → 最远两角点距离 vs 真值 307.45mm,残差 RMS 为毫米级(< 5mm,预期 ~1–2mm)
  4. 输出双目标定报告:每帧距离残差 + 整体 RMS

**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. 工程骨架 + 模块1 数据预处理 | 3/3 | Complete   | 2026-10-23 |
| 2. 模块2 内参标定 | 0/? | Not started | - |
| 3. 模块3 双目标定 | 0/? | Not started | - |

---
*Roadmap created: 2026-10-23. Granularity: coarse. Coverage: 17/17 v1 requirements mapped.*
