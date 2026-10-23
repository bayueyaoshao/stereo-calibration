# Phase 3: 模块3 双目标定 - Research

**Researched:** 2026-10-24
**Domain:** 双目外参标定(PnP-median 初值 + scipy 外参精化、180° 角点序消歧、undistortPoints 去畸变 + DLT 三角化、最远两角点物理距离残差验证)
**Confidence:** HIGH(核心结论全部来自对已验证参考工程 2 `.../multi_cameras/data_acquire/calibration` 源码的逐行阅读与行号引用,关键数值(真值距离、undistortPoints 语义、内参契约、对齐键覆盖)均在本机 uv 环境实测验证;该参考工程用同一类相机、同一棋盘规格端到端验证过本阶段的全部数学,其对角距离指标实测 0.23mm)

---

<user_constraints>

## User Constraints (from CLAUDE.md / PROJECT.md)

无 Phase 3 CONTEXT.md(未运行 discuss-phase),以下为 CLAUDE.md + PROJECT.md + STATE.md 锁定决策,规划必须逐条遵守:

- **Tech stack**: Python + OpenCV 4.x + numpy + scipy,参考工程1风格(配置文件驱动,命令行零算法参数)
- **配置文件**: 所有传参必须用配置文件控制,禁止硬编码(阈值/开关/路径全部进 config.yaml 的 stereo 段)
- **模块分离**: stereo 模块独立可运行调试,上游输出即下游输入(消费 Phase 1 npz/pairs CSV + Phase 2 intrinsics.json)
- **依赖**: 只读引用两个参考工程,绝不修改其任何内容
- **数据只读**: data/ 不可写,产物写独立输出目录(本阶段 rmtree 目标仅 output_root/stereo)
- **锁定决策(STATE.md / PROJECT.md Key Decisions)**:
  - 双目外参:**禁用 cv2.stereoCalibrate**(参考工程2验证该机型有 bug)→ PnP-median 初值 + scipy 外参精化
  - cam1 为参考系(R=I, t=0),输出 R/T + baseline 距离
  - 双目验证指标:test 集最远两角点物理距离残差 RMS,真值 **√(10²+7²)×25 = √149×25 ≈ 305.16mm**(2026-10-24 已修正原 307.45 算术误差;本机实测 `np.sqrt(10**2+7**2)*25 = 305.16389039334257`)
  - **OVERNIGHT AUTONOMY**: 计划必须含双目离群帧机制(配置阈值 + 剔除继续算 + 报告留痕),不阻塞人工复核
- **既有代码模式(Phase 1/2 确立)**: frozen dataclass 配置、中文报错收口、源码防线测试(grep count)、data 只读指纹、`--config` 唯一 CLI 选项、npz 读取一律 allow_pickle=False、`intrinsic/models.py` 是唯一数学源(残差只准 import 它)

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| STER-01 | 固定内参 + train 公共视场角点求双目外参:PnP-median 初值(每共视图两路 solvePnP → 相对位姿 → 全帧中位数,不用 stereoCalibrate)+ scipy 外参精化 | 参考工程2 `core/stereo.py:52-134`(PnP-median 全套)+ `core/multibundle.py:61-205`(JointBundleAdjustment 外参 BA)完整可移植;Port Map §2/§3/§4 |
| STER-02 | 输出双目外参 R/T(cam1 为参考系)及 baseline 距离 | `core/stereo.py:36-49` 相对位姿公式(语义 X_cam2 = R·X_cam1 + t,baseline = ‖t‖);输出契约见 Pattern 7 |
| STER-03 | test 集验证:180° 消歧 → 去畸变 → DLT 三角化 → 板系最远两角点距离 vs 真值 √149×25mm,残差 RMS | 参考 `cli/06_evaluate.py:177-269` + `core/triangulation.py` 全套(消歧/undist_pix/DLT 保号/板系反变换/对角距离)逐行可移植;评估链见 Pattern 5 |
| STER-04 | 双目标定报告(每帧距离残差 + 整体 RMS) | Phase 2 artifacts/report 模式直接延续(utf-8-sig CSV + markdown);契约见 Pattern 7 |

</phase_requirements>

---

## Summary

本阶段是把参考工程2 已验证的双目链路移植进本工程第三个模块包 `stereo/`:消费 Phase 1 的公共视场对清单 + 两路角点 npz 与 Phase 2 的 intrinsics.json(固定内参),产出 cam1 参考系下的双目外参 R/T 与 baseline,并用 test 集 15 对做端到端物理距离验证。参考工程的对应实现**每一环都有现成、经真实数据验证的代码**:`core/stereo.py`(PnP-median 相对位姿 + Rodrigues 向量逐分量中位,禁 stereoCalibrate 的完整实现)、`cli/05_extrinsic_multi.py` 的排序规范化(180° 消歧,evidence-threshold 0.5)与外参联合 BA(`core/multibundle.py` 的 JointBundleAdjustment:θ=[6 外参 | 6×N 位姿],trf + loss=linear + jac_sparsity)、`cli/06_evaluate.py` 的评估链(消歧 → per-frame 2 视图位姿 BA → undistortPoints(P=K)去畸变 → 多视图 DLT(保号除法)→ 板系反变换 → 对角距离 vs 305.16mm)。移植策略沿用 Phase 1/2 已验证的"移植 + 不变量测试锁定"模式,`intrinsic/models.py` 继续当唯一数学源(残差全部经 models.project,`_distort` 在 stereo/ 出现次数必须为 0)。

关键差异点(参考工程 → 本工程)有四个,均有明确解法:(1)**规模小且全 2 相机**——train 112 对 / test 15 对(参考 91 位姿 × 4 相机),所有多相机字典/组合逻辑裁剪为固定 2 路,PnP 全程亚秒,精化解析雅可比秒级;(2)**评估链有分叉**——参考 06_evaluate 在 DLT 前做 per-frame 2 视图位姿 BA(v1 独立 PnP+DLT 在 1m 处误差 24mm,v2 加 per-frame BA 后 0.23mm,这是参考实测的 100 倍改进,必须移植这半步);(3)**消歧锚点缺口**——参考用 Phase 2 单标定的逐帧位姿当锚,本工程实测 cam1 的 Phase 2 extrinsics.npz 只覆盖 112 对中的 109 对(7 帧被内参离群剔除吃掉),所以 train 求解侧推荐**两遍法自洽锚定**(pass-1 PnP-median 当锚 → 消歧 → pass-2 重解),不依赖 Phase 2 位姿产物;(4)**零图像读取**——双目链路全程消费角点数组(solvePnP/undistortPoints/DLT 都不需要 image_size),比 Phase 2 更快且天然不碰 data/。

精度预期有硬锚点:参考工程同类相机 cam1+cam2 组合(基线 1208mm)的对角距离残差实测 **abs mean 0.230mm / BA RMS 0.112px**(39 个 test 位姿,per-frame BA 版)。本工程支架基线未记载(planner 决策,见 Open Questions),深度误差按 σ_z ≈ z²/(B·fx)·σ_px 随基线缩短而放大,成功标准的 < 5mm(预期 1–2mm)已为短基线留出裕量,与参考锚点量级自洽。

**Primary recommendation:** 按"observation(对装载)→ solve(PnP-median + 两遍消歧 + scipy 精化)→ evaluate(per-frame BA + DLT 评估链)→ pipeline(离群剔除编排)→ artifacts(JSON/报告/CSV)"顺序移植 `stereo/` 包;残差一律经 `intrinsic.models`(单一数学源 + 源码防线锁定);消歧阈值用参考 05 的 0.5(证据充分才翻);离群剔除沿用 Phase 2 的"中位数乘数 + 双保险丝,只剔 train pair";测试三件套:合成真值恢复 e2e、真实数据门禁(距离 RMS < 5mm)、源码防线(grep stereoCalibrate == 0 等)。

---

## Architectural Responsibility Map

离线单机数值工具,按项目内部分层映射(每能力一个属主):

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| M2 投影/反投影/位姿变换数学 | `intrinsic/models.py`(复用,不改) | — | 全工程唯一数学源(Phase 2 锁定);stereo 残差只准 import |
| pairs CSV ↔ 两路 npz 对齐装载 | `stereo/observation.py`(新) | 复用 `preprocess.artifacts.load_corners`、`build_object_points`;`io.dataset` 只读校验 | 连接键 group_id+"_cam{n}" 的对齐与行序锁在此定死(见 Pattern 1) |
| PnP-median 初值 + 180° 消歧 | `stereo/solve.py`(新) | 只 import models + cv2 | 参考 core/stereo.py 移植;消歧/中位聚合/离散度诊断在此 |
| scipy 外参精化(θ=[6 ext \| 6N pose]) | `stereo/solve.py` | 只 import models | 参考 multibundle.JointBundleAdjustment 的 2-cam 特化 |
| 评估链(per-frame BA → DLT → 距离残差) | `stereo/evaluate.py`(新) | 只 import models | 参考 06_evaluate + core/triangulation 移植;**三角化只许住在这里**(源码防线) |
| 双目离群剔除编排(overnight) | `stereo/pipeline.py`(新) | 配置阈值 | solve → 逐帧 RMS → 剔除 → 重解循环,只剔 train pair |
| 产物落盘(JSON/报告/CSV/npz) | `stereo/artifacts.py`(新) | — | Phase 2 artifacts 模式延续;STERO-02/04 出口 |
| 配置解析(stereo 段) | `io/config.py` 扩展 | — | frozen dataclass + 中文校验既有模式延续 |
| CLI stereo 子命令 | `cli/main.py`(替换占位) | — | except 家族中文收口 + Exit(1) 既有模式 |

---

## Port Map(参考工程2 → 本工程 stereo/ 包,逐函数)

参考工程根:`/home/lx/Works/src/bolight_robot/vvd_daq/multi_cameras/data_acquire/calibration`(只读)。行号为 2026-10-24 实读核实。

### 1. `stereo/solve.py` — PnP-median 初值 + 消歧 + scipy 精化(STER-01/02)

| 参考来源 | 内容 | 去向与改法 |
|---|---|---|
| `core/stereo.py:36-49` `_relative_pose` | R_rel = R_b@R_a.T;t_rel = t_b − R_rel@t_a(含逐行推导 docstring) | `relative_pose(R_a,t_a,R_b,t_b)` 原样移植(5 行数学,语义:X_cam2 = R_rel·X_cam1 + t_rel) |
| `core/stereo.py:52-134` `stereo_calibrate_pair` | 逐对两路 `cv2.solvePnP(objp f32, corners f32, K f64, dist f64.reshape(-1,1), flags=cv2.SOLVEPNP_ITERATIVE)`(L82-91)→ `cv2.Rodrigues` 转矩阵(L96-97)→ 相对位姿收集;**中位聚合:t 逐分量中位、R 转 Rodrigues 向量逐分量中位再转回**(L113-121) | `pnp_median(pairs, K1, D1, K2, D2)` 适配:角点来自对齐装载(不是重检测);失败帧记 NaN 跳过;**新增离散度诊断**(rv_rels 相对 rv_med 的最大夹角,超阈值中文报错,防 Rodrigues 中位退化,见 Pitfall 1);`compute_per_view_stereo_rms`(L137-163)合并为逐帧重投影 RMS 输出,残差改经 models(单一数学源) |
| `cli/05_extrinsic_multi.py:102-131` 排序规范化 | 用锚位姿 + 当前外参预测投影,`e_flip < e_orig * 0.5` 才翻(L125-128,证据充分才翻) | `disambiguate_pairs(...)` 移植判定规则;锚改**两遍法自洽**(pass-1 对原始角点跑 pnp_median 得外参 + 逐帧 PnP 位姿当锚 → 消歧 → pass-2 重跑)。不采用参考的 Phase 2 位姿锚:实测 7/112 对缺 cam1 锚(内参剔除所致),见 Pitfall 12 |
| `core/multibundle.py:61-205` `JointBundleAdjustment` | θ = [非 ref cam (rod,t) 6 | 每 pose (rod,t) 6×N](L94-124 pack/unpack);残差 = 两路重投影(L133-150);`least_squares(method='trf', loss='linear', f_scale=1.0, x_scale='jac', jac_sparsity=lil)`(调用点 `cli/05_extrinsic_multi.py:265-266,293-294`;lil 稀疏结构 L152-170) | `refine_stereo(...)` 2-cam 特化:θ = [6 外参 | 6×N 位姿],cam1 恒 (I,0) 不进 θ。残差经 models.project + transform_to_camera(推荐**解析稀疏雅可比**,链式规则与 Phase 2 同构,见 Pattern 4;备选:照搬参考的数值雅可比 + jac_sparsity——零新数学但慢一个量级)。RMS 口径 float64(models 残差),绝不用 OpenCV 返回值 |

### 2. `stereo/evaluate.py` — test 验证链(STER-03)

| 参考来源 | 内容 | 去向与改法 |
|---|---|---|
| `cli/06_evaluate.py:180-203` 180° 消歧 | cam1 solvePnP 给板位姿初值 → `Xc = R_cam[cam]@(R1@objp.T).T + t_cam[cam]` → `cv2.projectPoints` → `err_rev < err_fwd` 选 arr/arr[::-1](L200-202,**阈值 1.0**) | `disambiguate_frame(...)` 移植;test 侧锚 = 当帧 cam1 PnP(自洽,无外部依赖);阈值统一走配置(0.5 vs 1.0 是 05/06 两处差异,planner 定,见 Decisions #3)。注意参考 06 对**含 cam1 在内的所有路**都消歧(L190 `for cam in CAMS:`),但 cam1 锚即自身 PnP 解,数学上恒不翻,照搬无害 |
| `core/multibundle.py:423-494` `multiview_pose_ba` | 固定外参下只优化 board→ref 的 (R,t) 6 参数:外参先换算到 ref 相对系(L454-457),ref 路 solvePnP 当初值(L459-460),残差 = 各路组合旋转投影(L462-478),`least_squares(trf)`(L481-482) | `frame_pose_ba(...)` 2-cam 移植。**初值路线改 Phase 2 scoring 模式**:models.unproject 反投影 → 单位阵 K + 零畸变 solvePnP(本工程 k1=-0.75 强畸变,Phase 2 实测直接解会进错误盆地误评 4.82px);残差经 models.project。参考此处 loss 默认 soft_l1,建议改 linear 与全工程一致(6 参数 176 点小问题,消歧后无需鲁棒核;planner 可否决) |
| `core/triangulation.py:24-29` `projection_matrix` | P = K @ [R \| t] | 原样(3×4) |
| `core/triangulation.py:32-40` `undist_pix` | `cv2.undistortPoints(p_f32, K, dist, None, K)`——**第 5 参传 K**,返回理想针孔**像素**坐标(不传 K 返回归一化坐标);本机实测 `undistortPoints(pts,K,dist,None,K) == K @ undistortPoints(pts,K,dist,None,None)`,逐位一致 | 原样移植;DLT 的 P 必须配同口径(见 Pitfall 4/5) |
| `core/triangulation.py:59-85` `triangulate_multi_view` | 逐点组 A(2·n_cams,4):`A[2i] = x·P[2] − P[0]`,`A[2i+1] = y·P[2] − P[1]` → SVD → **保号除法** `w = X_h[3] if abs(X_h[3])>1e-12 else 1e-12; out[k] = X_h[:3]/w`(L83-84,严禁 abs()) | `triangulate_dlt(projections, image_points)` 原样(2 视图特化);**只许住在本文件**(源码防线) |
| `cli/06_evaluate.py:245-263` 板系反变换 | ref 路 P=K1[I\|0],另一路 P=K2[R_rel\|t_rel](L245-254)→ `pts_board = (R_pose.T @ (pts_ref − t_pose).T).T`(L263) | 原样;R_pose/t_pose 来自 frame_pose_ba(board→cam1,与 DLT 输出系一致) |
| `core/triangulation.py:88-100` `farthest_corners_distance` + `cli/06_evaluate.py:265-269` | idx_a=0,idx_b=(rows−1)·cols+(cols−1)=87;`farthest_err = dist − true_dist`;真值 = `sqrt(((cols−1)·s)² + ((rows−1)·s)²)`(06 L103-104) | 原样;真值从 `build_object_points(cfg.board)` 对角距离**导出**(不写死数字,配置变了自动跟着对) |
| `core/triangulation.py:122-174` `kabsch_align` + `eval_axis_rms` | Kabsch 对齐分轴 RMS(mm,几何精度诊断) | **可选**(MVP 最小指标是对角距离残差;Kabsch 是锦上添花,planner 决策是否带) |

### 3. 不移植清单(明确负清单)

- `cli/04_extrinsic_pair.py` 的 `cv2.stereoCalibrate` 路线(本工程禁用;参考自己也已废弃——EXPERIMENTS.md:922)
- 多相机 3/4 路组合、全排列组合评估、渐进 BA(本工程恒 2 相机)
- `detect_one`/`detect_multiscale`(角点已由 Phase 1 产出,严禁重检测)
- pickle 产物(corners_cache/extrinsic_multi_ba.pkl)——本工程一律 npz + json(allow_pickle=False 纪律)
- 03_noise / 03b_fisher_crlb / diag_* / viz_*(研究性脚本)

---

## Standard Stack

### Core(零新增运行时依赖,uv.lock 不变)

| Library | Version(2026-10-24 实测) | Purpose | Why Standard |
|---------|--------------------------|---------|--------------|
| opencv-python | 4.14.0(锁 <5.0) | solvePnP(SOLVEPNP_ITERATIVE)、Rodrigues(含雅可比)、undistortPoints(P=K)、projectPoints(parity 基准) | 与参考工程2 同版本组合,本链路全部调用形态实测过 [VERIFIED: uv run python] |
| numpy | 2.5.4 | 数组数学、SVD(DLT)、npz 产物 | 与 scipy 1.18 成对锁定(Phase 1 决策)[VERIFIED: uv run python] |
| scipy | 1.18.1 | optimize.least_squares(trf;外参精化与 per-frame BA)、sparse(lil/coo→csr,若走解析稀疏雅可比) | Phase 2 同库同法实测收敛 [VERIFIED: uv run python] |
| Python | 3.12.13(uv venv) | 运行时 | pyproject requires-python >=3.12 [VERIFIED: uv run python] |

### Supporting

零新增。dev 组 pytest>=8 / ruff>=8 既有;当前 124 测试全绿(实测 collect)。数值雅可比测试基准(若走解析雅可比路线)用 `scipy.optimize._numdiff.approx_derivative`(scipy 自带,Phase 2 同款,零新依赖)。

**版本验证命令(本机 2026-10-24 实测输出):**

```bash
uv run python -c "import cv2,numpy,scipy,sys; print(sys.version.split()[0], cv2.__version__, numpy.__version__, scipy.__version__)"
# 实测:3.12.13 4.14.0 2.5.4 1.18.1
```

## Package Legitimacy Audit

> 本阶段零新增包(全部依赖 Phase 1 已入库锁定,uv.lock 不变)。无安装动作,无需 slopcheck 运行;按协议声明:零新增包 = 零供应链风险增量。

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| (none — 零新增) | — | — | — | — | — | — |

**Packages removed due to slopcheck [SLOP] verdict:** 无
**Packages flagged as suspicious [SUS]:** 无

---

## Architecture Patterns

### System Architecture Diagram

```
config.yaml(board / output_root / stereo 段:refine+outlier+flip_ratio+distance_rms_max+baseline_band)
        │  load_config(io/config.py 扩展,frozen dataclass,中文校验)
        ▼
Phase 1 产物(只读)                       Phase 2 产物(只读)
  output/{train,test}/common_fov_pairs_{split}.csv     output/intrinsic/cam{1,2}/intrinsics.json
    (group_id, cam{n}_path, route_cam{n};112/15 对)     (params 九名 + K + D + image_size;按名读取)
  output/{train,test}/cam{n}_{split}.npz(八数组,uv/objp/detected/source)
        │
        ▼
stereo/observation.py:对装载(行序契约在此定死)
  pairs CSV 逐行 → 连接键 group_id+"_cam{n}" 命中两路 npz 行 → detected/finite/exclude 三重防线
  → objp allclose 交叉核对 → Pairs{image_ids, uv1 (V,88,2), uv2 (V,88,2), objp, source, routes}
        │
        ▼  train 求解(stereo/solve.py)                        test 评估(stereo/evaluate.py)
┌──────────────────────────────────────────────┐   ┌────────────────────────────────────────────┐
│ pass-1:pnp_median(原始角点)                   │   │ 每对:                                      │
│   逐对两路 solvePnP(ITERATIVE,f32 点位)      │   │  1. 180° 消歧(锚 = 当帧 cam1 PnP,          │
│   → relative_pose 收集                        │   │     unproject 反投影初值路线)               │
│   → R 转 Rodrigues 向量逐分量中位 / t 中位     │   │  2. frame_pose_ba:固定外参,6 参数位姿       │
│   → 离散度诊断(最大夹角)                      │   │     trf 精化(board→cam1)                  │
│ pass-2:消歧(e_flip < ratio×e_orig)→ 重解     │   │  3. undist_pix(P=K)→ P=K1[I|0],           │
│ 剔除循环(overnight):逐帧重投影 RMS >           │   │     K2[R_rel|t_rel] → DLT(保号除法)        │
│   k×中位 → 剔对 → 重解(双保险丝,只剔 train)   │   │  4. pts_board = R_poseᵀ(X−t_pose)          │
│ (配置开关)refine_stereo:θ=[6 ext | 6N pose]   │   │  5. 对角距离(0↔87)vs 真值 √149×25          │
│   trf + 解析稀疏雅可比(经 models)             │   └────────────────────────────────────────────┘
└──────────────────────────────────────────────┘                │
        │                                                       ▼
        ▼                                        stereo/evaluate 汇总:逐帧距离残差 + pooled RMS
stereo/artifacts.py 落盘:output/stereo/
  extrinsics_stereo.json(R/t/baseline_mm/rot_deg + 每帧 RMS + 剔除留痕 + 配置快照)
  report.md(STER-04:逐帧距离残差表 + 整体 RMS + PASS/FAIL 判定)
  per_frame_{train,test}.csv(utf-8-sig)+ residuals npz(诊断)
```

读者可顺着箭头从 config + Phase 1/2 产物走到双目报告;两路(求解/评估)共用 `intrinsic/models.py` 这一个数学源,三角化只住在 evaluate.py,这就是结构保证。

### Recommended Project Structure

```
src/stereo_calib/stereo/              # 新包(与 preprocess/ intrinsic/ 平级,模块分离约束)
├── __init__.py
├── observation.py     # pairs CSV + 两路 npz 对齐装载(连接键/行序锁/三重防线)
├── solve.py           # relative_pose / pnp_median / disambiguate_pairs / refine_stereo
├── evaluate.py        # disambiguate_frame / frame_pose_ba / undist_pix / triangulate_dlt / 距离残差
├── pipeline.py        # run_stereo 编排:train 求解+剔除循环 → test 评估 → 中文汇总(零数学)
└── artifacts.py       # extrinsics_stereo.json / report.md / per_frame CSV / residuals npz
tests/
├── test_stereo_solve.py      # 合成真值恢复(R 角/t 误差)、消歧行为、中位离散度、精化开关
├── test_stereo_evaluate.py   # DLT 保号/合成板距离恢复、消歧翻转、frame_pose_ba、undist P=K 契约
├── test_stereo_pipeline.py   # 对装载契约、剔除行为、合成 npz e2e → 产物
├── test_stereo_config.py     # stereo 段中文校验
└── test_stereo_guards.py     # 源码防线:grep stereoCalibrate==0、_distort 只在 models、triangulate 只在 evaluate
config.yaml                   # 追加 stereo: 段
```

### Pattern 1: 对装载(pairs CSV ↔ 两路 npz,连接键与行序锁)

**What:** common_fov_pairs CSV 的 `group_id` 是 `<ts>_<us>`(如 `20261023_154025_519359`),npz 的 `image_id` 是 `group_id + "_cam{n}"`(实测核实:`20261023_154025_519359_cam1`)。逐行对清单 → 按连接键在两路 npz 的 image_ids 里定位行 → 取该行 uv。

**Why:** 这是唯一的两路角点对齐途径;对不齐 = 角点错位配对,外参静默错(不报错)。

```python
# 连接规则(实测核实于 output/train/cam1_train.npz + common_fov_pairs_train.csv)
id_to_row = {str(x): i for i, x in enumerate(npz_data["image_ids"])}
row1 = id_to_row[f"{pair.group_id}_cam1"]   # 不命中 → 中文报错(产物不一致)
row2 = id_to_row[f"{pair.group_id}_cam2"]
# 防线:detected 断言(清单说成对,npz 必须检出)、np.isfinite 断言、
# 路径交叉核对(csv cam{n}_path 与 npz paths[row] 一致)、objp allclose(照 Phase 2)
```

行序锁:装载后的 V 个对按 CSV 行序定死,后续 rvecs/残差/报告/剔除清单全部按该顺序,绝不重排(Phase 2 Pitfall 10 同款)。

### Pattern 2: PnP-median 初值(参考 core/stereo.py 移植)

**What:** 逐对两路各自 solvePnP(带畸变原始像素 + 全套 K/dist,`flags=cv2.SOLVEPNP_ITERATIVE`,点位 float32、K/dist float64——参考实测混合 dtype 组合),各自得 board→cam 位姿,消去 world 得相对位姿,跨帧取中位数。

```python
# 来源:参考工程 core/stereo.py:36-49,82-121(只读移植;公式推导见其 docstring)
def relative_pose(R_a, t_a, R_b, t_b):
    # X_a = R_a@X_w + t_a;X_b = R_b@X_w + t_b;消去 X_w:
    # X_b = (R_b@R_a.T)@(X_a - t_a) + t_b = R_rel@X_a + (t_b - R_rel@t_a)
    R_rel = R_b @ R_a.T
    t_rel = t_b.flatten() - R_rel @ t_a.flatten()
    return R_rel, t_rel

# solvePnP 调用形态(L82-91):objp float32 reshape(-1,1,3),角点 float32 reshape(-1,1,2),
# K float64,dist float64 reshape(-1,1),flags=cv2.SOLVEPNP_ITERATIVE(输入是带畸变像素!)
ok, rv, tv = cv2.solvePnP(objp_f32, corners_f32, K_f64, dist_f64_col, flags=cv2.SOLVEPNP_ITERATIVE)

# 中位聚合(L113-121):t 逐分量中位;R → Rodrigues 向量堆叠 (N,3) → 逐分量中位 → 转回
rv_rels = np.array([cv2.Rodrigues(R)[0].flatten() for R in R_rels])   # (N,3)
rv_med = np.median(rv_rels, axis=0)
R_med, _ = cv2.Rodrigues(rv_med.reshape(3, 1))
t_med = np.median(t_rels, axis=0)                                     # 单位 mm(objp 是 mm)
baseline_mm = float(np.linalg.norm(t_med))
```

**新增防线(参考没有,本工程加):** rv_rels 相对 rv_med 的最大夹角( radians)离散度诊断——刚架的相对位姿跨帧应紧密聚集(参考 91 帧实测 ~27.2° 附近小摆动);离散度过大说明有翻转帧/孪生解污染,中文报错而非静默出坏外参(阈值进配置,建议 1° 量级;planner 定)。

### Pattern 3: 180° 角点序消歧(两处应用,两个锚)

**What:** 棋盘 11×8 有 180° 旋转对称,检测器对同一物理位姿可能给出互为反转的角点序;Phase 1 的 canonicalize 只保证**单图形态规范**(行方向朝画面右),**不保证跨相机物理同一性**——cam1 与 cam2 视角不同,规范序可能对应相差 180° 的物理角点。不消歧:联合 PnP 残差 150-270px、相对位姿双峰、DLT 全错位(参考工程 v7 根因修复实录,05_extrinsic_multi.py:6-10)。

**判定规则(参考两处实现):** 用锚位姿把 objp 投到该相机 → 比较 `err_fwd = ‖proj − arr‖` 与 `err_rev = ‖proj − arr[::-1]‖`:

```python
# 来源:cli/06_evaluate.py:186-203(只读移植;05_extrinsic_multi.py:119-128 同构但阈值 0.5)
ok_pnp, rv1, tv1 = cv2.solvePnP(objp, cs_cam1, K[REF], dist[REF])   # 锚 = 当帧 cam1 PnP
R1, _ = cv2.Rodrigues(rv1)
Xw = (R1 @ objp.T).T + tv1.ravel()            # 板 → cam1
Xc = (R_rel @ Xw.T).T + t_rel                  # cam1 → cam(用当前外参估计)
proj, _ = cv2.projectPoints(Xc.astype(np.float32).reshape(-1,1,3), np.zeros(3), np.zeros(3), K[cam], dist[cam])
err_fwd = np.linalg.norm(proj.reshape(-1,2) - arr)
err_rev = np.linalg.norm(proj.reshape(-1,2) - arr[::-1])
arr = arr[::-1].copy() if err_rev < FLIP_RATIO * err_fwd else arr
```

**应用位置(两处,锚不同):**
- **train 求解侧**(`solve.py`):两遍法——pass-1 对原始角点跑 pnp_median 得外参 + 逐帧两路 PnP 位姿;pass-2 以 pass-1 外参 + 各帧位姿为锚逐对消歧,再重跑 pnp_median。自洽、无外部依赖。**不采用**参考的"Phase 2 位姿当锚":实测 cam1 extrinsics.npz 只覆盖 112 对中的 109 对(7 帧被 Phase 2 内参剔除),锚会缺(见 Pitfall 12)。
- **test 评估侧**(`evaluate.py`):单遍——锚 = 当帧 cam1 solvePnP(参考 06 原样)。自洽性保证:若 cam1 自身角点序整体翻转,PnP 解出的是 180° 转板位姿,全部相机一致地消歧到同一物理约定,而最终指标(板系对角距离)对该整体翻转**不变**(角点 0↔87 互换,距离相同)。
- 消歧阈值 FLIP_RATIO 进配置:参考 05 用 0.5(证据充分才翻,防误翻)、06 用 1.0(必选小者)。推荐 0.5(planner 锁)。

**确定性:** 规则是纯函数(同输入同输出),无随机性;锚来自确定性求解;消歧结果与翻转次数写入报告留痕(参考 05 打印 n_flips 的做法)。

### Pattern 4: scipy 外参精化(θ = [6 外参 | 6×N 位姿])

**What:** 固定两路内参,联合优化:cam1→cam2 外参 6 参数 + 每对 (board→cam1) 位姿 6 参数(N=112 → θ 共 678 维)。残差 = 两路重投影误差(models.project,float64):

```
X_cam1 = transform_to_camera(objp, ω_j, t_j)            # 第 j 对位姿
X_cam2 = R_rel @ X_cam1 + t_rel                          # 外参组合
residual_j = [models.project(X_cam1, p1) − uv1_j;  models.project(X_cam2, p2) − uv2_j]  (352×2 行/对)
```

scipy 配置(参考 multibundle.py:172-180 + 调用点 05:265-266):`method="trf"`、`loss="linear"`(参考全局 BA 调用时显式传 linear)、`x_scale="jac"`、`max_nfev` 配置(参考 50-60;本工程建议 200 留裕量)、ftol/xtol/gtol 沿 Phase 2 值。

**雅可比两条路线(planner 决策,推荐 A):**

- **A. 解析稀疏雅可比(推荐,Phase 2 风格):** 全部构件已存在——`transform_to_camera(want_jacobian=True)` 给 dX_cam1/dpose (N,3,6);外参列:`cv2.Rodrigues(r_rel)` 的 rot_jac 一次 einsum 给 dX_cam2/dω_rel = Σ_j rot_jac_rel[k,i,j]·X_cam1_j(与 models.transform_to_camera:141 同一模式,严禁手写 −[Xc]×)、dX_cam2/dt_rel = I;再链 `project_point_jacobian` (N,2,3)。行序契约照 Phase 2 `row_index` 函数(残差/雅可比共用一个函数)。稀疏 nnz = 2·N·2·88·12 ≈ 47 万,coo→csr。规模实测推算:单次残差 ~5ms、单次雅可比 ~30ms,精化全程秒级。
- **B. 数值雅可比 + jac_sparsity(参考 multibundle 原样):** 零新数学、移植最快,但有限差分每次雅可比要 679 次残差评估(每次 224 个视图变换+投影),单次精化预估分钟级;且 scipy 在 callable jac 下会静默忽略 jac_sparsity(Phase 2 WR-02 教训——数值路线才传 sparsity,解析路线绝不传)。

**成功标准 1 的"全程不调用 cv2.stereoCalibrate"由源码防线测试锁定**(grep 计数,见验收门)。

### Pattern 5: test 评估链(per-frame BA → undistort → DLT → 板系 → 对角距离)

**What:** 每个验证对的完整链路(参考 06_evaluate 的 v2 设计——v1"独立 PnP + DLT"实测 1m 处 24mm,v2 加 per-frame BA 后 0.23mm,这半步必须移植):

1. **消歧**(Pattern 3 test 侧);
2. **per-frame 2 视图位姿 BA**(`frame_pose_ba`):固定外参与内参,只优化 board→cam1 的 6 参数,残差 = 两路联合重投影(经 models);初值走 models.unproject 反投影 → 单位阵 K solvePnP(Phase 2 scoring 模式,防强畸变错误盆地);
3. **去畸变 + DLT**:`undist_pix`(P=K 针孔像素)两路观测;投影矩阵 P1 = K1[I|0]、P2 = K2[R_rel|t_rel](参考 06:245-254——**DLT 的 P 不带畸变、观测已去畸变,线性方程在针孔模型下严格成立**;参考注释明确记录:直接喂带畸变点会有板边缘系统性偏差);
4. **板系反变换**:`pts_board = (R_pose.T @ (pts_ref − t_pose).T).T`(06:263);
5. **对角距离残差**:idx 0 ↔ 87,`farthest_err = ‖pts_board[0] − pts_board[87]‖ − true_dist`;true_dist = `np.linalg.norm(objp[87] − objp[0])`(= √(10²+7²)×25 = **305.16389039334257** mm,本机实测;从 build_object_points 导出,不写死);
6. **汇总**:逐帧残差表 + pooled RMS = sqrt(全部帧残差平方均值)(mm);**test 帧永不剔除只标记**(考卷不可改,Phase 2 纪律延续)。

```python
# 来源:core/triangulation.py:59-85(只读移植,保号除法是命门)
for k in range(n_pts):
    A = np.zeros((2 * n_cams, 4))
    for i in range(n_cams):
        x, y = image_points[i][k]
        A[2*i]   = x * projections[i][2] - projections[i][0]
        A[2*i+1] = y * projections[i][2] - projections[i][1]
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    # 保号:SVD 零向量符号任意,必须除以 X_h[3] 本身;abs() 会在 X_h[3]<0 时整点取反,深度变负
    w = X_h[3] if abs(X_h[3]) > 1e-12 else 1e-12
    out[k] = X_h[:3] / w
```

### Pattern 6: 双目离群剔除(OVERNIGHT AUTONOMY,Phase 2 机制平移)

**设计要点(全部沿用 Phase 2 已验证模式,只换判据对象):**
- **判据尺:** 逐对重投影 RMS(两路 pooled,float64 回代,经 models)——在 pnp_median 阶段就有(solvePnP 后回代);对角距离残差(mm)作为第二判据可在精化后对 train 复算一次供报告与二次剔除(planner 决策,推荐:主判据重投影 RMS,距离残差只报告标记,避免"用指标自己训练指标"的循环耦合);
- **阈值:** `outlier.rms_multiplier`(默认建议 5.0,与 Phase 2 同);**保险丝:** max_drop_fraction=0.2、max_iterations=3、min_pairs=10(112 对裕量巨大);
- **范围:** 只剔 train pair;test 15 对永不剔除只标记;
- **留痕:** 每轮剔除的 group_id + 该轮 RMS + 阈值进 JSON dropped_pairs 与报告;
- **时序:** 剔除循环包住 pnp_median(亚秒级,可多轮);scipy 精化只跑幸存集定稿后一次(照 Phase 2"精化不进循环")。

### Pattern 7: 输出契约(STER-02/04)

```jsonc
// output/stereo/extrinsics_stereo.json —— 语义:X_cam2 = R @ X_cam1 + t(cam1 参考系,t 单位 mm)
{
  "ref_cam": 1,
  "R": [[...],[...],[...]],           // 3×3 cam1→cam2
  "t": [...],                          // (3,) mm,cam1 系下
  "rvec": [...],                       // R 的 Rodrigues 向量(报告直读)
  "rotation_angle_deg": ...,           // ‖rvec‖ 转角
  "baseline_mm": ...,                  // ‖t‖
  "K1": ..., "D1": ..., "K2": ..., "D2": ...,   // 所用固定内参快照(留痕)
  "n_pairs_train": ..., "n_pairs_test": 15,
  "rms_reproj_train": ...,             // 精化后逐对重投影 pooled RMS(px)
  "rms_reproj_test": ...,
  "distance_rms_test_mm": ...,         // 成功标准 3 的数字(pooled: sqrt(全帧残差平方均值))
  "true_distance_mm": 305.16389039334257,
  "dropped_pairs": [{"group_id": "...", "rms": ..., "round": 1, "threshold": ...}],
  "flip_log": {"train": {"pass2_flipped": N, "pairs": [...]}, "test": {...}},
  "solver_meta": {...},                // status/nfev/离散度诊断
  "board": {...}, "config_snapshot": {...}
}
```

配套:`report.md`(STER-04:指标汇总 + R/t/baseline 表 + **逐帧距离残差表** + 剔除留痕表 + PASS/FAIL 判定列 + 配置快照)、`per_frame_{train,test}.csv`(utf-8-sig:group_id / route / 重投影 RMS / 距离残差 mm / dropped / note)、`train_residuals.npz`(group_ids + 残差,allow_pickle=False,诊断)。运行前 rmtree `output_root/stereo`(D-11 模式;data/output 互斥校验既有,不重复实现)。

### Pattern 8: 配置段与 CLI(既有模式延伸)

```yaml
# config.yaml 追加段(全部算法参数/开关在此,CLI 零算法参数不变)
stereo:
  refine:
    enabled: true          # scipy 外参精化开关(成功标准 1 配置控制要求)
    method: "trf"          # 只收 trf(Phase 2 防呆照搬)
    max_nfev: 200
    ftol: 1.0e-10
    xtol: 1.0e-12
    gtol: 1.0e-10
  outlier:
    enabled: true          # 双目离群对自动剔除(OVERNIGHT AUTONOMY)
    rms_multiplier: 5.0    # 逐对重投影 RMS > k × 幸存中位 → 剔除重解
    max_drop_fraction: 0.2
    max_iterations: 3
    min_pairs: 10
  flip_ratio: 0.5          # 180° 消歧判定阈值(参考 05 用 0.5,06 用 1.0;planner 锁)
  median_spread_max_deg: 1.0   # Rodrigues 中位离散度防线(超过即报错,见 Pitfall 1)
  distance_rms_max_mm: 5.0     # 成功标准阈值:报告判 PASS/FAIL(不作硬退出条件)
  baseline_band_mm: [20, 2000] # baseline 量级 sanity 带(见 Open Questions 2;planner 锁)
  exclude_pairs: []            # 手动逻辑剔除对清单(与自动机制叠加,原数据不动)
```

解析进 `StereoConfig` frozen dataclass(io/config.py 扩展);CLI `stereo` 子命令替换占位,异常家族照 Phase 2 intrinsic 收口:`except (ValueError, OSError, ArithmeticError, RuntimeError, AssertionError, cv2.error)` → 中文一句 + Exit(1)。

### Anti-Patterns to Avoid

- **在 stereo/ 里写第二份投影/畸变公式**——残差只准 import `intrinsic.models`(`_distort` 在 stereo/ 计数必须为 0,源码防线)。
- **用 cv2.stereoCalibrate——哪怕当种子**——本工程禁用(机型 bug);参考工程自己也在 EXPERIMENTS.md:922 记录废弃原因(无鲁棒损失 LM,毒对应拖变形,|t| 标成 2500mm)。
- **DLT 喂带畸变像素 + P=K[R|t]**——参考实测板边缘系统性偏差;必须 undist_pix(P=K)后配无畸变 P。
- **重检测角点**——Phase 1 角点是唯一观测源;04_extrinsic_pair 的 multiscale 重检测模式不移植。
- **剔除 test 对**——考卷不可改;高残差只标记。
- **消歧"证据不足也翻"**(ratio=1.0 时边界帧抖动)——用 0.5 保证翻转有充分证据(参考 05 的教训结论)。
- **残差与雅可比各自维护行序**——共用 row_index(Phase 2 Pitfall 4 延续)。

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 双目外参求解 | cv2.stereoCalibrate / 自写交替优化 | PnP-median(参考 core/stereo.py)+ scipy 精化(参考 multibundle) | 机型 bug 实锤;参考全链路实测验证 |
| 非线性最小二乘 | 自写 LM/高斯牛顿 | scipy `least_squares`(trf) | Phase 2 同库同法;数值风险全在参考踩过 |
| 畸变去除 | 手推畸变逆 | `cv2.undistortPoints(p, K, dist, None, K)`(DLT 观测)+ `models.unproject`(PnP/BA 初值) | 两个已验证实现各司其职;undistortPoints 的 P=K 语义本机实测核实 |
| 三角化 | 自创解法/用 abs() 除齐次分量 | 参考 `triangulate_multi_view`(SVD + 保号除法) | abs() 符号坑参考注释明确记录(深度变负) |
| Rodrigues 及雅可比 | 手推旋转导数/中位数 | `cv2.Rodrigues`(含 rot_jac)+ numpy.median | 手推 −[Xc]× 是 Phase 2 已踩坑(0.1 rad 偏 70%) |
| npz/pairs 读取 | 新读取逻辑 | 既有 `load_corners`(allow_pickle=False)+ 标准库 csv | 安全防线与测试已在 Phase 1 锁定 |
| 物点网格/真值距离 | 写死 305.16 | `build_object_points(cfg.board)` 导出对角距离 | 配置单一来源;板规格变了真值自动跟随 |
| 数值雅可比基准 | numdifftools | `scipy.optimize._numdiff.approx_derivative` | Phase 2 决策延续,零新依赖 |
| 报告渲染 | 模板引擎/matplotlib | f-string markdown + csv 标准库 | STER-04 是表格类报告;Phase 2 模式直接延续 |

**Key insight:** 双目链路的每个"看起来要写数学"的环节(相对位姿、中位聚合、消歧、去畸变、DLT、板系反变换),参考工程 2 都有一份经真实数据验证、连坑都注释在代码里的实现。移植 + 不变量测试锁定是唯一低风险路径。

---

## Common Pitfalls

### Pitfall 1: Rodrigues 向量逐分量中位的适用域(轴向翻转 / 离散度)
**What goes wrong:** 转角接近 π 时微小噪声使 Rodrigues 向量方向整体翻转(θ 与 θ−2π、轴向反号表示同一旋转),逐分量中位数给出无意义旋转;或有少数翻转帧/孪生解混入时中位被拖偏,外参错且无报错。
**Why it happens:** 轴角表示在 π 处不连续;中位数对"聚类紧"的数据稳健、对多峰数据失效。
**How to avoid:** 求解后必算离散度诊断:每帧 rv_rels 与 rv_med 的夹角分布(刚架应 < 1° 量级),超 `median_spread_max_deg` 中文报错;对角度 > π/2 的相对旋转先检查语义(本支架不应出现)。
**Warning signs:** baseline/转角与邻帧抖动异常大;离散度直方图双峰。

### Pitfall 2: 180° 角点序跨相机翻转(Phase 1 canonicalize 不解决)
**What goes wrong:** cam1/cam2 对同一物理板位姿给出互为 180° 的角点序 → 两路 PnP 解出相差 180° 转板的两个位姿 → 相对位姿错、pnp_median 双峰、DLT 对应全错位。参考实录:联合残差 150-270px,v4 时代 16 个"假 outlier"的真根因。
**Why it happens:** Phase 1 canonicalize 只按**图内几何**规范单图形态(行方向朝画面右);两相机视角不同,规范序对应的**物理角点身份**可差 180°。
**How to avoid:** 求解(train 两遍法)与评估(test cam1-PnP 锚)两处都做重投影消歧(Pattern 3);翻转次数留痕。
**Warning signs:** pass-1 离散度异常高;某对消歧前后重投影 RMS 差一个量级。

### Pitfall 3: solvePnP 与 DLT 的输入形态相反,混用即错
**What goes wrong:** solvePnP 链路必须喂**带畸变原始像素** + 全套 K/dist(OpenCV 内部处理畸变);DLT 链路必须喂**去畸变针孔像素** + 无畸变 P=K[R|t]。把去畸变点喂 solvePnP(再叠加 dist)= 双重去畸变;把带畸变点喂 P=K[R|t] = 板边缘系统性偏差(参考注释原话记录)。
**Why it happens:** 两条链路"谁负责畸变"的约定相反。
**How to avoid:** 函数签名与 docstring 明确各自契约;测试分别锁定(solvePnP 路线 vs DLT 路线的合成往返)。
**Warning signs:** 评估残差随角点离画面中心距离系统性增大。

### Pitfall 4: undistortPoints 的 P 参数口径(归一化 vs 像素)
**What goes wrong:** `undistortPoints(p, K, dist)` 默认返回**归一化**坐标;若 DLT 的 P 却用 K[R|t](像素口径),单位错乱,三角化点尺度全错。
**Why it happens:** 默认dst 无 K;两种口径只差一个 P 参数。
**How to avoid:** 照参考 `undist_pix` 固定 `undistortPoints(p32, K, dist, None, K)`(第 5 参传 K → 返回像素);本机已实测 `undistortPoints(...,None,K) == K @ undistortPoints(...,None,None)` 逐位一致。全链路统一像素口径。
**Warning signs:** 三角化点深度 1e-3 量级(归一化尺度泄漏)。

### Pitfall 5: DLT 齐次除法的符号(严禁 abs)
**What goes wrong:** SVD 零空间向量符号任意;`X_h[:3]/abs(X_h[3])` 在 X_h[3]<0 时整点取反,点跑到相机后方,距离/深度全错。
**Why it happens:** "除以最后一个分量"被顺手写成 abs(参考工程 2 注释明确记录此坑,来自博客 geometry.py 的修复)。
**How to avoid:** 保号除法原样移植(Pattern 5 代码);测试加"两视图基线翻转符号仍正确"用例。
**Warning signs:** pts_ref 的 Z(cam1 前向)出现负值。

### Pitfall 6: 相对位姿方向/语义搞反
**What gets wrong:** R_rel=R_b@R_a.T 的语义是 X_cam2 = R_rel·X_cam1 + t_rel(cam1→cam2);若按反方向组合(X_cam2 = R_pose 位姿直接用),t_rel 全错、baseline 错;旋转差个转置,重投影 RMS 却可能看不出灾难性差异。
**Why it happens:** OpenCV PnP 的 rvec/tvec 是 world→camera 方向,与"相机→相机"的外参方向易混。
**How to avoid:** 唯一实现 `relative_pose`(带参考原样 docstring);合成 e2e 测试断言方向(已知 R/t 合成角点 → 恢复比较 R 与 R.T 的混淆会被抓住);报告同时给 rvec/转角/baseline 供人工核对量级。
**Warning signs:** baseline 数值对但 t 方向分量符号与支架几何直觉相反。

### Pitfall 7: OpenCV 绑定层 dtype 陷阱(Phase 2 Pitfall 1 延续)
**What goes wrong:** solvePnP/undistortPoints/projectPoints 的点参收 float32(传 float64 部分接口报 "Unsupported format" 或静默降精度);而 K/dist 传 float64 合法(参考实测组合:点位 f32 + K/dist f64)。
**How to avoid:** 装包规则集中在调用处显式 `.astype(np.float32)`;models 路径全 float64(单一数学源纪律);parity 测试锁 cv2 与 models 一致(<1e-9px,Phase 2 已锁)。
**Warning signs:** dtype 相关 "Unsupported format";或 cv2/models 双尺结果差 ~1e-4px。

### Pitfall 8: 共面 PnP 孪生解(参考 05 v3 真实发现)
**What goes wrong:** 大倾角位姿单路 PnP 存在两个几乎同残差(0.03-0.4px)的孪生解,组合出的 t_rel 偏差可达数百 mm(参考实录:741-3059mm);逐路独立选解不可行。
**How to avoid:** 三层缓解已内建:中位数聚合(孪生是少数帧时被淹没)、消歧(消灭排序性假解)、per-frame 2 视图 BA(评估侧联合两路残差使孪生消失);加上 Pattern 6 剔除兜底 + 离散度防线(Pitfall 1)。
**Warning signs:** t_rels 逐帧分布双峰;个别对 baseline 偏离中位数百 mm。

### Pitfall 9: NaN 传播(solvePnP 失败帧)
**What goes wrong:** solvePnP 返回 False 的帧若把 NaN 塞进堆叠数组,np.median 整体变 NaN,一次污染全部外参;精化残差含 NaN 则 scipy 报难懂错误。
**How to avoid:** 失败帧跳过不入堆叠(参考 per_view.append(np.nan) + `valid = per_view[~np.isnan(per_view)]` 模式);装载层 finite 断言(Pattern 1);幸存对数下限(min_pairs)防线。
**Warning signs:** median 结果 NaN;报告里 n_valid < n_pairs 未被解释。

### Pitfall 10: 连接键与行序漂移(pairs CSV ↔ npz)
**What goes wrong:** group_id 与 image_id 的 `"_cam{n}"` 后缀拼接规则写错(或拿路径当键)→ 配对错位/漏对;装载过滤后顺序与 CSV 脱钩 → 逐帧表错行,静默算错。
**How to avoid:** Pattern 1 的连接键函数唯一化 + 不命中即中文报错 + 路径交叉核对 + 行序锁;测试用合成 npz+CSV 验证对齐。
**Warning signs:** 对数 < CSV 行数;per_frame CSV 的 group_id 顺序与 pairs 文件不一致。

### Pitfall 11: 消歧阈值 ratio=1.0 的边界抖动
**What goes wrong:** 用"必选小者"(06 的阈值 1.0)时,err_fwd ≈ err_rev 的边界帧会因亚像素噪声来回翻,跨运行不确定,且可能把正确序翻错(证据不足时两序误差接近,选错概率 ~50%)。
**Why it happens:** 两序误差接近时决策无信息量;参考 05 用 0.5 正是为了"证据充分才翻"。
**How to avoid:** flip_ratio 配置默认 0.5;未达阈值时保留原序(Phase 1 canonicalize 已给规范形态);翻转对清单留痕。
**Warning signs:** 同一配置重复运行 flip 计数不稳定(不应发生——规则是确定性的;若发生说明锚不自洽)。

### Pitfall 12: 消歧锚依赖 Phase 2 位姿产物(7/112 缺口,实测)
**What goes wrong:** 照搬参考 05 用 Phase 2 单标定逐帧位姿当锚——实测 cam1 的 extrinsics.npz 只含 116 幸存帧,112 对中 **7 对缺锚**(20261023_154043_806856_cam1 等,被内参离群剔除);缺锚帧要么崩溃要么跳过消歧(翻转帧漏网)。
**How to avoid:** train 侧用两遍法自洽锚(pass-1 pnp_median 当锚),不依赖 Phase 2 位姿产物;cam2 侧覆盖完整(155 帧全含)但同理不依赖。
**Warning signs:** 锚查找 KeyError;消歧覆盖率 < 100%。

### Pitfall 13: 内参快照与 D 布局错位
**What goes wrong:** intrinsics.json 的 params 九名布局是 [fx,fy,cx,cy,k1,k2,p1,p2,k3],而 OpenCV dist 向量布局是 [k1,k2,p1,p2,k3]——按位置硬拼会把切向/径向塞错位,solvePnP/undistortPoints 全错。
**How to avoid:** 一律按名读 params → `models.intrinsics_matrix(params)` 出 K、`params[[4,5,6,7,8]]` 出 D(models.INTRINSIC_NAMES 常量索引导航,禁止魔数下标);Phase 2 已存冗余 K/D 字段,优先直读 K/D 并断言 D.shape==(5,)。
**Warning signs:** 消歧/评估误差整体巨大且与畸变强度同向。

### Pitfall 14: 精化雅可比路线与 jac_sparsity 的组合错误(Phase 2 WR-02 延续)
**What goes wrong:** 解析 callable jac 下传 jac_sparsity 被 scipy 静默忽略(死计算 + 文档口径不准);数值 jac 下忘传 sparsity 则有限差分按稠密算,内存/时间爆炸(39424×678 稠密 ≈ 200MB/次)。
**How to avoid:** 路线二选一并在 solver_meta 记录实际用的组合;源码防线测试断言组合正确(解析 → 无 sparsity;数值 → 有 sparsity + method=="trf")。
**Warning signs:** 精化耗时异常;scipy 文档与行为不一致的隐性警告。

---

## Code Examples(verbatim-critical,均已实读核实行号)

### 相对位姿公式(core/stereo.py:36-49,含原推导)

```python
def _relative_pose(R_a, t_a, R_b, t_b):
    """从两路各自的世界系位姿算 cam_a → cam_b 的相对位姿.
    世界系: X_world
    cam_a 系: X_a = R_a @ X_world + t_a
    cam_b 系: X_b = R_b @ X_world + t_b
    消去 X_world: X_b = R_b @ R_a.T @ (X_a - t_a) + t_b
                   = (R_b @ R_a.T) @ X_a + (t_b - R_b @ R_a.T @ t_a)
    所以 R_rel = R_b @ R_a.T,  t_rel = t_b - R_rel @ t_a.
    """
    R_rel = R_b @ R_a.T
    t_rel = t_b.flatten() - R_rel @ t_a.flatten()
    return R_rel, t_rel
```

### 中位聚合(core/stereo.py:113-121)

```python
R_rels = np.stack(R_rels, axis=0)   # (N, 3, 3)
t_rels = np.stack(t_rels, axis=0)   # (N, 3)
t_med = np.median(t_rels, axis=0)
# 旋转矩阵的中位数: 用 Rodrigues 转旋转向量 → 取中位 → 转回
rv_rels = np.array([cv2.Rodrigues(R)[0].flatten() for R in R_rels])
rv_med = np.median(rv_rels, axis=0)
R_med, _ = cv2.Rodrigues(rv_med.reshape(3, 1))
```

### 消歧锚定与翻转判定(cli/06_evaluate.py:186-202)

```python
ok_pnp, rv1, tv1 = cv2.solvePnP(
    objp.reshape(-1, 1, 3), cs_cam1,
    K[REF_CAM], np.asarray(dist[REF_CAM]).astype(np.float64).reshape(-1, 1))
R1, _ = cv2.Rodrigues(rv1)
t1 = tv1.ravel()
for cam in CAMS:
    arr = det[cam].reshape(-1, 2)
    # 用 cam1 位姿 + cam 外参预测板 → cam, 再选 arr 或 arr[::-1]
    Xw = (R1 @ objp.T).T + t1
    Xc = (R_cam[cam] @ Xw.T).T + t_cam[cam]
    proj, _ = cv2.projectPoints(
        Xc.astype(np.float32).reshape(-1, 1, 3),
        np.zeros(3), np.zeros(3), K[cam],
        np.asarray(dist[cam]).astype(np.float64).reshape(-1, 1))
    proj = proj.reshape(-1, 2)
    err_fwd = np.linalg.norm(proj - arr)
    err_rev = np.linalg.norm(proj - arr[::-1])
    det_canon[cam] = arr[::-1].copy() if err_rev < err_fwd else arr
```

### 去畸变 + 评估侧三角化装配(cli/06_evaluate.py:245-263)

```python
R_ref = R_cam[combo_ref]; t_ref = t_cam[combo_ref].flatten()
projs = []
for c in combo_list:
    if c == combo_ref:
        R_c = np.eye(3); t_c = np.zeros(3)
    else:
        R_c = R_cam[c] @ R_ref.T
        t_c = t_cam[c].flatten() - R_c @ t_ref
    projs.append(projection_matrix(K[c], R_c, t_c))
ips = [undist_pix(det[c].reshape(-1, 2), K[c], dist[c]) for c in combo_list]
pts_ref = triangulate_multi_view(projs, ips)
# 用 BA 位姿反变换到 board 系: pts_board = R_pose^T @ (pts_ref - t_pose)
pts_board = (R_pose.T @ (pts_ref - t_pose).T).T
```

### undist_pix 与对角距离(core/triangulation.py:32-40, 88-100)

```python
def undist_pix(pts2d, K, dist):
    # cv2.undistortPoints 默认返回归一化像坐标 (x_n, y_n); 传 P=K 让它直接
    # 返回针孔像素坐标 (用同一个 K 把归一化点反投回去).
    p = np.asarray(pts2d, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.undistortPoints(p, K, dist, None, K).reshape(-1, 2)

def farthest_corners_distance(points_3d, board):
    idx_a = 0                                                  # (0, 0)
    idx_b = (board.rows - 1) * board.cols + (board.cols - 1)   # last corner
    return float(np.linalg.norm(points_3d[idx_a] - points_3d[idx_b])), idx_a, idx_b
```

### 外参 BA 的残差结构(core/multibundle.py:133-150,2-cam 特化示意)

```python
def residuals(self, params) -> np.ndarray:
    R_cam, t_cam, R_pose, t_pose = self._unpack(params)   # cam1 恒 (I, 0)
    for j in range(self.n_poses):
        for c, cam in enumerate(self.cams):
            if cam == self.ref_cam:
                R_total = R_pose[j]; t_total = t_pose[j]
            else:
                R_total = R_cam[cam] @ R_pose[j]
                t_total = R_cam[cam] @ t_pose[j] + t_cam[cam]
            proj = self._project(R_total, t_total, cam)   # cv2.projectPoints(带畸变像素域)
            d = proj - cs.reshape(-1, 2)                  # (88, 2)
    # 展平为 (N*2*88*2,) —— 本工程移植时残差改经 models.project(单一数学源)
```

### per-frame 位姿 BA 的外参换算(core/multibundle.py:447-457,注释原文)

```python
# 把 R_cam/t_cam (相对全局系, 通常 cam1, 语义: X_cam_c = R_cam[c] @ X_cam1 + t_cam[c])
# 重算成相对 ref_cam 的: X_cam_c = R_rel[c] @ X_ref + t_rel[c].
# 推导: X_cam_c = R_cam[c] @ X_cam1 + t_cam[c]
#        X_cam1 = R_cam[ref].T @ X_ref - R_cam[ref].T @ t_cam[ref]
#        代入 → R_rel[c] = R_cam[c] @ R_cam[ref].T
#                t_rel[c] = t_cam[c] - R_rel[c] @ t_cam[ref]
# 若 ref_cam 已是全局系 (常见), R_cam[ref]=I, t_cam[ref]=0, R_rel=R_cam, t_rel=t_cam.
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| cv2.stereoCalibrate | PnP-median + 联合 BA(anipose/EasyMocap 同构) | 参考 2 已弃用(§8.5/EXPERIMENTS:922) | 本工程禁用,源码防线锁定 |
| 独立 PnP + DLT(v1) | per-frame BA + BA 位姿三角化(v2) | 参考 06 v2 | 1m 处 24mm → 0.23mm(百倍),必须带 per-frame BA |
| 人工审剔除(exclude_images) | 配置阈值自动剔除 + 留痕 | Phase 2 OVERNIGHT 授权 | 机制平移到双目(判据对象换对) |
| OpenCV 5.0 已发布 | 锁 4.14(<5.0) | Phase 1 决策延续 | 无影响 |

---

## Decisions Needed by Planner(显式清单)

1. **外参精化雅可比路线:** A 解析稀疏雅可比(models 风格,快、与 Phase 2 纪律一致,推荐)vs B 数值雅可比 + jac_sparsity(参考 multibundle 原样,零新数学但分钟级)。二选一,不both。
2. **剔除判据对象:** 逐对重投影 RMS 中位乘数(推荐主判据)+ 对角距离残差只报告标记;还是距离残差也进剔除判据(有"用指标炼指标"耦合风险)。
3. **flip_ratio 默认值:** 0.5(参考 05,证据充分才翻,推荐)vs 1.0(参考 06,必选小者)。
4. **median_spread_max_deg 默认值:** 建议量级 1.0°(刚架相对位姿跨帧摆动应远小于此);无参考先例,planner 定,真实分布出来后配置可调。
5. **baseline sanity 带宽:** `baseline_band_mm` 上下限(支架物理基线本仓无记载;参考 2 的 1208mm 是**另一台 4 相机宽基线 rig,不可当锚**)。建议硬带 [20, 2000] 防灾难(参考 stereoCalibrate bug 产出 2500mm),|t| 实际值进报告供人工核对量级;若用户可告知支架物理基线,收窄为 ±50% 带更好。
6. **Kabsch 分轴诊断(kabsch_align/eval_axis_rms)是否移植:** MVP 最小指标是对角距离;Kabsch z/xy RMS 是加分诊断(参考实测口径),planner 决定带不带。
7. **per-frame BA 的 loss:** linear(全工程一致,推荐)vs soft_l1(参考 multiview_pose_ba 默认)。
8. **输出契约字段:** Pattern 7 为提案;字段增删(如是否含逐对三角化板点 npz 导出)planner 定。
9. **train 侧是否也跑完整评估链**(距离残差):推荐跑(剔除参考 + 报告完整),开销 112 对 × 毫秒级可忽略。

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | 本支架 baseline 在几十~几百 mm 量级(典型双目刚架;参考 2 的 1208mm 属另一台宽基线 4 相机 rig 不可比) | Open Questions 2 / Decisions 5 | 带宽设错 → 误报/漏报;硬带宽 + 报告人工核对兜底,planner/用户可改配置 |
| A2 | 相对位姿转角远离 π(刚架两相机近同向),Rodrigues 中位不触翻转退化 | Pitfall 1 | 离散度防线会当场中文报错而非静默出错,风险转为显式失败 |
| A3 | 消歧默认 ratio=0.5 对本数据合理(参考 05 在同类检测器上实测的阈值) | Pattern 3 | 配置可调;误翻/漏翻会在 pass-2 重投影 RMS 与离散度上显形 |
| A4 | 全部 406 张图像同分辨率 2560×1440(Phase 2 A2 延续);双目链路不读图像,不新增分辨率风险 | Summary | 无(链路不消费 image_size) |
| A5 | Phase 2 intrinsics.json(锚点带外但 test RMS 0.14px,02-HUMAN-UAT 已路由人工评审)可直接冻结为 Phase 3 固定输入 | Summary | 若晨间评审结论是重标内参,Phase 3 需重跑但代码不变(输入按名读取) |
| A6 | test 15 对全部 sb 路线、train 仅 5 对含 classic(实测),混合角点进双目链路可接受(Phase 2 A5 同款逻辑) | Summary | 若 classic 系统差影响外参,报告 route 列已留痕,可配置过滤重跑 |
| A7 | 精化规模(678 参数/39424 残差)数值雅可比分钟级、解析雅可比秒级的推算 | Pattern 4 | 只影响耗时不影响正确性;参考在更大规模(564 参数/64064 残差)实测可跑 |

## Open Questions

1. **(已解决)真值距离数字** — √(10²+7²)×25 = **305.16389039334257mm** 本机复算核实(STATE.md 旧值 307.45 为算术误差,ROADMAP/STATE 已于 2026-10-24 修正)。**建议:** 真值从 `build_object_points` 导出,不写死数字。→ 已折入 Pattern 5。
2. **支架物理 baseline 是多少?** — What we know: 本仓无记载;参考 2 的 1208mm 是另一台 rig。What's unclear: 用户支架实际基线量级。Recommendation: 按 Decisions 5 的硬带 + 报告核对;若用户能提供,收窄带宽(A1)。
3. **(已解决,有发现)消歧锚来源** — 实测 Phase 2 位姿锚有 7/112 缺口 → 两遍法自洽锚。→ 已折入 Pattern 3 / Pitfall 12。
4. **7 对被内参剔除的 train 对,双目侧是否要同步排除?** — What we know: 它们的角点本身在 Phase 1 检出合格,只是内参标定时被判离群;双目链路有自己的剔除机制。Recommendation: 不预排除,交给双目自己的 outlier 机制 + `exclude_pairs` 手动通道;报告可对照留痕。planner 确认。

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | 全部 | ✓ | 3.12.13(uv venv)[VERIFIED: uv run python] | — |
| opencv-python | solvePnP / Rodrigues / undistortPoints / projectPoints | ✓ | 4.14.0(锁 <5.0)[VERIFIED: uv run python] | — |
| numpy | 数组数学 / SVD / npz | ✓ | 2.5.4 [VERIFIED: uv run python] | — |
| scipy | least_squares / sparse | ✓ | 1.18.1 [VERIFIED: uv run python] | — |
| pytest / ruff | dev | ✓ | >=8 / >=0.8(既有,124 测试实测 collect) | — |
| Phase 1 产物 | 对装载输入 | ✓ | common_fov_pairs_{train,test}.csv(112/15 对)+ 四份 npz 实测在位;连接键规则实测核实 | 缺失中文报错提示先跑 preprocess |
| Phase 2 产物 | 固定内参输入 | ✓ | intrinsics.json ×2(params 九名 + K/D + image_size)实测在位 | 缺失中文报错提示先跑 intrinsic |
| 真实数据 data/ | 本阶段不读图像 | ✓(不消费) | — | 双目链路纯角点级,零 imread(比 Phase 2 更强的 data/ 只读保证) |
| CPU | solvePnP / scipy 精化 | ✓ | 20 核(cv2_threads=0 确定性) | — |

**Missing dependencies with no fallback:** 无。
**Missing dependencies with fallback:** 无(零新增依赖)。
**已知环境注意事项:** worktree 检出无 data/(gitignored),真实数据冒烟需 symlink 主仓 data/(Phase 1 既有做法);.planning 默认 commit_docs=true。

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >=8(既有,当前 124 tests 全绿,实测 0.30s collect) |
| Config file | pyproject.toml `[tool.pytest.ini_options] testpaths=["tests"]`(既有) |
| Quick run command | `uv run pytest tests/test_stereo_solve.py -x` |
| Full suite command | `uv run pytest -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| STER-01 | 合成真值(R_rel/t_rel 已知)→ pnp_median + refine_stereo 恢复:转角误差 < 0.1°、t 误差 < 1mm 量级;精化开关关闭行为;离散度防线触发 | unit(合成) | `uv run pytest tests/test_stereo_solve.py -x` | ❌ Wave 0 |
| STER-01 | 源码防线:`grep -r stereoCalibrate src/` == 0;`_distort` 在 stereo/ 计数 0;三角化只在 evaluate.py | unit(grep 防线) | `uv run pytest tests/test_stereo_guards.py -x` | ❌ Wave 0 |
| STER-02 | 输出契约 JSON roundtrip(R/t/baseline 字段断言);baseline 进配置带宽 | unit | `uv run pytest tests/test_stereo_pipeline.py -x` | ❌ Wave 0 |
| STER-03 | 合成板 DLT 往返:已知位姿+外参合成两视观测 → 评估链恢复对角距离(误差 < 0.5mm);保号除法(负齐次分量用例);undist P=K 契约;消歧翻转行为(arr[::-1] 注入 → 检出并翻回) | unit(合成) | `uv run pytest tests/test_stereo_evaluate.py -x` | ❌ Wave 0 |
| STER-03/04 | 真实数据端到端:`uv run stereo-calib stereo` → test 距离 RMS < 5mm;报告/CSV/JSON 落盘且数值一致 | e2e(生产运行) | `uv run stereo-calib stereo`(生产验收) | ❌(CLI 占位待替换) |
| STER-04 | 对装载契约(连接键/行序/三重防线)+ 剔除行为(阈值/保险丝/test 永不剔)+ 报告逐帧表 | unit + e2e(合成 npz) | `uv run pytest tests/test_stereo_pipeline.py tests/test_stereo_config.py -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** 该 task 触达模块的 quick 命令(秒级)
- **Per wave merge:** `uv run pytest -q`(全量,预计 <40s)
- **Phase gate:** 全量绿 + 真实数据生产运行(distance_rms_test_mm < 5)后 `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_stereo_solve.py` — STER-01 合成恢复 + 防线
- [ ] `tests/test_stereo_evaluate.py` — STER-03 DLT/消歧/undist 契约
- [ ] `tests/test_stereo_pipeline.py` + `tests/test_stereo_config.py` — 装载/剔除/契约/配置
- [ ] `tests/test_stereo_guards.py` — 源码防线(stereoCalibrate==0 等)
- [ ] conftest 增加 stereo 合成夹具:已知双目几何(K/D 取 Phase 2 真实值量级 + R_rel/t_rel 假想真值)→ 两视角点合成器(复用 conftest 的 projectPoints 思路,不需要渲染图像)

## Security Domain

> security_enforcement 启用(ASVS Level 1)。离线单机数值工具,无网络面/认证面,与 Phase 2 同结论。

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication / V3 Session / V4 Access Control | no | 离线 CLI,单机单用户 |
| V5 Input Validation | yes | 配置经 io/config.py frozen dataclass + 中文校验(扩展 stereo 段延续);yaml.safe_load;npz 一律 allow_pickle=False(既有测试锁定);stereo 装载层形状/有限值/连接键断言(中文) |
| V6 Cryptography | no | 无 secrets;哈希仅 data/ 只读指纹(非安全用途) |

### Known Threat Patterns for 本 stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| 恶意 npz(pickle 载荷) | Tampering/Elevation | load_corners 全链 allow_pickle=False;新读盘代码照抄 |
| 配置注入(yaml 反序列化) | Elevation | yaml.safe_load(既有,绝不 yaml.load) |
| 产物写出 data/ | Tampering | 写路径全从 output_root 派生 + 互斥校验;rmtree 目标仅 output_root/stereo |
| 剔除/翻转留痕丢失 | Repudiation | dropped_pairs + flip_log 全落盘进 JSON 与报告,可审计 |

---

## Sources

### Primary (HIGH confidence)

- **参考工程2 源码逐行阅读(2026-10-24,只读,行号实核)**:
  - `.../calibration/core/stereo.py` — PnP-median 全套(_relative_pose:36-49 / solvePnP 调用形态:82-91 / 中位聚合:113-121 / 逐帧双目 RMS:137-163)
  - `.../calibration/cli/05_extrinsic_multi.py` — 排序规范化消歧(102-131,阈值 0.5)、外参 BA 调用形态(265-294)、共面孪生解发现(v3 注释 133-140)
  - `.../calibration/cli/06_evaluate.py` — 评估链全链(消歧 180-203 / per-frame BA 调用 229-232 / undist+DLT+板系+对角 245-269 / 真值公式 103-104)、v1→v2 24mm→0.23mm 记录(14-17)
  - `.../calibration/core/triangulation.py` — projection_matrix(24-29)/ undist_pix(32-40)/ DLT 保号(59-85)/ farthest(88-100)/ kabsch(122-174)
  - `.../calibration/core/multibundle.py` — JointBundleAdjustment(61-205:pack/unpack 94-124、残差 133-150、稀疏 152-170、run 172-180)/ multiview_pose_ba(423-494:外参换算 447-457、残差 462-478)
  - `.../calibration/core/board_config.py`(77-88 object_points 行主序契约)、`.../calibration/cli/diag_corner_ordering.py`(翻转判定实证)
  - `.../calibration/EXPERIMENTS.md`(:169 stereoCalibrate 崩坏实录 |t|≈2500mm;:922 弃用原因;:15-16 另一台 rig 基线 1208/802/529mm)
  - `.../calibration/outputs/eval_summary.json` — cam1+cam2 组合实测:farthest_err_abs_mean **0.2301mm**、BA RMS 0.1122px(n=39;真值 305.16389039334257 同值互证)
- **本工程代码与产物实测(2026-10-24,uv run)**:
  - `src/stereo_calib/intrinsic/models.py`(单一数学源:project/unproject/transform_to_camera/intrinsics_matrix 签名全读)、`scoring.py`(PnP 固定内参先例)、`observation.py`(装载三重防线)、`solve.py`(row_index/pack_theta/精化结构)、`artifacts.py`(extrinsics.npz 契约)、`cli/main.py`(stereo 占位:98-101)、`io/config.py`(IntrinsicConfig 模式 124-152)、`preprocess/artifacts.py`(八数组契约 + write_common_pairs_csv)
  - `config.yaml` 全文;`output/train/cam1_train.npz`(image_ids 形态实测)、`common_fov_pairs_train.csv`(group_id/路径列实测)、`output/intrinsic/cam1/intrinsics.json`(15 字段 + params 数值实测)、`output/intrinsic/cam{1,2}/extrinsics.npz`(锚覆盖 109/112 与 112/112 实测)
  - 数值实测:`np.sqrt(10**2+7**2)*25 = 305.16389039334257`;`undistortPoints(p,K,d,None,K) == K @ undistortPoints(p,K,d,None,None)` 逐位一致

### Secondary (MEDIUM confidence)

- 参考工程 2 的 EXPERIMENTS.md 精度理论段(σ_z = z²/(B·fx)·σ_px,基线/焦距主导;实测/理论比 0.6-0.75 稳定)——用于预期残差量级论证
- OpenCV solvePnP/undistortPoints 公式语义与 docs.opencv.org/4.x 一致(参考工程 parity 实测互证)

### Tertiary (LOW confidence)

- 无(全部结论有参考工程源码或本机实测支撑)

---

## Metadata

**Confidence breakdown:**
- 标准栈与移植映射: HIGH — 逐行移植已验证代码,行号实核,同版本环境,合成 e2e + 源码防线双层测试锁定
- 评估链(DLT/消歧/去畸变): HIGH — 参考 v2 版实测 0.23mm 锚点,关键坑(保号/P=K/输入形态)均有参考注释 + 本机实测互证
- 精化与剔除机制: MEDIUM-HIGH — 结构来自已验证模式;阈值默认值(A3/Decisions 2-4)需执行期按真实分布调,配置可调 + 留痕兜底
- baseline 量级: LOW(planner/用户决策)— 本仓无支架物理记载,硬带宽 + 报告人工核对兜底(A1)

**Research date:** 2026-10-24
**Valid until:** 2026-11-23(依赖锁定栈稳定;公式类结论长期有效;OpenCV/scipy 升级需复核 dtype 与 least_squares 行为)
