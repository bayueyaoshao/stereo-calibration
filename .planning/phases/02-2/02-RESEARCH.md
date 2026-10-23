# Phase 2: 模块2 内参标定 - Research

**Researched:** 2026-10-23
**Domain:** 单相机内参标定(M2 模型:calibrateCameraExtended 初值 + scipy least_squares 精化、float64 回代 RMS、test 集验证、离群帧处理)
**Confidence:** HIGH(核心结论全部来自对已验证参考工程 `/home/lx/Works/src/lx/github/intrinsic-calibration` 源码的逐行阅读,并在本机环境实测验证了 npz 契约、图像分辨率与依赖版本;该参考工程用同一类相机、同一棋盘规格端到端验证过本阶段的全部数学)

---

## Summary

本阶段是把参考工程已验证的内参标定链路移植到本工程的三个模块框架里:消费 Phase 1 的角点 npz(output/{train,test}/cam{n}_{split}.npz 八数组契约),对 cam1/cam2 分别做 M2 模型(fx,fy,cx,cy,k1,k2,p1,p2,k3)标定。参考工程的对应实现**每一环都有现成、经真实数据验证的代码**:`solve/opencv.py`(calibrateCameraExtended 初值,flags=0,float32 装包)、`solve/residuals.py` + `solve/scipy_solver.py`(θ 联合装箱 + 解析稀疏雅可比 + trf/lsmr 精化 + 实用收敛判定)、`solve/result.py` 的 `compute_result_metrics`(float64 回代 RMS,绝不用 OpenCV 返回值)、`solve/scoring.py` 的 `solve_pose_only`(test 帧固定内参只估位姿评分,含反投影初值防错误盆地)、`models/pinhole_brown.py`(M2 投影/解析雅可比/牛顿迭代反投影)。移植策略沿用 Phase 1 已验证的"移植 + 不变量测试锁定"模式。

关键差异点(参考工程 → 本工程)有四个,均有明确解法:(1)**规模小得多**——本工程 train 每路 141/155 帧(参考 640 帧),OpenCV 求解预估 ~25-35s/路、scipy 精化 ~3-5s/路,总耗时两路合计远小于 2 分钟;(2)**输入契约不同**——本工程 npz 自带 objp/uv/detected/source,不需要参考工程的 manifest 对齐,装载层更简单,但 npz 与 meta.json 都**没有图像分辨率**,需要 imread 一帧检出图像获取 (2560, 1440);(3)**离群帧处理是新增机制**——参考工程靠人工审阅 + 配置 `exclude_images` 清单逻辑剔除,本工程按 OVERNIGHT AUTONOMY 授权需要"配置阈值 + 自动剔除继续算 + 报告留痕"的自动机制(只作用于 train,test 帧永不剔除只标记);(4)**模型固定 M2**——不需要参考工程的 M1-M4 注册机制,移植时裁剪为单一 M2 实现即可。

精度预期有硬锚点:参考工程同类相机的实测结果为 train RMS ≈ 0.092px、test RMS ≈ 0.160px(65 张独立 test 帧),内参 fx≈1856 / fy≈1846 / cx≈1270 / cy≈724 / k1≈-0.406 / k2≈0.232 / k3≈-0.082(2560×1440)。本阶段成功标准的 < 1.0px 有约 6 倍裕量,预期 ~0.1-0.5px 与参考实测吻合。

**Primary recommendation:** 按"models(M2 数学)→ observation(npz 装载)→ solve(OpenCV 初值 + scipy 精化)→ scoring(test 只估位姿)→ pipeline(离群剔除编排)→ 报告与产物"顺序移植;残差与 RMS 一律走 float64 回代(同一把尺),test 帧评分必须走 unproject 反投影初值路线(参考工程实测:老路线在 40° 倾角帧上误评 4.82px,反投影路线 0.07px);离群剔除用"逐帧 RMS > k×中位数"的配置阈值 + 最大剔除比例保险丝,只剔 train。

---

## Architectural Responsibility Map

本工程是离线单机数值工具,无网络分层;按下表的项目内部分层映射(每个能力只有一个属主,防止数学被抄写两份)。

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| M2 投影/反投影/解析雅可比公式 | `intrinsic/models.py`(新) | — | 单一数学源:残差、精化、评分全部只准 import 这里的投影函数(参考工程 D-06 模式) |
| npz 装载与视图过滤 | `intrinsic/observation.py`(新) | 复用 `preprocess/artifacts.load_corners`、`build_object_points` | Phase 1 契约的标准读取入口已存在;装载层只做 detected 过滤 + 有限值断言 + 行序锁 |
| calibrateCameraExtended 初值 | `intrinsic/solve.py` 的 OpenCV 路径 | models 适配逻辑 | OpenCV 是黑盒初始化器;float32 装包规则隔离在调用处 |
| scipy 精化(装箱/残差/稀疏雅可比/收敛判定) | `intrinsic/solve.py` 的 scipy 路径 | 只 import models | 残差仅由 models.project 构成(参考 D-08 模式,源码防线测试锁定) |
| float64 回代 RMS(唯一指标尺) | `intrinsic/metrics.py`(新) | 只 import models | compute_result_metrics 移植;OpenCV 返回 RMS 只进 solver_meta 留参考 |
| test 帧只估位姿评分 | `intrinsic/scoring.py`(新) | 只 import models | solve_pose_only 移植(反投影初值 + 6 参数 trf 精化) |
| 离群帧剔除编排(overnight autonomy) | `intrinsic/pipeline.py`(新) | 配置阈值 | 标定 → 回代逐帧 RMS → 剔除超阈帧 → 重标定的循环,留痕进报告;只剔 train |
| 内参文件/报告/CSV 落盘 | `intrinsic/artifacts.py`(新) | — | 参数按名 + 分辨率 + K/D + RMS + 剔除清单;Phase 3 的输入接口在此定死 |
| 配置解析(intrinsic 段) | `io/config.py` 扩展 | — | frozen dataclass + 中文校验既有模式延续 |
| CLI intrinsic 子命令 | `cli/main.py`(替换占位) | — | except (ValueError, OSError) 中文收口 + Exit(1) 既有模式延续 |

---

<user_constraints>

## Project Constraints (from CLAUDE.md)

以下指令来自 CLAUDE.md 与 PROJECT.md,规划必须逐条遵守(无 CONTEXT.md,本节即约束来源):

- **Tech stack**: Python + OpenCV 4.x + numpy + scipy,参考工程1风格(配置文件驱动,命令行零算法参数)— 与已验证工程保持一致
- **配置文件**: 所有传参必须用配置文件控制(棋盘格规格、路径、算法参数、开关),禁止硬编码
- **模块分离**: 三个模块相互独立,每个可单独运行调试,输入输出文件明确;上游输出即下游输入
- **依赖**: 只读引用两个参考工程的方案与代码,绝不修改其任何内容(`intrinsic-calibration` 与 `multi_cameras` 均只读)
- **性能**: 生产用途,快速计算
- **数据只读**: data/ 目录图像不可修改,所有中间产物和结果写到独立输出目录
- **GSD Workflow**: 文件改动必须经 GSD 工作流(本阶段即 /gsd-plan-phase → /gsd-execute-phase)
- **OVERNIGHT AUTONOMY(用户授权)**: 计划必须包含离群帧处理机制(配置控制阈值 + 剔除继续算 + 报告留痕),不阻塞在人工复核上
- **既有代码模式**(Phase 1 确立):frozen dataclass 配置载体、中文报错收口、源码防线测试(grep count 锁定)、data 只读 mtime/md5 验证、`--config` 是唯一 CLI 选项(挂在 callback 上,置于子命令前)、npz 读取一律 allow_pickle=False

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| INTR-01 | 用 train 集每路相机全部可用图像标定 M2 模型:calibrateCameraExtended 初值 + scipy least_squares 精化 | 参考工程 `solve/opencv.py` `_solve_via_calibrate_camera`(flags=0,float32 装包)+ `solve/scipy_solver.py`(θ 装箱 + 解析稀疏雅可比 + trf/lsmr + 实用收敛)完整可移植;精化开关进配置(成功标准 1) |
| INTR-02 | 输出两路相机内参文件(参数按名保存 + 图像分辨率) | 输出契约 `output/intrinsic/cam{n}/intrinsics.json`:params 字典按名 + image_size + K/D + RMS + 剔除清单(见 Architecture Patterns §Pattern 7);模块 3 按名读取为固定内参 |
| INTR-03 | test 集整体重投影 RMS(float64 回代,不用 OpenCV 返回值) | `compute_result_metrics` 移植( pooled RMS,sqrt(全部平方残差均值))+ `solve_pose_only` 移植(test 帧固定内参只估位姿,unproject 反投影初值防错误盆地) |
| INTR-04 | 输出内参标定报告(逐帧误差 + 整体指标) | markdown 报告 + utf-8-sig CSV 逐帧表:image_id / source / train RMS / test RMS / dropped 标记 / 中文离群说明;离群帧可识别(requirement + OVERNIGHT AUTONOMY 双重要求) |

</phase_requirements>

---

## Standard Stack

### Core(全部已安装锁定,本阶段零新增运行时依赖)

| Library | Version(实测) | Purpose | Why Standard |
|---------|--------------|---------|--------------|
| opencv-python | 4.14.0(锁 <5.0) | calibrateCameraExtended 初值、solvePnP、Rodrigues(含雅可比)、projectPoints(parity 测试基准) | 与参考工程同版本组合,全部行为实测过 [VERIFIED: uv run python import] |
| numpy | 2.5.4 | 全部数组数学、npz 产物 | 与 scipy 1.18 成对锁定(Phase 1 决策)[VERIFIED: uv run python import] |
| scipy | 1.18.1 | optimize.least_squares(trf + lsmr + 稀疏雅可比)、sparse 结构矩阵 | 参考工程实测真实规模收敛 [VERIFIED: uv run python import] |
| Python | 3.12.13 | 运行时(uv venv) | pyproject requires-python >=3.12 [VERIFIED: uv run python] |

### Supporting

零新增。dev 组已有 pytest>=8 / ruff>=8。数值雅可比测试基准用 `scipy.optimize._numdiff.approx_derivative`(scipy 自带,不引入参考工程的 numdifftools dev 依赖——参考工程 D-10 锁 numdifftools 是它自己四模型研究项目的选择,本工程单一 M2 用 scipy 自带数值差分即可达到同样的"解析式对数值基准逐列校验"目的,零新依赖)。

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| 解析稀疏雅可比(移植) | scipy 数值雅可比 `jac="2-point"` + jac_sparsity | 本工程 141/155 视图规模数值雅可比可行(约 4s/次),但 ftol=1e-10/xtol=1e-12 的收敛精度达不到,且参考工程解析式已验证到 6.7e-10(对中心差分);移植解析式是复制粘贴已验证代码,零风险优势 |
| 稠密雅可比 | scipy.sparse 稀疏结构 | 本规模稠密 (24816, 855) ≈ 170MB 可行,但稀疏模式是参考工程实测验证过的(且随视图数平方内存增长),照搬已验证模式 |
| matplotlib 报告图 | 纯文本 markdown 表 + CSV | INTR-04 只要求逐帧误差 + 整体指标 + 可识别离群帧,文本/表格足够;零新依赖(参考工程画图是它存疑帧对比研究的需要,本工程 MVP 不需要) |
| 离群帧人工审阅(参考工程 exclude_images 模式) | 配置阈值自动剔除 + 报告留痕 | OVERNIGHT AUTONOMY 授权明确要求自动机制;配置保留手动剔除清单作为补充手段(两 mechanisms 可叠加) |

**Installation:**

```bash
# 零新增依赖,无需安装。环境验证命令:
uv run python -c "import cv2, numpy, scipy; print(cv2.__version__, numpy.__version__, scipy.__version__)"
# 实测输出:4.14.0 2.5.4 1.18.1
```

**版本验证:** 2026-10-23 在本机 uv 环境实测(python 3.12.13 / opencv 4.14.0 / numpy 2.5.4 / scipy 1.18.1),与参考工程完全同版本组合。[VERIFIED: 本机 uv run]

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
config.yaml(board / output_root / cv2_threads / intrinsic 段:精化开关+容差+离群阈值)
        │  load_config(中文校验,io/config.py 扩展)
        ▼
Phase 1 产物(只读输入契约):
  output/train/cam{n}_train.npz + output/test/cam{n}_test.npz
  (八数组:image_ids/paths/uv/objp/source/detected/sq_px/win_size,allow_pickle=False)
        │
        ▼
intrinsic/observation.py:装载(NaN 整行过滤 → detected 掩码 → 有限值断言
  → objp 与配置重建网格 allclose 交叉核对 → image_ids 行序锁 → image_size 从检出首帧 imread)
        │
        ▼  每路相机独立重复(cam1:train 141 帧 / cam2:train 155 帧;两路串行)
┌─────────────────────────────────────────────────────────────┐
│ intrinsic/pipeline.py:离群剔除循环(OVERNIGHT AUTONOMY)      │
│   1. solve/opencv 路径:calibrateCameraExtended(float32 进,  │
│      flags=0)→ 初值 params + rvecs/tvecs + opencv_rms(只留参考)│
│   2. metrics:float64 回代逐帧 RMS(同一把尺)                  │
│   3. 逐帧 RMS > k×中位数 → 记入剔除清单;超 max_drop_fraction   │
│      或 max_iterations 保险丝即停 → 回到 1 重标定              │
│   4. 幸存集定稿 → (配置开关)scipy 精化:                        │
│      θ=[9 内参 | 每视图 6 外参] 联合装箱                       │
│      残差 = models.project(transform_to_camera(objp,ω,t))−uv  │
│      解析稀疏雅可比(trf+lsmr+x_scale=jac)+ 实用收敛判定        │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
intrinsic/scoring.py:test 15 帧逐帧只估位姿(内参加锁):
  model.unproject(反投影=天然去畸变)→ 归一化平面 solvePnP(单位阵 K)
  → 6 参数 trf 精化 → 每帧 RMS(失败帧记 NaN + 中文备注,不中断)
        │
        ▼
metrics:test 集整体 RMS = float64 回代 pooled(全部平方残差均值开根)
        │
        ▼
intrinsic/artifacts.py 落盘:
  output/intrinsic/cam{n}/intrinsics.json   ← 模块 3 的固定内参输入(参数按名+分辨率)
  output/intrinsic/cam{n}/{train,test}_residuals.npz + extrinsics.npz(诊断)
  output/intrinsic/report.md + per_frame_cam{n}.csv(逐帧表+离群识别,INTR-04)
```

读者可顺着箭头从 config + npz 走到报告;唯一的数学源 `intrinsic/models.py` 被求解、评分、指标三侧同时 import,这就是"一把尺量到底"的结构保证。

### Recommended Project Structure

```
src/stereo_calib/intrinsic/           # 新包(与 preprocess/ 平级,模块分离约束)
├── __init__.py
├── models.py          # M2 投影/解析雅可比/牛顿反投影/畸变核(移植 pinhole_brown.py,裁剪为 M2 单模型)
├── observation.py     # npz 装载 → 视图列表(过滤/断言/行序锁/image_size 派生)
├── solve.py           # OpenCV 初值路径 + scipy 精化路径(装箱/残差/雅可比/实用收敛)
├── metrics.py         # float64 回代 RMS(compute_result_metrics 移植)
├── scoring.py         # test 帧只估位姿评分(solve_pose_only 移植)
├── pipeline.py        # run_intrinsic 编排:两路循环 + 离群剔除 + 中文汇总
└── artifacts.py       # intrinsics.json / residuals.npz / report.md / CSV 落盘
tests/
├── test_intrinsic_models.py     # parity vs cv2.projectPoints(float64)+ 雅可比数值校验 + 往返投影
├── test_intrinsic_solve.py      # 合成真值恢复、精化开关、失败契约、RMS 不用 OpenCV 返回值(源码防线)
├── test_intrinsic_scoring.py    # 只估位姿(含大倾角错误盆地回归用例)
├── test_intrinsic_pipeline.py   # npz 装载契约、离群剔除行为、端到端合成 npz → 产物
└── test_intrinsic_config.py     # intrinsic 段中文校验
config.yaml                        # 扩展 intrinsic: 段
```

### Pattern 1: M2 模型移植(单一数学源)

**What:** 移植 `pinhole_brown.py` 中 M2 相关的 `_distort` / `_radial` / `_radial_and_derivs` / project / project_jacobian / project_point_jacobian / unproject / intrinsics_matrix,裁掉 M1/M3 分支与注册机制(本工程模型固定 M2,无注册需求)。

**When to use:** 一切投影、残差、评分、报告回代的唯一入口。

**Example(投影核心,畸变公式与 OpenCV 4.14.0 parity 到 ~5e-13px,参考工程实测):**

```python
# 来源:参考工程 src/calib/models/pinhole_brown.py(只读移植)
def _distort(x, y, dist):  # dist = [k1,k2,p1,p2,k3]
    r2 = x * x + y * y
    rad = 1.0 + dist[0] * r2 + dist[1] * r2**2 + dist[4] * r2**3
    p1, p2 = dist[2], dist[3]
    xd = x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd   # 切向项加在径向缩放之外(嵌套方式经数值判定对齐 cv2)

def project(pts_cam, params):  # (N,3) + (9,) → (N,2),float64;断言 Z 全 > 0
    x, y = pts_cam[:, 0] / pts_cam[:, 2], pts_cam[:, 1] / pts_cam[:, 2]
    xd, yd = _distort(x, y, params[4:])
    return np.stack([params[0] * xd + params[2], params[1] * yd + params[3]], axis=1)
```

**unproject(牛顿迭代反投影,~20 行):** 像素 → 归一化畸变坐标 → 解畸变正映射方程的逆(2×2 解析雅可比迭代到修正量 < 1e-15)→ 单位射线。project → unproject → project 往返 < 1e-9px(参考工程测试锁定)。**它是 test 帧评分正确性的关键**(见 Pitfall 7),必须移植,不能用"零畸变直接解 PnP"替代。

### Pattern 2: OpenCV 初值路径(calibrateCameraExtended)

**What:** flags=0(估全 5 个畸变系数 = M2),输入物点/角点显式降 float32(绑定层只收 32 位),image_size 传 (宽, 高) 元组。

```python
# 来源:参考工程 src/calib/solve/opencv.py L122-151(只读移植)
# OpenCV 4.14 绑定只收 32 位浮点,传 float64 报 "Unsupported format";
# 像素坐标 float32 表达精度 ~1e-4px,远小于 0.5px 断言线,可接受
objs32 = [objp.astype(np.float32).reshape(n, 1, 3) for _ in views]
imgs32 = [uv.astype(np.float32).reshape(n, 1, 2) for uv in views]
rms, K, D, rvecs, tvecs, _std_int, _std_ext, _per_view = cv2.calibrateCameraExtended(
    objs32, imgs32, (width, height), None, None, flags=0)
# K/D/rvecs/tvecs 全部转回 float64;rms 只进 solver_meta["opencv_rms"] 留参考
```

D 返回 5 长 [k1,k2,p1,p2,k3] 直接全取(flags=0 无切片陷阱;参考工程的 FIX_K3=4 长 / RATIONAL=14 长切片问题在本工程不存在,但解包后仍要断言 `D.shape == (5,)` 防御)。

### Pattern 3: scipy 精化(θ 联合装箱 + 解析稀疏雅可比 + 实用收敛)

**What:** θ = [9 内参 | 视图 0 的 (ω,t) 6 个 | 视图 1 的 6 个 | …](141 视图 → 855 维)。残差与雅可比共用同一 row_index 函数(行序契约);trf + lsmr + jac_sparsity + x_scale="jac";loss 固定 "linear"(离群帧由剔除机制处理,不需要鲁棒核;见 Open Questions)。

```python
# 来源:参考工程 src/calib/solve/scipy_solver.py + residuals.py(只读移植)
sol = least_squares(traced_residual, theta0,
                    jac=make_jacobian_fn(model, obs),       # 解析稀疏 csr
                    jac_sparsity=make_sparsity(model, n_views, n_points),
                    method="trf", tr_solver="lsmr", x_scale="jac",
                    loss="linear", max_nfev=cfg.max_nfev,
                    ftol=cfg.ftol, xtol=cfg.xtol, gtol=cfg.gtol)
# 实用收敛判定(纯函数移植):status>0 → 收敛;status<=-1 → 失败;
# status==0 → 尾部窗口(10 次)代价包络相对改进 < 1e-6 视为走平即收敛。
# 判定依据(status/nfev/final_cost/tail_improvement)全进 solver_meta,不静默。
```

雅可比链式三块(全部移植,公式经参考工程数值验证 6.7e-10):
1. `project_jacobian`(N,2,9)——内参列,模型层;
2. `project_point_jacobian`(N,2,3)·`transform_to_camera` 的 dXc_d_pose(N,3,6)——einsum 链式;
3. **dXc/dω 必须用 `cv2.Rodrigues(r)` 返回的现成雅可比**(`rot_jac`),绝不写 ∂Xc/∂ω=−[Xc]×——那只是 ω=0 处的极限形式,0.1 rad 就偏 70%(参考工程 02-04 的 Rule 1 修复教训,现 base.py L185-191 是修复后版本)。

**本工程规模实测推算:** 141 视图稠密雅可比 (24816, 855) float64 ≈ 170MB(参考工程 640 视图时是 3.47GB 不可行);稀疏 nnz = 2·V·P·15 ≈ 37 万,单次雅可比组装 < 0.05s,精化全程预估 3-5s。稀疏模式照搬参考(已验证 + 随视图数可扩展)。

### Pattern 4: float64 回代 RMS(唯一指标尺)

**What:** rms_overall / per_view_rms 一律由 `compute_result_metrics` 用 models.project 以 float64 回代重算;OpenCV 返回的 rms/per_view 只进 solver_meta 作对账参考。 pooled 整体 RMS = sqrt(全部视图全部角点平方欧氏残差的均值),逐帧 RMS = sqrt(该帧均值)。

```python
# 来源:参考工程 src/calib/solve/result.py L114-155(只读移植)
residuals = np.empty((n_views, n_points, 2), dtype=np.float64)
for i in range(n_views):
    pts_cam, _ = transform_to_camera(obs.object_points, rvecs[i], tvecs[i])
    residuals[i] = model.project(pts_cam, params) - obs.image_points[i]
squared = np.sum(residuals * residuals, axis=2)          # (V, P)
per_view_rms = np.sqrt(squared.mean(axis=1))             # (V,)
rms_overall = float(np.sqrt(squared.mean()))             # pooled
```

**两把尺的差异是实测的:** OpenCV 内部 RMS 基于 float32 角点,与 float64 回代差 ~1e-4px。成功标准 3 明确要求"不用 OpenCV 返回值"——测试用源码防线锁定(如:metrics/solve 模块中报告路径不引用 calibrateCameraExtended 返回的 rms 变量,grep 断言模式照 Phase 1)。

### Pattern 5: test 帧只估位姿评分(unproject 反投影初值)

**What:** test 帧没参与标定,位姿未知。内参当尺子:每帧先 unproject 反投影成单位射线(反投影天然完成去畸变),取归一化平面坐标后用单位阵 K + 零畸变解 solvePnP 得位姿初值,再 6 参数 trf 精化(内参加锁),回代得该帧 RMS。

```python
# 来源:参考工程 src/calib/solve/scoring.py L79-158(只读移植)
rays = model.unproject(uv, params)                        # (P,3) 单位射线
ok, rvec0, tvec0 = cv2.solvePnP(
    objp, rays[:, :2] / rays[:, 2:3],
    np.eye(3, dtype=np.float64), np.zeros(5, dtype=np.float64))
def residual(theta):
    pts_cam, _ = transform_to_camera(objp, theta[:3], theta[3:])
    return (model.project(pts_cam, params) - uv).ravel()  # 内参 params 固定不动
sol = least_squares(residual, np.concatenate([rvec0, tvec0]), method="trf", x_scale="jac")
rms = float(np.sqrt(np.mean(np.sum(sol.fun.reshape(-1, 2) ** 2, axis=1))))
```

单帧解不出(solvePnP 失败 / 射线 Z 非正)记 (NaN, 中文备注) 不中断整批(参考 T-02-06-04 模式)。15 帧 × 毫秒级,开销可忽略。test 整体 RMS 由全帧残差 pooled 重算(与 train 同一把尺),不是逐帧 RMS 的均方平均的简写(88 点等帧下两者数值相同,但 pooled 公式口径唯一)。

### Pattern 6: 离群帧自动剔除(OVERNIGHT AUTONOMY 机制)

**What:** 配置控制的"标定 → 回代逐帧 RMS → 剔除超阈帧 → 重标定"循环,全程留痕。

**设计要点(全部从参考工程模式推导):**
- **判据尺:** 逐帧 RMS 用 float64 回代(同一把尺),不用 OpenCV per-view;
- **阈值:** `outlier.rms_multiplier`(默认建议 5.0)——逐帧 RMS > multiplier × 幸存集中位数判离群。中位数基准稳健(不受离群自身拖动);备选口径是参考工程的 MAD 稳健 z 分数 ≥ 3.5(同源惯例),实现二者取一即可,阈值进配置可调;
- **保险丝:** `outlier.max_drop_fraction`(默认建议 0.2,防剔除循环吃掉过多视图;同时保护 calibrateCamera 的最少视图需求)与 `outlier.max_iterations`(默认建议 3,防中位数随剔除漂移的迭代不收敛);
- **范围:** 只剔 train;test 帧永不剔除,只在高 RMS 时报告标记(它是考卷,剔了就作弊);
- **留痕:** 每轮剔除的 image_id + 该轮 RMS + 触发阈值全部进报告与 intrinsics.json 的 `dropped_frames` 字段;
- **时序:** 剔除循环在 OpenCV 初值阶段做(每次 ~25-35s,可承受多轮);scipy 精化只跑幸存集定稿后的一次(精化贵,不进循环);
- **配置补充:** 参考工程的 `exclude_images` 手动清单模式可以作为并行机制保留(配置里列 image_id,装载层跳过)——人工复核通道与自动机制不冲突。

### Pattern 7: 输出契约(模块 3 的固定内参输入,INTR-02)

```jsonc
// output/intrinsic/cam1/intrinsics.json —— Phase 3 按名读取的唯一接口
{
  "cam": 1,
  "image_size": [2560, 1440],            // (宽, 高),calibrateCameraExtended 同源
  "params": {                             // M2 按名,布局即参考工程 param_names
    "fx": ..., "fy": ..., "cx": ..., "cy": ...,
    "k1": ..., "k2": ..., "p1": ..., "p2": ..., "k3": ...
  },
  "K": [[...],[...],[...]],               // 3×3,冗余存储避免模块 3 重建出错
  "D": [...],                             // [k1,k2,p1,p2,k3] OpenCV 布局,同理
  "solver": "scipy",                      // "opencv" | "scipy"(精化开关关闭时)
  "refine_enabled": true,
  "rms_train_overall": ...,               // float64 回代,标定集(幸存帧)
  "rms_test_overall": ...,                // float64 回代,test 15 帧(成功标准 3 的数字)
  "n_views_train": ..., "n_views_test": ...,
  "dropped_frames": [{"image_id": "...", "rms": ..., "round": 1, "threshold": ...}],
  "solver_meta": {...},                   // status/nfev/opencv_rms 等对账信息
  "board": {...}, "config_snapshot": {...} // 留痕(照 Phase 1 meta 模式)
}
```

配套诊断产物(参考工程三件套模式):`train_residuals.npz` / `test_residuals.npz`(image_ids 定长 unicode + residuals (V,88,2) + per_view_rms,allow_pickle=False 可读)、`extrinsics.npz`(train rvecs/tvecs,诊断用;模块 3 自己做 PnP,不消费它)。整体报告 `output/intrinsic/report.md`(INTR-04:两路对比 + 逐帧表 + 离群识别 + 中文结论)+ `per_frame_cam{n}.csv`(utf-8-sig,Excel 直开,Phase 1 既有模式)。运行前 rmtree `output_root/intrinsic`(D-11 固定覆盖模式,安全前提是既有 output/data 互斥校验,不重复实现)。

### Pattern 8: 配置段与 CLI(既有模式延伸)

```yaml
# config.yaml 追加段(全部算法参数/开关在此,CLI 零算法参数不变)
intrinsic:
  refine:
    enabled: true          # 精化开关(成功标准 1 的配置控制要求)
    max_nfev: 300
    ftol: 1.0e-10
    xtol: 1.0e-12
    gtol: 1.0e-10
  outlier:
    enabled: true          # 自动离群剔除开关(OVERNIGHT AUTONOMY)
    rms_multiplier: 5.0    # 逐帧 RMS > k × 幸存集中位数 → 剔除(下轮重标定)
    max_drop_fraction: 0.2 # 保险丝:累计剔除不得超过该比例
    max_iterations: 3      # 保险丝:剔除-重标定循环上限
  rms_test_max: 1.0        # 成功标准阈值:报告按此判 PASS/FAIL(不作硬退出条件)
  exclude_images: []       # 手动逻辑剔除清单(参考工程 D-03 模式,与自动机制叠加)
```

解析进 `IntrinsicConfig` frozen dataclass(io/config.py 扩展,中文校验:bool 冒充 int 拦截、正数校验、乘数 > 1 等)。CLI:`intrinsic` 子命令替换占位,坏输入 `except (ValueError, OSError)` 中文收口 + Exit(1)(既有模式)。method 白名单只收 "trf"('lm' 只支持稠密雅可比,参考工程配置层直接拒收的防呆照搬)。

### Anti-Patterns to Avoid

- **在 solve/metrics 里重写投影或畸变公式**——哪怕一行。数学只准 import models.py(parity 测试 + 源码防线双保险)。
- **把 OpenCV 返回的 rms / per_view 当报告指标**——float32 尺,与 float64 回代差 ~1e-4px;成功标准 3 明确禁止。
- **test 帧评分走"零畸变直接解 PnP"老路线**——大倾角 + 强畸变 + 棋盘偏画面中心时把精化带进错误盆地(参考工程真实案例:40° 倾角帧误评 4.82px,unproject 路线 0.07px)。
- **手写 ∂Xc/∂ω = −[Xc]×**——只在 ω=0 成立;用 cv2.Rodrigues 的现成雅可比。
- **剔除 test 帧**——考卷不可改;test 高 RMS 只标记。
- **残差与雅可比各自维护行序**——共用一个 row_index 函数,否则梯度接错行、不报错只是收敛奇慢。
- **逐点 Python 循环组装残差/雅可比**——按视图循环、视图内全 numpy 向量化(参考实测快 8 倍以上)。

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| 非线性最小二乘优化器 | 自写 LM/高斯牛顿/分块交替 | scipy `least_squares`(trf+lsmr) | 参考实测真实规模收敛;自建只有引入数值 bug 的空间 |
| 稀疏矩阵结构 | 手写稀疏索引 | scipy.sparse(coo→csr)+ tr_solver="lsmr" | jac_sparsity 由 scipy 自动分组校验 |
| Rodrigues 及其雅可比 | 手推旋转导数 | `cv2.Rodrigues(r)` 返回 (rot, rot_jac) | 语义与 OpenCV 全兼容;rot_jac 对中心差分实测 ~1e-10;手推 −[Xc]× 是已踩过的坑 |
| 物点网格 | 再写一遍 | 既有 `stereo_calib.preprocess.artifacts.build_object_points` | 本工程 Phase 1 已有同一实现;npz 内 objp 与其 allclose 交叉核对即可 |
| npz 读取 | 新读取逻辑 | 既有 `load_corners`(allow_pickle=False) | 安全防线与测试已在 Phase 1 锁定 |
| 畸变逆解 | 现场推导 | 移植参考 unproject 牛顿迭代(~20 行,往返 <1e-9px 已验证) | 数值反解是手写数学里最容易出错的;这份已经过 parity 锁定 |
| 数值雅可比测试基准 | 新增 numdifftools 依赖 | `scipy.optimize._numdiff.approx_derivative` | scipy 自带;参考工程加 numdifftools 是它四模型研究项目的需要,本工程零新依赖达成同一验证目的 |
| 报告渲染 | 引入模板引擎/matplotlib | f-string markdown + csv 标准库 | INTR-04 是表格类报告;零依赖,Phase 1 CSV 模式直接延续 |

**Key insight:** 本阶段每个"看起来要写数学"的环节,参考工程都有一份对齐到机器精度、经真实数据验证的实现。移植 + 不变量测试锁定(Phase 1 已验证的工作模式)比任何"重新推导"都更快更稳。

---

## Common Pitfalls

### Pitfall 1: OpenCV 绑定层 dtype 陷阱
**What goes wrong:** `calibrateCameraExtended` 拒收 float64(报 "Unsupported format");直接把 npz 的 float64 数组传进去会当场失败。
**Why it happens:** OpenCV Python 绑定的 Point2f/Point3f 只收 32 位浮点。
**How to avoid:** 调用前显式 `.astype(np.float32)`;返回值全部转回 float64;像素 float32 表达精度 ~1e-4px 远小于 0.5px 线,可接受。
**Warning signs:** "Unsupported format" 报错;或 parity 差 ~1e-4px 而非 ~1e-13px。

### Pitfall 2: 图像分辨率 (宽,高) 与 numpy shape (高,宽) 写反
**What goes wrong:** image_size 传错 → calibrateCameraExtended 主点初值/归一化全错,RMS 高得离谱。
**Why it happens:** numpy `img.shape` 是 (h, w),OpenCV 要 (w, h)(参考工程阶段一陷阱 8 原话:"直觉写 numpy 的 shape 会把宽高写反")。
**How to avoid:** 统一 `height, width = img.shape[:2]` 后传 `(width, height)`;npz/meta 不含分辨率,需 imread 一帧检出图像派生(见 Open Questions 3)。
**Warning signs:** RMS 在百像素量级;主点恰为 (720, 1280) 这类转置值。

### Pitfall 3: 两把尺(RMS 口径)混用
**What goes wrong:** 报告或判停逻辑一边用 OpenCV 返回 RMS 一边用回代 RMS,~1e-4px 口径差污染比较;直接违反成功标准 3。
**How to avoid:** 全部指标(逐帧/整体/train/test)只经 compute_result_metrics 一条路径;OpenCV rms 只进 solver_meta["opencv_rms"];源码防线测试锁定。
**Warning signs:** review 时发现 report 里出现 opencv_rms 字样进了指标字段。

### Pitfall 4: 残差/雅可比行序错位(静默失败)
**What goes wrong:** 残差按 (视图,点,(u,v) 交错) 排、雅可比按别的排——least_squares 拿到错误梯度,**不报任何异常,只是收敛奇慢或原地不动**。
**Why it happens:** 两侧各自为政拼装下标。
**How to avoid:** 移植参考 residuals.py 的 row_index 函数,残差与雅可比共用;测试用数值差分对解析雅可比逐项校验(approx_derivative 基准)。
**Warning signs:** 精化迭代次数暴涨、RMS 停在初值水平。

### Pitfall 5: 雅可比/残差组装的逐点 Python 循环
**What goes wrong:** 逐点 append 式组装,慢一个数量级以上。
**How to avoid:** 按视图循环、视图内全 numpy;雅可比用 COO 三元组向量化拼接(参考实测 0.15s/次 @ 640 视图)。
**Warning signs:** 单次残差评估 > 0.1s(本规模应 < 10ms)。

### Pitfall 6: test 帧评分的错误盆地
**What goes wrong:** 固定内参后直接拿像素坐标按标定 K/D 解 PnP 当初值,大倾角 + 强畸变 + 棋盘偏心时落入错误盆地,精化回不来,单帧 RMS 虚高几十倍。
**Why it happens:** 初值本身带畸变偏差(参考工程真实案例:40° 倾角帧 4.82px vs 正确 0.07px,"四模型同坑")。
**How to avoid:** 必须 unproject 反投影初值路线(Pattern 5);测试加大倾角合成帧回归用例。
**Warning signs:** 个别 test 帧 RMS 孤高(> 1px)而相邻帧正常。

### Pitfall 7: 旋转导数手写错误
**What goes wrong:** ∂Xc/∂ω 写成 −[Xc]×(反对称),一般转角下系统性错误(0.1 rad 偏 70%),雅可比测试炸或不收敛。
**How to avoid:** 用 `cv2.Rodrigues(r)` 第二返回值(rot_jac,布局 rot_jac[k,3i+j]=∂R_ij/∂ω_k),einsum 一次算完 ∂Xc_i/∂ω_k = Σ_j rot_jac[k,i,j]·Xb_j(移植 base.py L185-191 修复后版本)。
**Warning signs:** 解析雅可比对数值差分的偏差 > 1e-6。

### Pitfall 8: 离群剔除循环的反馈漂移与视图耗尽
**What goes wrong:** 剔除改变中位数 → 阈值漂移 → 循环不收敛;或剔除过多 → 视图数逼近 calibrateCamera 下限(OpenCV 要求足够多视图,惯例 ≥10)→ 求解失败或病态。
**How to avoid:** max_iterations 与 max_drop_fraction 双保险丝(配置控制);剔除只作用于 train;每轮剔除清单落盘;幸存视图数下限断言(建议 ≥10,远高于参考的 ≥2,本工程 141/155 帧裕量巨大)。
**Warning signs:** 循环轮数打满 max_iterations;剔除清单超过 20% 帧(说明阈值或数据有问题,如实报告而非静默)。

### Pitfall 9: 装载层 NaN 泄漏进求解器
**What goes wrong:** 未检出帧整行 NaN,若过滤遗漏(如只看 detected 不查 finite),NaN 毒化整个求解且报错难懂。
**How to avoid:** 移植参考 observation.py 三重防线:detected=False 跳过 → 有限值断言(检出但含 NaN/Inf 直接中文报错)→ 幸存视图数下限;npz 的 uv (N,88,2) 形状断言对 board 规格。
**Warning signs:** scipy 报 LinAlgError 或结果全 NaN。

### Pitfall 10: 行序漂移(逐帧表对不上)
**What goes wrong:** 装载过滤后视图顺序与 image_ids 脱钩,per_view_rms 表错行——不报错、悄悄算错(参考 Pitfall 10 原话)。
**How to avoid:** image_ids 元组是一等字段,装载时定死行序;结果/残差/报告全部按该元组顺序;落盘 npz 携带 image_ids 列供对账。
**Warning signs:** 逐帧 CSV 的 image_id 与 npz 原始顺序不一致。

### Pitfall 11: objp 契约漂移
**What goes wrong:** config.yaml 的 board 规格在 Phase 1 运行后被改过,npz 内 objp 与新配置重建的网格不一致 → 角点与物点错位,RMS 虚高且难排查。
**How to avoid:** 装载时 `build_object_points(cfg.board)` 与 npz["objp"] 做 `np.allclose` 断言,不一致中文报错提示重跑 preprocess。
**Warning signs:** 报错信息里 objp 不一致;训练 RMS 反常地高。

### Pitfall 12: 剔除判据用了 OpenCV float32 逐帧表
**What goes wrong:** calibrateCameraExtended 第 8 个返回值 per_view 是 float32 口径,剔除决策与最终报告口径不一致。
**How to avoid:** 剔除决策与报告一律用 float64 回代逐帧 RMS(Pattern 4);OpenCV per-view 不读(下划线接收)。
**Warning signs:** 代码 review 发现 per_view 返回值被消费。

---

## Code Examples

### 合成真值恢复测试的数据合成(不需要渲染图像!)

Phase 2 消费的是角点数组,不是图像——测试直接用 cv2.projectPoints 合成 (V, 88, 2) 角点,按 Phase 1 契约写 npz,跑完整 pipeline 断言参数恢复:

```python
# 测试策略(参考工程 test_solve_solvers.py 模式 + 本工程契约)
rng = np.random.default_rng(42)
K_true = np.array([[1850.0, 0, 1280.0], [0, 1840.0, 720.0], [0, 0, 1.0]])
dist_true = np.array([-0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])  # M2 布局
views = []
for _ in range(12):  # 12 个随机位姿(参考工程合成测试的量级)
    rvec = rng.normal(0, 0.3, 3); tvec = np.array([rng.normal(0, 50), rng.normal(0, 50), rng.uniform(400, 900)])
    proj, _ = cv2.projectPoints(objp.reshape(-1, 1, 3), rvec, tvec, K_true, dist_true)
    views.append(proj.reshape(-1, 2) + rng.normal(0, 0.05, (88, 2)))  # 0.05px 检测噪声
# 写成 Phase 1 契约 npz → run_intrinsic → 断言 |fx_est - fx_true|/fx_true < 1e-3 量级
```

既有 `conftest.py make_chessboard_image`(带真值角点的渲染夹具)留给"角点级"测试;标定级测试用上面的直接合成,秒级完成。

### 离群剔除循环(编排层伪代码)

```python
# intrinsic/pipeline.py 编排(设计示意)
survived = list(range(n_views))
dropped = []
for round_i in range(1, cfg.outlier.max_iterations + 1):
    result = solve_opencv(views[survived], image_size)         # flags=0, float32 进
    params, rvecs, tvecs = unpack(result)
    per_rms = float64_backsub_rms(views[survived], params, rvecs, tvecs)  # 唯一判据尺
    med = np.median(per_rms)
    bad = np.where(per_rms > cfg.outlier.rms_multiplier * med)[0]
    if bad.size == 0:
        break
    if len(survived) - bad.size < max(min_views, (1 - cfg.outlier.max_drop_fraction) * n_views):
        break  # 保险丝:保留视图数不足,停止剔除并如实记录
    for i in bad:
        dropped.append({"image_id": ids[survived[i]], "rms": per_rms[i],
                        "round": round_i, "threshold": cfg.outlier.rms_multiplier * med})
        survived.remove(survived[i])
# 幸存集定稿 → cfg.refine.enabled 时 scipy 精化一次 → test 评分 → 落盘与报告
```

### 中文报错收口(CLI,既有模式)

```python
# cli/main.py intrinsic 子命令(照 preprocess 体例)
@app.command(help="模块2 单相机内参标定:...")
def intrinsic() -> None:
    try:
        from stereo_calib.intrinsic.pipeline import run_intrinsic
        run_intrinsic(load_config(_config_path))
    except (ValueError, OSError) as e:
        typer.echo(f"内参标定失败:{e}")
        typer.echo("  请检查 config.yaml 配置与 Phase 1 产物(output/ 下的角点 npz),修正后重跑。")
        raise typer.Exit(code=1) from e
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| OpenCV 5.0(已发布) | 本工程锁 4.14(<5.0) | 5.0.0 于 2026-08-19 发布 | 无影响;pyproject 已锁 <5.0,calib3d 行为稳定性优先(Phase 1 决策延续) |
| scipy method="lm" 心智 | trf + lsmr + jac_sparsity | 长期(lm 只支持稠密) | 配置白名单拒收 "lm"(参考工程防呆照搬) |
| cv2.stereoCalibrate | (Phase 3)PnP-median | 本项目决策 | 与本阶段无关,但内参输出格式要服务该路线(K/D 按名给全) |
| 手工审阅剔除帧(参考工程 exclude_images) | 配置阈值自动剔除 + 留痕 | 本阶段 OVERNIGHT AUTONOMY 授权 | 两机制叠加,人工清单保留 |

**Deprecated/outdated:**
- "OpenCV 返回 RMS 可直接用":float32 绑定口径,成功标准 3 明确禁止。
- "标定单次秒级":参考实测 640 视图 ~107s;本工程 141/155 视图预估 ~25-35s/路,仍非秒级,排计划按分钟级总预算。

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | 内参预期锚点(fx≈1856/fy≈1846/cx≈1270/cy≈724/k1≈-0.406/k2≈0.232/k3≈-0.082)适用于本数据(同类相机、同为 2560×1440) | Summary | 仅作 sanity 参考不作门禁;若实际差异大,提示数据或装载有病,排查方向不变 |
| A2 | 全部 406 张图像同分辨率 2560×1440(仅实测首帧) | Pitfall 2 | 若混分辨率,calibrateCameraExtended 单 image_size 语义被破坏;装载层可加抽查断言兜底(见 Open Questions 4) |
| A3 | 离群阈值默认值(multiplier=5.0 / max_drop_fraction=0.2 / max_iterations=3)对本数据合理 | Pattern 6 | 阈值进配置,错了改配置重跑即可(overnight 授权场景下报告留痕保底);真实分布出来后按参考工程"迭代回环校准"惯例调 |
| A4 | scipy/OpenCV 单线程(cv2_threads=0)下确定性可复现 | Standard Stack | 若不确定,复现断言改为 1e-9 容差(参考工程 A4 同款兜底) |
| A5 | 混合 sb/classic 角点进标定可接受(Phase 1 D-01"检出即可用";参考 D-03 量化两类角点系统差 ~0.135px,评估明确 out of scope) | Summary | 若用户后续要求分开评估,报告的 source 列已留痕,重跑按配置过滤即可,无结构返工 |
| A6 | scipy 精化 loss 固定 "linear"(离群由剔除机制处理)满足需求 | Pattern 3 / Open Questions 2 | 若 test RMS 不达标需要鲁棒核对照,配置化扩展是局部改动(白名单 + f_scale auto 移植),非结构返工 |

---

## Open Questions (RESOLVED)

> 全部问题已在规划阶段裁决:OQ1 → 中位数乘数(rms_multiplier=5.0,02-03 落地);OQ2 → loss 固定 "linear" 写死(02-02 落地);OQ3 → imread 首帧派生 + 首/中/末 3 帧抽查(02-01 落地);OQ4 → 单份 report.md + 每路 per_frame CSV(02-03 落地);OQ5 → 属 Phase 3 范围,Phase 3 规划时修正 STATE 真值数字。

1. **离群剔除判据口径:中位数乘数 vs MAD 稳健 z 分数**
   - What we know: 参考工程两套惯例并存——存疑帧判定用 MAD z ≥ 3.5(ingest 段),人工剔除用 exclude_images 清单;本工程需要自动机制。
   - What's unclear: multiplier(中位数倍数)与 MAD z 在本数据分布上的行为差异。
   - Recommendation: 实现中位数乘数(更直白,报告可写"RMS 超中位数 5 倍"),阈值/保险丝全进配置;真实分布出来后按 A3 惯例调。planner 定一个即可,不both实现。

2. **scipy loss 是否配置化**
   - What we know: 参考工程 losses: [linear, soft_l1] 双核对照是它的研究使命;本工程是生产工具,离群已由剔除机制处理。
   - What's unclear: 无——这是取舍不是未知。
   - Recommendation: loss 固定 "linear" 写死在代码(文档注明),精化开关/容差/max_nfev 进配置;若未来需要鲁棒核,按 A6 扩展。

3. **image_size 派生方式**
   - What we know: npz 与 meta.json 都不含分辨率;imread 单帧 ~30ms。
   - What's unclear: 是否需要防御"数据里混入不同分辨率帧"。
   - Recommendation: imread 第一张检出帧派生 (w,h);可选对首/中/末 3 帧抽查断言一致(总开销 ~0.1s)。不建议改 Phase 1 把分辨率写进 meta(会触发预处理重跑,违反模块边界)。

4. **报告形态:单文件 report.md vs 每路一份**
   - What we know: INTR-04 未规定;Phase 1 报告类产物是 CSV + meta。
   - What's unclear: 用户阅读偏好。
   - Recommendation: 单份 `output/intrinsic/report.md`(两路并排对比,一目了然)+ 每路一份 per_frame CSV(Excel 抽查惯例)。MVP 文本表格,不加图。

5. **(顺带发现,Phase 3 范围)STATE.md 双目真值数字疑似算术误差**
   - What we know: STATE.md 记"真值 √(10²+7²)×25mm ≈ 307.45mm";按 PROJECT.md 公式 sqrt(100+49)×25 = 305.16mm,两处不一致。
   - What's unclear: 以哪个为准(公式给出的 305.16mm 是算术确定的)。
   - Recommendation: 不影响本阶段;Phase 3 规划时按公式重算并修正 STATE(此处仅留痕,避免错误真值流入 Phase 3 指标)。

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python | 全部 | ✓ | 3.12.13(uv venv) | — |
| opencv-python | calibrateCameraExtended / solvePnP / Rodrigues / projectPoints | ✓ | 4.14.0(锁 <5.0) | — |
| numpy | 全部数组数学 / npz | ✓ | 2.5.4 | — |
| scipy | least_squares / sparse | ✓ | 1.18.1 | — |
| pytest / ruff | dev | ✓ | >=8 / >=0.8(既有) | — |
| Phase 1 产物 npz | 装载输入 | ✓ | output/{train,test}/cam{n}_{split}.npz 实测在位(cam1 train 141 检出 / test 15/15;objp[87]=[250,175,0]) | 缺失时中文报错提示先跑 preprocess |
| 真实数据 data/ | image_size 派生(imread 1 帧) | ✓ | 2560×1440 实测 | — |
| CPU | OpenCV 求解 / scipy 精化 | ✓ | 20 核(cv2_threads=0 单线程确定性) | — |

**Missing dependencies with no fallback:** 无。
**Missing dependencies with fallback:** 无(零新增依赖)。
**已知环境注意事项(Phase 1 遗留):** worktree 检出无 data/(gitignored),真实数据冒烟需 symlink 主仓 data/(Phase 1 既有做法);.planning 默认 commit_docs=true。

## Security Domain

> security_enforcement 启用(ASVS Level 1)。离线单机数值工具,无网络面、无认证面。

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | 无用户/会话概念(离线 CLI) |
| V3 Session Management | no | 无会话 |
| V4 Access Control | no | 单机单用户;文件系统权限即边界 |
| V5 Input Validation | yes | 配置经 io/config.py frozen dataclass + 中文校验(既有,扩展 intrinsic 段时延续);yaml.safe_load;npz 读取强制 allow_pickle=False(防 pickle 任意代码执行,既有测试锁定);solver 输入形状/有限值断言(中文) |
| V6 Cryptography | no | 无 secrets;哈希仅作配置留痕指纹(非安全用途) |

### Known Threat Patterns for 本 stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| 恶意 npz(pickle 载荷) | Tampering/Elevation | load_corners 全链 allow_pickle=False(Phase 1 测试锁定,新读盘代码照抄该模式) |
| 配置注入(yaml 反序列化) | Elevation | yaml.safe_load(既有,绝不 yaml.load) |
| 产物写出 data/(数据只读击穿) | Tampering | 全部写路径从 cfg.output_root 派生 + 既有 output/data 互斥校验;本阶段新增 rmtree 目标仅 output_root/intrinsic |
| 剔除循环误删证据 | Repudiation | 剔除清单(帧号/RMS/轮次/阈值)全部落盘进报告与 JSON,可审计 |

---

## Sources

### Primary (HIGH confidence)

- **参考工程源码逐行阅读(2026-10-23,只读)**:
  - `/home/lx/Works/src/lx/github/intrinsic-calibration/src/calib/solve/opencv.py` — calibrateCameraExtended 调用形态、float32 装包、flags 语义、失败契约
  - `.../src/calib/solve/scipy_solver.py` — θ 装箱、trf+lsmr 配置、实用收敛判定(纯函数)、f_scale auto
  - `.../src/calib/solve/residuals.py` — row_index 行序契约、解析稀疏雅可比组装、COO→csr
  - `.../src/calib/solve/result.py` — compute_result_metrics(float64 回代)、CalibrationResult、三件套落盘
  - `.../src/calib/solve/observation.py` — 装载过滤/有限值断言/行序锁/≥2 视图保护
  - `.../src/calib/solve/scoring.py` — solve_pose_only(unproject 反投影初值 + 错误盆地真实案例)、test 帧评分
  - `.../src/calib/models/pinhole_brown.py` + `base.py` — M2 投影/解析雅可比/牛顿反投影/transform_to_camera(Rodrigues 雅可比修复版)
  - `.../src/calib/solve/baseline.py` — build_object_points、针孔基线(per_frame 表口径)
  - `.../config/calibration.yaml` + `io/config.py` — scipy 段配置键、白名单防呆(lm 拒收)、exclude_images 模式
- **参考工程 `.planning/phases/02-m1-m4/02-RESEARCH.md`** — 12 条实测陷阱(dtype/行序/鲁棒核假阴性/耗时/稠密内存)、flags↔D 布局实测表、scipy 收敛实测(linear 15 迭代 13s @640 视图)
- **参考工程 `docs/findings.md`** — 同类相机实测锚点(m2: train 0.0919px / test 0.1598px;内参逐项数值;k2/k3 晃动量级;解不唯一性)
- **本机实测(2026-10-23)**:`uv run python` 验证依赖版本;npz 契约逐键验证(八数组/uv shape (188,88,2)/objp[87]/source 取值);首帧分辨率 2560×1440;output/ 产物在位

### Secondary (MEDIUM confidence)

- 参考工程 `src/calib/report/build.py`(报告结构参考,本工程 MVP 大幅裁剪)
- 参考工程 OpenCV/scipy 公式与 docs.opencv.org/4.x、docs.scipy.org 一致的声明(其 02-RESEARCH 已做 parity 实测互证)

### Tertiary (LOW confidence)

- 无(全部结论有参考工程实测或本机验证支撑)

---

## Metadata

**Confidence breakdown:**
- 标准栈与数学实现: HIGH — 逐行移植已验证代码,同版本环境,parity/雅可比/真值恢复三层测试锁定
- 架构与输出契约: HIGH — 沿用 Phase 1 已验证的模块/配置/产物模式,接口双向对账(输入 npz 实测、输出 JSON 为 Phase 3 定义的新契约)
- 离群剔除机制: MEDIUM — 机制设计来自已验证模式,但默认阈值(A3)与真实分布行为需执行期观察;配置可调 + 留痕兜底
- 性能预估: MEDIUM — 按参考 640 视图实测线性折算,量级可信(±2×),不影响正确性

**Research date:** 2026-10-23
**Valid until:** 2026-11-22(依赖锁定栈稳定;公式类结论长期有效;若 OpenCV/numpy/scipy 升级需复核 Pitfall 1/5)
