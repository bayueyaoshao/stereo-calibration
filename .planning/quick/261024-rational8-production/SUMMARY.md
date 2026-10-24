---
type: quick
status: completed
priority: high
created: 2026-10-24
completed: 2026-10-24
source: 用户指令「双目测量误差是参考工程2的10倍,再想办法优化」;归因 = 唯一优化路径是产线支持 rational8 畸变模型
plan: 261024-rational8-production
---

# Quick Task Summary: 产线 M8 rational 畸变模型 + Phase 2 重标 + Phase 3 重跑

**一句话(最终,含追加轮):** M8(rational 8)畸变模型全链落地为产线一等公民,首轮 cam1 转绿、cam2 honest-stop;**追加轮定位 cam2 根因为 fx-D 简并谷**(scipy 精化的第二落盆,爆系数近相消、标量 RMS 与好盆相当),以「精化后 M8 洗盆」修复——**两相机体检 reliable 全绿**(红检 0 项,cam1 fx=2037.04/cam2 fx=2037.52,均落参考锚点 2037±1% 带),**双目重跑 SC1-SC4 全过**(test 距离 RMS 0.2001mm,为 5mm 阈值的 1/25;baseline 798.11mm;剔除 0 对、翻转 0),全链验收通过,退出码 0。273 测试全绿(270 基线 + 3 新增),阈值/闸门零改动。

## 任务完成度

| Task | 内容 | 状态 |
|---|---|---|
| 1 | models.py 双模型化(M8 布局+导数+牛顿逆)+ parity/雅可比/往返测试 | 完成(`6d570ac`) |
| 2 | solve.py + config.py + config.yaml(distortion_model 选项、flags、D 解包) | 完成(`fc2d7f7`) |
| 3 | artifacts/observation/pipeline/joint 契约适配 + diagnostics M8 适配 | 完成(`63023fc`) |
| 4 | 合成端到端 M8 求解测试 + M2 回归全绿 | 完成(`63023fc`,270 passed) |
| 5 | 生产重跑 intrinsic(M8) | 完成(首轮 cam1 reliable/cam2 honest-stop;**追加轮 Task 10 转全绿**,见下) |
| 6 | 生产重跑 stereo | 完成(**追加轮 Task 11 执行**,SC1-4 全过,见下) |
| 7 | REQUIREMENTS/deferred-items/SUMMARY 登记 | 完成(本文件) |
| 8 | solve.py M8 两遍拟合防退化谷(洗盆)+ solver_meta 留痕 + 合成简并谷测试 | 完成(`342f1d2` 初版 + `575d6cf` 位置修正) |
| 9 | diagnostics A3 切片重标 M8 冻结协议(D 8 长时冻结全部畸变系数) | 完成(`8266e3e`) |
| 10 | 生产重跑 intrinsic(追加轮)——两机体检 reliable 全绿、fx 落锚点带、退出码 0 | 完成(`575d6cf` 后重跑) |
| 11 | 生产重跑 stereo(追加轮)——闸门零改动,SC1-4 验收 | 完成(test 距离 RMS 0.2001mm,剔除 0 对) |
| 12 | SUMMARY/REQUIREMENTS/deferred-items/STATE/ROADMAP 追加轮登记 | 完成(本文件) |

## 交付内容(提交按序)

| 提交 | 内容 |
|---|---|
| `6d570ac` | `models.py` 双模型化:`INTRINSIC_NAMES_M8`(12 参,D 长 8 = [k1,k2,p1,p2,k3,k4,k5,k6],num 用下标 0/1/4、den 用 5/6/7)、`intrinsic_names(model)` 唯一出口、`_as_params` 按 len 分派(9→M2/12→M8)、`_radial_and_derivs` 返回 ratio/d_ratio_dr2(den≡1 时数值退化为 M2,单一公式体纪律不动)、den<1e-6 中文防呆;测试:M2≡M8 逐位一致、cv2.projectPoints parity <1e-9px、牛顿往返 <1e-9px |
| `fc2d7f7` | `solve.py`:`model: str = "M2"` 全链透传,M8 时 flags=`cv2.CALIB_RATIONAL_MODEL`(实测 flags_used=16384)、D 解包 ravel 后断言 len≥8 且余位全零取前 8(OpenCV 4.14 实测返回 14 长)、pack/unpack_theta 9/12 自动跟随、稀疏雅可比结构按参数长度;`io/config.py`:`distortion_model` 字段(默认 "M2")+ `_ALLOWED_DISTORTION_MODELS` 中文守卫;config.yaml 加开关(生产值即 M8,注释写明 M2 legacy);测试:合成 M8 端到端 fx 误差 <0.5%、雅可比逐列 <1e-6、θ 往返、config 非法值四态 |
| `63023fc` | `artifacts.py`:intrinsics.json 新增顶层键 `distortion_model`(只增不改)、params 按模型取名、report 模型名入头部与体检节;`stereo/observation.py` load_intrinsics 读键缺省 "M2"(旧文件向后兼容,有测试)、未知模型中文拒绝;`diagnostics.py`:D shape (5,)/(8,) 双收、A3/A4 重标 flags 与 guess 长度按模型对齐、A4 在 M8 下语义自动变为「同阶 rational8 独立重标一致性探针」(note/detail 如实标注)、`_jsonify_value` 字符串直通修复;`stereo/joint.py`:联合精化入口对 M8 fail-closed 中文拒收(联合 BA 是 M2 时代失败方案的坑,M8 内参语义未验证前不放行);scoring/metrics docstring 12 参口径;五个测试文件契约/边界补齐 |
| `1d52d86` | config.yaml `exclude_images` 30 帧(cam1 27 + cam2 3)证据驱动剔除,含逐轮证据注释;阈值/闸门零改动 |

**质量口径:** 270 passed(244 基线 + 26 新增,存量 M2 测试零修改);本任务全部新建/改动文件 ruff 干净(存量 5 处 RUF007/C408 为 Phase 1 遗留,Scope Boundary 不修;`.codebuddy/` 为用户自有未跟踪目录,非本任务产物);data/ 指纹 `8a430c9395e91f256923b2a6eabb4354` 提交前后一致;参考工程零接触;output/ 未提交。

## 生产验收数值(Task 5)

### cam1 — 验收通过侧

| 指标 | M2 基线(2026-10-24 晨) | M8 定稿(本任务) |
|---|---|---|
| train 幸存帧 | 141(无剔除) | 114(手动剔 27) |
| fx / fy | 2265(**虚高 +11%**) | **2042.89 / 2030.59** |
| cx / cy | — | 1265.93 / 759.48 |
| D8 | (5 参) | [-0.2507, 8.0761, 0.0009, -0.0006, 2.9680, 0.4920, 7.5037, 8.6672] |
| test RMS(px) | 0.1407 | **0.0652**(2.15× 改善,与预期 ~0.075 吻合) |
| A0 残差能量比 | 1.4730 | 1.4318(通过) |
| A1 残差径向结构 | 0.0220 红 | **0.0121**(通过) |
| A3 切片 fx 稳定性 | 3.6567 红 | **0.1462**(通过;切片 fx 2034.6–2038.0 全落锚点邻域) |
| A4 探针 fx 跳变 | 11.2404 红 | **0.2872**(通过;产线 2042.89 vs 独立重标 2037.04) |
| verdict | 不可靠(A1/A3/A4) | **可靠(红灯 0 项)** |

fx=2042.89 距 PLAN 验收锚点(参考工程 rational8 重拟合 cam1 fx≈2037.4)偏差 **+0.27%,在 ±1% 带内 → PASS**。

### cam2 — honest-stop 侧

| 指标 | M2 基线 | M8 R0(0 剔) | M8 R1(剔 1 帧) | M8 R2(剔 3 帧,定稿) |
|---|---|---|---|---|
| train 幸存帧 | 155 | 155 | 154 | 152 |
| fx / fy | — | 2036.17 / 2025.18 | 2035.04 / 2024.12 | **2034.88 / 2024.09** |
| D8 | (5 参) | [2.0399, 84.79, 0.0000, -0.0002, 27.34, 2.72, 87.04, 81.29] | [1.8101, 77.85, …] | [1.7754, 74.5826, 0.0000, -0.0002, 23.9136, 2.4600, 76.5223, 71.2976] |
| train RMS(px) | — | 0.1384 | 0.1077 | 0.0948 |
| test RMS(px) | 0.1361 | 0.0760 | 0.0761 | **0.0760**(1.79× 改善) |
| A1(阈值 0.02) | 0.0772 红 | 0.0284 红 | 0.0283 红 | **0.0280 红** |
| A3(阈值 0.5%) | 1.2795 红 | 0.9849 红 | 0.9558 红 | **0.9392 红** |
| A0 / A2 / A4 | 1.0578 / 0.1835 / 0.7395 | 全过 | 全过 | 1.2024 / 0.1608 / **0.1292**(fx 探针 2037.52) |
| verdict | 不可靠(A1/A3) | 不可靠 | 不可靠 | **不可靠(A1/A3)→ exit 1** |

cam2 定稿 A3 切片:左半 2040.0 / 右半 2034.3 / 上半 2031.3 / 下半 2038.0 / 内圈 2038.6 / **外圈 2020.9**;A1 分桶 [0.0280, -0.0115, 0.0084, -0.0059, 0.0061](中心桶唯一越阈)。

### 剔除留痕(机制口径)

- **cam1 27 帧(R1)**:M8 首轮逐帧精化 RMS 2.9–7.2px(中位 0.09px),占 train 141 帧的 19.1%(守 20% 保险丝)。这批帧是 cam1 fx 被拖至 2088.89(偏离 cv2 全切片一致锚点 ~2037)的直接原因:scipy 联合精化沿 rational 近简并谷底方向走,坏帧在几乎不伤 RMS 的情况下搬走 fx;剔除后立即回到锚点(独立重标探针 2037.04),自动 5×中位循环随后零剔除。
- **cam2 3 帧(R1 剔 1.08px 一帧;R2 剔 0.460/0.447px 两帧,第三名仅 0.220 自然断点)**:train RMS 0.1384→0.0948px 持续改善,但 A1/A3 红灯对剔除**不敏感**(轨迹见上表)——中心桶 0.028px 量级的径向结构残差与外圈切片 fx 低压(2020.9 vs 内圈 2038.6)是**数据/光学层面的结构性效应**,不是离群帧。M8 已把两项较 M2 改善 2.8×/1.4×,但未过严格阈值。
- test 集 15 帧与剔除清单零重叠(逐项核对),评分集全程不动;两轮为自限上限,未做第三轮。

### Task 6(双目重跑)首轮未执行的理由

体检 verdict=不可靠时管线自身 fail-closed:产物全部落盘留痕后中文报错退出码 1,报告明示「禁止消费该内参进入双目标定」。在此契约下运行 stereo-calib stereo 等于带病消费,且 PLAN 验收前置条件(两路 reliable)未满足。SC1–4 数值首轮**无新数据**(旧值 SC3=16.93mm 系 M2 内参产物,已失效,不具可比性)——追加轮已补上,见下节。

## 追加轮(Task 8-12)——cam2 归因、洗盆修复、双目全过

### 追加轮交付(提交按序)

| 提交 | 内容 |
|---|---|
| `342f1d2` | `solve.py` M8 两遍拟合:`_solve_opencv_m8_pass2`(pass1 fresh → pass2 `RAT|USE_INTRINSIC_GUESS` 自再拟合)、`_unpack_opencv_solution` 单点解包/断言、solver_meta 记 `opencv_rms_pass1/pass2` 双口径、M2 路径零改动;合成简并谷落盆测试(爆炸谷系数 guess → 两遍落好盆) |
| `8266e3e` | `diagnostics.py` A3 切片重标 M8 冻结协议:生产 D 长 8 时切片重标追加 `FIX_K1..K6|FIX_TANGENT_DIST`(fx 成为唯一自由 field 信号),M2 逐位不变;A0/A1/A2/A4 零触碰;冻结协议单元测试(monkeypatch 捕 flags:M8 带 freeze bits、A4 不带、M2 bitwise==GUESS) |
| `575d6cf` | **洗盆位置修正**:真实数据探针证明 fresh cv2 初值在本数据上是远谷垃圾(fx 5892/8908),pass2 从它出发洗不动;可洗输入是**精化后**的谷解(2034.88+爆系数)。洗盆移入 `solve_calibration` 编排位(scipy 精化之后、仅 M8),`solve_opencv` 回退单遍;新 solver 标签 `opencv+wash`,meta 链 `pre_wash_solver/pre_wash_fx/pre_wash_meta` 全留痕;e2e 测试断言机械同步 |

**追加轮质量口径:** 273 passed(270 基线 + 3 新增,存量 M2 测试零修改);改动文件 ruff 干净(5 处 Phase 1 遗留不修,data/ 指纹不变,参考工程零接触,output/ 未提交)。

### Task 10 生产重跑 intrinsic(追加轮)——两机全绿

| 指标 | cam1 | cam2 |
|---|---|---|
| train 幸存帧 | 114(手动剔 27,本轮自动剔 0) | 152(手动剔 3,本轮自动剔 0) |
| **fx / fy** | **2037.04 / 2026.66** | **2037.52 / 2026.54** |
| cx / cy | 1263.74 / 766.83 | 1268.03 / 732.65 |
| D8 | [5.4135, 3.6938, 0.0, -0.0, 0.2295, 6.206, 6.9342, 1.3606] | [11.4917, 19.3973, 0.0001, 0.0, 2.7679, 12.2905, 27.3293, 11.3491] |
| solver | opencv+wash(pre_wash_fx=2042.89) | opencv+wash(pre_wash_fx=2034.88) |
| train / test RMS(px) | 0.0694 / 0.0618 | 0.0844 / 0.0686 |
| A0 残差能量比 | 1.4646 通过 | 1.7266 通过 |
| A1 残差径向结构(阈 0.02) | **0.0005** 通过 | **0.0011** 通过 |
| A2 去畸变直线度 | 0.1335 通过 | 0.1617 通过 |
| A3 切片 fx 稳定性(阈 0.5%,冻结协议) | **0.1301** 通过 | **0.1974** 通过 |
| A4 同阶独立重标(阈 2%) | 0.0000 通过 | 0.0026 通过 |
| **verdict** | **可靠(红灯 0 项)** | **可靠(红灯 0 项)** |

- fx 落锚点:cam1 +0.002%、cam2 +0.025%(参考锚点 2037,±1% 带)→ PASS。cam2 A1=0.0011 与主会话洗盆预测(~0.001)吻合。
- 首轮 cam2 红 A1=0.0280/A3=0.9392 在洗盆后熄灭,证实首轮红灯是**求解落盆问题**而非数据层结构性残差(首轮归因结论部分修正:数据无恙,错在 solver 谷底)。
- **循环评分膨胀观察(如实记录,未改阈值)**:剔除循环仍走单遍 fresh cv2 初值评分,膨胀现象**依旧**(cam1 中位 1.4332px→阈值 7.17px、cam2 1.8579px→9.29px);本轮两机自动剔除 0 帧,零实际影响,登记不变(deferred-items)。

### Task 11 生产重跑 stereo(追加轮)——SC1-SC4 全过

| 闸门 | 结果 | 判定 |
|---|---|---|
| SC1 外参精化收敛 | refine 开,solver=scipy,status=2,nfev=15,converged=true | PASS |
| SC2 位姿一致性 | 中位偏移闸门(1.0°)未触发;180° 翻转 train 0 对 / test 0 对 | PASS |
| SC3 板距验证(阈 5mm) | 距离 RMS train **0.2321mm** / test **0.2001mm**(真值 305.1639mm) | **PASS(阈值的 1/25)** |
| SC4 baseline 合理带 | **798.1140mm**(‖t‖=[798.03, -9.25, 7.44]mm,带 [20,2000]) | PASS |

- 对数:train 112/112 幸存(自动剔除 **0 对**,exclude_pairs 空,未动用)、test 15;重投影 RMS train 0.0985 / test 0.0681px;退出码 0。
- R(旋转角 1.8504°)与 t 语义:X_cam2 = R @ X_cam1 + t,cam1 为参考系;产物 `output/stereo/extrinsics_stereo.json` + `report.md`(未提交,output/ 留盘)。
- 0.2001mm 优于参考工程 2 同协议 0.23mm,且为「双目测量误差是参考工程2的10倍」这一任务源头的最终回收。

## 已知局限与遗留

1. **cam2 结构性残差(~~核心遗留~~ 追加轮已销账)**:首轮「数据层结构性效应」归因经追加轮真实数据探针**部分修正**——红灯真因是 scipy 精化在 rational 模型下的 fx-D 简并谷落盆(cam2 pre_wash fx=2034.88 + 爆系数近相消),洗盆后 A1=0.0011/A3=0.1974 全绿且 fx 钉锚点,数据无恙。原登记与销账记录见 deferred-items(03-3)。
2. **自动离群循环在 M8 下的语义保真度(已知限制,未改)**:M8 的 OpenCV 初值逐帧评分中位 ~1.4–1.8px(M2 ~0.6px),5×中位阈值随之膨胀到 7–9px,坏帧若不超过该宽阈值则自动循环漏剔(cam1 首轮自动循环剔 26 帧,>1px 的 27 帧里最轻的 1 帧漏网,靠手动清单兜住)。**追加轮观察:现象依旧、本轮零实际影响(两机自动剔 0 帧)**。修法(精化后逐帧 RMS 驱动循环)属管线行为变更,按验收纪律不在本任务中途改,已记 deferred-items。
3. **D8 逐系数不可单独解读**:rational num/den 存在近简并方向(本数据 r² 覆盖 ≤0.364),产线解落在大系数谷底但 fx 被强钉死——追加轮正是利用这一点洗盆(guess 只需「近解」,fx 语义由精化守恒)。消费侧只应使用 fx/fy/cx/cy + D8 整体映射,不应逐系数单独解读。
4. **stereo/joint 拒收 M8**:联合 BA 是 M2 时代失败方案(03-05 裁决),M8 内参语义下的联合精化未验证,joint.py 入口 fail-closed 中文拒收;若未来要启用,需专门计划验证。

## Deviations from Plan

1. **[Rule 1 - Bug] `_jsonify_value` 字符串崩溃**:生产首轮全链崩在体检 JSON 化(A4 detail 新增 `probe: "same_order_rational8"` 字符串被 fall-through 到 `float(v)`)。修复:字符串直通返回。`63023fc`。
2. **PLAN 测试 #4 断言设计变更(如实记录)**:PLAN 要求合成 M8 端到端「D 各系数符号量级对」。实测(噪声自由探针 + 带噪合成)发现:σ0.05px 噪声下 D 逐系数不可稳定恢复(近简并谷底),而 fx 恒被钉死(<0.5% 误差)。断言改为 **fx 误差 <5e-3 + 回代 RMS + 契约元数据**,放弃逐系数符号断言——不是降低标准,是把断言放在物理可辨识的量上(噪声自由探针证明逐系数本身可辨识到 ~1e-4,是噪声层不可辨识)。
3. **[Task 5] 剔除协议为执行期自主决策**:PLAN 未预写 exclude_images 用法。依据:①`exclude_images` 是 Phase 2 就有的设计通道(手动逻辑剔除,原数据不动);②过夜授权「误差大可剔除」;③证据驱动(逐帧 RMS 留痕)+ ≤2 轮自限 + 20% 保险丝口径 + 阈值/闸门零改动。全过程与数值已如上表留痕。
4. **[Task 5] 验收结论反转如实上报**:PLAN 预期「两路 reliable」;实测 cam1 转绿、cam2 不转。按用户指令「不许放宽断言或阈值硬凑」选择 honest-stop,不做第三轮、不动阈值、不跳过体检。
5. **INTR-06 措辞收窄**:PLAN 验收登记措辞为「M8 下体检 reliable」;实际仅 cam1 reliable。REQUIREMENTS.md 按功能事实登记(M8 全链支持完成),生产结论(部分通过)记在 traceability 表与本文,不冒领「reliable」。
6. **测试机械同步**:`_JSON_KEYS` 增 distortion_model、`flaky_solve` 测试替身补 model 参数、体检 M8 测试用 `a3_min_frames=5`(24 合成视图低于默认 20 帧/切片会整节跳过)。均为契约跟随,非行为变更。
7. **[Rule 3,追加轮 Task 8→10] 洗盆位置偏离任务书字面措辞**:追加轮 PLAN 写「solve_opencv 做两遍拟合」;literal 实现导致生产重跑**更糟**(cam1 A3 冻结口径 0.596 红、cam2 不可靠 A1=0.0329/A3=0.8519,且 cam2 剔除循环出现第 2 轮中位 4.24px 异常)。真实数据探针归因:fresh cv2 初值在本数据是远谷垃圾(fx 5892/8908),pass2 从它出发洗不动;可洗输入是精化后的谷解。按任务书自身验证过的实质(2034.88+爆系数谷 guess 必洗入好盆)把洗盆移到 `solve_calibration` 精化之后,`solve_opencv` 回退单遍。探针证据与逐级归因数值已写入提交 `575d6cf` 信息,修正后两机全绿。判定:忠实任务书实质、偏离字面编排——如实登记。
8. **[契约跟随,追加轮] e2e 测试断言机械同步**:洗盆移位后 solver 标签变为 `opencv+wash`、`solver_meta` 结构变为 `pre_wash_meta` 链,既有 e2e 断言(首轮引入)2 行同步(`pre_wash_meta["converged"]`、solver 标签)。与偏差 #6 同类,非行为变更。
9. **[追加轮 Task 10 首跑失败如实上报]**:Task 8 literal 版本的生产重跑验收失败(cam1 1 红/cam2 2 红),未硬凑——停下探针归因→改编排位→重跑转绿,全程提交留痕(`342f1d2`→生产跑→`575d6cf`→生产跑)。
10. **[登记口径,追加轮 Task 12] REQUIREMENTS.md 同步修正**:首轮登记的 INTR-06 措辞(cam1 转绿/cam2 不可靠)与 STER-01..04 Pending 已被追加轮事实反转,一并更新为最终口径(两机全绿 + STER complete),避免文档自相矛盾;非新增需求,仅状态登记。
11. **[测试协议恢复,追加轮 Task 12] 真实数据双目引擎验收恢复 plan 原门限**:全量回归发现 `test_real_data_stereo_engine_acceptance` 翻红——该用例是 M2 时代写的「哨兵 + 诊断」双段式(断言 solve_stereo 必须被离散度防线拦下),其 docstring 与断言消息**预埋了恢复协议**:「一旦上游修复(重标内参/重采数据),本断言翻红,强迫把本用例恢复成 plan 原门限(pooled rms_reproj < 1.0px)验收并重新记录数值」。追加轮 M8+洗盆内参正是预埋协议所指的「上游修复」:防线不再触发、哨兵按设计翻红。按协议恢复用例为全链验收(solve_stereo 经真实守门直接通过 + pooled RMS < 1.0px + baseline 落带/转角/收敛断言 + 数值留档 print),阈值/闸门零改动,历史沿革完整留档于 docstring。273 测试全绿维持。

## 验收登记(第 7、12 项)

- REQUIREMENTS.md:INTR-06 增补(追加轮更新为两机全绿终态口径)+ STER-01..04 勾选并附生产实证数字 + traceability 四行 Pending→Complete。
- deferred-items.md(03-3):rational8 三条待办**全部销账完成**;cam2 结构性残差条目追加轮销账(保留原始登记供归因追溯);自动循环 M8 评分保真度条目追加观察注记(现象依旧、零实际影响)。
- STATE.md:Quick Tasks 表、Session Continuity、Blockers 清空、Metrics 更新。
- ROADMAP.md:Phase 3 行更新(双目重跑完成,test 距离 RMS 0.2001mm)。
