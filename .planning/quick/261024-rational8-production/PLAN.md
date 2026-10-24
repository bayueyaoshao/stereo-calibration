---
type: quick
status: blocked
priority: high
created: 2026-10-24
source: 用户指令「双目测量误差是参考工程2的10倍,再想办法优化」(2026-10-24);归因结论 = 唯一优化路径是产线支持 rational8 畸变模型(见 .planning/phases/03-3/deferred-items.md 登记)
---

# Quick Task: 产线支持 M8 rational 畸变模型 + Phase 2 重标 + Phase 3 重跑

## 背景(为什么)

2026-10-24 晨根因确认:本相机(8mm C-mount 4MP)M2 五参畸变欠拟合,体检闸门(quick 261024-intrinsic-adequacy-battery)已 fail-closed 拦截。本会话用真实数据完成的消融证据链(/tmp/newton4.py,OpenCV 4.14.0 实测):

- rational8 重拟合内参下,双目评测链:**逐对相对位姿口径 test RMS 0.105mm / 刚性 R/T 口径 0.40mm**(参考工程2 同协议 0.23mm);此前 M2 下 16.93mm。
- 产线评测链 evaluate.py **无 bug**:P 矩阵用各自 K、去畸变走 models.unproject 精确牛顿逆。之前 /tmp 脚本报的 2.58mm 是脚本自身 K1 混搭 P2 的 bug。
- 因此**唯一堵点是 models.py 的 M2 硬契约**(9 参、D shape (5,))。本任务把 M8(rational 8 畸变系数,12 参)做成产线一等公民,然后重标内参、重跑双目。

## 关键技术事实(已实测锁定,不要重新发明)

1. **OpenCV 4.14.0 rational 布局**(位置探针实证,/tmp/layout_probe2.py):D 向量前 8 位 = `[k1,k2,p1,p2,k3,k4,k5,k6]`,其中 num = 1+k1·r²+k2·r⁴+k3·r⁶(下标 0,1,4),den = 1+k4·r²+k5·r⁴+k6·r⁶(下标 5,6,7)。**文档里的 k4,k5,k6 是 den 项**。flags=CALIB_RATIONAL_MODEL 时 cv2 可能返回 8/12/14 长(s1..s4、τx,τy 仅在 THIN_PRISM/TILTED 标志下非零)——执行时先实证返回长度,取前 8,若 >8 断言余项全零。
2. **M8 畸变正映射**(与 cv2.projectPoints 实测 parity 1.35e-4px,/tmp/newton4.py):`xd = x·(num/den) + 2·p1·x·y + p2·(r2+2x²)`,`yd = y·(num/den) + p1·(r2+2y²) + 2·p2·x·y`——切向加在 rational 缩放**之外**,与 M2 同构(把 M2 的 rad 换成 num/den 即得)。
3. **M8 解析导数**(沿用 M2 现有结构):令 `ratio = num/den`,`d_ratio_dr2 = (num'·den − num·den')/den²`,则 2×2 雅可比公式与 M2 逐字同构(a11/a12/a22 里 rad→ratio、drad_dr2→d_ratio_dr2)。对 k 的导数列:∂u/∂k_num_j = fx·x·r^(2j)/den,∂u/∂k_den_j = −fx·x·num·r^(2j)/den²(num 槽位 j=0,1,2 ↔ r²,r⁴,r⁶;den 槽位同)。scipy 数值差分基准必须逐列锁定 <1e-6。
4. **unproject 牛顿逆**:同一 2×2 公式结构;初值 (xd,yd) 起步,收敛阈值 1e-15 不变。强 rational 畸变下 cv2 迭代逆误差可达 1.9px——绝不允许换 cv2 逆(models.unproject 的精确逆是本任务的价值所在,evaluate.py undist_pix 已在用它,零改动)。
5. **M8 参考真值**(本机数据 rational8 重拟合,cam1):K≈[[2037.4,0,1279.5],[0,2037.2,719.6],[0,0,1]],D 前 8 ≈ [2.7411,−1.3171,0.0002,−0.0000,−0.4289,3.5219,−0.0652,−1.4662]。产线重标数值应在此邻域(fx 容差 ~1%);cam2 fx 同样应落在 ~2037 邻域(M2 假值 2265 的对照)。**这是验收锚点**(memory 规则:同硬件跨工程一致性,参考工程2 同相机 fx=2036)。
6. **预期指标**:内参 test 回代 RMS ~0.075px;双目逐帧距离残差(产品口径,per-pair 相对位姿 + DLT + 固定对角)test RMS ~0.1~0.4mm ≪ SC3 闸门 5mm;相对位姿散布中位 ~0.08°。已知风险:散布 max ~1.38° 可能超 SC2 闸门 1.0°(个别离群对)——处理方式见任务 6。

## 算法/契约规格

**模型名与参数布局**(config 值,禁止其他拼写):
- `"M2"`:12 不变,9 参 `INTRINSIC_NAMES_M2 = (fx,fy,cx,cy,k1,k2,p1,p2,k3)`,D 长 5。
- `"M8"`:12 参 `INTRINSIC_NAMES_M8 = (fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6)`,D 长 8(k4,k5,k6 = den r²,r⁴,r⁶)。

**models.py 改法**(单一数学源纪律不动,只做最小双模型支持,不做参考工程式注册表):
- `_as_params` 按长度分派:9→M2、12→M8,其余中文 AssertionError(点名两种合法布局)。`INTRINSIC_NAMES` 保留为 M2 别名(存量测试/observation 兼容),新增 `intrinsic_names(model: str)` 唯一出口。
- `_radial_and_derivs` 泛化:返回 (ratio, d_ratio_dr2, d_ratio_dk_columns 所需原料);M2 时 den≡1 数值退化为现公式(现有 M2 测试全绿即证明兼容)。注意 den 在工作域不取零(本机 |den−1| 最大 ~0.5,远离 0),不需正则化保护,但 den 极小(<1e-6)时中文 AssertionError 防静默垃圾。
- `project` / `project_jacobian` / `project_point_jacobian` / `unproject`:全部按 params 长度分派(9/12 无歧义);**残差/雅可比/牛顿公式在本文件仍各只有一份**(M2 是 M8 的 den≡1 特例,共享同一公式体,不许出现第二份径向公式)。
- 模块 docstring 更新:M2 独占声明改为 M2/M8 双模型;注明「按参考工程方式扩展模型」的用户指令 supersession(2026-10-24 优化指令),仍不做注册表。

**solve.py**:
- `solve_opencv(uvs, objp, image_size, model)`:model="M8" 时 flags=`cv2.CALIB_RATIONAL_MODEL`(int 16384),M2 时 flags=0 不变;D 解包断言按模型:M2 恰 5;M8 先 ravel,断言 len≥8 且(若 >8)余位全零,取前 8。默认参数 model="M2" 保持现有调用兼容。
- `solve_calibration` 透传 model(从 IntrinsicConfig 读)。pack/unpack_theta、雅可比结构自动跟随 len(INTRINSIC_NAMES(model))——检查这几处当前引用的是模块常量 `INTRINSIC_NAMES`,要改成按 model 取名。

**config**(io/config.py + config.yaml):`intrinsic:` 新增 `distortion_model: M8`(解析守卫:只收 "M2"/"M8" 字符串,其余中文 ValueError;dataclass IntrinsicConfig 加字段,默认 "M2")。config.yaml 注释写清:M2=五参 Brown-Conrady(legacy)、M8=rational 8(本相机实测充分,体检对照)。

**artifacts.py**:write_intrinsics_json 按模型取名装 params dict;D 断言 (5,) 或 (8,);**intrinsics.json 新增顶层键 `distortion_model`("M2"|"M8",只增不改)**;report 头部标注模型名。旧文件无此键——读取侧默认 "M2"。

**stereo/observation.py load_intrinsics**:读 `distortion_model`(缺省 "M2");按 `intrinsic_names(model)` 取名构建 params(禁魔数下标);D 形状断言 (5,)/(8,) 按模型;IntrinsicsView 加 `model: str` 字段。旧 M2 JSON 必须继续可读(向后兼容测试)。

**stereo/evaluate.py、joint.py**:预期零算法改动(evaluate 全走 models 函数,params 12 长自动通);joint.py 的 `(9,)` 形状断言改为 `len(intrinsic_names(model))` 口径。undist_pix 的 params 拼装(np.diag(K)[:2]+center+D)对 8 长 D 自动得 12 参,验证即可。

**diagnostics.py(体检适配)**:
- A3 切片重标与 A4 探针的 `cv2.calibrateCameraExtended` 调用:当产线 D 为 8 长(M8)时 flags 必须含 `CALIB_RATIONAL_MODEL`(否则 cv2 对 8 长 guess 报错/静默截断)。A4 探针本来就恒 rational,只需保证 guess 长度与 flags 一致。
- report 体检节标注当前模型名;判定规则、阈值、config 不动。
- 语义检查:M8 产线下 A4「升阶探针」变成「同阶一致性探针」(fx 跳变应≈0)——这是预期行为,报告 note 里说清,不判红逻辑不变。

**pipeline.py**:run_intrinsic 把 cfg.distortion_model 传给 solve 与 artifacts;report.md 模型名入头部;params dict 按 12 名。

## 测试(全部进 tests/,预期 244 → ≥265)

1. **models parity(最关键)**:M8 project vs cv2.projectPoints(用实测布局构造 D12,如 [0.1,-0.05,0.001,-0.002,0.01,0.05,-0.02,0.03] 量级 + 高 r² 点),parity <1e-9 px;M8 project→unproject→project 往返 <1e-9 px;unproject 收敛 ≤50 步。
2. **M8 雅可比基准**:project_jacobian/project_point_jacobian 对 scipy 数值差分逐列 <1e-6(含 k4,k5,k6 列)。
3. **M2 回归**:现有全部 M2 测试不动一行保持绿(INTRINSIC_NAMES 别名兼容、solve/scoring/metrics/artifacts/observation M2 路径)。
4. **solve M8 端到端(合成)**:用 M8 真值(参考第 5 条量级)合成 ~20 视图多样位姿 + σ0.05px 噪声 → solve_calibration(model="M8") → fx 误差 <0.5%、D 各系数符号量级对;M2 合成数据用 M8 解 → 体检思路下 A4 跳变大(此断言可选,不作硬性)。
5. **config 守卫**:distortion_model 非法值中文 ValueError;缺省 "M2"。
6. **artifacts/observation 契约**:M8 JSON 写→读 roundtrip;旧式无 distortion_model 键的 M2 JSON 可读(默认 M2);D 形状错仍中文报错。
7. **diagnostics**:M8 D(8 长)下 A3/A4 不崩、red/green 语义与 M2 一致(小合成数据)。
8. 现有 244 测试全绿;ruff 干净(存量 5 处 RUF007/C408 告警为 Phase 1 遗留,Scope Boundary 不修)。

## 生产重跑与验收

1. `uv run stereo-calib intrinsic`(config distortion_model: M8):两路求解成功,体检 verdict=**reliable**(预期 cam1/cam2 全绿;M2 时代红 A1/A3/A4 应熄灭),退出码 0,fx 落在 ~2037 邻域(±1%);产物覆盖 output/intrinsic/(gitignored,只增键不改键)。
2. `uv run stereo-calib stereo`:消费新内参;SC 闸门阈值**一律不改**。预期 SC3 距离残差 ≪5mm;若 SC2 散布 max 超 1.0°(预期个别离群对 ~1.38°):走既有 `stereo.outlier.exclude_pairs` 剔除流程(最多 2 轮,每轮剔除报告 top-1 离群对),report 如实记录剔除清单与理由;3 轮内过不了 → 停,如实报告失败数值,不许改阈值、不许静默。
3. 验收登记:REQUIREMENTS.md Phase 2 增补一条「distortion_model 可配置 M2/M8,M8 下体检 reliable」;deferred-items.md 该条目标记完成(引用本 quick 任务);SUMMARY.md 记录最终数值表(fx、体检 verdict、SC1-4、剔除清单)。

## Constraints

- 不改 data/(sorted md5 `8a430c9395e91f256923b2a6eabb4354` 不变)
- intrinsics.json / npz 契约**只增不改**(distortion_model 是新增键;params dict 按模型 9 或 12 名)
- 参考工程只读(两个参考路径绝不修改)
- 全部模型选择走 config,禁止硬编码;阈值/闸门一律不动
- 原子提交 `feat(02)`(模型+config)、`feat(02)`(管线+契约+诊断适配)、`test(02)`/`feat(03)`(重跑产物登记)按逻辑拆分
- GSD 纪律:执行中发现的偏差如实记 SUMMARY,不许静默扩scope

## Tasks

1. models.py 双模型化(M8 布局+导数+牛顿逆)+ parity/雅可比/往返测试
2. solve.py + config.py + config.yaml(distortion_model 选项、flags、D 解包)
3. artifacts/observation/pipeline/joint 契约适配 + diagnostics M8 适配 + 契约测试
4. 合成端到端 M8 求解测试 + M2 回归全绿
5. 生产重跑 intrinsic(M8,体检须 reliable,fx~2037 邻域)
6. 生产重跑 stereo(闸门不动;散布离群对走 exclude_pairs 流程,≤2 轮)
7. REQUIREMENTS/deferred-items/SUMMARY 登记 + 原子提交收尾
