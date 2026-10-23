---
phase: 03-3
depth: standard
status: issues_found
reviewed_at: 2026-10-23T23:34:13Z
files_reviewed: 14
files_reviewed_list:
  - src/stereo_calib/stereo/observation.py
  - src/stereo_calib/stereo/solve.py
  - src/stereo_calib/stereo/evaluate.py
  - src/stereo_calib/stereo/pipeline.py
  - src/stereo_calib/stereo/artifacts.py
  - src/stereo_calib/stereo/joint.py
  - src/stereo_calib/cli/main.py
  - src/stereo_calib/io/config.py
  - src/stereo_calib/intrinsic/models.py
  - src/stereo_calib/intrinsic/solve.py
  - src/stereo_calib/preprocess/artifacts.py
  - tests/conftest.py
  - tests/test_stereo_solve.py
  - tests/test_stereo_pipeline.py
  - tests/test_stereo_joint.py
  - tests/test_stereo_evaluate.py
  - tests/test_stereo_guards.py
  - tests/test_stereo_config.py
findings:
  critical: 1
  warning: 2
  info: 6
  total: 9
---

# Phase 03-3: Code Review Report

**Reviewed:** 2026-10-23T23:34:13Z
**Depth:** standard
**Files Reviewed:** 18(评审范围 8 个源文件 + 抽样测试/上游依赖 10 个)
**Status:** issues_found

## Summary

对 Phase 3(双目标定)全部交付物做了 standard 深度评审:数学链路(relative_pose / PnP-median / 两遍法消歧 / 解析稀疏雅可比 / DLT / 板系反变换 / joint 三变体)、异常防线、22 键 JSON 契约、配置校验、测试防线均逐项核验。整体质量高:方向语义在点变换层面被测试锁死、雅可比有全维差分基准(< 1e-6)、行号契约/装箱互逆有专测、失败契约(NaN + 中文备注)与 fail-closed 防线到位、209 测试全绿确认。

发现 1 项 Critical:`stereo.joint.enabled` 开关有配置段、有引擎、有引擎级测试,但 `run_stereo` 从不读取它——开关打开是静默 no-op(fail-open),违反本工程自己的「不静默」红线。2 项 Warning:train 评估失败无单段降级、pairs CSV 畸形列抛 KeyError 击穿 CLI 异常网。其余 6 项 Info。

数学正确性结论:未发现符号约定、单位(mm/px)、坐标变换链、中位数/消歧逻辑、雅可比索引方面的错误;joint.py 大参数向量(112 对 696 维)索引逐块核对无误;float64 纪律与 OpenCV 调用点 float32 装包符合声明。

## Critical Issues

### CR-01: `stereo.joint.enabled` 开关无任何消费者,置 true 是静默 no-op(fail-open)

**File:** `src/stereo_calib/stereo/pipeline.py:158-370`(run_stereo 全文无 joint 分支);`src/stereo_calib/io/config.py:549-585`(配置层照常解析并放行 `enabled: true`);`config.yaml:89-95`(生产配置出厂即带 joint 段,注释写「03-05 裁决后置 true」)

**Issue:** `solve_stereo_joint` / `refine_joint` / `JointSolveOutput` 在 `src/` 全树中只有 `joint.py` 自身与 `tests/test_stereo_joint.py` 引用,`run_stereo` 及 CLI 从不读取 `cfg.stereo.joint.enabled`(grep 全 src/ 零命中)。后果:产线操作员按 config.yaml 注释把 `enabled` 置 true 后,程序照常跑 rigid 路径、照常出 22 键 JSON 与 PASS/FAIL 报告——没有任何日志、告警或契约键提示「joint 实际没跑」。操作员会以为联合 BA 已生效。这正是本工程自定红线(Pitfall 8:触发原因全部如实留痕,绝不静默)所禁止的 fail-open 形态。

背景核实:03-05-PLAN.md Task 3(评估链接线 + 24 键契约)与 Task 4 因「真实数据三变体裁决全灭」按失败协议跳过,03-05-SUMMARY.md L51 有留痕——跳过决策本身合规;问题在于**开关与其邀请性注释随生产 config.yaml 一并出厂**,而失败协议的收尾(fail-closed 拦截或 deferred 登记)没有落地:`.planning/phases/03-3/deferred-items.md` 只登记了 ruff 告警,未登记「joint 接线缺口」。

**Fix:**

```python
# pipeline.py run_stereo() 入口、清理目录之前加一道 fail-closed 硬门:
if cfg.stereo.joint.enabled:
    raise ValueError(
        "stereo.joint.enabled=true 不合法:联合 BA 引擎已交付但生产接线"
        "(03-05 Task 3,评估链接线 + 24 键契约)因三变体裁决失败按失败协议"
        "跳过,当前开关没有消费者——置 true 只会静默跑 rigid 路径。"
        "请保持 enabled: false,等待接线任务落地。")
```

或者(若裁决结论反转)完成 PLAN Task 3 的接线。二选一必须做其一;同时把该缺口登记进 deferred-items.md。

## Warnings

### WR-01: train 评估失败不在单段降级防线内,精化成功后仍会整批炸掉、零产物

**File:** `src/stereo_calib/stereo/pipeline.py:226-227`

**Issue:** test 评估有 `try/except _SEGMENT_EXC` 降级留痕(pipeline.py:219-225),但紧随其后的 `train_eval = evaluate_split(train_survivors, ...)` 裸奔在外。若 train 评估全帧失败(如全部帧触发消歧锚失败/DLT 退化),ValueError 直接穿到 CLI 最外层:已成功且昂贵的 train 求解+精化结果(R/t)全部丢弃,JSON/CSV/npz/报告一件都不落盘,退出码 1。这违背模块 docstring 自我声明「单段降级:train 求解或 test 评估某段失败 → 中文留痕降级,绝不炸掉已成功段的产物」(pipeline.py:26-27)的后半句。

**Fix:**

```python
train_eval = None
train_error = ""
try:
    train_eval = evaluate_split(train_survivors, intr1, intr2, final.R, final.t,
                                cfg.board, stereo.flip_ratio)
except _SEGMENT_EXC as e:
    train_error = f"{type(e).__name__}——{e}"
    typer.echo(f"  train 评估失败,降级留痕:{train_error}")
```

后续 `distance_rms_train`、`true_distance_mm`、summary 组装对 `train_eval is None` 给 NaN/占位(照 test 侧同款体例),并在 summary/report 追加 train_error 降级留痕。注意 `true_distance_mm` 可改从 `evaluate.true_board_distance(build_object_points(cfg.board))` 直接导出,不依赖 train_eval 存活。

### WR-02: pairs CSV 列缺失/空文件时抛 KeyError,击穿 CLI 中文异常网吐堆栈

**File:** `src/stereo_calib/stereo/observation.py:151`(`row["group_id"]`)、`:174-175`(`row["cam1_path"]`/`row["cam2_path"]`)、`:196`(`row["route_cam1"]`/`row["route_cam2"]`)

**Issue:** `csv.DictReader` 对缺失列名抛 `KeyError`。pairs CSV 是磁盘产物,被截断/手编辑/空文件(表头都丢)时,`row["group_id"]` 抛出的 KeyError 不在 CLI 异常网 `(ValueError, OSError, ArithmeticError, RuntimeError, AssertionError, cv2.error)`(cli/main.py:103-104)内——用户看到的是英文堆栈而不是本工程 everywhere else 的中文一句收口。与本文件其余防线(连接键不命中、detected 不一致、objp 漂移全部 ValueError 中文报错)的失败契约不一致。短行(值缺失)倒是安全:DictReader 补 None → 连接键不命中 → 走既有中文防线,唯独「列名级」损坏裸奔。

**Fix:** 装载前核对表头,缺列给中文 ValueError:

```python
reader = csv.DictReader(f)
required_cols = ("group_id", "cam1_path", "cam2_path", "route_cam1", "route_cam2")
missing_cols = [c for c in required_cols if reader.fieldnames is None
                or c not in reader.fieldnames]
if missing_cols:
    raise ValueError(
        f"对清单 {Path(pairs_csv).name} 不合法:缺列 {missing_cols}"
        f"(实际表头 {reader.fieldnames})——它不是 Phase 1 对清单契约的产物,"
        f"请重跑 preprocess。")
```

## Info

### IN-01: frame_pose_ba 的射线 Z>0 检查恒真,是死防线

**File:** `src/stereo_calib/stereo/evaluate.py:288-290`

**Issue:** `models.unproject` 返回单位射线 `[x, y, 1]/‖·‖`,第三分量恒为 `1/‖·‖ > 0`,故 `if not bool((rays[:, 2] > 0).all())` 永不触发。该 RuntimeError 分支是不可达代码,给读者「此处有 Z≤0 防护」的虚假信心;真实的板点 Z≤0 防护由 BA 残差里 `models.project` 的断言 + solvePnP 失败承担。

**Fix:** 删除该死分支,或改注释如实声明「射线 Z 恒正,Z≤0 防护在 project 断言与 solvePnP 失败出口」。

### IN-02: 报告「距离 RMS」行 3 个单元格塞进 2 列表格,Markdown 渲染错位

**File:** `src/stereo_calib/stereo/artifacts.py:261-262`

**Issue:** 指标汇总表头是 `| 指标 | 值 |`(2 列,分隔行也 2 列),但距离 RMS 行写成 `| 距离 RMS train/test (mm) | ... | {verdict} |`(3 个单元格)。Markdown 渲染时 verdict 会溢出成第三列,整列错位;PASS/FAIL 判定也无处安放表头。

**Fix:** 要么把该行裁回 2 格(verdict 并入值格:`... / ... — PASS`),要么把表头与分隔行改成 3 列(`| 指标 | 值 | 判定 |`)。

### IN-03: 失败帧进 CSV 是 "nan" 字符串,与「失败对留空」文档契约漂移

**File:** `src/stereo_calib/stereo/pipeline.py:287`、`:293-294`、`:302-303`;对照 `src/stereo_calib/stereo/artifacts.py:156-157`

**Issue:** artifacts docstring 写明「reproj_rms / dist_err_mm 列为格式化字符串,失败对留空」,但编排层用 `f"{ev.dist_err_mm:.4f}"` 格式化,NaN 会产出字面 `"nan"` 而非空串。单帧失败时(train/test CSV 均可能)Excel 侧看到 nan,与文档承诺不符。

**Fix:** 照 report 的 `_fmt` 体例给 CSV 加同款小工具:`"" if not np.isfinite(v) else f"{v:.4f}"`。

### IN-04: load_intrinsics 不交叉核对 K 与 params 的 f/c 冗余字段

**File:** `src/stereo_calib/stereo/observation.py:254-311`

**Issue:** IntrinsicsView 的 K/D 与九名 params 是「双轨」(docstring 有意为之),但两轨之间没有任何 `np.allclose` 交叉核对——文件被部分篡改/手改(K 改了、params 没改)时,PnP/DLT 链路消费 K、models 残差链路消费 params,两链路静默分叉,重投影指标与三角化结果互相矛盾且无告警。本工程同类风险(objp 三方交叉核对)都有防线,这里是唯一的冗余双轨无核对点。

**Fix:** 读取时补一道:`np.allclose(K[:2,:2].diagonal(), params[:2]) and np.allclose(K[:2,2], params[2:4])`,不一致中文报错提示重跑 intrinsic。

### IN-05: 降级路径 NaN 进 JSON 产出非严格 JSON 字面量 `NaN`

**File:** `src/stereo_calib/stereo/pipeline.py:255-256`;`src/stereo_calib/stereo/artifacts.py:125-128`

**Issue:** test 评估降级时 `rms_reproj_test` / `distance_rms_test_mm` 为 NaN,`json.dump` 默认 `allow_nan=True` 会写出 `NaN` 字面量——ECMA-404 严格解析器(Python 之外如 serde/Rust、多数 Java 库)会拒绝整个契约文件。降级场景恰恰是下游最需要机器可读契约的时刻。

**Fix:** 写侧对非有限值显式转 `None`(JSON null)并在 report 标注,或 `json.dump(..., allow_nan=False)` 配合写前归一化,把「合法 JSON」变成写侧不变量。

### IN-06: deferred-items.md 未登记 joint 接线缺口,后续 wave 无追踪入口

**File:** `.planning/phases/03-3/deferred-items.md`(全文只有 ruff 告警两条)

**Issue:** 03-05 Task 3/4(评估链接线 + 24 键契约 + 生产重跑)按失败协议跳过,留痕只在 03-05-SUMMARY.md;但 deferred-items.md 作为跨 wave 追踪文件没有登记该缺口。结合 CR-01(config.yaml 注释邀请置 true),这是缺口被遗忘的直接通道。

**Fix:** 在 deferred-items.md 增补一条:joint 生产接线(含 fail-closed 硬门或完整接线 + 24 键契约 + SC1-4 验收)待三变体裁决结论反转后执行,引用 03-05-SUMMARY 的裁决数值表。

## 评审确认(专项核验通过项)

- **数学正确性**:`relative_pose` 方向推导逐项复核无误且有点变换级测试锁定(test_stereo_solve.py:72-98);消歧锚组合(cam1 自身 PnP + pass-1 中位外参)、DLT 保号除法、板系反变换 `Rᵀ(X−t)`、pooled RMS 口径(逐对/整体/初值出口三处公式等价性)全部核对一致。
- **雅可比与索引**:`row_index`/`_structure_positions`/`pack/unpack_theta` 与 joint 版逐一对照残差布局核验,块序、列序(fc: pj[:, :, :4];full: 全 9 列与 θ 列 9-17 的 fx..k3 顺序)匹配;joint 雅可比有全维 approx_derivative 基准 < 1e-6。
- **异常防线**:Phase 2 教训已吸收——`_PNP_EXC`/`_FRAME_EXC`/`_SEGMENT_EXC` 三层均含 AssertionError 与 cv2.error;无一处传 jac_sparsity(源码防线测试锁定);容器型 yaml 值先判 str 再进集合(config.py 三处)。
- **契约**:22 键断言 `_EXTRINSICS_STEREO_JSON_KEYS` 与 e2e 测试双向锁定;npz 定长 unicode/float64/allow_pickle=False 有 roundtrip 测试;CSV utf-8-sig 读写两侧一致。
- **配置**:joint 段 enabled 默认 false(有缺段默认值测试)、变体白名单/先验权重正性/spread_ratio_max>1 全部中文校验;baseline_band_mm 有序校验;数据/输出目录互斥校验在加载层。
- **测试防线**:209 passed(本次评审实测复跑);源码防线(内建例程零出现、SVD 单住址、零读图、单一数学源)覆盖到位。

---

_Reviewed: 2026-10-23T23:34:13Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
