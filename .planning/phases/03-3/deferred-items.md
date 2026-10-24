# Phase 3 Deferred Items

## Out-of-scope discoveries (logged per deviation rules, not fixed)

### 2026-10-24 · 03-01 执行期 · 既有测试文件的 ruff 新版规则告警(5 处,均与本计划改动无关)

- **发现于:** Task 3 收尾全仓 `uv run ruff check src/ tests/`
- **文件:**
  - `tests/test_corners.py:181` — RUF007(zip(ladder, ladder[1:]) → itertools.pairwise)
  - `tests/test_preprocess_pipeline.py:48-51` — C408 × 4(dict(...) → 字面量)
- **原因:** Phase 1 落地的旧测试文件;仓库只锁 `ruff>=0.8`,当前解析版本 0.17.2
  的新规则(RUF007/C408)对存量代码报出。`git log` 确认两文件在本计划三个
  commit 中从未被触碰(基线 8e403c4 即已如此)。
- **不修的理由:** Scope Boundary——预存在告警与当前任务改动无因果;修了反而
  让本计划的 diff 混入无关文件。
- **建议:** 后续 wave(或 Phase 3 收尾)统一跑一次 `ruff check --fix --unsafe-fixes`
  或在 pyproject 里显式冻结 ruff 版本/规则集,避免基线漂移。

### 2026-10-24 · 03-03 执行期 · 上述 5 处 ruff 告警复核(未变化,仍未修)

- **发现于:** Task 3 收尾全仓 `uv run ruff check .`(5 处,同上一条:RUF007 × 1
  + C408 × 4,均在 Phase 1 旧测试文件)。`git log` 复核两文件仍非本计划触碰
  (`45a6b75` / `df044e0` 之后无改动),按 Scope Boundary 不修。
- **本计划侧状态:** 03-03 全部新建/改动文件 ruff 干净;全量测试 191 passed。

### 2026-10-24 · phase-03 code review(CR-01/IN-06)· joint 生产接线缺口(跨 wave 追踪)

- **来源:** `.planning/phases/03-3/03-REVIEW.md` CR-01 + IN-06;03-05 Task 3/4
  (评估链接线 + 24 键契约 + 生产重跑)因「真实数据三变体裁决全灭」按失败协议
  跳过(留痕见 03-05-SUMMARY.md 的裁决数值表)。
- **缺口:** `stereo.joint` 段与引擎(joint.py)随生产 config.yaml 出厂,但
  `run_stereo` 不读该开关——评估链接线 + 24 键契约未实施。
- **已落地的临时收口(CR-01 修复):** `run_stereo` 入口 fail-closed 硬门——
  `enabled: true` 中文 ValueError 拦截(先于目录清理);config.yaml 注释如实化
  (默认 false、当前不可置 true)。
- **待办(接线任务落地时执行):** 完成 03-05 Task 3 全量(评估链接线 + 24 键
  契约 + spread 防线生产验证 + 生产重跑 SC1-4 验收),移除 fail-closed 硬门;
  前提 = 三变体裁决结论反转(引用 03-05-SUMMARY.md 裁决数值表复核)。

### 2026-10-24 · phase-03 code review IN-04 · K/params 双轨无交叉核对(不修,登记)

- **位置:** `src/stereo_calib/stereo/observation.py` load_intrinsics——IntrinsicsView
  的 K/D 与九名 params 是有意双轨,但两轨之间无 allclose 交叉核对;文件被部分
  篡改(K 改了 params 没改)时 PnP/DLT 链路(消费 K)与 models 残差链路(消费
  params)静默分叉。属 review Info 级(非 Critical/Warning),本期不修;
  后续如加固,补 `K[:2,:2] 对角/平移列 vs params[:4]` 核对 + 中文报错。

### 2026-10-24 · phase-03 code review IN-05 · 降级 NaN 产出非严格 JSON(不修,登记)

- **位置:** `src/stereo_calib/stereo/pipeline.py` 降级路径 + `artifacts.py`
  json.dump 默认 allow_nan=True——test/train 评估降级时 NaN 会写成 `NaN`
  字面量,ECMA-404 严格解析器(serde/Rust、多数 Java 库)会拒绝整个契约
  文件。本期 Python 侧消费不受影响,按 review 结论不修;后续如加固,写侧
  非有限值显式转 null 并在 report 标注。

### 2026-10-24 · quick 261024-intrinsic-adequacy-battery · rational8 产线支持 + 重标定待办(登记)

- **来源:** 内参模型充分性体检(quick 261024)生产实证——M2 五参畸变模型对
  本相机欠拟合,cam1/cam2 体检均判「内参不可靠」(cam1 红 A1/A3/A4、cam2 红
  A1/A3;A4 升阶 rational8 探针 cam1 fx 跳变 11.19%、cam2 0.74%,说明 M2 的
  fx 存在系统性虚高,标量 RMS 看不出)。体检已 fail-closed 拦截内参进入
  双目标定(明细见 `output/intrinsic/report.md` 体检表格)。
- **待办(需用户/后续计划决策与排期):**
  1. 产线支持 rational8 畸变模型(扩展 models.py M2 契约或新增 M8 模型,
     含 INTRINSIC_NAMES/契约键/精化雅可比适配);
  2. 用 rational8 对两路相机重新标定内参(Phase 2 重标,体检复核应转绿);
  3. 内参就绪后 Phase 3 双目标定重跑(SC3 距离残差验收)。
- **说明:** 本期只做体检闸门(可换模型不依赖),不改 M2 契约与
  INTRINSIC_NAMES;合成端到端已验证 rational 真值必判不可靠、M2 一致真值
  必判可靠。

