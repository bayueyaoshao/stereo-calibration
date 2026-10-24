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

### 2026-10-24 · quick 261024-intrinsic-adequacy-battery · rational8 产线支持 + 重标定待办(已销账 2026-10-24)

- **来源:** 内参模型充分性体检(quick 261024)生产实证——M2 五参畸变模型对
  本相机欠拟合,cam1/cam2 体检均判「内参不可靠」(cam1 红 A1/A3/A4、cam2 红
  A1/A3;A4 升阶 rational8 探针 cam1 fx 跳变 11.19%、cam2 0.74%,说明 M2 的
  fx 存在系统性虚高,标量 RMS 看不出)。体检已 fail-closed 拦截内参进入
  双目标定(明细见 `output/intrinsic/report.md` 体检表格)。
- **待办销账状态(2026-10-24,quick 261024-rational8-production):**
  1. 产线支持 rational8 畸变模型 → **完成**(M8 全链一等公民,提交
     6d570ac/fc2d7f7/63023fc,270 测试,契约只增不改);
  2. rational8 重标内参 → **部分完成**:cam1 体检转绿(红灯 0,fx=2042.89
     落参考锚点 2037±1% 带,test RMS 0.0652px);cam2 两轮证据剔除后仍红
     A1/A3(0.0280/0.9392),fail-closed 拦停(honest-stop,见下条新登记);
  3. Phase 3 双目重跑 → **阻塞**(前置 = cam2 内参 reliable)。
  数值全表见 `.planning/quick/261024-rational8-production/SUMMARY.md`。

### 2026-10-24 · quick 261024-rational8-production · cam2 结构性残差(阻塞,待用户决策)

- **现象:** cam2 在 M8 rational8 模型下体检仍判「不可靠」(红 A1=0.0280
  超阈 0.02、A3=0.9392% 超阈 0.5%):中心桶径向结构残差 ~0.028px + 外圈
  切片 fx 低压(2020.9 vs 内圈 2038.6、左半 2040.0)。
- **排除项(实证):** 非离群帧——两轮证据驱动剔除(1.08/0.460/0.447px
  三帧)后 A1/A3 对剔除不敏感(0.0284/0.9849→0.0280/0.9392);非 M8 特异
  ——M2 下同判不可靠(0.0772/1.2795),M8 已改善 2.8×/1.4×;非 fx 漂移
  ——A4 同阶独立重标 2037.52 与产线 2034.88 差 0.13%,fx 本身钉在锚点。
- **方向(需用户决策):** Phase 1 外圈角点检出质量分层复核;采集侧排查
  (光轴/传感器倾斜、C-mount 接口机械应力);或更高阶畸变模型(风险:
  rational 近简并随阶数加剧)。
- **影响:** cam2 内参 不可靠 → fail-closed 禁入双目 → Phase 3 双目重跑
  (PLAN Task 6)未执行。

### 2026-10-24 · quick 261024-rational8-production · 自动离群循环在 M8 下的评分保真度(已知限制,未修)

- **现象:** 自动剔除循环以 OpenCV 初值的逐帧 RMS 为评分(5×中位阈值)。
  M8 下 calibrateCamera 初值逐帧评分中位 ~1.4–1.8px(M2 ~0.6px),阈值随之
  膨胀到 7–9px:cam1 首轮 27 帧 >1px 坏帧中 26 帧被剔、最轻 1 帧(1–2.9px
  带)漏网,靠手动 exclude_images 兜住。
- **不修的理由:** 验收中途不改管线循环语义(行为变更需独立验证);本任务
  已用设计内通道(手动剔除清单)达成 cam1 转绿。
- **修法建议(后续):** 循环评分改用精化后逐帧 RMS(每轮 scipy 精化后再
  算逐帧残差),或阈值改为「中位数 + 绝对上限」双界;需配套回归测试。


