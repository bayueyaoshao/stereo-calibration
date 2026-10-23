---
phase: 03-3
plan: 03
subsystem: stereo-calibration
tags: [stereo, dlt, bundle-adjustment, disambiguation, outlier-rejection, opencv, artifacts]

# Dependency graph
requires:
  - phase: 03-2
    provides: solve_stereo 引擎(pnp_median 两遍法 + 解析雅可比精化)、disambiguate_pairs、Pairs/IntrinsicsView 装载契约
  - phase: 03-1
    provides: load_pairs 三重防线、load_intrinsics 按名读取、合成夹具 make_stereo_synth/write_stereo_synth_files
  - phase: 02
    provides: intrinsics.json 契约产物(Phase 3 固定内参输入)、intrinsic/pipeline 剔除循环与报告体例
provides:
  - src/stereo_calib/stereo/evaluate.py:评估全链(消歧→per-frame 2 视图 BA→去畸变→DLT 保号除法→板系对角距离残差),三角化唯一住址
  - src/stereo_calib/stereo/artifacts.py:Pattern 7 契约五件套(JSON/CSV×2/npz/report.md)
  - src/stereo_calib/stereo/pipeline.py:run_stereo 编排(剔除循环三保险丝 + baseline 硬门 + test 保护 + 中文汇总)
  - tests/test_stereo_guards.py:全树五条源码防线常态化运行
affects: [03-4 (CLI 验收), real-data UAT, STER 验收]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "评估链双口径畸变契约:PnP/BA 吃带畸变原始像素,DLT 吃精确逆去畸变像素配无畸变 P=K[R|t]"
    - "剔除循环全程快解(refine 临时关闭),精化只跑幸存集定稿一次(时序照 Phase 2)"
    - "pattern 6 保险丝三道:max_drop_fraction / max_iterations / min_pairs,触发必留痕"
    - "全树源码防线:grep 计数族测试锁五条工程纪律,被锁串原文不进测试 docstring(串拼接规避自伤)"

key-files:
  created:
    - src/stereo_calib/stereo/evaluate.py
    - src/stereo_calib/stereo/artifacts.py
    - src/stereo_calib/stereo/pipeline.py
    - tests/test_stereo_evaluate.py
    - tests/test_stereo_guards.py
  modified:
    - tests/test_stereo_pipeline.py

key-decisions:
  - "Rule 1 修复:undist_pix 弃 cv2.undistortPoints(定点迭代,r²=0.5 处逆误差实测 0.28px、r²=1.0 处 8px),改走 models.unproject 精确牛顿逆回乘 K——P=K 像素口径不变,畸变数学单一住址 intrinsic/models"
  - "evaluate_split 返回 SplitEval 冻结数据类(逐帧 evals + pooled 距离 RMS + 逐角点残差 + 真值),满足 write_residuals_npz 的逐点残差数据流"
  - "整体翻转帧消歧不翻(锚自洽性):cam1 PnP 锚解出 180° 转板位姿与翻转序自洽,对角距离指标天然翻转不变——测试锁自洽性论证而非'恢复'标志"
  - "test split 装载下限用几何下限 2,不用 config min_pairs(那是 train 求解素材的保险丝,不是考卷的尺)"
  - "离群注入用逐点散斑而非恒定偏移:恒定偏移被该对自身 PnP 位姿吸收(per_pair_rms 度的是自洽残差),test_intrinsic_pipeline 同款教训"
  - "flip_log 键归一在编排层做:solve_stereo 出 pair_ids,Pattern 7 契约要求 pairs,pipeline 写 JSON 时映射"
  - "baseline 量级硬门放降级收口之外:带外必抛中文 ValueError,绝不被单段降级吞掉"

patterns-established:
  - "Pattern 7 双目产物契约:extrinsics_stereo.json 22 键全集 + 写前断言(字段集防漂移、baseline_mm=‖t‖ 一致性)"
  - "report.md 人工核对出口:基线量级核对行(|t| 分量 + 落带说明)+ PASS/FAIL 判定列(报告结论非退出条件)"
  - "e2e 数据只读自动化:_tree_snapshot 快照前后逐字节对比 + 新增文件必须 ⊆ output/stereo/"

requirements-completed: [STER-01, STER-03, STER-04]

# Metrics
duration: 26min
completed: 2026-10-24
---

# Phase 03-3 Plan 03: 双目评估链 + Pattern 7 产物 + OVERNIGHT 编排 Summary

**test 集评估全链(消歧 → per-frame 2 视图 BA → 精确逆去畸变 → DLT 保号除法 → 板系对角距离残差,合成板恢复 < 0.5mm/对)+ Pattern 7 五件套契约 + run_stereo 离群剔除编排(三保险丝 + baseline 硬门 + test 保护),全树五条源码防线进测试**

## Performance

- **Duration:** 26 min(RED 首提交 → 末提交;含前置研读)
- **Started:** 2026-10-24T05:49:54+08:00(RED)
- **Completed:** 2026-10-24T06:15:40+08:00
- **Tasks:** 3/3
- **Files modified:** 6(4 新建 + 2 扩展)

## Accomplishments

- evaluate.py(492 行):`disambiguate_frame`(锚=当帧 cam1 PnP,证据充分才翻,绝不原地改输入)→ `frame_pose_ba`(unproject 反投影初值防错误盆地,trf/x_scale=jac,残差只经 models)→ `undist_pix`(models.unproject 精确牛顿逆回乘 K,P=K 像素口径)→ `triangulate_dlt`(保号除法,SVD 唯一住址)→ 板系反变换 → 对角距离残差(真值 `norm(objp[87]−objp[0])` 从 build_object_points 导出,零写死常数);单帧失败 NaN+中文备注不中断整批,全帧失败中文 ValueError
- artifacts.py(323 行):Pattern 7 全集 + distance_rms_train_mm(决策 8)的 22 键 JSON 契约(写前断言字段集与 baseline=‖t‖ 一致性)、utf-8-sig 逐帧 CSV(8 列契约)、allow_pickle=False 可读残差 npz、report.md(基线量级核对行 + 逐帧距离残差表 + 剔除/翻转留痕 + PASS/FAIL 报告结论非退出条件 + 配置快照)
- pipeline.py(370 行):run_stereo 零数学编排——剔除循环(阈值 = rms_multiplier × 幸存中位;三保险丝触发必留痕;全程快解,精化定稿一次)→ baseline 硬门(带外中文 ValueError 拒绝带病出报告)→ test 全对评估永不剔除只标记 + train 幸存集评估(距离残差只进报告,决策 9)→ 五件产物落盘 → 中文汇总 + 产物绝对路径;单段降级留痕不炸整批
- 测试 +34 例:evaluate 链 10 例、guards 全树防线 5 例、writer roundtrip 6 例、合成 e2e 4 例(全契约/离群/保险丝/零数学防线)、03-01 既有 9 例照旧;全量 `uv run pytest -q` **191 passed** 零回归

## Task Commits

1. **Task 1: evaluate.py 评估链** — `bebaea0`(test RED)→ `e876ec0`(feat GREEN,TDD 双门齐备)
2. **Task 2: artifacts.py Pattern 7 契约 + test_stereo_guards.py 全树防线** — `623382a`(feat)
3. **Task 3: pipeline.py run_stereo 编排 + 合成 e2e** — `f3076c5`(feat)

**Plan metadata:**(SUMMARY 由编排层收口提交)

## Files Created/Modified

- `src/stereo_calib/stereo/evaluate.py` — 评估全链数学(三角化唯一住址;SVD 只在此文件)
- `src/stereo_calib/stereo/artifacts.py` — Pattern 7 契约落盘(零数学)
- `src/stereo_calib/stereo/pipeline.py` — run_stereo 编排(零数学,源码防线锁死)
- `tests/test_stereo_evaluate.py` — DLT 往返/保号除法/undist 契约/消歧/BA/失败契约(10 例)
- `tests/test_stereo_guards.py` — 全树五防线(禁内建例程/单数学源/三角化单文件/雅可比组合/零读图)
- `tests/test_stereo_pipeline.py` — 装载契约 + writer roundtrip + 合成 e2e + 离群行为 + 零数学防线(21 例)

## Decisions Made

- **cv2.undistortPoints → models.unproject(精确逆):** 强畸变域实测 cv2 定点迭代逆误差随半径爆炸(r²=0.5 → 0.28px,r²=1.0 → 8px;models.unproject 解析雅可比牛顿逆 1e-12px),进 DLT 即 mm 级假距离残差——合成强倾角帧当场抓住。修复后全链畸变数学单一住址 intrinsic/models。返回口径不变:归一化坐标乘 K 回像素框(P=K 契约保持)
- **SplitEval 返回形状:** 计划文本的 `evaluate_split → (list[FrameEval], distance_rms_mm)` 装不下 `write_residuals_npz` 需要的逐角点残差;返回冻结数据类补齐数据流(evals/distance_rms_mm/residuals_cam1/residuals_cam2/true_distance_mm)
- **test split 装载下限 = 2:** config min_pairs 是 train 求解素材保险丝(一对定不了相对位姿);对 test 用同一把尺会让小样本考卷被误杀(保险丝用例 8 train/2 test × min_pairs=7 当场暴露)
- **离群注入用 3px 散斑:** 计划写的"恒定 +15px 偏移"会被该对自身 PnP 位姿吸收(per_pair_rms 度的是自洽残差,solve.py L187-193 实证),永远不会触发剔除——正是 test_intrinsic_pipeline 记过的同一教训
- **report 增加 test 降级留痕位:** test 评估单段失败时 train 产物照常落盘,report 写明降级原因(绝不编造 test 指标)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] cv2.undistortPoints 定点迭代逆误差污染 DLT 链**
- **Found during:** Task 1(GREEN 验证,合成强倾角帧 dist_err 0.55mm > 0.5mm 门)
- **Issue:** cv2.undistortPoints 是定点迭代,本工程强畸变域(k1=-0.40/k2=0.23)实测逆误差 r²=0.5 处 0.28px、r²=0.8 处 2.4px、r²=1.0 处 8.4px(生产 2560×1440 板缘可达 r²≈0.8)——进 DLT 即 mm 级假距离残差,且真实数据同样受害
- **Fix:** undist_pix 改经 models.unproject(精确牛顿逆,1e-12px)解归一化坐标再乘 K;P=K 像素口径、函数签名、调用方全部不变
- **Files modified:** src/stereo_calib/stereo/evaluate.py,tests/test_stereo_evaluate.py(契约基准换成精确逆,cv2 降为 <1px 量级 sanity;源码防线断言 undistortPoints→unproject)
- **Verification:** 全 6 对合成板恢复 |dist_err| < 0.5mm(修复前强倾角对 0.55mm);test_undist_pix_pixel_frame_contract 锁 1e-6px 精确性
- **Committed in:** e876ec0

**2. [Rule 3 - Blocking] evaluate_split 返回形状不满足 npz 数据流**
- **Found during:** Task 1(write_residuals_npz 需要逐角点残差,计划签名只回 (list, rms))
- **Issue:** 逐点残差无处安放,train_residuals.npz 契约无法兑现
- **Fix:** 新增 SplitEval 冻结数据类(residuals_cam1/cam2 (V,88,2) + true_distance_mm),失败行整行 NaN 留痕
- **Files modified:** src/stereo_calib/stereo/evaluate.py
- **Verification:** e2e 断言 npz 形状 (10,88,2) float64、allow_pickle=False 可读
- **Committed in:** e876ec0 / f3076c5

---

**Total deviations:** 2 auto-fixed(1 bug + 1 blocking)+ 3 处计划内语义澄清(见 Decisions Made:test 下限、散斑注入、整体翻转自洽性)
**Impact on plan:** 两处 auto-fix 都是正确性必需(Rule 1 修复让真实数据评估链免于 mm 级系统误差);无范围蔓延,零新增依赖

## Issues Encountered

- 研究文档"undist parity 逐位一致"的口径经实测不成立(cv2 自身浮点装包 6e-5px,且定点迭代随半径发散到 px 级)——已按实测改写测试基准与 docstring,研究侧结论以本 SUMMARY 为准
- 整体翻转帧的消歧行为:计划行为描述为"检出并恢复",几何事实是 cam1 PnP 锚对整体翻转自洽(解出 180° 转板位姿),消歧判定不翻且对角距离指标天然不变——测试按自洽性论证锁定(test_disambiguate_frame_full_flip_metric_invariant)
- `_STEREO_YAML` 模板内含 YAML flow-map 花括号,`str.format` 撞键——改用 token 替换(测试基础设施问题,一次修掉)
- 全仓 `ruff check .` 的 5 处告警(RUF007×1 + C408×4)全部位于 Phase 1 旧测试文件(git log 复核非本计划触碰),按 Scope Boundary 不修,已在 `.planning/phases/03-3/deferred-items.md` 追加复核记录;本计划全部新建/改动文件 ruff 干净

## Known Stubs

None——所有产物字段真实落盘,无占位数据流。

## User Setup Required

None - 无外部服务配置。

## Next Phase Readiness

- 03-04(CLI 验收)接口已定死:`run_stereo(cfg)` 唯一编排入口;产物五件在 `output/stereo/`(extrinsics_stereo.json / per_frame_{train,test}.csv / train_residuals.npz / report.md)
- extrinsics_stereo.json 22 键契约即 03-04 与真实数据 UAT 的验收接口(Pattern 7 全集 + distance_rms_train_mm)
- 已知前置(03-02 遗留,如实留痕不绕过):真实数据 solve_stereo 触发离散度防线(spread 12.16° > 1.0°,fail-closed);合成链全绿,真实数据放行属 UAT 决策
- 全量测试 191 passed(基线 167 + 本计划 24 净增);ruff 本计划文件全干净

## Self-Check: PASSED

- 7/7 声称文件存在(evaluate.py / artifacts.py / pipeline.py / 三个测试文件 / 本 SUMMARY)
- 4/4 任务提交在案:bebaea0(RED)→ e876ec0(GREEN)→ 623382a(Task 2)→ f3076c5(Task 3)
- 计划跨度 4 提交共 1978 行新增,零删除

---
*Phase: 03-3*
*Completed: 2026-10-24*
