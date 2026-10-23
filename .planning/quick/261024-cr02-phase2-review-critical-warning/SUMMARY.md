---
phase: 261024-cr02-phase2-review-critical-warning
plan: 01
subsystem: intrinsic (scoring/solve/pipeline/artifacts) + io.config + cli
tags: [bugfix, code-review, CR-01, WR-01, WR-02, exception-containment, config-validation, scipy]
requires:
  - "Phase 2 intrinsic module (src/stereo_calib/intrinsic)"
  - "Phase 1 config layer (src/stereo_calib/io/config.py)"
provides:
  - "三层异常收口:评分边界(AssertionError 纳入捕获面)→ 编排层(单路失败降级留痕)→ CLI(数值异常家族中文一句 + 退出码 1)"
  - "report.md 失败留痕节与 FAIL 行渲染(write_report_md 支持整路失败记录,不编造指标)"
  - "配置集合成员检查的 str 类型守卫(method / window_granularity 容器型值给中文 ValueError)"
  - "精化调用的稀疏机制描述归位:callable jac 返回 csr + tr_solver='lsmr',死结构矩阵代码已移除"
affects:
  - "Phase 3 双目标定(intrinsics.json / npz / CSV 契约字段集未变,消费接口不受影响)"
  - "真实生产路径(重跑验证与基线同量级,数据指纹不变)"
tech-stack:
  added: []
  patterns:
    - "多层异常收口(评分边界单帧 NaN → 编排层单路降级 → CLI 最外层中文一句)"
    - "源码防线自伤规避体例延续(被计数串不出现在 solve.py 注释)"
key-files:
  created: []
  modified:
    - src/stereo_calib/intrinsic/scoring.py
    - src/stereo_calib/intrinsic/pipeline.py
    - src/stereo_calib/intrinsic/solve.py
    - src/stereo_calib/intrinsic/artifacts.py
    - src/stereo_calib/cli/main.py
    - src/stereo_calib/io/config.py
    - tests/test_intrinsic_scoring.py
    - tests/test_intrinsic_pipeline.py
    - tests/test_intrinsic_config.py
    - tests/test_intrinsic_solve.py
decisions:
  - "CR-01 收口取三层而非只改评分边界:plan 明确要求核查 solve.py「上层收口」声明——核查结论是不属实(run_intrinsic 与 CLI 均不捕数值异常家族),按 plan 在 run_intrinsic 调用边界补单路降级,CLI 同步扩网"
  - "CLI/编排捕获面在 review 建议元组之外加 cv2.error:实测 MRO 是 Exception 直系(非 ValueError/RuntimeError 子类),scoring.py 既有捕获面含它,边界一致才不留新洞"
  - "make_sparsity 整体删除而非保留:全仓零引用(仅 solve.py 自身),保留即死代码;源码防线测试锁定不得回潮"
  - "write_report_md 扩展失败记录渲染(failed 标记 + error 键):「报告如实留痕」的必要载体;JSON/npz/CSV 契约字段集未动"
metrics:
  duration: "~21 min"
  completed: 2026-10-23T18:42:16Z
---

# Phase 261024-cr02 Plan 01: Phase 2 Code Review CR-01/WR-01/WR-02 Fixes Summary

修复 phase 2 code review 三个高价值发现:评分器「绝不抛异常」契约的 AssertionError 洞三层收口(CR-01),配置集合成员检查的容器型值 TypeError 洞加 str 守卫(WR-01),精化调用里被 scipy 静默忽略的死 jac_sparsity 连同失实机制描述一并移除(WR-02);npz/JSON/CSV 契约字段集未变,生产重跑两路 test RMS 0.1407/0.1361 与基线逐位同值,data/ 指纹不变。

## What Was Built

### Task 1 — CR-01: 评分/编排/CLI 三层异常收口 (5f014d1)

核查结论先行:review 指出 solve.py:364-367 docstring 声称的「编排层中文收口(02-03)」**不属实**——`run_intrinsic` 的两路循环对 `_calibrate_one_cam` 无任何捕获,CLI 只捕 `(ValueError, OSError)`。按 plan 指示在不属实的调用边界补齐收口:

- **scoring.py**:捕获面从 `(ValueError, ArithmeticError, RuntimeError, cv2.error)` 扩入 `AssertionError`——`models.project` 的 Z>0 数据防卫依赖精化期优化器位姿,入口防线拦不住;单帧失败照旧返回 `(NaN, None, 中文备注)`,注释写明为何不吞代码契约(形状/参数仍在入口拦)。
- **pipeline.py**:`run_intrinsic` 每路捕获数值异常家族,失败路降级为带 `failed` 标记 + `error` 的最小记录(中性空字段),另一路继续;失败路无产物、报告如实留痕;docstring 与末尾中文汇总同步。
- **cli/main.py**:`intrinsic()` 收口面扩为 `(ValueError, OSError, ArithmeticError, RuntimeError, AssertionError, cv2.error)`,兜最外层的底——中文一句 + 退出码 1,绝不吐堆栈。
- **solve.py**:失败契约 docstring 从编辑残留的自问自答改为平实陈述,「上层收口」现在真实存在。
- **artifacts.py**:`write_report_md` 支持整路失败记录——指标表 FAIL 行(失败路无指标,不编造)、新增「失败留痕」节写病因、参数表只列成功路(空时显式说明)。

回归测试 3 个:① 真实路径触发(6 角点 +3000px 劣化 → solvePnP 仍 ok=True → 精化试验步 Z<=0 → 断言从 least_squares 内部抛出;修复前实测整体堆栈逃逸,修复后返回 NaN + 含异常类型的备注);② 编排层单路失败降级(monkeypatch 第 2 次精化调用注入 AssertionError,cam1 照常落盘、cam2 无半套产物、report.md 含失败留痕);③ CLI 异常家族退出码 1。

### Task 2 — WR-01: 配置集合成员检查 str 守卫 (b7db951)

`config.py` 两处 `in set` 检查(`window_granularity`、`refine.method`)前加 `isinstance(x, str)` 守卫:容器型 yaml 值(如 `method: [trf]`)不可哈希,旧写法抛 `TypeError: unhashable type` 击穿 CLI 异常网;现在收口为该文件统一的中文 `_fail` ValueError。回归测试 4 例(method/granularity × list/dict;`window_granularity` 在 flow 式行内,`_mutate` 够不到,用模板定向替换)。

### Task 3 — WR-02: 死 jac_sparsity 移除 + 机制描述归位 (1fdce91)

`refine_scipy` 的 `least_squares` 调用移除 `jac_sparsity=make_sparsity(...)`(jac 为 callable 时 scipy 静默忽略,实测 1.15+);`make_sparsity` 全仓零引用一并删除;调用处注释注明稀疏性真实机制 = callable jac 返回 csr + `tr_solver="lsmr"`;`_structure_positions` docstring 同步(不再提结构矩阵消费方)。回归测试:源码防线锁定两个死代码符号不得回潮(注释措辞按本文件自伤规避体例避开被计数串)。

### 生产重跑验证(无提交,产物 gitignored)

临时 `output` → 主仓 output/、`data` → 主仓 data/ 双 symlink(02-04 惯例,npz 内 paths 为相对路径需 imread 派生分辨率):

- 退出码 0,总耗时 7:31.72;
- cam1:test RMS **0.1407px**(train 0.5645px,幸存 116/141,剔 25 帧,第 2 轮收敛);
- cam2:test RMS **0.1361px**(train 0.2389px,幸存 155/155,零剔除);
- 两路均 solver=scipy、refine 生效、阈值 1.0 判 PASS——与基线(02-04)同量级且 test RMS 逐位同值;
- 11 件产物落主仓 `output/intrinsic/`;
- data/ 指纹运行前后逐字节一致:`find data/ -type f -printf '%T@ %s %p\n' | sort | md5sum` = `a183699da79875fc63f7cbfe75874398`(Phase 1 基线同值);
- 两枚 symlink 已在收尾提交前移除,worktree 树干净。

## Verification

- `uv run pytest -q`:**124 passed**(基线 116 + 新增 8:CR-01 三个、WR-01 参数化四例、WR-02 源码防线一个),零失败零跳过;
- 三项修复各自原子提交 fix(02):`5f014d1` / `b7db951` / `1fdce91`;
- 生产重跑退出码 0,test RMS 0.1407/0.1361 与基线同量级(逐位同值);
- data/ 指纹 a183699d… 前后一致,npz/JSON/CSV 契约字段集未动。

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] worktree 基线错误:从根提交起步,缺全部 Phase 2 代码**
- **Found during:** 启动(worktree_branch_check 之后)
- **Issue:** worktree 分支创建在根提交 `dd04569 add_readme` 上(无 src/、无 .planning/、无测试),orchestrator 提示的基线 c17c4dd 对象不存在;分支自身零提交。
- **Fix:** `git merge --ff-only main`(纯快进,非破坏;工作树由 79 个提交补齐)。
- **Files modified:** 无源文件(仅分支指针)
- **Commit:** 无(指针移动)

**2. [Rule 2 - Missing critical] CLI/编排捕获面在 review 元组外加 cv2.error**
- **Found during:** Task 1
- **Issue:** review 建议 CLI 元组 `(ValueError, OSError, ArithmeticError, RuntimeError, AssertionError)` 不含 cv2.error;实测其 MRO 是 `Exception` 直系,不被前三者覆盖;scoring.py 既有捕获面含它。
- **Fix:** 编排层与 CLI 捕获面统一纳入 cv2.error,与评分边界一致,不留新洞。
- **Files modified:** src/stereo_calib/intrinsic/pipeline.py, src/stereo_calib/cli/main.py
- **Commit:** 5f014d1

### 计划内裁量(非偏差)

- `make_sparsity` 选择整体删除(review 给了「删参数修 docstring」或「连函数一起删」两条路):全仓零引用,保留即死代码,与 WR-02 意图一致。
- `write_report_md` 扩展失败渲染:plan 要求「报告如实留痕」,这是载体;JSON/npz/CSV 契约未动(plan 红线)。

## Known Stubs

None — 本任务为缺陷修复,无占位/无未接线数据。

## TDD Gate Compliance

本 quick 任务非 `type: tdd` 计划,不适用 RED/GREEN 门;但每项修复均按「先证缺陷可达、再修、回归测试锁定」执行(CR-01 修复前实测过 AssertionError 整堆栈逃逸)。

## Self-Check: PASSED

- 10 个修改文件全部在位(6 源文件 + 4 测试文件)+ SUMMARY.md 本体;
- 三个 fix(02) 提交 5f014d1 / b7db951 / 1fdce91 均在分支历史中;
- 124 tests passed;生产重跑退出码 0(RMS 0.1407/0.1361);data/ 指纹 a183699d… 前后一致;symlink 已移除。
