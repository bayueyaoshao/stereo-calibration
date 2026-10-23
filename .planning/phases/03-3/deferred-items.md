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
