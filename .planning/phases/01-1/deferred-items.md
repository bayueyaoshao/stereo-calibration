# Deferred Items — Phase 01-1

Out-of-scope discoveries logged during execution (not fixed, per executor scope boundary).

## 2026-10-23 (found during 01-02, Task 2)

- **Pre-existing ruff debt from 01-01 (9 errors, present at commit 4bd253c):**
  - `src/stereo_calib/cli/main.py:34` — B008 `typer.Option` in argument defaults.
    This is the idiomatic and required typer pattern (FastAPI/typer DI convention);
    typical resolution is a per-file ignore or repo ruff config
    (`[tool.ruff.lint.flake8-bugbear] extend-immutable-calls = ["typer.Option"]`),
    not a code change.
  - `tests/test_preprocess_pipeline.py:47-50` — 8 × C408 `dict()` calls in the
    `_VIEWS` fixture table (rewrite as literals). 01-02 kept the existing file's
    pattern untouched (only source/route assertions were relaxed) and used dict
    literals in its own new `tests/test_parallel.py` to avoid adding new debt.
  - Full-repo `ruff check .` was failing before 01-02 started; 01-02's own
    changed/created files are ruff-clean.

## 2026-10-23 (found during 01-03, Task 2 / REQUIREMENTS.md bookkeeping)

- **REQUIREMENTS.md checkbox gap from 01-01:** 01-01-SUMMARY.md frontmatter records
  `requirements-completed: [PREP-01, PREP-02, PREP-03, PREP-05, INFRA-01, INFRA-02,
  INFRA-03]`, but REQUIREMENTS.md at the start of 01-03 only had PREP-02 checked
  (by 01-02). PREP-01/PREP-03/PREP-05/INFRA-01/INFRA-02 remain unchecked despite
  being implemented and test-locked by 01-01. 01-03 marked only its own plan's
  requirements (PREP-04/PREP-06/INFRA-03) to stay scope-bounded; the rest should
  be back-filled by the orchestrator or phase verifier.
- Note: ruff C408 debt in `tests/test_preprocess_pipeline.py:47-50` still stands
  (now exactly 4 × C408 after 01-03's edits kept the `_VIEWS` table untouched);
  01-03's own changed/created files are ruff-clean.
