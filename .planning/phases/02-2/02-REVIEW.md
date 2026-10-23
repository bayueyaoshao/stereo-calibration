---
phase: 02-2
reviewed: 2026-10-23T18:07:26Z
depth: standard
files_reviewed: 15
files_reviewed_list:
  - src/stereo_calib/cli/main.py
  - src/stereo_calib/intrinsic/__init__.py
  - src/stereo_calib/intrinsic/artifacts.py
  - src/stereo_calib/intrinsic/metrics.py
  - src/stereo_calib/intrinsic/models.py
  - src/stereo_calib/intrinsic/observation.py
  - src/stereo_calib/intrinsic/pipeline.py
  - src/stereo_calib/intrinsic/scoring.py
  - src/stereo_calib/intrinsic/solve.py
  - src/stereo_calib/io/config.py
  - tests/test_intrinsic_config.py
  - tests/test_intrinsic_models.py
  - tests/test_intrinsic_pipeline.py
  - tests/test_intrinsic_scoring.py
  - tests/test_intrinsic_solve.py
findings:
  critical: 1
  warning: 2
  info: 6
  total: 9
status: issues_found
---

# Phase 02-2: Code Review Report

**Reviewed:** 2026-10-23T18:07:26Z
**Depth:** standard
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Reviewed the Phase 2 intrinsic-calibration module slice at standard depth: solver engine (OpenCV init + scipy refine), float64 metrics, M2 model math, view loading, test-frame pose-only scoring, orchestration/pipeline, artifact writers, config parsing, CLI entry, and the five test files.

Overall quality is high and the advertised defense lines largely hold up under adversarial tracing: M2 projection matches OpenCV conventions (tangential nesting verified against OpenCV formula and locked by parity tests); float64 back-substitution is the single metrics outlet; the row-order contract is enforced end-to-end (load_views → drop loop → final solve → npz/CSV alignment all index from the same `image_ids` order); `allow_pickle=False` is never bypassed; the 15-field intrinsics.json contract is asserted at write time and in tests; the data-read-only and rmtree-scoping guarantees check out (`load_config` disjointness guard + `output_root/intrinsic`-only rmtree).

However, tracing the failure paths uncovered one genuine contract break: the "single frame failure never aborts the batch / never throws" guarantee in `score_pose_only` has a reachable hole (`AssertionError` from the Z>0 guard in `models.project` escapes both the scoring `except` tuple and the CLI's `(ValueError, OSError)` net), plus two warnings: a `TypeError` crack in config validation for container-typed yaml values, and a scipy API misuse (`jac_sparsity` is silently ignored when `jac` is callable — verified against the installed scipy 1.15.3 source and empirically). Findings below; line numbers verified against working tree at commit 7396767.

## Critical Issues

### CR-01: `score_pose_only` no-throw contract is broken — a mid-refinement Z<=0 AssertionError escapes and aborts the whole production run

**File:** `src/stereo_calib/intrinsic/scoring.py:103-117` (with `src/stereo_calib/intrinsic/models.py:153-157`, `src/stereo_calib/cli/main.py:87-91`, `src/stereo_calib/intrinsic/solve.py:396-405`)

**Issue:** The module contract states "任何失败路径返回 (NaN, None, 中文备注),绝不抛异常" and the pipeline depends on it (`_score_test_set` has no per-frame try — by design, since the scorer promises not to throw). But the refinement closure at scoring.py:103-106 calls `models.project`, whose Z>0 guard (models.py:153-157) raises `AssertionError`. Inside `least_squares`, `theta` is optimizer-controlled: a trial step (or even `theta0` itself, if a pathological detected-but-garbage corner set makes `cv2.solvePnP` return `ok=True` with a nonsense pose) can put board points behind the camera. `AssertionError` is a subclass of none of `(ValueError, ArithmeticError, RuntimeError, cv2.error)`, so it escapes `score_pose_only` (verified: `AssertionError` is not caught by those types; empirically confirmed that exceptions raised inside a residual function propagate out of `scipy.optimize.least_squares`). From there nothing catches it: `_score_test_set` → `_calibrate_one_cam` → `run_intrinsic` → CLI `intrinsic()` catches only `(ValueError, OSError)` (main.py:87). Result: one bad test frame kills the entire two-camera overnight run with a raw stack trace — violating three explicitly written contracts at once (scoring's "绝不抛异常", pipeline's "单帧失败不中断整批" T-02-06, CLI's "坏输入收口...不吐堆栈"), and leaving cam2 without artifacts and neither camera with a report.md.

The same hole exists in the train-refinement path: `solve.refine_scipy`'s docstring (solve.py:364-367) says numerical exceptions are deliberately re-raised "交编排层中文收口(02-03)", but the 编排层 and CLI do not catch `AssertionError`/`ArithmeticError`/`RuntimeError` either — the declared hand-off target does not exist. The scoring.py:116 comment claims the except list is safe because "形状/参数错误已在入口防线拦下" — true for shape/param contracts, false for the Z>0 guard, which is data-dependent on pose values and unknowable at entry.

Probablity is low (the real production run passed), but the trigger — a detected-but-degenerate frame — is exactly the input class this module's whole design exists to survive, and the blast radius is a crashed unattended run plus partial/missing artifacts.

**Fix:**

```python
# scoring.py — extend the except tuple; the only data-dependent assertion
# reachable mid-solve is the Z>0 guard in models.project (shape/param
# contracts are pre-guarded at the entry, so this cannot mask code bugs):
    except (ValueError, ArithmeticError, RuntimeError, AssertionError, cv2.error) as e:
        return _failure(f"只估位姿求解失败:{type(e).__name__}——{e}")
```

Additionally, honor solve.py's declared hand-off by catching the numeric exception family at the orchestration boundary so a refine-phase blow-up produces the Chinese one-liner instead of a stack trace:

```python
# cli/main.py — intrinsic() (and symmetrically preprocess() if applicable)
    except (ValueError, OSError, ArithmeticError, RuntimeError, AssertionError) as e:
```

A regression test should drive `score_pose_only` toward a pose with Z<=0 (e.g., monkeypatch/seed a trial path or call the residual closure directly) and assert a `PoseScore` failure is returned instead of a raise.

## Warnings

### WR-01: Config membership checks crash with TypeError on container-typed yaml values — stack trace instead of the Chinese one-liner

**File:** `src/stereo_calib/io/config.py:232-235` (`window_granularity`), `src/stereo_calib/io/config.py:308-312` (`refine.method`)

**Issue:** `granularity not in _ALLOWED_GRANULARITY` and `method not in _ALLOWED_REFINE_METHODS` perform membership tests against `set` objects without first checking the value is a (hashable) string. A hand-edited yaml typo like `method: [trf]` or `window_granularity: {a: b}` yields a list/dict, and `list in set` raises `TypeError: unhashable type: 'list'` (verified on Python 3). `TypeError` is caught by neither the CLI's `(ValueError, OSError)` net nor the callback's, so the user gets a full stack trace — the exact outcome the CLI contract ("中文一句 + 退出码 1,不吐堆栈") forbids, on the config-malformation path the config layer exists to make friendly. Every other field in this file uses `isinstance` first and is immune; these two membership checks are the only cracks.

**Fix:**

```python
    method = section.get("method", cfg.method)
    if not isinstance(method, str) or method not in _ALLOWED_REFINE_METHODS:
        _fail("intrinsic.refine.method", method,
              f"只能是 {' 或 '.join(sorted(_ALLOWED_REFINE_METHODS))}(...)")
```

Same pattern for `window_granularity` in `_parse_subpix`. Add parametrized cases (`("method", "[trf]")`) to `tests/test_intrinsic_config.py`.

### WR-02: `jac_sparsity` is silently ignored by scipy when `jac` is callable — `make_sparsity` is dead computation resting on a factually wrong docstring claim

**File:** `src/stereo_calib/intrinsic/solve.py:396-405` (the `least_squares` call), `src/stereo_calib/intrinsic/solve.py:311-321` (`make_sparsity` + docstring)

**Issue:** `refine_scipy` passes `jac=make_jacobian_fn(...)` (callable) together with `jac_sparsity=make_sparsity(...)`. In scipy (verified against installed 1.15.3 source and empirically — a deliberately wrong-shaped `jac_sparsity` is accepted without error), `jac_sparsity` is consumed only in the finite-difference branch of `least_squares`; with a callable `jac` it is never validated or used. The real sparsity exploitation comes from the callable returning a `csr_matrix` under `tr_solver="lsmr"` — which works, so there is no correctness bug today. But: (a) `make_sparsity` builds a `2VP × (9+6V)` COO→CSR matrix on every refine call for nothing; (b) the `make_sparsity` docstring asserts "结构外的位置必须永远为 0,scipy 按结构分组解子问题" — a mechanism that does not exist in this call configuration. In a solver file that stakes numerical claims on documented mechanics (row-index contract, Rodrigues jacobian provenance, plateau detection), a wrong mechanism claim is a maintenance trap: a future change (e.g., switching `jac` to a string method for debugging) would silently activate `jac_sparsity` semantics that were never actually exercised.

**Fix:** Drop `jac_sparsity=make_sparsity(...)` from the `least_squares` call (and `make_sparsity` itself, or keep it solely for the row/col structure shared with `_structure_positions` if desired), and correct the docstring to attribute sparsity exploitation to "callable jac 返回 csr + tr_solver='lsmr'", not to `jac_sparsity`.

## Info

### IN-01: `compute_metrics` lacks a point-count cross-check between `objp` and `uvs`

**File:** `src/stereo_calib/intrinsic/metrics.py:48-71`

**Issue:** The function validates `uvs` shape and pose shapes with Chinese contract errors, but never checks `uvs_arr.shape[1] == objp_arr.shape[0]`. A mismatch surfaces as a raw numpy broadcast `ValueError` ("operands could not be broadcast together") instead of the module's established named-contract error — the same check `solve._check_common_inputs` does perform (solve.py:81-85). All current callers are aligned via `load_views`, so impact is limited to a cryptic message, but the self-described "唯一指标尺" is the one entry missing the alignment guard its sibling module has.

**Fix:** Add `if uvs_arr.shape[1] != objp_arr.shape[0]: raise ValueError(...)` alongside the existing shape checks, mirroring solve.py's wording.

### IN-02: `project_jacobian` is the only projection-path function without the Z>0 guard

**File:** `src/stereo_calib/intrinsic/models.py:165-202`

**Issue:** `project` (line 153) and `project_point_jacobian` (line 216) both assert all Z>0; `project_jacobian` silently computes inf/nan for Z<=0 input. models.py's own dtype/shape contract says "非法输入抛中文 AssertionError...绝不返回半成品". Today the invariant holds only transitively (in `make_jacobian_fn`, `project_point_jacobian`'s guard fires in the same call before the result is consumed), but the math layer's public surface is inconsistent.

**Fix:** Add the same `if not np.all(pts[:, 2] > 0.0): raise AssertionError(...)` guard at the top of `project_jacobian`.

### IN-03: Editing residue in `refine_scipy` docstring

**File:** `src/stereo_calib/intrinsic/solve.py:364-367`

**Issue:** "数值路径异常(ValueError/ArithmeticError/RuntimeError,含线性代数炸掉)同样不向上抛?——抛。" is a leftover rhetorical self-question from an editing decision, shipped in a docstring. It also documents a hand-off to the 编排层 that, per CR-01, does not actually exist for these exception types. Confusing for maintainers deciding where failures are handled.

**Fix:** Rewrite as a plain statement: "数值路径异常(ValueError/ArithmeticError/RuntimeError/AssertionError)如实上抛,由 CLI 层中文收口(见 main.py 的 except 清单)" — and make the CLI actually catch them (CR-01 fix).

### IN-04: `cv2_threads` is type-validated only — negative values pass, unlike every other numeric field

**File:** `src/stereo_calib/io/config.py:433-435`

**Issue:** All other numeric fields carry range validation (`>= 1`, `> 0`, etc.); `cv2_threads` checks only `isinstance(int) and not bool`, so `cv2_threads: -5` loads cleanly. `cv2.setNumThreads(-5)` does not raise on OpenCV 4.11.0 (verified), but the semantics of negative thread counts are backend-dependent and undocumented in this project. Low impact; a consistency gap in the validation layer.

**Fix:** Add `or cv2_threads < 0` to the `_fail` condition, matching the file's established style.

### IN-05: `write_intrinsics_json` asserts D's length but not params' — `zip` truncates silently

**File:** `src/stereo_calib/intrinsic/artifacts.py:89` (vs. the explicit D assert at lines 82-85)

**Issue:** The writer asserts `d_arr.shape == (5,)` with a dedicated Chinese contract error, but `zip(INTRINSIC_NAMES, params)` would silently truncate if `params` carried more than 9 entries. The current caller is guarded transitively (`intrinsics_matrix` asserts `(9,)`), but the Phase 3 contract writer should assert its own input symmetrically rather than depending on a side effect.

**Fix:** `assert np.asarray(params).ravel().shape == (len(INTRINSIC_NAMES),), "..."` next to the D assert.

### IN-06: `write_residuals_npz` / `write_extrinsics_npz` never cross-check `image_ids` length against the arrays' first dimension

**File:** `src/stereo_calib/intrinsic/artifacts.py:117-152`

**Issue:** Both writers' docstrings promise "行序与调用方幸存视图一一对应(对账列)", but a caller passing a mismatched id list vs. residual/pose rows would produce silently misaligned audit artifacts — precisely the "悄悄算错、不报错" failure class the project's row-order contract (Pitfall 10) exists to prevent. Current callers in `_calibrate_one_cam` are aligned by construction; this hardens the writer to match its own stated contract.

**Fix:** Assert `len(image_ids) == arr.shape[0]` for residuals/per_view_rms/rvecs/tvecs before `np.savez`, with a Chinese contract error naming the arrays.

---

_Notes on deliberate non-findings (checked, judged correct or adjudicated design):_
- _Median-based outlier threshold, `5.0` multiplier, fuse ordering (`floor = max(min_views, (1-frac)·n_initial)`), and survivor re-indexing in `_drop_outliers` traced round-by-round — indexing, fuse semantics, and notes are correct._
- _`models._distort` matches OpenCV's tangential nesting; parity and analytic-vs-numeric jacobian tests lock the math._
- _`LinAlgError ⊂ ValueError` claim in scoring.py:115 is correct (verified MRO)._
- _`_unicode_array` empty-list `"<U1"` trap, `allow_pickle=False` discipline, `image_size` (宽,高) flip, pooled-vs-per-view RMS formulas, and the 15-field JSON contract all verified consistent across modules and tests._
- _Tests: `make_chessboard_image` fixture comes from conftest (outside scope); `_mutate`'s first-hit-key limitation for `outlier.enabled` is documented in the test docstring and does not affect reliability._

_Reviewed: 2026-10-23T18:07:26Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
