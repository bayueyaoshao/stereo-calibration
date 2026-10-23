---
phase: 01-1
reviewed: 2026-10-23T15:05:23Z
depth: standard
files_reviewed: 17
files_reviewed_list:
  - src/stereo_calib/cli/main.py
  - src/stereo_calib/corners/canonical.py
  - src/stereo_calib/corners/detect.py
  - src/stereo_calib/corners/window.py
  - src/stereo_calib/io/config.py
  - src/stereo_calib/io/dataset.py
  - src/stereo_calib/preprocess/artifacts.py
  - src/stereo_calib/preprocess/parallel.py
  - src/stereo_calib/preprocess/pipeline.py
  - src/stereo_calib/visualization/__init__.py
  - src/stereo_calib/visualization/overlay.py
  - tests/conftest.py
  - tests/test_corners.py
  - tests/test_io_config.py
  - tests/test_io_dataset.py
  - tests/test_parallel.py
  - tests/test_preprocess_pipeline.py
findings:
  critical: 1
  warning: 5
  info: 8
  total: 14
status: issues_found
---

# Phase 01-1: Code Review Report

**Reviewed:** 2026-10-23T15:05:23Z
**Depth:** standard
**Files Reviewed:** 17
**Status:** issues_found

## Summary

Reviewed the Phase 1 preprocessing module at standard depth: CLI entry, corner
detection (SB + classic fallback), canonicalization, window derivation, config
loading, dataset scanning, artifact writing, parallel execution, orchestration
pipeline, visualization overlay, and all six test files.

Overall the code is unusually defensive and well-documented: yaml safe_load,
`np.load(allow_pickle=False)`, image_id path-traversal whitelist, config-level
output/data disjointness check, exception-envelope parallelism with a batch
completeness check, and data-read-only test locks are all present and coherent.
The findings below are edge-condition and contract-drift defects, not missing
defenses.

The one Critical finding is a silent data-merge bug in the dataset scanner:
when two different parent directories contain leaf directories with the same
name (the documented duplicate-image_id scenario), the `by_leaf` dict keyed by
leaf name alone merges them before the dedup warning can ever see a duplicate —
one image is silently overwritten, or worse, cam1/cam2 from two different
captures get paired as one stereo group. The docstring explicitly promises a
warning for this case; the implementation makes that warning unreachable within
a single split.

Known deferred debt (9 ruff findings: 8×C408, 1×B008, per
`deferred-items.md`) is not re-reported.

## Critical Issues

### CR-01: Same-split duplicate leaf names silently merge groups and defeat the documented dedup warning

**File:** `src/stereo_calib/io/dataset.py:79-88` (contract at 125-141)
**Issue:** `by_leaf` is keyed by `jpg.parent.name` (leaf directory name) only.
If two different parent directories under one split contain leaf directories
with the same name (e.g. `train/<ts1>/g/cam1.jpg` and `train/<ts2>/g/cam1.jpg`),
`by_leaf.setdefault(leaf, {})[cam] = jpg` silently overwrites the first path
with the second — no warning is emitted and one image vanishes from the
calibration set. Worse for a stereo tool: `train/<ts1>/g/cam1.jpg` and
`train/<ts2>/g/cam2.jpg` merge into a single "group g", pairing images from two
different captures; that poisoned pair flows into `common_fov_pairs_*.csv` and
would corrupt Phase 3 stereo calibration input.

`scan_dataset`'s own docstring promises: "叶子目录重名会撞编号:重复记中文
警告并跳过后出现者" — but the image_id dedup (lines 136-141) can only trigger
across splits, because within a split the collision is already merged in
`by_leaf` and only one record is ever produced. The protective warning is
dead code for the same-split case, i.e. exactly the case the project's
"绝不静默丢结果" invariant exists for.

**Fix:** Key by the full relative directory (or detect the collision) instead
of leaf name alone; emit the promised warning and skip the later occurrence:

```python
by_leaf: dict[str, dict[int, Path]] = {}
for jpg in sorted(split_dir.glob("*/*/cam*.jpg")):
    ...
    key = str(jpg.parent.relative_to(split_dir))  # or jpg.parent.name + collision check
    if jpg.parent.name in by_leaf and str(by_leaf[jpg.parent.name].get(1, jpg.parent)) != ...:
        # simplest: keep leaf as pairing key but warn on cross-parent reuse
        ...
```

A minimal fix that preserves the group_id contract: track
`leaf -> parent_dir` and warn + skip when a leaf name reappears under a
different parent, mirroring the existing duplicate-image_id warning wording.
Add a test with `data/train/<ts1>/<leaf>/cam1.jpg` +
`data/train/<ts2>/<leaf>/cam1.jpg` asserting the warning fires and only one
record is kept.

## Warnings

### WR-01: Recovery window ladder never reaches window 3 when cap is even (documented invariant violated)

**File:** `src/stereo_calib/corners/detect.py:161-162` (and 172)
**Issue:** `ladder = list(range(cap, 2, -2))`. For odd `cap` (e.g. 11) the
ladder is `[11, 9, 7, 5, 3]` as documented; for even `cap` (e.g. 12 or 4) it
is `[12, 10, 8, 6, 4]` — window 3 is never tried, contradicting the docstring
"往下每级减 2 试到 3" and the assert message "下限 3 恒在阶梯里". Verified:
`range(4, 2, -2) == [4]`, `range(6, 2, -2) == [6, 4]`. A defective frame that
would only recover at window 3 is left with defects whenever
`int(np.percentile(pool, 2) // 2) - 1` happens to be even (statistically ~half
of recovery invocations with cap > 3).

**Fix:**
```python
ladder = list(range(cap, 2, -2))
if ladder[-1] != 3:
    ladder.append(3)  # 保证下限 3 恒在阶梯里(偶数 cap 时 range 不含 3)
```
Or step the ladder over odd values only: `wins = [w for w in range(cap, 2, -1) if w % 2 == 1] or [3]` plus cap itself if even. Either way, add a unit test
parametrized over even and odd caps asserting `3 in ladder`.

### WR-02: TaskFailed reason hardcoded as "读图失败" for every worker exception

**File:** `src/stereo_calib/preprocess/pipeline.py:171`
**Issue:** `except`-all envelope branch records
`f"读图失败:{outcome.error}"`. The envelope can carry any worker exception —
`AssertionError` from `derive_window_frame` (WR-03), `cv2.error` from the
detectors, pickling/transport errors — all get labeled "读图失败" (image read
failed) in `fail_*.csv` and `meta.json.fail_reasons`. This is the D-02
human-review entry point: a reviewer told "read failed" for a frame that
actually failed a window-safety assertion will hunt for filesystem problems in
the wrong place, and may "confirm" a false root cause.

**Fix:** Label by exception type, or at minimum use a generic wording:
```python
reason = (f"读图失败:{outcome.error}"
          if isinstance(outcome.error, (OSError, ValueError)) and "照片读取失败" in str(outcome.error)
          else f"处理失败({type(outcome.error).__name__}):{outcome.error}")
_record_fail(slot, rec, reason)
```
Since `imread_checked` already raises ValueError with "照片读取失败" in the
message, matching on that (or having `detect_one` raise a dedicated
`ImageReadError` type) keeps read failures accurately labeled while everything
else gets an honest generic label.

### WR-03: detect_sb can discard valid SB detections via a window assertion on a value SB never uses

**File:** `src/stereo_calib/corners/detect.py:58`
**Issue:** `win_size = derive_window_frame(sq_px, cfg.subpix.min_win)` calls
the hard-asserting derivation purely "for artifact structural alignment" (the
docstring says SB does not use a refinement window). If SB successfully
detects a board whose estimated square size is too small for a safe classic
window (e.g. `sq_px <= 2*min_win+1`, a small/far board), the `assert` fires,
the exception is enveloped, and a frame with perfectly good subpixel SB
corners is dropped entirely and recorded as a failure (with the misleading
"读图失败" label per WR-02). The head-of-file defense rationale for the
assertion applies to *refinement*; SB output never passes through refinement.

**Fix:** Record the derived value without the safety assert on the SB path
(the assert must stay on the classic path where the window is actually used):
```python
win_size = max(int(cfg.subpix.min_win), int(sq_px // 2) - 1)  # 仅记录口径,SB 不做二次精化,断言不适用
```
Or record `np.nan`/0 for the SB route and document that `win_size` is
meaningful only for the classic route — but then the npz audit contract
(see WR-04) needs updating to match.

### WR-04: In corner-granularity mode the recorded win_size/sq_px pair can violate the documented npz audit assertion

**File:** `src/stereo_calib/corners/detect.py:197-202`; contract at
`src/stereo_calib/preprocess/artifacts.py:13`
**Issue:** The npz layout documents `win_size` as "供审查窗口断言
2*win+1 < sq_px". In `corner` granularity mode, `win_size = int(wins.max())`
is the largest per-corner window (derived from the largest *local* spacing,
typically the near side under perspective), while `sq_px` is the frame-level
conservative `min(median_h, median_v)`. The per-corner assertion correctly
holds against each corner's *local* spacing, but the recorded pair can violate
`2*win_size+1 < sq_px` by construction: e.g. frame sq_px=20 (frame win 9,
19<20 OK) with a near-side local spacing of 26 → that corner's win 12 →
recorded win_size 12 → 2*12+1 = 25 ≥ 20. Any downstream or human audit of the
npz using the documented rule will flag valid frames as defective in corner
mode. (Default config is `frame`, so the default path is unaffected.)

**Fix:** Either record the frame-level conservative window alongside the
per-corner max (e.g. store the frame-derived value in `win_size` and the max
per-corner value in a separate meta field), or amend the npz contract doc to
state that in corner mode the assertion is `2*win+1 < local spacing at that
corner` and per-corner windows are not recoverable from the npz. Also add a
test locking whichever contract is chosen.

### WR-05: cv2_threads accepts any int including negatives — validation gap inconsistent with the module's own rigor

**File:** `src/stereo_calib/io/config.py:290-292`
**Issue:** Every other numeric field is validated with an explicit bound and a
Chinese error (`max_workers >= 1`, `min_win >= 1`, `max_iter >= 1`,
`epsilon > 0`, `jpeg_quality` in 1..100, `sb_flags >= 0`). `cv2_threads` is
only checked for int-not-bool; a negative value (a plausible yaml typo) passes
`load_config` and its semantics are silently delegated to
`cv2.setNumThreads` internals in both the CLI callback and every worker —
behavior then depends on the OpenCV build rather than on config validation,
and the config docstring's own claim "0 = 关闭" implies the valid domain is
non-negative.

**Fix:**
```python
cv2_threads = raw.get("cv2_threads", StereoCalibConfig.cv2_threads)
if not isinstance(cv2_threads, int) or isinstance(cv2_threads, bool) or cv2_threads < 0:
    _fail("cv2_threads", cv2_threads, "必须是 >= 0 的整数(OpenCV 内部线程数,0 = 关闭)")
```

## Info

### IN-01: Empty split still produces header-only CSVs and the split directory, contradicting the inline comment

**File:** `src/stereo_calib/preprocess/pipeline.py:227-228` vs 250-253
**Issue:** The comment says "空划分(如缺 test 目录)不产出文件", and npz/usable
CSV are indeed skipped via `continue`, but `write_fail_csv` /
`write_common_pairs_csv` sit outside the cam loop and run for every split — a
missing test split still yields `output/test/` with header-only
`fail_test.csv` and `common_fov_pairs_test.csv`. Either the comment or the
behavior should change; header-only files are harmless but the comment is what
future maintainers will trust.
**Fix:** Move the two writes inside an `if any(groups[(split, cam)]["records"] for cam in _CAMS):` guard, or fix the comment to say only npz/usable CSV are skipped.

### IN-02: slot["fail_rows"] is write-only dead storage

**File:** `src/stereo_calib/preprocess/pipeline.py:99, 157`
**Issue:** `_record_fail` appends to `slot["fail_rows"]`, but the fail CSV is
written from `fail_by_split` and meta uses `slot["fail_reasons"]`; grep
confirms `fail_rows` is never read. Dead code that suggests a per-(split,cam)
fail artifact that does not exist.
**Fix:** Delete the `"fail_rows": []` initialization and the append (the split-level `fail_by_split` row is the same dict object, so nothing is lost).

### IN-03: npz stores cwd-relative image paths while tasks use resolved absolute paths

**File:** `src/stereo_calib/preprocess/pipeline.py:92, 139-143`
**Issue:** `slot["paths"]` records `str(r.path)` (relative to the scan-time
cwd), while `detect_one` receives `str(rec.path.resolve())`. Phase 2/3
consumers that open `paths` from a different working directory (artifacts
record absolute resolved paths in the CLI summary, so the mix invites this)
will fail to locate originals. Determinism is fine; portability of the
artifact is not.
**Fix:** Store `str(r.path.resolve())` in the slot paths (and usable/common CSV path columns) so every artifact is cwd-independent.

### IN-04: _CAM_STEM accepts cam01/cam0002 as cam1/cam2

**File:** `src/stereo_calib/io/dataset.py:40, 82-88`
**Issue:** `^cam(\d+)$` + `int(m.group(1))` accepts `cam01.jpg` as camera 1,
deviating from the documented exact `cam1.jpg`/`cam2.jpg` contract; and if a
leaf contains both `cam1.jpg` and `cam01.jpg`, the later-sorted one silently
overwrites the earlier in `by_leaf` (same silent-overwrite mechanism as
CR-01, narrower trigger).
**Fix:** Match exact names: `_CAM_STEM = re.compile(r"^cam([12])$")` and drop the `_SUPPORTED_CAMS` int conversion (keep the unsupported-name warning branch for cam3 etc., which then covers cam01 too).

### IN-05: sb_flags is an unvalidated arbitrary bitmask

**File:** `src/stereo_calib/io/config.py:209-211`
**Issue:** `classic_flags` has a name whitelist translated to cv2 constants,
but `sb_flags` accepts any `>= 0` int (e.g. 999). Invalid bits only surface
inside workers as per-frame failures (labeled "读图失败" per WR-02) instead of
being rejected at load time with a field-named Chinese error — inconsistent
with the config layer's stated design (名字在配置层已白名单校验).
**Fix:** Either expose named options like classic_flags (SB has few flags), or at least validate against the known-bit mask and `_fail` on unknown bits.

### IN-06: Source-guard test depends on cwd-relative source path

**File:** `tests/test_corners.py:117`
**Issue:** `pathlib.Path("src/stereo_calib/corners/detect.py").read_text()`
resolves against the process cwd; running pytest from any directory other than
the repo root (e.g. `pytest /abs/path/to/tests`) fails the guard test with
FileNotFoundError, not a defense failure.
**Fix:** Anchor to the module file: `pathlib.Path(detect.__file__).with_name("detect.py")` — note this imports `detect` but must keep counting only the target file's text, which it does.

### IN-07: meta.json path derived twice with potential drift

**File:** `src/stereo_calib/preprocess/pipeline.py:248`
**Issue:** `save_corners_npz` internally writes `out_dir / f"{name}.meta.json"`
but returns only the npz path; the caller reconstructs the meta path
independently. If the sidecar naming ever changes, the CLI artifact listing
silently points at a nonexistent file.
**Fix:** Have `save_corners_npz` return both paths (tuple or small dataclass) and use the returned values in the `artifacts` list.

### IN-08: No guard against output_root being a filesystem root

**File:** `src/stereo_calib/io/config.py:248-271`; consumer at
`src/stereo_calib/preprocess/pipeline.py:123-124`
**Issue:** `_check_output_data_disjoint` only protects the two data dirs (as
documented). `output_root: "/"` (or a drive root) with data dirs elsewhere
passes validation, and `shutil.rmtree("/train", ignore_errors=True)` is then
attempted — ignore_errors plus later write failures turn a config typo into a
confusing partial run rather than a load-time Chinese error. Low likelihood,
but this sits on the project's own top-priority axis (data/write safety).
**Fix:** In `_check_output_data_disjoint` (or load_config), additionally reject `out == out.anchor` / `out.parent == out` (filesystem root) with the standard `_fail`-style message.

---

_Reviewed: 2026-10-23T15:05:23Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
