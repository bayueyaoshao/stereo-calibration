---
phase: 261023-wp3
plan: 01
subsystem: preprocess (dataset scan / corner detection / pipeline)
tags: [bugfix, code-review, CR-01, WR-01, WR-02, dataset-scan, window-ladder, fail-reasons]
requires:
  - "Phase 1 preprocessing module (src/stereo_calib/{io,corners,preprocess})"
provides:
  - "同划分同名叶子目录的去重警告可达性(CR-01):by_leaf 按「父/叶子」相对路径分组"
  - "_window_ladder 辅助函数:任意 cap >= 3 阶梯恒含下限 3 且严格降序(WR-01)"
  - "_fail_reason 辅助函数:TaskFailed 失败标签按异常类型区分(WR-02)"
affects:
  - "Phase 2 单相机内参标定(消费 preprocess npz/schema 未变)"
  - "Phase 3 双目标定(common_fov_pairs 配对键 group_id 契约未变)"
tech-stack:
  added: []
  patterns:
    - "路径键分组(dict 键 = split 相对路径 as_posix)"
    - "可测辅助函数收拢阶梯/标签生成逻辑"
key-files:
  created: []
  modified:
    - src/stereo_calib/io/dataset.py
    - src/stereo_calib/corners/detect.py
    - src/stereo_calib/preprocess/pipeline.py
    - tests/test_io_dataset.py
    - tests/test_corners.py
    - tests/test_preprocess_pipeline.py
decisions:
  - "CR-01 采用路径键分组而非「叶子名 + 碰撞检测」:实现更小,同 image_id 去重复用 scan_dataset 既有逻辑(警告 + 跳过后出现者)"
  - "WR-02 用「消息含『照片读取失败』的 ValueError」识别读图失败,与 imread_checked 既有错误消息耦合,零新异常类型"
metrics:
  duration: "~8.5 min"
  completed: 2026-10-23T15:47:54Z
---

# Phase 261023-wp3 Plan 01: Phase 1 Code Review CR-01/WR-01/WR-02 Fixes Summary

修复 phase 1 code review 三个高价值发现:dataset 扫描按「父/叶子」路径分组使同划分叶子重名的去重警告真实可达(CR-01),窗口阶梯补 3 修复偶数 cap 漏试下限(WR-01),TaskFailed 失败标签按异常类型诚实区分(WR-02);npz/fail CSV schema 与 image_id/group_id 契约均未变,当前数据(188+15 组)行为与修复前一致。

## What Was Built

### Task 1 — CR-01: dataset.py 分组键加入父目录路径 (e1076fb)

`_scan_split` 的 `by_leaf` 分组键从叶子目录名(`jpg.parent.name`)改为 split 相对路径
`jpg.parent.relative_to(split_dir).as_posix()`(「父/叶子」)。同划分不同父目录下的
同名叶子目录现在成为两个独立组,产生的同 `image_id` 由 `scan_dataset` 既有去重逻辑
接管:记「叶子目录重名」中文警告并跳过后出现者——修复前该场景被 `by_leaf` 静默合并,
跨拍摄 cam1/cam2 错配成毒对直接流入 `common_fov_pairs`。

- 空目录警告循环成员判断同步改用同形路径键;单边缺文件警告消息现打印相对路径(更精确)
- 记录构造循环从排序后的键取 `leaf = Path(key).name`,`image_id`/`group_id` 格式不变
- 行为保持:叶子名全局唯一时路径键分组与原叶子名分组完全等价;键排序因
  `<ts>_<us>` 以 `<ts>` 开头而给出相同记录顺序
- 回归测试 `test_same_split_duplicate_leaf_dirs_warn_and_keep_first`:dup_leaf 两父目录,
  断言警告触发、恰留先出现者 2 条、group_id 契约不变

### Task 2 — WR-01: 窗口阶梯恒降到 3 (45a6b75)

新增模块级 `_window_ladder(cap)`:`list(range(cap, 2, -2))` 不含 3 时补上 3。
修复前偶数 cap(如 4、6)的阶梯是 `[4]`/`[6, 4]`,窗口 3 永不试,击穿文档化不变量
「往下每级减 2 试到 3」与 `_recover_grid_defects` 尾部 assert 的前提。奇数 cap 输出
逐字节不变。`_recover_grid_defects` 改调该辅助函数,其余(含 `_subpix_refine` 源码防线)未动。

- 回归测试:cap∈{3,4,5,6,11,12} 参数化断言恒含 3/严格降序/顶格 cap/min==3,
  另加精确值用例锁定 cap=11 → [11,9,7,5,3](奇数不变)、cap=4 → [4,3](偶数补 3)

### Task 3 — WR-02: TaskFailed 失败标签按异常类型区分 (df044e0)

新增模块级 `_fail_reason(error)`:消息含「照片读取失败」的 ValueError(imread_checked
的读图错误)标「读图失败:{原文}」;其余 worker 异常(窗口断言 AssertionError、检测器
cv2.error、传输错误等)标「检测失败({类型名}):{原文}」。`run_preprocess` 消费循环
改调该函数。修复前所有异常一律误标「读图失败」,误导 D-02 人工复核方向。fail CSV 列
schema 与 meta.fail_reasons 结构未变;坏 JPEG 帧 e2e 锁定(test_parallel.py:181)保持绿色。

- 回归测试:读图 ValueError/AssertionError/RuntimeError 三用例锁定前缀分类与原文保留

## Verification Results

- `uv run pytest -q`:**61 passed**(含全部既有测试与 10 个新回归用例)
- npz 八数组 schema:`test_npz_schema_and_masked_rows` 绿,schema 未变
- fail CSV 列 schema:`test_fail_csv_contains_noise_frame_with_reason` 等绿,未变
- `git diff 7d045f7..HEAD -- data/`:空,data/ 未被触碰
- 三个原子提交均以 `fix(01)` 开头:`git log --oneline -3` 确认
- 真实数据冒烟(可选项,已跑):188 组 train + 15 组 test 全部扫描入册,
  cam1/train 141/188、cam2/train 155/188、两路 test 15/15,公共对 112+15,
  无任何「重名/重复」警告——当前数据行为与修复前一致(冒烟产物已清理,
  data 以只读符号链接挂入后移除)

## Commits

| Task | Commit | Files |
| ---- | ------ | ----- |
| 1 (CR-01) | e1076fb | src/stereo_calib/io/dataset.py, tests/test_io_dataset.py |
| 2 (WR-01) | 45a6b75 | src/stereo_calib/corners/detect.py, tests/test_corners.py |
| 3 (WR-02) | df044e0 | src/stereo_calib/preprocess/pipeline.py, tests/test_preprocess_pipeline.py |

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None.

## Self-Check: PASSED

- src/stereo_calib/io/dataset.py / corners/detect.py / preprocess/pipeline.py:FOUND(均已提交)
- tests/test_io_dataset.py / test_corners.py / test_preprocess_pipeline.py:FOUND(均已提交)
- Commits e1076fb / 45a6b75 / df044e0:FOUND in `git log`
