---
status: partial
phase: 01-1
source: [01-VERIFICATION.md]
started: 2026-10-23T15:20:00Z
updated: 2026-10-23T15:20:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. 人工抽查角点标注图
expected: 打开 output/train/visualization/ 与 output/test/visualization/ 抽 10-20 张(建议混入 common_fov_pairs 中 route 为 classic 的帧,如 npz source 列为 classic 的 9 帧);角点圆点落在棋盘内角点上;连线顺序一致且 SB 帧与 classic 帧连线走向相同(canonicalize 生效,无 180° 错序);烧录的路径与 [sb/classic] 文字可读
result: [pending]

### 2. 失败帧真假复核(D-02)
expected: 逐张查看 output/train/fail_originals/ 的 80 张 fail_ 前缀原图;区分「真失败」(棋盘出画/严重遮挡/过曝)与「假失败」(棋盘在画面内但未检出);假失败帧应记录下来供后续处理决策
result: [pending]

### 3. CSV 可用性抽查
expected: 用 Excel/LibreOffice 双击打开 usable_cam1_train.csv、fail_train.csv、common_fov_pairs_train.csv;中文列头与内容不乱码(utf-8-sig BOM),筛选/排序可用
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
