---
status: partial
phase: 02-2
source: [02-VERIFICATION.md]
started: 2026-10-24T01:20:00Z
updated: 2026-10-24T01:20:00Z
---

## Current Test

[awaiting human testing]

## Tests

### 1. 剔除帧视觉抽查(cam1 离群 25 帧)
expected: 打开 output/intrinsic/per_frame_cam1.csv 中 dropped=True 的 25 帧 + 幸存帧中 4 帧 RMS > 1.0px(最高 4.87px)的图像,对照 output/train/visualization/ 角点标注图;确认被剔帧确有质量缺陷(角点误检/棋盘倾斜退化为平面/遮挡),幸存高误差帧属真实噪声而非系统性错标;report.md 离群表 25 行与 CSV dropped 列一致
result: [pending]

### 2. 内参锚点偏差评审(领域专家判断)
expected: 对照 output/intrinsic/cam{1,2}/intrinsics.json 的 fx/fy(2021–2265,参考带 1800–1950)与 k1(-0.63~-0.75,参考带 -0.5~-0.3);结合 SUMMARY 中 fx↔k1 耦合分析,判断该组内参可否冻结为 Phase 3 固定输入;若否,记录后续处理方向(如 exclude_images 重标)
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
