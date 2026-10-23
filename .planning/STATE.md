---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-23T19:24:17.170Z"
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 11
  completed_plans: 7
  percent: 64
---

# STATE: 双目相机标定生产工具

## Project Reference

- **Core Value:** 快速、可靠地产出经过测试集指标验证的双目标定结果(内参 + 双目外参 R/T)
- **Current Focus:** Phase 02 — 模块2-内参标定
- **Roadmap:** .planning/ROADMAP.md
- **Requirements:** .planning/REQUIREMENTS.md

## Current Position

Phase: 02 (模块2-内参标定) — EXECUTING
Plan: 1 of 4

- **Phase:** 3
- **Plan:** Not started
- **Status:** Ready to execute
- **Progress:** ▢▢▢ 0/3 phases complete

```
[----------] 0%  Phase 1 of 3
```

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Requirements shipped | 17 | 0 |
| Phases complete | 3 | 0 |
| 内参 test RMS | < 1.0 px | — |
| 双目距离残差 RMS | < 5 mm | — |

## Accumulated Context

### Decisions

- 角点:SB 主路线(严禁二次亚像素)+ classic 兜底(窗口从本帧方格像素边长推导)
- 内参:M2 模型,calibrateCameraExtended 初值 + scipy least_squares 精化,RMS float64 回代
- 双目:禁用 cv2.stereoCalibrate;PnP-median 初值 + 外参精化
- 双目验证指标:test 集最远两角点物理距离残差 RMS,真值 √(10²+7²)×25mm = √149×25 ≈ 305.16mm(2026-10-24 修正原 307.45 算术误差)
- 参考工程只读:`/home/lx/Works/src/lx/github/intrinsic-calibration`、`/home/lx/Works/src/bolight_robot/vvd_daq/multi_cameras`

### TODOs (from roadmap)

- Phase 1 规划与实现(含 INFRA 工程基础)

### Blockers

- None

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261023-wp3 | 修复 phase 1 code review 三个高价值发现(CR-01 分组键、WR-01 窗口阶梯、WR-02 失败标签) | 2026-10-23 | de3daae | [261023-wp3-phase-1-code-review-cr-01-wr-01-wr-02](./quick/261023-wp3-phase-1-code-review-cr-01-wr-01-wr-02/) |
| 261024-cr02 | 修复 phase 2 code review CR-01/WR-01/WR-02(异常收口三层防线、配置类型守卫、jac_sparsity 死代码),124 测试,生产重跑逐位复现 0.1407/0.1361 | 2026-10-24 | 8cd1231 | [261024-cr02-phase2-review-critical-warning](./quick/261024-cr02-phase2-review-critical-warning/) |

## Session Continuity

- **Last action:** Phase 2 complete(内参 test RMS cam1 0.1407px / cam2 0.1361px,双路 PASS)+ quick 261024-cr02 修复收口 (2026-10-24)
- **Next action:** `/gsd-plan-phase 3`
- **Notes:** data/ 指纹 a183699d 全程不变;Phase 2 产物在 output/intrinsic/(intrinsics.json×2 + report.md + per_frame CSV×2);晨间人工项:01-HUMAN-UAT(3) + 02-HUMAN-UAT(2:剔除帧抽查、内参锚点评审)

---
*Initialized 2026-10-23 during roadmap creation*
