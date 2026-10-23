---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-23T16:37:15.626Z"
progress:
  total_phases: 3
  completed_phases: 1
  total_plans: 7
  completed_plans: 3
  percent: 33
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

- **Phase:** 2
- **Plan:** Not started
- **Status:** Executing Phase 02
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
- 双目验证指标:test 集最远两角点物理距离残差 RMS,真值 √(10²+7²)×25mm ≈ 307.45mm
- 参考工程只读:`/home/lx/Works/src/lx/github/intrinsic-calibration`、`/home/lx/Works/src/bolight_robot/vvd_daq/multi_cameras`

### TODOs (from roadmap)

- Phase 1 规划与实现(含 INFRA 工程基础)

### Blockers

- None

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261023-wp3 | 修复 phase 1 code review 三个高价值发现(CR-01 分组键、WR-01 窗口阶梯、WR-02 失败标签) | 2026-10-23 | de3daae | [261023-wp3-phase-1-code-review-cr-01-wr-01-wr-02](./quick/261023-wp3-phase-1-code-review-cr-01-wr-01-wr-02/) |

## Session Continuity

- **Last action:** Completed quick task 261023-wp3: CR-01/WR-01/WR-02 fixes (2026-10-23)
- **Next action:** `/gsd-plan-phase 2`
- **Notes:** data/ 已存在(train 188 组 / test 15 组);granularity=coarse,mode=yolo;Phase 1 complete + 修复后预处理重跑计数不变(141/155/15/15,classic 9,公共对 112/15)

---
*Initialized 2026-10-23 during roadmap creation*
