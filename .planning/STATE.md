---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
last_updated: "2026-10-23T23:30:00.000Z"
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 11
  completed_plans: 11
  percent: 100
---

# STATE: 双目相机标定生产工具

## Project Reference

- **Core Value:** 快速、可靠地产出经过测试集指标验证的双目标定结果(内参 + 双目外参 R/T)
- **Current Focus:** Phase 3 — 模块3-双目标定
- **Roadmap:** .planning/ROADMAP.md
- **Requirements:** .planning/REQUIREMENTS.md

## Current Position

Phase: 3 (模块3-双目标定) — BLOCKED(联合 BA gap-closure 03-05 收敛裁决三变体全灭,失败协议停止)
Plan: 5 of 5 executed

- **Phase:** 3
- **Plan:** 03-05(blocked,失败协议)
- **Status:** Phase 3 gap-closure 裁决完毕,交 orchestrator 决策
- **Progress:** ▩▩▩▩▩ 11/11 plans executed(03-05 联合 BA 引擎落地但裁决全灭:内参漂移全部越诊断界,SC3 达标与内参可信结构性不可兼得)

```
[----------] 0%  Phase 1 of 3
```

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Requirements shipped | 17 | 0 |
| Phases complete | 3 | 0 |
| 内参 test RMS | < 1.0 px | cam1 0.1407 / cam2 0.1361 |
| 双目距离残差 RMS | < 5 mm | 16.93(FAIL,SC3 生产实测;03-05 联合 BA 裁决全灭未落地)|

## Accumulated Context

### Decisions

- 角点:SB 主路线(严禁二次亚像素)+ classic 兜底(窗口从本帧方格像素边长推导)
- 内参:M2 模型,calibrateCameraExtended 初值 + scipy least_squares 精化,RMS float64 回代
- 双目:禁用 cv2.stereoCalibrate;PnP-median 初值 + 外参精化
- 双目验证指标:test 集最远两角点物理距离残差 RMS,真值 √(10²+7²)×25mm = √149×25 ≈ 305.16mm(2026-10-24 修正原 307.45 算术误差)
- 参考工程只读:`/home/lx/Works/src/lx/github/intrinsic-calibration`、`/home/lx/Works/src/bolight_robot/vvd_daq/multi_cameras`
- 03-05 失败协议:联合 BA 三变体(fc/fc_prior/full_prior)在原样流与生产同构流下全部死于内参漂移越诊断界(f 漂 7.7–10.6% vs ±3% 界、主点漂 33–59px vs ±30px 界、D 漂 0.09–0.18 vs ≤0.05 界)——SC3 预演 1.53mm 的收益来自参数漂移而非物理解;joint.enabled 保持 false,引擎代码+测试留库待命
- 03-05 裁决证据:权重网格 w∈{0.1,1,10} 先验拉回力无效(先验行 scale=w/θ0,θ0~2265px,弱于数据项约 5 个数量级);真绑定需 w~1e4–1e5 属调权重凑数,计划明令禁止

### TODOs (from roadmap)

- Phase 1 规划与实现(含 INFRA 工程基础)

### Blockers

- SC3 生产验收 FAIL:test 距离残差 16.93mm ≫ 5mm(03-04-SUMMARY 完整诊断)。根因:两路独立内参在双目精度下互不兼容(相对位姿逐对散布中位 2.5°/max 12.2°)+ 支架跨帧摆动;刚性外参+固定内参模型在本数据不可达
- 联合 BA 预案已被 03-05 收敛裁决否决:三变体全灭(漂移越界为致命判据,SC3 达标与内参物理可信结构性不可兼得)。可选后续(需新计划,交 orchestrator/用户决策):(a) Phase 2 协方差作先验强度的内参不确定度传播;(b) 只放主点(冻结 f/D)的第四变体消融;(c) 接受 rigid 16.93mm 现状改验收口径(需用户决策)

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261023-wp3 | 修复 phase 1 code review 三个高价值发现(CR-01 分组键、WR-01 窗口阶梯、WR-02 失败标签) | 2026-10-23 | de3daae | [261023-wp3-phase-1-code-review-cr-01-wr-01-wr-02](./quick/261023-wp3-phase-1-code-review-cr-01-wr-01-wr-02/) |
| 261024-cr02 | 修复 phase 2 code review CR-01/WR-01/WR-02(异常收口三层防线、配置类型守卫、jac_sparsity 死代码),124 测试,生产重跑逐位复现 0.1407/0.1361 | 2026-10-24 | 8cd1231 | [261024-cr02-phase2-review-critical-warning](./quick/261024-cr02-phase2-review-critical-warning/) |

## Session Continuity

- **Last action:** Phase 3 wave 5(03-05)执行完毕:联合 BA 引擎 TDD 落地(209 测试绿),收敛裁决三变体全灭(漂移越界),失败协议——config.yaml 零 diff、跳过 T3/T4、REQUIREMENTS 未勾选 (2026-10-24)
- **Next action:** 晨间向用户交 03-05 诊断报告;Phase 3 后续方向(内参不确定度传播 / 只放主点变体 / 验收口径变更)待决策
- **Notes:** data/ 指纹 a183699d 全程不变;output/stereo/ 保持 03-04 正式配置诚实失败留痕;裁决数值表在 /tmp/joint_trial_0305_result.md(已引用进 03-05-SUMMARY);晨间人工项:01-HUMAN-UAT(3) + 02-HUMAN-UAT(2)

---
*Initialized 2026-10-23 during roadmap creation*
