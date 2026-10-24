---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: blocked
last_updated: "2026-10-24T12:00:00.000Z"
progress:
  total_phases: 3
  completed_phases: 2
  total_plans: 12
  completed_plans: 10
  percent: 83
---

# STATE: 双目相机标定生产工具

## Project Reference

- **Core Value:** 快速、可靠地产出经过测试集指标验证的双目标定结果(内参 + 双目外参 R/T)
- **Current Focus:** Phase 3 — 模块3-双目标定
- **Roadmap:** .planning/ROADMAP.md
- **Requirements:** .planning/REQUIREMENTS.md

## Current Position

Phase: 3 (模块3-双目标定) — BLOCKED(SC3 16.93mm FAIL,夜间全链诊断完毕,待用户晨间决策)
Plan: 5 of 5 executed + code review 修复 + verification 完成

- **Phase:** 3
- **Plan:** 03-04/03-05(blocked,失败协议,诊断完整)
- **Status:** 过夜工作收口,晨间报告在 .planning/MORNING-REPORT-2026-10-24.md
- **Progress:** ▩▩▩▩▢ 10/12 plans complete(03-04/03-05 诚实 blocked)

```
[----------] 0%  Phase 1 of 3
```

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Requirements shipped | 17 | 0 |
| Phases complete | 3 | 0 |
| 内参 test RMS | < 1.0 px | cam1 0.0652 / cam2 0.0760(M8;体检:cam1 可靠 / cam2 不可靠)|
| 双目距离残差 RMS | < 5 mm | 16.93(M2 时代实测,已失效不可比;M8 下因 cam2 内参不可靠阻塞未测)|

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
- 联合 BA 预案已被 03-05 收敛裁决否决:三变体全灭(漂移越界为致命判据,SC3 达标与内参物理可信结构性不可兼得)
- **内参可辨识性诊断(2026-10-24 晨,/tmp/ident_diag_0326.py)终审:Phase 2 内参被数据强钉死(σ(fx)/fx ≈ 0.3–0.9%,profile likelihood),联合 BA 漂移 12σ–156σ 非物理解**——方向 (a) 协方差先验、(b) 只放主点均被数字否决;结论:问题在采集物理(支架刚性/同步),推荐采集整改后重跑现有链;备选逐对外参管线(放弃公共 R/T)需用户决策(03-HUMAN-UAT HUMAN-4)
- **cam2 内参 M8 下仍不可靠(2026-10-24,quick 261024-rational8-production honest-stop)**:红 A1=0.0280(阈 0.02)/A3=0.9392%(阈 0.5%)——两轮证据驱动剔除(1.08/0.460/0.447px 三帧)证明非离群帧,M2/M8 双模型下同判证明非单模型失配,A4 同阶独立重标 fx 2037.52 vs 产线 2034.88(0.13%)证明 fx 未漂;中心桶 0.028px 径向结构 + 外圈切片 fx 低压(2020.9 vs 内圈 2038.6)为数据/光学层效应 → fail-closed 禁入双目,Phase 3 重跑阻塞(方向决策见 deferred-items)

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261023-wp3 | 修复 phase 1 code review 三个高价值发现(CR-01 分组键、WR-01 窗口阶梯、WR-02 失败标签) | 2026-10-23 | de3daae | [261023-wp3-phase-1-code-review-cr-01-wr-01-wr-02](./quick/261023-wp3-phase-1-code-review-cr-01-wr-01-wr-02/) |
| 261024-cr02 | 修复 phase 2 code review CR-01/WR-01/WR-02(异常收口三层防线、配置类型守卫、jac_sparsity 死代码),124 测试,生产重跑逐位复现 0.1407/0.1361 | 2026-10-24 | 8cd1231 | [261024-cr02-phase2-review-critical-warning](./quick/261024-cr02-phase2-review-critical-warning/) |
| 261024-cr03 | 修复 phase 3 code review CR-01/WR-01/WR-02(joint.enabled fail-closed 硬门、train 评估降级收口、pairs CSV 中文收口),214 测试,生产重跑留痕形态不变 | 2026-10-24 | 6910b96 | [261024-cr03-phase3-review-critical-warning](./quick/261024-cr03-phase3-review-critical-warning/) |
| 261024-intrinsic-adequacy | 内参模型充分性体检(A0/A1/A3/A4)fail-closed 接入内参管线:INTR-05 增补,244 测试,生产双机判不可靠(cam1 红 A1/A3/A4、cam2 红 A1/A3)exit 1,rational8 重标待办已登记 | 2026-10-24 | f2bb429 | [261024-intrinsic-adequacy-battery](./quick/261024-intrinsic-adequacy-battery/) |
| 261024-rational8 | 产线 M8 rational8 畸变模型全链一等公民 + Phase 2 重标:INTR-06 增补,270 测试,cam1 体检转绿(fx=2042.89 锚点 ±1% 带,test RMS 0.0652)、cam2 两轮证据剔除后仍红 A1/A3(结构性残差)honest-stop,双目重跑阻塞 | 2026-10-24 | 6d570ac/fc2d7f7/63023fc/1d52d86 | [261024-rational8-production](./quick/261024-rational8-production/) |

## Session Continuity

- **Last action:** quick 261024-rational8-production(2026-10-24):M8 rational8 畸变模型产线一等公民全链落地(6d570ac models 双模型化 / fc2d7f7 solve+config / 63023fc 契约+体检+双目装载,契约只增不改、270 测试、存量 M2 测试零修改);Phase 2 重标:cam1 证据剔除 27 帧后体检 5 项全绿(fx=2042.89 落参考锚点 2037±1% 带,独立重标探针 2037.04,test RMS 0.1407→0.0652px),cam2 两轮剔除 3 帧后仍红 A1=0.0280/A3=0.9392 → 按授权「不许放宽阈值硬凑」honest-stop(1d52d86 剔除配置),双目重跑未执行(fail-closed);INTR-06 增补,deferred-items rational8 待办销账 + 两条新登记,数值全表见 quick/261024-rational8-production/SUMMARY.md
- **Next action:** 用户决策 cam2 结构性残差方向(deferred-items 新登记:Phase 1 外圈角点质量分层复核 / 采集侧排查光轴-传感器倾斜与 C-mount 机械应力 / 更高阶模型风险自认);cam2 内参 reliable 后重跑 stereo-calib intrinsic → stereo(闸门不动);原晨报四项(03-HUMAN-UAT 重点 HUMAN-4)继续有效
- **Notes:** data/ 指纹:排序版可复现 8a430c9395e91f256923b2a6eabb4354,本任务提交前后一致;output/intrinsic/ 现为 M8 产物留痕(cam1 可靠 / cam2 不可靠,体检节 + diagnostics 键),cam2 路仍禁止消费进双目;M8 已知限制:自动离群循环评分用 OpenCV 初值(M8 中位 ~1.4–1.8px,阈值膨胀 7–9px,靠手动 exclude_images 兜底,修法已登记);rational D8 逐系数近简并不可单独解读(fx 被强钉死,消费侧用整体映射);.codebuddy/ 为用户自有未跟踪目录(未提交)

---
*Initialized 2026-10-23 during roadmap creation*
