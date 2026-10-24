---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
last_updated: "2026-10-24T16:40:00.000Z"
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 12
  completed_plans: 12
  percent: 100
---

# STATE: 双目相机标定生产工具

## Project Reference

- **Core Value:** 快速、可靠地产出经过测试集指标验证的双目标定结果(内参 + 双目外参 R/T)
- **Current Focus:** 全部完成(v1.0)——内参两机 reliable 全绿,双目 SC1-4 全过
- **Roadmap:** .planning/ROADMAP.md
- **Requirements:** .planning/REQUIREMENTS.md

## Current Position

Phase: 3 (模块3-双目标定) — **COMPLETE**(quick 261024-rational8-production 追加轮:M8 洗盆修复后双目重跑验收全过)
Plan: 5 of 5 executed + code review 修复 + verification 完成 + quick 追加轮收口

- **Phase:** 3
- **Plan:** 03-04/03-05(blocked 记录保留,由 quick 追加轮达成验收)
- **Status:** v1.0 交付完成:cam1 fx=2037.04 / cam2 fx=2037.52(两机体检 reliable 全绿),双目 test 距离 RMS 0.2001mm ≪ 5mm 阈值
- **Progress:** ▩▩▩▩▩ 12/12 plans complete(03-04/03-05 的 STER 验收由 quick 追加轮补齐)

```
[##########] 100%  Phase 3 of 3
```

## Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Requirements shipped | 19 | 19(v1 全勾) |
| Phases complete | 3 | 3 |
| 内参 test RMS | < 1.0 px | cam1 0.0618 / cam2 0.0686(M8+洗盆;体检两机 reliable 红灯 0 项)|
| 双目距离残差 RMS | < 5 mm | **0.2001**(test,M8 内参;train 0.2321;baseline 798.11mm)|

## Accumulated Context

### Decisions

- 角点:SB 主路线(严禁二次亚像素)+ classic 兜底(窗口从本帧方格像素边长推导)
- 内参:M8 rational 8 模型(2026-10-24 追加轮定稿;M2 为 config 可选 legacy),calibrateCameraExtended 初值 + scipy least_squares 精化 + **M8 洗盆**(精化后以精化解作 guess 追加一遍 RAT|GUESS cv2 拟合,防 fx-D 简并谷;solver 标签 opencv+wash),RMS float64 回代
- 内参体检:A3 切片重标在 D 长 8 时冻结全部畸变系数(fx 成唯一自由 field 信号)
- 双目:禁用 cv2.stereoCalibrate;PnP-median 初值 + 外参精化
- 双目验证指标:test 集最远两角点物理距离残差 RMS,真值 √(10²+7²)×25mm = √149×25 ≈ 305.16mm(2026-10-24 修正原 307.45 算术误差)
- 参考工程只读:`/home/lx/Works/src/lx/github/intrinsic-calibration`、`/home/lx/Works/src/bolight_robot/vvd_daq/multi_cameras`
- 03-05 失败协议:联合 BA 三变体(fc/fc_prior/full_prior)在原样流与生产同构流下全部死于内参漂移越诊断界(f 漂 7.7–10.6% vs ±3% 界、主点漂 33–59px vs ±30px 界、D 漂 0.09–0.18 vs ≤0.05 界)——SC3 预演 1.53mm 的收益来自参数漂移而非物理解;joint.enabled 保持 false,引擎代码+测试留库待命
- 03-05 裁决证据:权重网格 w∈{0.1,1,10} 先验拉回力无效(先验行 scale=w/θ0,θ0~2265px,弱于数据项约 5 个数量级);真绑定需 w~1e4–1e5 属调权重凑数,计划明令禁止

### TODOs (from roadmap)

- Phase 1 规划与实现(含 INFRA 工程基础)

### Blockers

- 无(2026-10-24 追加轮收口)。历史 blocker 全部销账:
  - SC3 16.93mm FAIL(M2 内参产物)→ M8+洗盆内参重跑双目,test 距离 RMS **0.2001mm** PASS
  - cam2 内参不可靠(honest-stop)→ 根因 fx-D 简并谷(非数据层),洗盆后 A1=0.0011/A3=0.1974 全绿
  - 联合 BA 三变体全灭/内参可辨识性终审(03-04/03-05 诊断)→ 结论保留有效:M8 独立内参 + 固定内参双目链即达 0.2mm 级,无需联合 BA;诊断结论与 joint fail-closed 决策留档不变

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 261023-wp3 | 修复 phase 1 code review 三个高价值发现(CR-01 分组键、WR-01 窗口阶梯、WR-02 失败标签) | 2026-10-23 | de3daae | [261023-wp3-phase-1-code-review-cr-01-wr-01-wr-02](./quick/261023-wp3-phase-1-code-review-cr-01-wr-01-wr-02/) |
| 261024-cr02 | 修复 phase 2 code review CR-01/WR-01/WR-02(异常收口三层防线、配置类型守卫、jac_sparsity 死代码),124 测试,生产重跑逐位复现 0.1407/0.1361 | 2026-10-24 | 8cd1231 | [261024-cr02-phase2-review-critical-warning](./quick/261024-cr02-phase2-review-critical-warning/) |
| 261024-cr03 | 修复 phase 3 code review CR-01/WR-01/WR-02(joint.enabled fail-closed 硬门、train 评估降级收口、pairs CSV 中文收口),214 测试,生产重跑留痕形态不变 | 2026-10-24 | 6910b96 | [261024-cr03-phase3-review-critical-warning](./quick/261024-cr03-phase3-review-critical-warning/) |
| 261024-intrinsic-adequacy | 内参模型充分性体检(A0/A1/A3/A4)fail-closed 接入内参管线:INTR-05 增补,244 测试,生产双机判不可靠(cam1 红 A1/A3/A4、cam2 红 A1/A3)exit 1,rational8 重标待办已登记 | 2026-10-24 | f2bb429 | [261024-intrinsic-adequacy-battery](./quick/261024-intrinsic-adequacy-battery/) |
| 261024-rational8 | 产线 M8 rational8 全链 + Phase 2 重标 + Phase 3 双目重跑(含追加轮洗盆收口):INTR-06 增补,273 测试;追加轮定位 cam2 根因为 fx-D 简并谷并以精化后洗盆修复(342f1d2/8266e3e/575d6cf),两机体检 reliable 全绿(fx=2037.04/2037.52 落锚点带),双目 SC1-4 全过(test 距离 RMS 0.2001mm,baseline 798.11mm,剔除 0 对),v1.0 交付完成 | 2026-10-24 | 6d570ac/fc2d7f7/63023fc/1d52d86/342f1d2/8266e3e/575d6cf | [261024-rational8-production](./quick/261024-rational8-production/) |

## Session Continuity

- **Last action:** quick 261024-rational8-production 追加轮(2026-10-24):真实数据探针归因 cam2 红灯为 scipy 精化的 fx-D 简并谷落盆(非数据层),落地「精化后 M8 洗盆」(342f1d2 两遍拟合 / 8266e3e A3 冻结协议 / 575d6cf 洗盆位置修正);生产重跑 intrinsic 两机体检 reliable 全绿(cam1 fx=2037.04/A 全过,cam2 fx=2037.52/A 全过,红检 0),双目重跑 SC1-4 全过(test 距离 RMS 0.2001mm ≪ 5mm,baseline 798.11mm,翻转 0/剔除 0 对,退出码 0);REQUIREMENTS INTR-06 更新 + STER-01..04 勾选,deferred-items 三条销账,273 测试全绿,阈值/闸门零改动
- **Next action:** 无待办——v1.0 全部交付(19/19 需求)。后续可选项(非阻塞):自动离群循环 M8 评分保真度修法(登记 deferred-items,本轮零实际影响);stereo/joint M8 语义验证(现 fail-closed 拒收);03-HUMAN-UAT 人工验收项可按需执行
- **Notes:** data/ 指纹:排序版可复现 8a430c9395e91f256923b2a6eabb4354,追加轮提交前后一致;output/(intrinsic + stereo)为 M8+洗盆最终产物留痕(未提交);M8 已知限制:自动离群循环评分用 OpenCV 初值(膨胀现象依旧,本轮自动剔 0 帧零影响,修法登记);rational D8 逐系数近简并不可单独解读(fx 被强钉死,消费侧用整体映射);.codebuddy/ 为用户自有未跟踪目录(未提交)

---
*Initialized 2026-10-23 during roadmap creation*
