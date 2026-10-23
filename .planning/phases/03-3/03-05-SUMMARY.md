---
phase: 03-3
plan: 05
status: blocked
subsystem: stereo-calibration
tags: [joint-ba, gap-closure, convergence-verdict, failure-protocol, drift-bound, tdd]

# Dependency graph
requires:
  - phase: 03-3 (03-04)
    provides: solve_stereo 引擎母本、rigid 基线数值(test 重投影 3.0159px / SC3 16.9256mm)、SC3 FAIL 根因诊断(两路内参互不兼容)
  - phase: 01/02
    provides: data/train 112 对角点产物、output/intrinsic/cam{1,2}/intrinsics.json 固定内参
provides:
  - src/stereo_calib/stereo/joint.py:联合 BA 引擎(三结构化变体 fc/fc_prior/full_prior + Tikhonov 先验 + 解析稀疏雅可比 + fail-closed 散布防线),feature-flag 关闭待命(joint.enabled: false,rigid 路径零改动)
  - tests/test_stereo_joint.py:5 引擎用例(合成内参扰动恢复真值 / 雅可比全维数值基准 <1e-6 / 变体布局往返 / 先验拉回 / 散布防线)
  - 03-05 收敛裁决数值表:三变体在「原样全量」与「生产同构流」两种流下全灭的完整证据(漂移界为致命判据)
affects: [STER-01..04 验收, orchestrator 决策(联合 BA 后续预案), 后续 gap-closure 规划]

# Tech tracking
tech-stack:
  added: []  # 零新增依赖(scipy least_squares / numpy 既有)
  patterns:
    - "联合 BA θ 布局:内参|公共外参|逐对位姿,冻结块不进 θ(fc 变体 D 原样回填),先验行独立编号"
    - "COO 稀疏雅可比行主序交错纪律:多块列值必须沿 axis=2 拼接后统一 ravel(逐块 ravel 会把值接错行,数值差分基准当场抓住)"
    - "结构化变体裁决 + 失败协议体例:先 /tmp 时间盒试跑后落地;全灭即 config 零 diff 停止,绝不调权重/放宽阈值凑数"

key-files:
  created:
    - src/stereo_calib/stereo/joint.py
    - tests/test_stereo_joint.py
    - .planning/phases/03-3/03-05-SUMMARY.md
  modified:
    - src/stereo_calib/io/config.py
    - config.yaml
    - tests/test_stereo_config.py
    - tests/test_stereo_guards.py

key-decisions:
  - "三变体在两种裁决流下全灭(漂移界为致命判据)→ 失败协议:joint.enabled 保持 false,config.yaml 零 diff,跳过 Task 3/4,REQUIREMENTS.md 零 diff"
  - "补充生产同构流裁决(刚性快解剔除循环 2 对 → 110 幸存 → 联合解)复核,全灭结论不变,证据链双流闭合"
  - "权重标定发现计划网格 w∈{0.1,1,10} 先验拉回力无效(先验行 scale=w/θ0,θ0~2265px,三档解逐 8 位一致);真绑定需 w~1e4–1e5,属计划明令禁止的「调权重凑数」,不越界"
  - "试验型 ValueError(V1/V2 散布防线 5.0589 薄触发)不改判据不重跑原样流;按 Task 3 生产编排语义(快解防线移交)补生产同构流裁决"

requirements-completed: []  # 失败协议:STER-01..04 全部保持不勾选

---

# Phase 03-3 Plan 05: 双目联合 BA gap-closure Summary

**联合 BA 引擎全链落地并测试锁定,但真实数据收敛裁决三变体全灭(内参漂移全部越过诊断界),按失败协议停止:配置零 diff、跳过接线与生产重跑、STER-01..04 保持不勾选。**

## Performance

- **Duration:** ≈40min(06:50–07:30 +0800;T1 25min / T2 15min,T3/T4 按失败协议跳过)
- **Tasks:** 2/4 完成(T1 GREEN、T2 裁决),2/4 跳过(T3/T4,失败协议)
- **测试:** 209 passed(194 基线零回归 + 15 新增),ruff 干净
- **data/ 指纹:** 试验前后均 `a183699da79875fc63f7cbfe75874398`,只读硬约束达标

## Accomplishments

- T1(GREEN 1e62798):`stereo/joint.py` 联合 BA 求解器——θ=[内参|公共外参|逐对位姿],三结构化变体(fc 冻结 D / fc_prior fc+先验 / full_prior 全放开+先验),Tikhonov 先验(f/c 相对偏差 + D 绝对偏差),解析稀疏雅可比(COO→csr、tr_solver=lsmr、x_scale=jac,零稀疏性结构参数——WR-02 教训),joint 散布防线 fail-closed;`StereoJointConfig` 中文校验 + config.yaml joint 子段(默认关);rigid 既有代码与 194 基线测试零改动
- T1 合成验证:两路内参互不兼容扰动(f/c +1.5%、主点 +6px、k1 +0.03)下 full_prior 恢复真值(f/c 相对 <0.3%、k1 绝对 <0.05、t <2mm、转角 <0.05°),且同扰动 rigid 路径劣于 joint——联合路线必要性的引擎侧反证在案;雅可比全维数值差分 <1e-6;强先验(w=1e6)拉回 <1e-6
- T2 裁决(计划字面流 + 生产同构流双流):三变体全灭,失败协议触发,全部数值留档 `/tmp/joint_trial_0305_result.md` 与 `/tmp/joint_trial_0305_result.json`

## Task Commits

| Task | Commit | 内容 |
| ---- | ------ | ---- |
| T1 RED | ad33082 | test(03-05): 5 引擎用例 + 9 配置用例 + 1 guard 用例(先失败) |
| T1 GREEN | 1e62798 | feat(03-05): joint.py 引擎 + StereoJointConfig + config.yaml joint 段 |
| T2 | 无提交 | /tmp 裁决(计划规定脚本与产物不进仓库);三变体全灭 → config 零 diff |
| T3/T4 | 跳过 | 失败协议(不接线、不生产重跑、不勾 REQUIREMENTS) |

## 收敛裁决数值表

### 流一:计划字面(原样 train 112 对直跑联合解;权重标定 w∈{0.1,1,10} 于 20 对子集)

权重标定:三档全部 converged、train RMS 逐 8 位一致(0.9142461x px,max 差 1.6e-7)、SC3 预演全部 1.2764mm,但内参漂移全部越界(cam1 fx +7.97%/fy +6.98%、cx +33.5px;cam2 cx +59.2px)——网格内先验无实际拉回力,取 w=1.0 进正式裁决。

| 变体 | converged | 墙钟(s) | train RMS(px) | test 重投影(px) | SC3 预演(mm) | 漂移 max | 判定 |
| ---- | --------- | ------- | -------------- | ---------------- | ------------- | -------- | ---- |
| V1 fc | 防线拦截(spread 5.0589 > 5.0) | 45.2 | — | — | — | — | ERROR(fail-closed) |
| V2 fc_prior w=1.0 | 防线拦截(spread 5.0589 > 5.0) | 45.2 | — | — | — | — | ERROR(fail-closed) |
| V3 full_prior w=1.0 | True(nfev=200 顶格) | 77.7 | 0.4154 | 0.302 | 1.5484 | cam1 fx 10.6%、D 0.184 | FAIL(漂移越界) |

### 流二:生产同构(刚性快解剔除循环[防线移交,内存构造]剔 2 对 → 110 幸存 → 联合解)

| 变体 | converged | 墙钟(s) | train RMS(px) | test 重投影(px) | SC3 预演(mm) | 漂移 max | 判定 |
| ---- | --------- | ------- | -------------- | ---------------- | ------------- | -------- | ---- |
| B1 fc | True | 39.5 | 1.6972 | 1.028 | 5.5790 | cam1 fx 7.7%、cx 43.0px | FAIL(漂移+SC3) |
| B2 fc_prior w=1.0 | True | 41.2 | 1.6972 | 1.028 | 5.5790 | cam1 fx 7.7%、cx 43.0px | FAIL(漂移+SC3) |
| B3 full_prior w=1.0 | True | 79.1 | 0.4148 | 0.304 | 1.5322 | cam1 fx 10.5%、D 0.180 | FAIL(漂移越界) |

**落地判据三缺一不可(收敛 → 漂移全界内 → SC3 <5.0mm),六次运行零过闸 → 三变体全灭,失败协议成立。**

## 根因定论(诊断证据)

1. **漂移与精度的不可兼得是结构性的**:联合解靠把内参拉离 Phase 2 值 7–10%(f)、33–59px(主点)、0.09–0.18(D)换取 SC3 1.53mm(相对 rigid 16.93mm);而内参诊断界(03-02:f ±3%、主点 ±30px、D ≤0.05)恰是「内参还是物理标定结果」的底线。要 SC3 达标必须漂移越界,守界内则退回 rigid 的 16.93mm——**低残差来自参数漂移而非物理解,裁决判据正确地拦下了它**。
2. **先验网格无绑定力**:先验行 scale = w/θ0(θ0~2265px),w=10 时对 f 的单位相对偏差罚仅 ~0.004(px 当量),弱于数据项约 5 个数量级(数据项 4×20×88 行、雅可比 O(100–1000)px)。真绑定需 w~1e4–1e5,但计划明令「SC 不达标调权重凑数 = 篡改验收证据」(T-03-21)、绝不「先验权重凑数」(Task 4)——不越界,按计划网格裁决并如实留痕。
3. **V1/V2 散布防线薄触发(5.0589 vs 5.0)是坏对吸收效应**:原样流坏对被逐对位姿吸进联合解(中位被压低、max 抬高);剔除循环(2 对:20261023_154252_947367 / 20261023_155711_088510)后 spread 降到 4.94 过线,但 B1/B2 仍死于漂移+SC3——防线不是全灭的原因,漂移才是。
4. **联合 BA 引擎本身无罪**:合成数据上扰动恢复真值全指标达标(引擎测试),真实数据上是数据/模型层面的不可行,与 03-02「两路内参在双目精度下互不兼容」的诊断一致且更精确:互不兼容的量(f 差 12%、主点差 ~460px)超出了「保持内参物理可信」所能吸收的范围。

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] 剔除循环被 rigid 1.0° 防线拦死,按计划 Task 3 语义内存内移交防线**
- **Found during:** Task 2(生产同构流补充裁决)
- **Issue:** 直接调 pipeline._drop_outliers 在 pass-2 触发「双目初值离散度防线 12.1560° > 1.0°」(正是 03-04 生产失败点),剔除循环无法完成
- **Fix:** 裁决脚本内 `dataclasses.replace(stereo, median_spread_max_deg=None)` 内存构造(零 yaml 变更)——这正是计划 Task 3 规定的 joint 生产编排语义
- **Files modified:** 仅 /tmp/joint_trial_0305b.py(不进仓库)

**2. [Rule 1 - Bug] 裁决脚本 Pairs 长度取错(select_pairs 返回 Pairs 无 len)**
- **Found during:** Task 2 补充裁决
- **Fix:** `len(survivors.image_ids)`;仅 /tmp 脚本,仓库零影响

### 自主决策(夜间授权范畴,如实留痕)

- **补充生产同构流裁决**:计划 Task 2 字面是「原样全量直跑」,但 Task 3 生产编排是「剔除循环 → 幸存集定稿联合解」。原样流 V1/V2 被防线拦下拿不到漂移/SC3 数值,裁决表缺列。补生产同构流让六列齐全、结论双流复核——目的是证据完整性,非重跑凑绿(结论未变)。
- **权重标定胜者取法**:计划判据 =「收敛且 train RMS 最小」应取 w=0.1(与 w=1.0 差 1.6e-7px,数值上无差别);脚本按 SC3 兜底取 w=1.0。三档解逐 8 位一致,对裁决结论零影响,如实留痕。

## Issues Encountered

- 计划网格 w∈{0.1,1,10} 对 f~2265px 量级的先验行无绑定力(见根因 2)——后续若重启联合 BA 预案,权重语义需重新设计(如以数据项 RMS 量级为基准的自适应权重,或对数网格 1e2–1e6),这属于方案变更,交 orchestrator 决策
- nfev=200 顶格(status=0)但 cost 实用收敛(引擎判 converged=true):112 对 678 参数规模下 max_nfev=200 裕量不足,若重启预案建议 400+ 并核实收敛尾迹

## Known Stubs

无。joint.py 为完整引擎实现(feature-flag 关闭待命),无占位/TODO/硬编码空值;rigid 生产路径零改动,主仓 output/stereo/ 保持 03-04 的诚实失败留痕状态。

## User Setup Required

无。

## Next Steps(交 orchestrator 决策,executor 未自行实施)

- STER-01..04 保持 Pending;Phase 3 与 ROADMAP Wave 5 状态维持 blocked,诊断结论已更新到 STATE/ROADMAP
- 可选预案方向(均需新的 gap-closure 计划,未实施):(a) 内参不确定度传播——把 Phase 2 协方差作为先验强度依据替代统一标量权重;(b) 联合 BA 只放主点(冻结 f/D)的第四变体——主点漂移 33–59px 是最大越界项,f 漂移 7.7% 与之耦合,值得单独消融;(c) 接受刚性模型 + 16.93mm 现状,改产线验收口径(需用户决策)

## Self-Check: PASSED
