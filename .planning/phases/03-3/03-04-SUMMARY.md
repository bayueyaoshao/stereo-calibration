---
phase: 03-3
plan: 04
status: blocked
subsystem: stereo-calibration
tags: [cli, production-run, acceptance, spread-defense, fail-closed, sc3-failure]

# Dependency graph
requires:
  - phase: 03-3 (01-03 波次)
    provides: run_stereo 编排、Pattern 7 五件套产物、评估全链、solve_stereo 引擎
  - phase: 01/02
    provides: data/train 112 对角点产物、output/intrinsic/cam{1,2}/intrinsics.json 固定内参
provides:
  - src/stereo_calib/cli/main.py:stereo 真实子命令(占位替换,六类异常中文收口 + Exit(1),CLI 行为测试 3 例锁定)
  - 真实数据生产运行实测诊断:SC3 距离残差 16.93mm(test)/20.04mm(train),FAIL 3.4 倍——刚性外参模型在本数据上不可达的最终证据
  - 失败协议执行记录:REQUIREMENTS.md STER-01..04 保持不勾选,交 orchestrator 决策
affects: [联合 BA 预案(orchestrator 决策), STER 验收, 真实数据 UAT]

# Tech tracking
tech-stack:
  added: []  # 零新增依赖
  patterns:
    - "SC3 裁决试验体例:防线 fail-closed 时以最终验收标准(SC3)实测裁决放行与否,不以拍脑袋放宽"

key-files:
  created:
    - .planning/phases/03-3/03-04-SUMMARY.md
  modified:
    - src/stereo_calib/cli/main.py
    - tests/test_stereo_pipeline.py

key-decisions:
  - "删除已无调用者的 _not_implemented 占位函数:计划 action(保留)与 acceptance(全文件无占位残留)冲突,按 acceptance 执行"
  - "SC3 裁决试验(夜间授权):临时 median_spread_max_deg 1.0→15.0 跑真实生产,SC3 实测 16.93mm FAIL → 判定链路不可用,配置精确恢复(git diff 干净),试验配置绝不提交"
  - "正式产物恢复诚实状态:主仓 output/stereo/ 以正式配置(防线 1.0)重跑,只留防线拦截的中文失败留痕 report.md,不放带病报告"
  - "STER-01..04 不勾选:SC3 未达标,按 PLAN Task 3 失败协议执行,绝不修改阈值凑数"

requirements-completed: []  # SC3 FAIL,STER-01..04 全部保持 Pending

# Metrics
duration: 8min
completed: 2026-10-24
---

# Phase 03-3 Plan 04: CLI 验收 + 真实数据生产运行 Summary

**stereo CLI 真实子命令交付(194 测试全绿);真实数据生产运行被散布防线 fail-closed 拦截(spread 12.16° > 1.0°),SC3 裁决试验实测距离残差 16.93mm FAIL(阈值 5mm)——刚性外参模型在本数据上不可达的最终定论,STER-01..04 按失败协议保持不勾选,交 orchestrator 联合 BA 预案决策**

## Performance

- **Duration:** 8 min
- **Started:** 2026-10-23T22:22:38Z
- **Completed:** 2026-10-23T22:30:22Z
- **Tasks:** Task 1 完成(RED+GREEN)/ Task 2 执行完毕但验收 FAIL / Task 3 按失败协议执行(不勾选)
- **测试:** 194 passed(基线 191 + 本计划 3 例 CLI 用例),ruff 全干净

## Accomplishments

- **Task 1(CLI,完成):** `src/stereo_calib/cli/main.py` stereo 占位替换为真命令——help 全链文案、局部导入 `run_stereo(load_config(_config_path))`、六类异常 `(ValueError, OSError, ArithmeticError, RuntimeError, AssertionError, cv2.error)` 中文收口 + `typer.Exit(code=1)`,与 intrinsic 命令逐行同体例;`_not_implemented` 死函数删除(见 Decisions);TDD RED(`019978d`)→ GREEN(`44c3078`),CLI 行为测试 3 例(help 真文案锁定 / 坏环境 Exit(1) / 合成环境一条命令产契约 JSON)
- **Task 2(生产运行,执行完毕,SC3 FAIL):** 正式配置运行被散布防线 fail-closed(spread 12.1560° > 1.0°,降级留痕退出码 0);裁决试验放行后全链 12.2s 跑通,SC3 实测 **test 16.9256mm / train 20.0398mm,FAIL 3.4 倍**;data/ md5 指纹运行前后逐字节一致(`a183699da79875fc63f7cbfe75874398`)
- **Task 3(失败协议):** 成功标准逐条核验见下;REQUIREMENTS.md STER-01..04 保持 `[ ]` 不动,Traceability 保持 Pending;失败证据全部留档本 SUMMARY

## Task Commits

1. **Task 1: CLI stereo 子命令** — `019978d`(test RED,3 例失败)→ `44c3078`(feat GREEN,194 全绿)
2. **Task 2: 真实生产运行** — 运行类任务无代码产物;诊断证据留档本 SUMMARY(试验配置未提交,精确恢复)
3. **Task 3: 验收 + 勾选** — SC3 FAIL,按失败协议不勾选、不提交 REQUIREMENTS 改动(无 diff)

## 生产运行指标(裁决试验,median_spread_max_deg 临时 15.0,未采纳)

| 指标 | 值 | 判定 |
|---|---|---|
| train 剔除 | 幸存 110/112(第 1 轮剔 `20261023_154252_947367` 0.879px、`20261023_155711_088510` 1.537px,阈 0.860px;第 2 轮收敛) | 机制正常 |
| 重投影 RMS | train 6.5783px / test 3.0159px | 与 03-02 刚性外参地板 6.64px 一致 |
| **距离 RMS(SC3)** | **train 20.0398mm / test 16.9256mm(阈值 5mm)** | **FAIL(3.4 倍)** |
| baseline | 871.64mm,t = [858.33, −11.92, −151.26]mm,转角 4.8346° | 落带 [20, 2000],但 SC3 连带不成立 |
| true_distance_mm | 305.16389039334257(repr 逐位一致) | 真值链从 build_object_points 导出正确,排除真值错误 |
| 消歧翻转 | train 0 / test 0 | 无翻转 |
| solver | scipy trf status=2 nfev=64 converged=True | 精化正常收敛 |
| test 逐帧 dist_err(mm) | [5.51, 6.43, 8.06, −3.71, 5.14, −14.16, −29.38, −1.62, 25.73, −5.87, −22.87, −18.83, 25.97, 18.97, −21.11] | **散乱型**:mean −1.45 / std 16.86 / ±26mm 双向分布,非系统性偏移 |
| train 逐帧 | 110 帧中 83 帧 \|err\|>5mm,max \|75.23\|mm | 同样散乱 |
| data/ md5 | 运行前后 = `a183699da79875fc63f7cbfe75874398` | INFRA-03 硬证明(双目链路零 imread) |
| 产物五件 | JSON/CSV×2/npz/report.md 齐全,JSON 22 键契约、K/D 快照与 Phase 2 产物一致 | 契约审计通过 |

**正式配置(median_spread_max_deg 1.0)下的生产状态:** `uv run stereo-calib stereo` 退出码 0,中文防线报错(`双目初值离散度防线触发:……实测 12.1560°,超过配置阈值 1.0°`),降级留痕 `output/stereo/report.md`(仅此一件,无 JSON)——编排层单段降级按设计工作,主仓产物为诚实的失败留痕。

## 成功标准逐条证据

| 标准 | 结果 | 证据 |
|---|---|---|
| SC1 一条命令两路 PnP→中位→scipy 精化,零内建双目例程 | **未达成**(端到端被防线拦) | CLI 真命令在案(退出码 0);guards 五防线全绿(test_stereo_guards 194 套件内);正式配置下 solve_stereo 防线中文 ValueError,无 R/T 产物 |
| SC2 R/T + baseline 落 [20,2000] 带 | **未达成**(连带) | 正式配置无产物;放行试验 baseline 871.64mm 落带但量级核对的前提(SC3 有效性)不成立 |
| SC3 test 距离残差 RMS < 5mm(真值 305.16389039334257) | **FAIL:16.9256mm(3.4 倍)** | 放行试验 JSON `distance_rms_test_mm`;true_distance_mm repr 逐位一致;逐帧残差散乱 ±26mm(std 16.86)非系统性偏移 |
| SC4 报告逐帧距离残差 + 整体 RMS | **契约已验证,验收形态不成立** | 放行试验产物五件齐全可审计(report 逐帧表/量级核对行/留痕节全);正式配置仅失败留痕 |

## 根因定论(本次运行补齐的最后一块证据)

03-02 已建立四层诊断链(逐对独立拟合亚像素、无孪生解、三优化器同地板、焦距扫描排除内参失准、全内参联合 descent 34.9→1.64px 不收敛、相邻对位姿差最小 0.077° 证明噪声地板可达),结论:两路独立标定的内参在双目精度下互不兼容 + 支架相对位姿跨帧摆动(逐对散布中位 2.5°/max 12.2°)。本次生产运行补齐**端到端最后一环**:即便放行散布防线强行产出 R/T,SC3 距离链(逐帧 2 视图 BA + DLT,外参为固定约束)仍实测 16.93mm——**刚性外参 + 固定内参模型在本数据上从求解到验证全链不可达**,防线 fail-closed 是正确行为。逐帧残差散乱型分布(非系统性偏移)进一步排除"距离链实现 bug"(实现 bug 通常产生同号偏置;03-03 合成链板恢复 <0.5mm/对 也已验证实现正确)。

## Deviations from Plan

### Auto-fixed Issues

**1. [计划内矛盾消解] 删除 _not_implemented 死函数**
- **Found during:** Task 1(GREEN 实现)
- **Issue:** action 说"_not_implemented 若无其他引用则保留(避免无关 diff)",acceptance 说"全文件应无占位残留"——替换后该函数体仍含「由后续阶段实现」字样,保留则 acceptance 字面不可达
- **Fix:** 删除该函数(4 行,无任何调用者,diff 完全相关)
- **Files modified:** src/stereo_calib/cli/main.py
- **Committed in:** 44c3078

### 自主决策(夜间授权范畴,如实留痕)

**2. SC3 裁决试验(防线放行的实测裁决)**
- **Found during:** Task 2(正式运行防线 fail-closed,SC 无从验收)
- **Issue:** 防线拦截使整条链无产物;用户提示预期"SC3 距离链可过 <5mm",该预期只有在链路能跑通时才可检验——需要用 SC3 实测裁决"防线是否阻断了唯一可用路径"
- **Action:** 临时将 median_spread_max_deg 1.0→15.0(仅工作区,不提交)跑真实生产;SC3 实测 16.93mm FAIL → 判定链路不可用;配置精确恢复(`git diff config.yaml` 干净),主仓产物以正式配置重跑恢复为失败留痕
- **不采纳理由:** 放行只产出带病报告(17mm 残差),不满足任何成功标准;distance_rms_max_mm 验收闸门从未动过
- **Evidence:** 本 SUMMARY 生产运行指标表

---

**Total deviations:** 1 处计划内矛盾消解 + 1 项夜间授权自主决策(均有完整留痕)
**Impact on plan:** Task 1 全部达成;Task 2/3 验收目标(STER 勾选)不可达——根因是数据/上游缺陷(03-02 已定性),非本计划实现缺陷

## Issues Encountered

- **SC3 失败 → 执行停止(按用户失败协议):** "若 SC3 失败(<5mm 不过):如实记录完整诊断信息到 SUMMARY 后停止,主仓编排层有联合 BA 预案,不要自行发明修复"——本 SUMMARY 即完整诊断记录,执行到此停止
- 未自行发明的修复(明确禁止):内参+外参联合 BA 移植、散布判据剔除循环、放宽 distance_rms_max_mm、逐帧位姿平均替代公共 R/T

## Known Stubs

None——CLI 全链真实实现,无占位数据流。

## User Setup Required

None - 无外部服务配置。

## Next Steps(交 orchestrator 决策,executor 未自行实施)

按 03-02 SUMMARY 预案 + 本次端到端证据,可选路线:
- **(a) 内参+外参联合 BA(主预案):** 移植参考工程 2 `core/multibundle.py` IntrinsicJointBA(只读引用);03-02 实测全内参放开始联合 descent 34.9→1.64px(120 轮未收敛、内参持续漂移),移植前须先做本数据收敛性验证(参考工程在其数据上结论是"联合优化无收益",与本数据症状不同);需新立计划
- **(b) 物理侧复查:** 采集支架刚性/同步(相邻对相对位姿差 min 0.077° 证明偶发一致可达,但中位 3.0° 说明采集间支架在变)
- **(c) Phase 2 重标:** 更近/更斜板位姿打破焦距-位姿退化(当前板距 1.0–3.4m、离轴最大 38°),再重跑本链
- CLI/编排/产物契约/评估链全部就绪(194 测试 + 合成 e2e 全绿),上游 R/T 质量修复后重跑 `uv run stereo-calib stereo` 即可完成 STER 验收

## Self-Check: PASSED

- 文件:src/stereo_calib/cli/main.py(真命令)、tests/test_stereo_pipeline.py(+3 CLI 用例)、本 SUMMARY 均在位
- 提交:019978d(RED)→ 44c3078(GREEN)在案;config.yaml 零残留 diff;worktree 临时 data/output symlink 已移除
- 测试:194 passed(symlink 在位时);ruff src/ 干净
- REQUIREMENTS.md:STER-01..04 保持 [ ](失败协议,无 diff)
- data/ md5 指纹:`a183699da79875fc63f7cbfe75874398` 运行前后一致

---
*Phase: 03-3*
*Completed: 2026-10-24*
