---
phase: 03-3
status: gaps_found
score: 4/8(实现交付 4/4 VERIFIED;指标达标 0/4:SC3 FAILED,SC1/2/4 BLOCKED)
phase_goal_achieved: false
verified_at: 2026-10-24T07:54:23+08:00
overrides_applied: 0
re_verification:
  previous_status: none
  note: 初始核验,无既有 VERIFICATION.md
gaps:
  - truth: "SC3:test 集最远两角点物理距离残差 RMS 为毫米级(< 5mm,真值 305.16389039334257mm)"
    status: failed
    reason: "真实数据实测 16.9256mm(阈值的 3.4 倍),逐帧残差散乱 ±26mm(std 16.86)非系统性偏移。根因在数据/物理层而非实现:两路独立标定内参在双目精度下互不兼容(f 差 12%、主点差 ~460px)+ 支架相对位姿跨帧摆动(中位 2.5°/max 12.2°);03-05 联合 BA 三变体裁决全灭证实「SC3 达标」与「内参物理可信」在本数据上不可兼得。实现侧链路合成验证恢复板距 <0.5mm/对,实现正确。"
    artifacts:
      - path: "src/stereo_calib/stereo/evaluate.py"
        issue: "无实现问题——消歧/undist/DLT/板系距离链完整,合成链可证;真实数据指标受上游内参质量钳制"
    missing:
      - "方案决策(交 orchestrator/用户):(a) 内参不确定度传播先验替代统一标量权重;(b) 只放主点冻结 f/D 的第四变体消融;(c) 接受刚性模型 16.93mm 现状、改产线验收口径;(d) 物理侧复查支架刚性/同步或 Phase 2 更近/更斜板位姿重标"
  - truth: "SC1/SC2/SC4 验收形态:正式配置一条命令产出 R/T、baseline、完整双目标定报告"
    status: partial
    reason: "实现全部交付(CLI/求解/评估/契约/报告五件套,合成 e2e 可证,214 测试全绿),但真实数据被散布防线 fail-closed 拦截(spread 12.1560° > 1.0°),正式产物仅中文失败留痕 report.md——这是设计正确行为,非缺陷。上游 R/T 质量修复后重跑即可完成验收。"
    artifacts:
      - path: "output/stereo/report.md"
        issue: "无——诚实失败留痕,正是防线设计意图"
    missing:
      - "前置依赖:SC3 根因方案落地后重跑 uv run stereo-calib stereo"
human_verification:
  - test: "抽查 output/stereo/report.md 失败留痕形态"
    expected: "中文防线文案(spread 12.1560° > 1.0°)+ 配置快照;目录仅此一件、无带病 JSON"
    why_human: "留痕形态是否满足产线审计习惯需人工过目;机检已确认内容与结构"
  - test: "复核 03-05-SUMMARY.md 收敛裁决数值表(三变体 × 双流,六次运行零过闸)"
    expected: "落地判据三缺一不可(收敛→漂移全界内→SC3<5mm)的判定逻辑成立;原始证据 /tmp/joint_trial_0305_result.md|.json 仍在(可对照,注意 /tmp 易失)"
    why_human: "「漂移越界即非物理解」是方案决策的根据,数值裁决应经人工确认而非仅 agent 判读"
  - test: "复核内参可辨识性诊断(/tmp/ident_diag_0326.py:profile likelihood,σ(fx)/fx≈0.3-0.9%,联合 BA 漂移 12σ-156σ)"
    expected: "确认「Phase 2 内参被数据强钉死、漂移非物理」结论支持方案 (a)/(b) 方向"
    why_human: "统计诊断结论决定下一步方案取舍,属于决策判断"
  - test: "下一步方案决策(SC3 gap 的 a/b/c/d 四选)"
    expected: "用户/orchestrator 在 03-04/03-05 SUMMARY 列出的预案中拍板,或更新验收口径"
    why_human: "本质是产品/物理决策,自动化无法代判"
---

# Phase 3: 模块3 双目标定 — Verification Report

**Phase Goal:** 双目标定模块可独立运行:消费 Phase 1 公共视场角点 + Phase 2 固定内参,产出双目外参 R/T 与 baseline,并用 test 集最远两角点物理距离残差完成端到端验证
**Verified:** 2026-10-24T07:54:23+08:00
**Status:** gaps_found(phase_goal_achieved: false —— SC3 真实数据指标 FAIL,根因数据/物理层,非实现层)
**Re-verification:** No — 初始核验

**核验方法论说明:** 本报告对每条 SC 区分「实现交付」(功能存在、防线真实、合成链可证)与「指标达标」(真实数据验收)两个维度。所有 SUMMARY 数值均经本机命令实测复核,未轻信文档。

## Goal Achievement

### Observable Truths(4 条 SC × 两维度)

| # | Truth(SC 原文要义) | 实现交付 | 指标达标/验收 | 证据 |
|---|---|---|---|---|
| 1 | SC1:一条命令双目外参求解:PnP-median(每共视图两路 solvePnP → Rodrigues 中位)→ scipy 精化,零 `cv2.stereoCalibrate` | ✓ VERIFIED | ✗ BLOCKED(防线 fail-closed 拦截真实数据端到端) | 实现:solve.py L93-227(pnp_median 两路 solvePnP L156/159、Rodrigues 逐分量中位 L208-211、least_squares 精化)、pipeline.py 剔除循环、CLI stereo 真命令实测运行;防线:`grep -rn stereoCalibrate src/` = 0 命中;验收:正式配置 spread 12.1560° > 1.0° 中文 ValueError 降级留痕,无 R/T 产物(设计正确) |
| 2 | SC2:输出 R/T(cam1 参考系)+ baseline,与支架物理量级相符 | ✓ VERIFIED | ✗ BLOCKED(无正式产物;试验值前提不成立) | 实现:artifacts.py baseline_mm/rotation_angle_deg 序列化 + pipeline.py L223-227 baseline 硬带 [20,2000] 检查;验收:放行试验 baseline 871.64mm 落带,但 SC3 连带不成立且本仓无支架物理记载(config 注释如实标注),量级核对前提不成立 |
| 3 | SC3:test 集验证链端到端(消歧 → 去畸变 → DLT → 板系)距离残差 RMS < 5mm(真值 305.1639mm) | ✓ VERIFIED(链路正确) | ✗ FAILED:16.9256mm,3.4 倍阈值 | 实现:evaluate.py 消歧 L172 / undist_pix L76(牛顿逆)/ triangulate_dlt L99 / farthest_corners_distance L143 / true_board_distance L157 / per-frame 2 视图 BA L249;合成链恢复板距 <0.5mm;真值链 repr 逐位一致。指标:03-04 放行试验 test 16.9256mm / train 20.0398mm,逐帧散乱 ±26mm 非系统偏移(排除实现 bug);03-05 联合 BA 三变体全灭证实结构性不可达 |
| 4 | SC4:输出双目标定报告(每帧距离残差 + 整体 RMS) | ✓ VERIFIED | ✗ BLOCKED(验收形态为失败留痕) | 实现:write_report_md/write_per_frame_csv,FrameEval 含 dist_err_mm/dist_measured_mm/pooled RMS;放行试验产物五件齐全(03-04 表);正式配置下 report.md 为中文防线留痕 + 配置快照(本机实测在位) |

**Score:** 实现交付 4/4 VERIFIED;指标达标 0/4(SC3 FAILED、SC1/2/4 BLOCKED)→ 合计 4/8

**核心结论:** 五个计划(03-01..03-05)的实现全部真实交付且可证(本机实测 214 测试全绿、CLI 实跑、防线实触发、data/ 指纹运行前后一致)。Phase Goal 未达成的原因是唯一硬指标 SC3 在真实数据上 FAIL 16.93mm ≫ 5mm,且 gap-closure(03-05 联合 BA)证实该失败在本数据 + 内参物理可信约束下不可修复——这是数据/物理问题,不是实现问题。防线 fail-closed 与 REQUIREMENTS 不勾选均为正确行为。

### 已交付且验证的资产清单(供晨间报告)

| 资产 | 内容 | 本机验证方式 |
|---|---|---|
| `src/stereo_calib/stereo/solve.py` | PnP-median 初值 + 两遍法 180° 消歧 + scipy 外参精化(解析稀疏雅可比) | 代码审读 + 214 套件 + 防线实测触发 |
| `src/stereo_calib/stereo/evaluate.py` | 评估全链:消歧 → per-frame 2 视图 BA → undist_pix → DLT → 板系对角距离残差 | 代码审读 + 合成 e2e 测试(tests/test_stereo_pipeline.py L433+) |
| `src/stereo_calib/stereo/pipeline.py` | run_stereo 编排:剔除循环 + baseline 硬带 + 单段降级 + joint fail-closed 硬门 | 实跑(降级路径按设计工作)+ 代码审读 |
| `src/stereo_calib/stereo/artifacts.py` | extrinsics_stereo.json 22 键契约(assert 断言锁死字段集)+ CSV×2 + npz + report.md | 逐键清点 = 22,L146-150 契约断言在码 |
| `src/stereo_calib/stereo/joint.py` | 联合 BA 引擎(三变体 fc/fc_prior/full_prior + Tikhonov 先验 + COO 稀疏雅可比),feature-flag 关闭待命 | 代码审读 + test_stereo_joint 引擎用例;合成恢复真值见 03-05 |
| `src/stereo_calib/cli/main.py` | stereo 真实子命令,六类异常中文收口 + Exit(1) | `uv run stereo-calib --help` 实测含 stereo;实跑退出码 0 |
| `config.yaml` stereo/joint 段 | 9 项决策参数 + 防线(spread 1.0° / distance 5.0mm / baseline 带)+ joint.enabled: false | grep 实读;git 树零残留 diff |
| `output/stereo/report.md` | 正式配置诚实失败留痕(中文防线 12.1560° > 1.0° + 配置快照) | 实跑后 head -30 核验 |
| 测试资产 | 214 passed(实测 48.38s,零回归) | 本机 `uv run pytest -q` |
| 诊断证据链 | 03-02 四层根因 / 03-04 端到端 16.93mm / 03-05 裁决表(/tmp 原始件在位)/ /tmp/ident_diag_0326.py 内参可辨识性 | 文件存在性核实;数值表待人工复核(见 human_verification) |

### Required Artifacts

| Artifact | Expected | Status | Details |
|---|---|---|---|
| `src/stereo_calib/stereo/solve.py` | SC1 求解引擎 | ✓ VERIFIED | PnP×2/对 + Rodrigues 中位 + least_squares + 离散度防线 |
| `src/stereo_calib/stereo/evaluate.py` | SC3 验证链 | ✓ VERIFIED | 全链函数在位,合成 <0.5mm 可证 |
| `src/stereo_calib/stereo/pipeline.py` | 编排/剔除/防线/降级 | ✓ VERIFIED | fail-closed 硬门 L172-177 + 单段降级实测工作 |
| `src/stereo_calib/stereo/artifacts.py` | 22 键 JSON 契约 | ✓ VERIFIED | 22 键逐一清点 + 契约断言 |
| `src/stereo_calib/stereo/joint.py` | 联合 BA 引擎(关闭待命) | ✓ VERIFIED | 完整引擎,fail-closed,无占位 |
| `src/stereo_calib/cli/main.py` | stereo 子命令 | ✓ VERIFIED | 实跑通过;_not_implemented 已删 |
| `output/stereo/extrinsics_stereo.json` | SC2 正式外参产物 | ✗ MISSING(设计使然) | 防线拦截,无带病产物——诚实状态,依赖 SC3 根因解决 |

### Key Link Verification

| From | To | Via | Status | Details |
|---|---|---|---|---|
| cli/main.py stereo | pipeline.run_stereo | 局部导入 run_stereo(load_config(...)) | WIRED | 实跑触发完整链 |
| pipeline | data/train npz + pairs | 装载器(连接键/行序锁/按名读内参) | WIRED | 实跑日志「train 112 对、test 15 对」 |
| pipeline | output/intrinsic/cam{1,2}/intrinsics.json | 按名读取固定内参 | WIRED | 实跑确认按名读取 |
| run_stereo | output/stereo/ 五件套 | artifacts.py 落盘 | WIRED(合成 e2e 证明) | 真实数据降级态仅 report.md |
| config stereo.joint | joint 引擎 | run_stereo fail-closed 硬门 | WIRED | enabled=true 中文 ValueError(先于目录清理) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|---|---|---|---|---|
| 真实数据运行 | R/T/baseline/距离残差 | data/train 112 对 + Phase 2 固定内参 | 是(03-04 试验实测 16.93mm 等) | ✓ FLOWING |
| 正式配置产物 | report.md | 防线异常 + config 快照 | 是(失败留痕即真实数据流) | ✓ FLOWING(降级形态) |
| 合成 e2e 测试 | 契约 JSON 22 键 | conftest make_stereo_synth 真值双目几何 | 是(测试锁定) | ✓ FLOWING |

无 HOLLOW/STATIC/DISCONNECTED:唯一「缺产物」处是防线设计意图,非数据断链。

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|---|---|---|---|
| CLI 含 stereo 子命令 | `uv run stereo-calib --help` | stereo 命令在列,中文全链文案 | ✓ PASS |
| 正式配置一条命令 | `uv run stereo-calib stereo` | 退出码 0;防线中文报错 12.1560° > 1.0°,与 SUMMARY 数值逐位一致 | ✓ PASS(fail-closed 按设计) |
| 失败留痕形态 | head -30 output/stereo/report.md | 中文失败留痕 + 配置快照;目录仅此一件 | ✓ PASS |
| 零内建双目例程 | `grep -rn stereoCalibrate src/ \| wc -l` | 0 | ✓ PASS |
| data/ 只读 | md5 指纹(实跑 stereo 前后各一次) | 均 `a183699da79875fc63f7cbfe75874398` | ✓ PASS |
| 测试套件 | `uv run pytest -q` | 214 passed in 48.38s | ✓ PASS |
| REQUIREMENTS 失败协议 | grep STER- | 4 项均 `[ ]`,Traceability Pending | ✓ PASS(正确未勾选) |
| config 零残留 | `git status --porcelain` + `git diff --stat` | 均空 | ✓ PASS |

### Probe Execution

| Probe | Command | Result | Status |
|---|---|---|---|
| (无 probe 脚本) | `find scripts -path '*probe*'` = 0 | N/A | SKIP — 本仓 probe 等价物为 pytest + CLI 实跑,已在上方全绿 |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|---|---|---|---|---|
| STER-01 | 03-01/02/03/04/05 | PnP-median 初值 + scipy 精化,零 cv2.stereoCalibrate | 实现 ✓ / 验收 BLOCKED | solve.py + grep=0;真实数据被防线拦,无法完成验收 |
| STER-02 | 03-02/03/04 | 输出 R/T + baseline | 实现 ✓ / 验收 BLOCKED | artifacts.py 契约在位;正式配置无产物 |
| STER-03 | 03-03/04 | test 验证链 + 距离残差 RMS | 实现 ✓ / 指标 FAILED | evaluate.py 链完整;实测 16.9256mm > 5mm |
| STER-04 | 03-03/04 | 双目标定报告 | 实现 ✓ / 验收形态 BLOCKED | 报告生成器在位;正式态为失败留痕 |

无 ORPHANED:REQUIREMENTS.md 映射到 Phase 3 的 4 项(STER-01..04)全部被计划声明。4 项保持 `[ ]` 未勾选 = 失败协议正确执行。

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|---|---|---|---|---|
| src/stereo_calib/stereo/solve.py | 175 | 「占位」字样 | ℹ️ Info | 指失败对 NaN/False 诊断位(结构保留),非实现占位,良性 |
| src/stereo_calib/cli/main.py | 62/78/99 | 「占位命令」字样 | ℹ️ Info | 懒加载历史注释(占位命令已被真命令替换),良性 |

无 TBD/FIXME/XXX/PLACEHOLDER 真实命中;无 console.log 型桩;无空返回桩。deferred-items.md 登记 5 处 Phase 1 旧测试 ruff 告警(非本 Phase 引入,scope 边界正确)。

### Human Verification Required

见 frontmatter `human_verification` 4 项:失败留痕抽查、03-05 裁决数值表复核、内参可辨识性诊断复核、SC3 方案四选决策。**建议 orchestrator 据此生成 03-HUMAN-UAT.md**(既有 01:3 项、02:2 项,Phase 3 目前为空)。

### Gaps Summary

两条 gap,同一根因(SC3 真实数据 16.93mm FAIL,数据/物理层不可达,03-05 gap-closure 已证联合 BA 三变体全灭),非实现缺陷。实现资产完整交付且全部经本机实测验证;下一步是方案决策(内参不确定度传播 / 只放主点变体 / 改验收口径 / 物理侧复查),不是代码修补。deferred 区为空:Phase 3 是里程碑末阶段,joint 生产接线缺口(deferred-items.md 已登记)无后续 phase 覆盖,保持为 gap 的前置依赖之一。

---

_Verified: 2026-10-24T07:54:23+08:00_
_Verifier: Claude (gsd-verifier)_
