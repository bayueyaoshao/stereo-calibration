---
type: quick
status: pending
priority: high
created: 2026-10-24
source: phase-03 code review (.planning/phases/03-3/03-REVIEW.md)
---

# Quick Task: 修复 phase 3 code review CR-01/WR-01/WR-02

## Description

修复 phase 3 code review (.planning/phases/03-3/03-REVIEW.md) 中的 Critical + Warning 发现,自主模式(用户已授权过夜自主运行,无需 checkpoint):

1. **CR-01 (Critical)**: `src/stereo_calib/stereo/joint.py` 已交付但 `run_stereo`(pipeline.py)从不读取 `stereo.joint.enabled` 开关——置 true 会静默跑 rigid 路径(fail-open);且 config.yaml joint 段注释含「裁决后置 true」误导性文案。修复:在 run_stereo 入口加 fail-closed 硬门——`joint.enabled=true` 时抛中文 ValueError(「联合模式评估接线未完成(03-05 Task 3 按失败协议跳过),禁止启用」),附最小回归测试;config.yaml 注释改为如实描述(引擎已落地、接线待 03-06、默认 false 不可置 true);在 `.planning/phases/03-3/deferred-items.md` 登记「joint 评估链接线」待办。
2. **WR-01 (Warning)**: `src/stereo_calib/stereo/pipeline.py:226` train 评估在单段降级 try 之外,全帧评估失败会在精化成功后炸掉整批、零产物。修复:纳入降级收口(该路评估失败降级为失败留痕,产物五件尽量齐),附最小回归测试(合成环境模拟评估抛异常)。
3. **WR-02 (Warning)**: `src/stereo_calib/stereo/observation.py:151/174/196` pairs CSV 列缺失/空文件抛 KeyError,不在 CLI 六类异常网内,用户见英文堆栈。修复:列缺失/空文件转中文 ValueError(与既有 triple-guard 风格一致),附最小回归测试。

## Constraints

- 全部测试须绿(`uv run pytest -q`,当前 209 个);每项修复附最小回归测试
- 提交按 fix(03) 前缀原子提交;不改 JSON/npz 契约字段集;不修改 data/
- 不改 joint.enabled 默认值(false);不实施联合模式接线(那是 03-06 决策范畴)
- 修复后重跑 `uv run stereo-calib stereo` 验证生产路径无回归:仍为 fail-closed 中文防线留痕、退出码 0、`output/stereo/` 保持诚实失败留痕形态;md5 指纹 a183699da79875fc63f7cbfe75874398 不变
- Info 6 项不修,但把「降级 NaN 产出非严格 JSON」「K/params 双轨无交叉核对」两项补登进 deferred-items.md(其余 4 项 reviewer 已在 REVIEW.md 留档即可)

## Tasks

1. CR-01 joint.enabled fail-closed 硬门 + config 注释如实化 + deferred-items 登记(含回归测试)
2. WR-01 train 评估纳入降级收口(含回归测试)
3. WR-02 pairs CSV 中文 ValueError 收口(含回归测试)+ 生产重跑验证

## Success Criteria

- [ ] uv run pytest -q 全绿(≥209 + 3 个新回归测试)
- [ ] 三项修复各自原子提交 fix(03)
- [ ] 生产重跑退出码 0,输出保持诚实失败留痕(中文防线报错 + report.md 留痕)
- [ ] data/ md5 指纹不变(a183699da79875fc63f7cbfe75874398)
