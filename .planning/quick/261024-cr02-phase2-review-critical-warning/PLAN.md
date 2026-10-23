---
type: quick
status: pending
priority: high
created: 2026-10-24
source: phase-02 code review (.planning/phases/02-2/02-REVIEW.md)
---

# Quick Task: 修复 phase 2 code review CR-01/WR-01/WR-02

## Description

修复 phase 2 code review (.planning/phases/02-2/02-REVIEW.md) 中的 Critical + Warning 发现，自主模式（用户已授权过夜自主运行，无需 checkpoint）：

1. **CR-01 (Critical)**: `src/stereo_calib/intrinsic/scoring.py:103-117` 单帧评分契约「绝不抛异常」可被 AssertionError 绕过——`models.project` 的 Z>0 防卫断言在精化期优化器控制的位姿上可触发，AssertionError 不在评分器 except 元组内，且会逃逸 `scipy.optimize.least_squares`；CLI `main.py:87-91` 只捕 (ValueError, OSError)，一个退化检出帧会中断整批双路运行且无 report.md。修复：在 scoring 评分边界把 AssertionError（以及 models.project 能抛的断言类异常）纳入捕获面，单帧失败记 NaN + note；同时核查 `solve.py` 精化路径 L364-367 docstring 声称「上层收口」是否属实，不属实则在 run_intrinsic 调用边界收口（每路求解失败降级为该路失败记录，另一路继续，报告如实留痕）。
2. **WR-01 (Warning)**: `src/stereo_calib/io/config.py:232-235, 308-312` method/window_granularity 的集合成员检查遇容器型 yaml 值（如 `method: [trf]`）抛 TypeError(unhashable) 逃逸 CLI 异常网。修复：成员检查前做类型守卫，str 之外类型给中文 ValueError 配置错误。
3. **WR-02 (Warning)**: `src/stereo_calib/intrinsic/solve.py:396-405` scipy 在 jac 为 callable 时静默忽略 jac_sparsity（实测 1.15.3），make_sparsity 成死计算且 docstring 机制描述失实。修复：移除该调用配置下的 jac_sparsity 死代码并修正 docstring（或保留但注明 scipy 忽略、以注释说明为何保留）——按最小改动取一。

## Constraints

- 全部测试须绿（`uv run pytest -q`，当前 116 个）；每项修复附最小回归测试
- 提交按 fix(02) 前缀原子提交；不改 npz/JSON 契约字段集；不修改 data/
- 修复后重跑 `uv run stereo-calib intrinsic` 验证生产路径无回归（test RMS 与 0.1407/0.1361 同量级即可，允许数值微差），产物落主仓 output/intrinsic/

## Tasks

1. CR-01 scoring/solve/pipeline 异常收口（含回归测试）
2. WR-01 config 类型守卫（含回归测试）
3. WR-02 jac_sparsity 死代码最小处理（含回归测试）+ 生产重跑验证

## Success Criteria

- [ ] uv run pytest -q 全绿（≥116 + 3 个新回归测试）
- [ ] 三项修复各自原子提交 fix(02)
- [ ] 生产重跑退出码 0，两路 test RMS 与 0.1407/0.1361 同量级
- [ ] data/ md5 指纹不变（a183699da79875fc63f7cbfe75874398）
