---
status: complete
priority: high
created: 2026-10-24
completed: 2026-10-24
source: phase-03 code review (.planning/phases/03-3/03-REVIEW.md) CR-01/WR-01/WR-02
plan: 261024-cr03-phase3-review-critical-warning
---

# Quick Task Summary: 修复 phase 3 code review CR-01/WR-01/WR-02

**一句话:** 三项评审发现全部修复并各带回归测试——joint.enabled 从 fail-open 静默 no-op 变为 fail-closed 硬门、train 评估失败从整批炸掉零产物变为单段降级留痕、pairs CSV 列名级损坏从英文 KeyError 堆栈变为中文一句收口。

## 修复摘要

### CR-01 (Critical): joint.enabled fail-closed 硬门 — 提交 `6910b96`

- `src/stereo_calib/stereo/pipeline.py`:run_stereo 入口(先于目录清理)拦截
  `stereo.joint.enabled=true` → 中文 ValueError(收口进 CLI 六类异常网,
  退出码 1)。此前开关无任何消费者,置 true 会静默跑 rigid 路径(fail-open)。
- `config.yaml` joint 段注释如实化:移除「03-05 裁决后置 true」误导文案,
  改为「默认 false 且当前不可置 true;置 true 被硬门拦截,待接线任务落地」。
- `.planning/phases/03-3/deferred-items.md`:登记 joint 接线缺口(CR-01/IN-06
  跨 wave 追踪,含解除条件)+ IN-04(K/params 双轨无交叉核对)、IN-05(降级
  NaN 产出非严格 JSON)两项 Info 补登(按 PLAN 约束不修只登记)。
- 回归测试 `test_run_stereo_joint_enabled_fail_closed`:先证 fail-open(RED:
  静默落盘五件产物),修复后断言中文拦截 + 拦截发生在 rmtree 之前(既有产物
  原样保留)。

### WR-01 (Warning): train 评估纳入单段降级收口 — 提交 `8cbc7bb`

- `src/stereo_calib/stereo/pipeline.py`:train 评估从裸奔改为
  try/except `_SEGMENT_EXC` 降级(与 test 侧同体例)——全帧失败时中文留痕,
  已成功且昂贵的 train 求解+精化结果绝不丢弃(此前 ValueError 穿到 CLI
  最外层:整批炸掉、零产物、退出码 1)。
- 降级侧产物五件尽量齐:`distance_rms_train_mm` = NaN 留痕;npz 残差整块
  NaN(形状契约 (V,88,2) 不破);train CSV 幸存行缺席(表头仍在);
  `write_report_md` 新增可选键 `train_error` 渲染「train 评估降级留痕」
  (体例同既有 `test_error`)。
- `true_distance_mm` 改从配置棋盘直接导出(`true_board_distance ∘
  build_object_points`),不再依赖 train 评估存活——JSON 契约键照常给真值。
- 回归测试 `test_run_stereo_train_eval_failure_degrades_not_explodes`:
  monkeypatch evaluate_split 让 train 侧全帧失败(RED:RuntimeError 穿出、
  零产物),修复后断言五件齐、test 指标不受影响、npz 全 NaN、报告留痕。

### WR-02 (Warning): pairs CSV 列名级损坏中文收口 — 提交 `2a264c5`

- `src/stereo_calib/stereo/observation.py`:load_pairs 装载前核对五列表头
  契约(group_id/cam1_path/cam2_path/route_cam1/route_cam2)——缺列、空文件
  (表头全丢)、表头行丢失(数据行被当表头)统一中文 ValueError「缺列 +
  实际表头 + 重跑 preprocess」。此前 DictReader 抛 KeyError,不在 CLI 六类
  异常网 (ValueError, OSError, ArithmeticError, RuntimeError, AssertionError,
  cv2.error) 内,用户见英文堆栈。
- 值缺失的短行不在新防线拦(仍走既有连接键/路径核对中文防线);模块
  docstring 防线清单同步。
- 回归测试 `test_load_pairs_corrupt_header_raises_chinese`(3 形态参数化):
  RED 时 missing_cols 抛 KeyError、empty_file/headerless 误入 min_pairs 报错,
  修复后三形态全部「缺列」中文收口。

## 测试

- 基线:209 passed(与 PLAN 记录一致)
- 结束:**214 passed,0 failed**(`uv run pytest -q`,新增 5 个回归用例:
  CR-01 x1 + WR-01 x1 + WR-02 x3)
- 全部改动文件 `ruff check` 干净

## 生产重跑验证(Task 3 后,正式配置 config.yaml)

- `uv run stereo-calib stereo` 退出码 **0**;
- 中文散布防线报错:双目初值离散度防线触发(逐对相对位姿最大夹角实测
  **12.1560°** > 阈值 1.0°)——与 03-04 记录同数值同形态;
- `output/stereo/report.md` 诚实失败留痕(「## 失败留痕」节 + 完整中文错误
  消息 + 配置快照),与 03-04 的失败协议留痕形态一致;
- data/ md5 指纹运行前后不变:`find data/ -type f -printf '%T@ %s %p\n' |
  sort | md5sum` = `a183699da79875fc63f7cbfe75874398`(Phase 1 基线同值)。

## 约束遵守

- data/ 未改动(md5 前后一致);参考工程零触碰
- JSON 22 键契约字段集未动(`_EXTRINSICS_STEREO_JSON_KEYS` 断言测试仍绿)
- joint.enabled 默认值保持 false;未实施联合模式接线(03-06 决策范畴)
- worktree 缺 data//output/ 期间用临时 symlink 指主仓,提交前已移除

## 提交列表

| 提交 | 内容 |
| ---- | ---- |
| `6910b96` | fix(03): CR-01 joint.enabled fail-closed 硬门 + config 注释如实化 + 缺口登记(含回归测试) |
| `8cbc7bb` | fix(03): WR-01 train 评估纳入单段降级收口(含回归测试) |
| `2a264c5` | fix(03): WR-02 pairs CSV 列名级损坏收口中文 ValueError(含回归测试) |

## 偏离清单

无——PLAN 三项任务按序执行,无额外发现、无范围外改动。
