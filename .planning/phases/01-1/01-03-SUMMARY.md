---
phase: 01-1
plan: 3
subsystem: preprocess
tags: [opencv, chessboard-detection, drawchessboardcorners, visualization, common-fov-pairing, csv-manifest, fail-review, data-readonly, rmtree-cleanup]

# Dependency graph
requires:
  - phase: 01-1/01-01(walking skeleton)
    provides: 配置层互斥校验(rmtree 安全前提)、npz/CSV 契约、copy_fail_originals/jpeg_quality 配置键
  - phase: 01-1/01-02(classic 兜底 + 并行)
    provides: 双检测器 detect_one worker、并行流水线、source 列 {"sb","classic",""}
provides:
  - visualization/overlay.py(render_marked 标准角点叠加 + 烧录路径/路线 + copy_fail_original 逐字节副本)
  - 全量角点标注图(output/{split}/visualization/{image_id}.jpg,D-07 无抽样,D-08 烧录)
  - common_fov_pairs_{split}.csv(group_id 配对两路均检出,Phase 3 双目标定输入清单)
  - fail_originals/(fail_ 前缀原图副本,开关控制,D-02 假失败复核入口)
  - 运行前 rmtree 清理产物子目录(D-11 无残留)
affects: [Phase 2 内参标定(usable/npz 消费), Phase 3 双目外参标定(common_fov_pairs 消费)]

# Tech tracking
tech-stack:
  added: []  # 零新增依赖(T-01-SC):OpenCV drawChessboardCorners/putText + 标准库 shutil/csv
  patterns:
    - "drawChessboardCorners 一行标准叠加(Don't Hand-Roll,不自绘圆圈连线),烧录文字 ASCII only(Pitfall 9)"
    - "字号/线宽按画面尺寸比例推导(max(1.0, min(h, w)/480)),不写死像素(换分辨率不失效)"
    - "失败帧原图 shutil.copy2 逐字节副本:fail 标记只体现在输出文件名前缀,data/ 源文件零改动"
    - "配对在主进程消费端做(检出结果已在内存,不进 worker、不二次读 npz)"
    - "运行前 rmtree(output_root/{train,test}) 固定覆盖;安全前提是加载层 output/data 互斥校验,不重复实现"

key-files:
  created:
    - src/stereo_calib/visualization/__init__.py
    - src/stereo_calib/visualization/overlay.py
  modified:
    - src/stereo_calib/preprocess/artifacts.py
    - src/stereo_calib/preprocess/pipeline.py
    - tests/test_preprocess_pipeline.py
    - .planning/REQUIREMENTS.md
    - .planning/phases/01-1/deferred-items.md

key-decisions:
  - "render_marked 按本计划签名(返回 None,route 为参数):不复刻参考工程的 MISS 红框/note 分支——本工程失败帧走 fail_originals 原图副本(D-02),检出帧才画标注图"
  - "可视化与 fail 拷贝留在主进程串行:全量 326 张 2560×1440 渲染 + 80 张拷贝仅约 9s(全流程 107s 中检测占 98s),并行复杂度不值得;且主进程写集中与 T-01-08(子进程零写)防线一致"
  - "配对结果在落盘循环之前算好(common_pairs 计数要进各 split 两个 meta.json);CSV 只在主进程写"
  - "rmtree 只删 output_root/{train} 与 output_root/{test} 两个产物子目录,output_root 本身与未知内容不动;ignore_errors=True 容忍首跑目录不存在"
  - "单边存在/单边检出不入公共清单也不记 fail:fail 清单只记检测失败(该帧检出侧已在 usable、失败侧已在 fail),误标只会带来无谓原图拷贝与复核噪音(D-01/Pitfall 8 定案)"

patterns-established:
  - "全量可视化 = 检出循环 + render_marked,张数恒等于 detected 计数和(测试锁定,兼防 Pitfall 7 残留)"
  - "公共对数 <= min(两路检出)作为永久断言(Pitfall 8 警告信号永不触发)"

requirements-completed: [PREP-04, PREP-06, INFRA-03]

# Metrics
duration: 10min
completed: 2026-10-23
---

# Phase 1 Plan 3: 可视化 + 公共视场对 + 全量生产运行 Summary

**全量角点标注图(drawChessboardCorners 标准叠加 + 烧录路径/路线 ASCII 文字,326 张无抽样)、失败帧 fail_ 前缀原图副本(开关控制)、group_id 公共视场配对 CSV(train 112/test 15 对)与运行前 rmtree 无残留清理落地;真实 406 张全量生产运行 107s,visualization 326 == detected 总和、公共对 <= min(两路检出)、fail_originals 80 == fail 行数,data/ md5 指纹运行前后逐字节一致(INFRA-03 硬证明)**

## Performance

- **Duration:** 10 min(14:45:35Z → 14:55:07Z)
- **Tasks:** 2/2
- **Files modified:** 5(2 新建:visualization 包;3 修改)+ REQUIREMENTS.md/deferred-items.md 补记
- **Tests:** 50 通过(01-02 的 44 + 本计划新增 6),全量绿,零回归
- **真实全量(406 张,2560×1440):** 107s 墙钟 / 12 进程(1066% CPU)——检测 98s(与 01-02 基线一致)+ 渲染 326 张标注图与 80 张原图拷贝约 9s(主进程串行)

## 真实数据全量运行关键计数(SUMMARY 硬要求)

| split | cam1 检出 | cam2 检出 | classic 帧 | 公共视场对 | fail(原图副本) | visualization |
|-------|-----------|-----------|------------|------------|------------------|---------------|
| train | 141/188(75.0%) | 155/188(82.4%) | 9 | 112 | 80(80) | 296 |
| test | 15/15(100%) | 15/15(100%) | 0 | 15 | 0(0) | 30 |
| 合计 | 156/203 | 170/203 | 9 | 127 | 80(80) | 326 |

- 检测路线分布:sb 317、classic 9——与 01-02 全量运行逐数一致(确定性跨计划复现)
- data/ 只读硬证明:`find data/ -type f -printf '%T@ %s %p\n' | sort | md5sum` 运行前后均为 `a183699da79875fc63f7cbfe75874398`(与 01-02 记录的指纹同值)
- 产物计数自洽:visualization JPEG 总数 326 == 四个 npz detected 总和;train 公共对 112 <= min(141, 155);test 15 <= 15;fail_originals 文件数 == fail CSV 行数(80/0)
- fail reason 全部为「SB 与 classic 均未检出」(真实数据无读图失败);方格像素边长中位 18.9px(范围 8.5~69.9)
- 公共对里同时含 sb 与 classic 路线的帧(classic 兜底帧在两路配对中可用,canonicalize 跨路线同序的证据)

## Task Commits

1. **Task 1: 角点标注图渲染 + 失败帧原图拷贝(D-02/D-07/D-08/PREP-06)** - `8b55d84` (feat)
2. **Task 2: 公共视场配对 + 运行前清理 + 全量生产运行与 data/ 只读锁定(PREP-04/D-05/D-11/INFRA-03)** - `d25c3fe` (feat)

## Files Created/Modified
- `src/stereo_calib/visualization/overlay.py` - 新建:render_marked(imread 只读 → drawChessboardCorners 标准叠加 → putText 烧录 f"{path} [route]" ASCII、字号/线宽按 min(h,w) 推导 → imwrite 带质量参数,读写失败中文 ValueError)+ copy_fail_original(shutil.copy2 → fail_{image_id}.jpg,源文件绝不改写)
- `src/stereo_calib/preprocess/pipeline.py` - 运行前 rmtree(output_root/{train,test}) 清理;公共视场配对(两路均检出才成对,主进程);meta 加 common_pairs;汇总加公共对/标注图/副本三段
- `src/stereo_calib/preprocess/artifacts.py` - 新增 write_common_pairs_csv(列 group_id, cam1_path, cam2_path, route_cam1, route_cam2;utf-8-sig)
- `tests/test_preprocess_pipeline.py` - 树工厂 _build_synthetic_dataset 抽取 + 新增 6 测试(全量标注覆盖/字节一致副本/开关关闭/配对规则/公共对上限/孤儿清理)
- `.planning/REQUIREMENTS.md` - 勾选 PREP-04/PREP-06/INFRA-03(traceability 同步)
- `.planning/phases/01-1/deferred-items.md` - 记录 01-01 的 REQUIREMENTS 勾选缺口

## Decisions Made
- 失败帧不画 MISS 红框(参考工程做法),改走 fail_originals 原图副本:本工程的复核工作流是"看原图判断真假失败"(D-02),检出帧标注图只服务"角点偏移"复核(D-07),两类证据分开
- 配对计数写入 meta 用"该 split 的两个 cam meta 都带同一 common_pairs 值":Phase 2/3 任读一份 meta 即知 split 级配对规模
- render_marked 对已检出帧读图失败抛 ValueError 让 CLI 收口退出 1:检测刚成功的图渲染时读不到属异常环境(文件被移动/权限变化),静默跳过会破坏"全量无抽样"承诺

## Deviations from Plan

### 计划草案微调(不改变行为面)

- **测试夹具重构**:把 synthetic_run 的造树逻辑抽成 `_build_synthetic_dataset` 模块级工厂——任务 2 的清理测试(需重跑两次)与开关关闭测试(改配置重跑)复用同一棵标准树;既有 6 个测试的夹具返回值形态(out, before)不变
- **开关关闭测试用最小树**(纯噪声单图)而非复跑全量合成树:fail 行存在 + 目录不产出 + 汇总含「已关闭」三个断言点全部覆盖,省一次完整合成树运行
- **清理测试额外验证 train/ 直下的孤儿文件**(计划只要求 visualization/ 内 stale.jpg):rmtree 目标是整个产物子目录,顺带锁定"train/ 下任何残留都会清掉"

---

**Total deviations:** 3 处测试形态微调(零行为面变化)
**Impact on plan:** 无范围蔓延;must_haves 全部达成,npz 八数组 schema 不变。

## Known Stubs(intentional,已在计划中定义后续归属)
- intrinsic / stereo 子命令:仍为「由后续阶段实现」占位,Phase 2/3 实现(D-09 骨架预留;本计划未触碰)
- 本计划自身无新增 stub:可视化/配对/拷贝/清理全部真实数据通路,真实全量运行验证闭环

## Issues Encountered
- worktree 为新检出(无 01-02 遗留 symlink):重新 `ln -s` 主仓 data/(gitignored);`find data/`(尾斜杠)跟随 symlink 取指纹
- REQUIREMENTS.md 勾选缺口:01-01 完成的 PREP-01/03/05/INFRA-01/02 未勾选(01-02 只勾了自己的 PREP-02)——超出本计划范围不代勾,已记 deferred-items.md 供 orchestrator/verifier 补记

## Threat Mitigations Implemented
- T-01-09(data/ 意外写入):全部写路径从 output_root 派生;rmtree 仅命中 output_root/{train,test}(安全前提 = 01-01 加载层互斥校验,不重复实现);生产运行 md5 指纹前后一致 + e2e 快照测试(含清理重跑路径)双锁定
- T-01-10(残留误导复核):运行前 rmtree 重建;孤儿文件测试锁定(stale.jpg 与 train/ 直下孤儿均随重跑消失)
- T-01-11(角点序可追溯):每图烧录路径 + [sb/classic](D-08);npz source 列 + meta CANONICAL_RULE_VERSION(01-01 既有,公共对 CSV 的 route_cam1/route_cam2 逐对延续该留痕)
- T-01-SC(依赖):零新增依赖,未发生任何包安装

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 1 全部产物链路闭合:每 split 有 npz×2 + meta×2 + usable×2 + fail + common_fov_pairs + visualization/ + fail_originals/,固定覆盖无残留
- Phase 2(内参标定):每相机 usable 清单 + cam{n}_{split}.npz(八数组带掩码)即输入接口,load_corners(allow_pickle=False) 为标准读取入口;train 可用帧 cam1 141、cam2 155
- Phase 3(双目外参):common_fov_pairs_{split}.csv(127 对:train 112 + test 15)+ 两路 npz 按 group_id/路径对齐即输入;两路路线可混合(sb/classic 均已规范化同序)
- 人工核验(end-of-phase,phase 验证阶段):抽 10-20 张 visualization 看角点贴合与连线走向;fail_originals 80 张逐张判真假失败(D-02);CSV 用 Excel 抽查

## Self-Check: PASSED

- src/stereo_calib/visualization/{__init__.py, overlay.py} 存在;artifacts.py/pipeline.py/test_preprocess_pipeline.py 均已修改
- 提交 8b55d84、d25c3fe 均在 git log;两个提交零文件删除
- `uv run pytest -q` 50 例全绿;`uv run stereo-calib preprocess` 真实 406 张全量成功
- 产物计数自洽与 data/ md5 一致性见上表与「真实数据全量运行关键计数」节

---
*Phase: 01-1*
*Completed: 2026-10-23*
