---
phase: 01-1
plan: 2
subsystem: preprocess
tags: [opencv, chessboard-detection, corner-detection, classic-fallback, defect-gate, window-ladder-recovery, joblib, loky, parallel, taskfailed-envelope]

# Dependency graph
requires:
  - phase: 01-1/01-01(walking skeleton)
    provides: detect_sb SB 主路线、window/canonical 断言防线、串行流水线与 npz/CSV 契约
provides:
  - 完整双检测器 detect.py(detect_sb + detect_classic:缺陷闸门 + 窗口阶梯重提,源码防线 == 1)
  - preprocess/parallel.py(joblib loky 并行 + TaskFailed 异常信封 + max_workers=1 串行回退)
  - 并行化 pipeline(detect_one worker,SB→classic 两级,npz source 列取值 {"sb","classic",""})
  - 单帧坏图只记 fail 不杀批(结果数 == 任务数)的批完整性契约
affects: [01-3 可视化与公共视场对, Phase 2 内参标定(更多可用帧), Phase 3 双目外参标定]

# Tech tracking
tech-stack:
  added: []  # 零新增依赖:joblib 已在 01-01 入锁(T-01-SC)
  patterns:
    - "异常信封(worker 内全捕获包 TaskFailed,joblib 结果流永不被单任务异常杀掉)"
    - "loky + parallel_config(inner_max_num_threads=核数//进程数) + return_as=generator(有序逐个消费)"
    - "任务元组只带绝对路径/标识/配置(像素数据不跨进程,worker 自己 imread)"
    - "窗口阶梯恢复(间距 2% 分位定顶格,逐级 -2 收严到 3,更严收敛判据 100/1e-6,恢复留痕)"
    - "zip(任务, 结果) 对回任务身份(信封不带身份,靠同位置任务元组识别)"

key-files:
  created:
    - src/stereo_calib/preprocess/parallel.py
    - tests/test_parallel.py
    - .planning/phases/01-1/deferred-items.md
  modified:
    - src/stereo_calib/corners/detect.py
    - src/stereo_calib/preprocess/pipeline.py
    - tests/test_corners.py
    - tests/test_preprocess_pipeline.py

key-decisions:
  - "run_parallel 签名 max_workers 前置(体例照计划 key_link:run_parallel(max_workers, tasks, worker=detect_one)),其余照参考工程逐行移植"
  - "_recover_grid_defects 照参考工程签名不带 cfg(恢复用模块常量 _RECOVERY_MAX_ITER/_RECOVERY_EPSILON,cfg 参数无用途;计划草案里的 cfg 形参放弃)"
  - "detect_one 任务元组 = (绝对路径, image_id, cam, group_id, split, cols, rows, CornersConfig, cv2_threads)——frozen dataclass 可靠序列化,且 worker 内补设 cv2.setNumThreads(子进程没有 CLI callback)"
  - "消费端用 zip(任务, 结果) 而非结果自带身份:TaskFailed 信封只装异常,任务识别靠同位置元组;末尾结果数==任务数断言兜底(信封被击穿时宁可全批报错)"
  - "'读图失败:' 前缀统一用于一切 worker 异常(含检测层意外异常),原文随 reason 落 fail CSV——查有实据,不静默"

requirements-completed: [PREP-02]

# Metrics
duration: 13min
completed: 2026-10-23
---

# Phase 1 Plan 2: classic 兜底 + joblib 并行 Summary

**classic 兜底半边(缺陷闸门 + 窗口阶梯重提)与 joblib loky 并行层(TaskFailed 异常信封)移植落地,流水线替换为 run_parallel + detect_one 两级检测 worker:406 张全量 12 进程 98s(10.6x CPU 效率),SB 拒检帧获第二次检出机会(cam1/train 135→141、cam2/train 152→155,classic 兜住 9 帧),单帧坏图只记 fail 不杀批,data/ md5 运行前后一致**

## Performance

- **Duration:** 13 min(14:26:37Z → 14:40:07Z)
- **Tasks:** 2/2
- **Files modified:** 5(2 新建源码/测试对 + 3 修改)+ deferred-items.md
- **Tests:** 44 通过(01-01 的 35 + 本计划新增 9),全量绿
- **真实全量(406 张,2560×1440):** 12 进程 98.4s 墙钟 / 1039 CPU-s(CPU 效率 10.6x);01-01 SB-only 串行约 120s,同负载串行(SB+classic)推算约 17 分钟

## Accomplishments
- detect_classic 完整移植:findChessboardCorners 粗定位 → 窗口逐帧(或逐角点)推导精化 → grid_defects 无条件过闸门(重复点 <0.25×中位 / 锯齿二阶差分 >0.3×中位)→ 带病帧从粗定位网格走窗口阶梯(2% 分位定顶格,cap→cap-2→…→3,收敛判据 100 次/1e-6)重提,恢复过程全部留痕
- 三条硬不变量全部锁定:源码防线升级为「精化调用形式全文件恰好 1 次」(grep 实测 == 1);全部检出帧(含 classic 兜底帧)2w+1<sq_px 成立(真实数据 npz 全帧复核);两路输出必过 canonicalize(SB/classic 同图规范化后 allclose 测试化)
- parallel.py:loky 常驻进程 + inner_max_num_threads=核数//进程数(防多进程叠满线程互抢带宽,T-01-07)+ return_as="generator" 有序逐个消费 + max_workers<=1 同函数串行回退
- TaskFailed 异常信封:worker 内任何异常(含 imread ValueError、cv2.error)包信封交回,单帧坏图记 fail 行绝不杀整批(T-01-06);批完整性双保险(测试锁定结果数==任务数 + 消费端断言)
- 真实数据成果:检出 326/406(01-01 基线 317,+9 全部来自 classic 兜底,只增不减);fail 80 张逐张记因;source 列 317 sb / 9 classic,与汇总打印一致
- data/ 只读复核:`find data/ -type f -printf '%T@ %s %p\n' | sort | md5sum` 运行前后一致(a183699d…);worker 只 imread,全部写操作在主进程(T-01-08)

## Task Commits

1. **Task 1: classic 兜底移植——缺陷闸门 + 窗口阶梯重提 + 源码防线升级为 == 1** - `e3b743d` (feat)
2. **Task 2: joblib loky 并行移植 + 流水线并行化 + 异常信封** - `f7eb479` (feat)

## Files Created/Modified
- `src/stereo_calib/corners/detect.py` - 追加 classic 半边:_CLASSIC_FLAG_VALUES / _classic_flag_value / _subpix_refine(全文件唯一精化调用点)/ _spacing_pool / grid_defects / _recover_grid_defects(_RECOVERY_MAX_ITER=100、_RECOVERY_EPSILON=1e-6)/ detect_classic(去掉参考工程 force_win 实验分支);detect_sb 逐字节不变
- `src/stereo_calib/preprocess/parallel.py` - 新建:TaskFailed 信封、_guarded_worker、run_parallel(串行回退 + loky 生成器,体例照参考工程 calib.evaluation.parallel)
- `src/stereo_calib/preprocess/pipeline.py` - detect_one 顶层 worker(两级检测 + worker 内 cv2.setNumThreads 同源补设);串行循环替换为 run_parallel + zip 对回任务;fail reason 双措辞;汇总加「检测路线分布」一行;结果数==任务数断言
- `tests/test_corners.py` - 源码防线 == 1 升级;新增 classic 精度 <0.5px / 纯噪声 None / SB-classic 规范化 allclose / 干净网格零缺陷 / 缺陷注入(重复点+锯齿)被抓
- `tests/test_parallel.py` - 新建:坏 JPEG 信封不杀批 / 串行并行逐帧 allclose / 进度计数 1..N / 端到端串行 vs 并行 npz 一致 + 两类 fail reason 锁定(185 行)
- `tests/test_preprocess_pipeline.py` - source/route 断言放宽为 {"sb","classic"}(classic 接入后语义更新)
- `.planning/phases/01-1/deferred-items.md` - 记录 01-01 遗留的 9 处 ruff 债务(超出本计划范围)

## Decisions Made
- 消费端不依赖结果顺序的稳健映射:slot_of[绝对路径] 索引 + zip(任务, 结果) 对位——信封路径与正常路径同一套识别逻辑
- worker 内 `cv2.setNumThreads(cv2_threads)`:子进程不经 CLI callback,线程限制必须带进 worker(值同源 config,防 12 进程 × 20 线程互抢)
- 全量并行实际 98s(12 进程)/ 93s(18 进程):瓶颈是 classic 兜底在 89 张硬帧上的 ~10 CPU-s/张,非调度问题;max_workers 配置杠杆在此负载下 ~12 后收益见底
- `--config` 选项挂在 typer callback 上,须置于子命令前(`stereo-calib --config X preprocess`)——01-01 既有形态,非本计划改动

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] test_parallel.py 初版 ruff C408(照抄了 01-01 测试的 dict() 写法)**
- **Found during:** Task 2 全仓 ruff 检查
- **Issue:** 新文件照抄 test_preprocess_pipeline.py 的 `_VIEWS = dict(...)` 体例,ruff C408 × 4
- **Fix:** 新文件改字典字面量;01-01 遗留的 8 处 C408(同文件)+ 1 处 B008(cli/main.py,typer.Option 惯用法)超出范围不修,记入 deferred-items.md
- **Files modified:** tests/test_parallel.py
- **Verification:** `uv run ruff check` 本计划全部改动文件绿
- **Committed in:** f7eb479(Task 2 commit)

### 计划草案微调(不改变行为面)

- **_recover_grid_defects 签名不带 cfg**:计划草案写了 `(gray, coarse, cols, rows, cfg)`,但恢复全程用模块常量(_RECOVERY_MAX_ITER/_RECOVERY_EPSILON),cfg 无用途;照参考工程忠实移植省去死参数
- **detect.py 模块 docstring 更新**:01-01 版写「本文件不得出现二次精化的调用形式」,classic 落地后该表述与源码防线(== 1)直接矛盾,不改会误导;detect_sb 函数体逐字节不变(已 diff 验证)
- **detect_one 任务元组/返回字典为计划规格的超集**:多带 image_id(消费端稳健识别)与 cv2_threads(worker 内补设线程限制),其余照计划
- **verify_real_smoke 的 30-60s 预期未在 12 进程达成(实际 98s)**:研究阶段 0.29s/图按 SB-only 工作负载估计;classic 兜底在 89 张硬帧(2560×1440)上 ~10 CPU-s/张,总 CPU 工作 1039s 决定墙钟下限。10.6x CPU 效率 + 结果与 18 进程一致,证明并行层本身达标;「分钟级→亚分钟级」未逐字满足,亚分钟可在未来用 classic_flags 加 fast_check(检测行为变更,须用户定夺)或减小图像分辨率预处理(精度风险)探索——两根杠杆都超出本计划授权,如实记录
- **tests/test_preprocess_pipeline.py 的 source/route 断言放宽**:classic 兜底接入后 source 列语义为 {"sb","classic",""},锁死 =="sb" 是过约束(前序 wave 执行者交接意见同此);合成图实际仍全 sb,断言放宽不弱化契约

---

**Total deviations:** 1 auto-fixed(C408)+ 5 处计划草案微调(签名/文档/元组超集/性能预期如实修正/断言语义更新)
**Impact on plan:** 无范围蔓延;npz 八数组 schema 与 01-01 完全一致(source 列按契约仅新增 "classic" 取值)。

## Known Stubs(intentional,已在计划中定义后续归属)
- copy_fail_originals 与可视化标注图(jpeg_quality):配置键已解析校验,**01-03 消费**
- 公共视场对清单:**01-03 落地**(npz 的 group_id/paths 已含推导所需信息)
- intrinsic / stereo 子命令:Phase 2/3 实现(D-09 骨架预留)

## Issues Encountered
- worktree 无 data/(gitignored):以 symlink 接入主仓 data/ 供真实冒烟,symlink 本身被 /data 规则忽略,不入库;`find data` 需尾斜杠才跟随 symlink
- 18 进程全量复跑(93.5s)与 12 进程结果完全一致(检出 326、路线分布 317+9),确定性得证;CPU 利用率 1548% 说明 OpenCV 线程限制在 worker 内生效

## Threat Mitigations Implemented
- T-01-06(并行异常杀批):_guarded_worker 全捕获包 TaskFailed;test_parallel 坏 JPEG 用例锁定结果数 == 任务数;消费端 consumed==total 断言兜底
- T-01-07(线程互抢):parallel_config(backend="loky", inner_max_num_threads=max(1, 核数//进程数));worker 内 cv2.setNumThreads(cv2_threads) 同源补设
- T-01-08(子进程写竞争):worker 只 imread;全部写路径在主进程汇总层,从 cfg.output_root 派生
- T-01-SC(依赖):零新增依赖(joblib 01-01 已入锁),未发生任何包安装

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 01-03(可视化 + 公共视场对):input npz 契约就绪(source 列现含 "classic",标注图 route 文字直接用);copy_fail_originals/jpeg_quality 配置键待消费;运行前 rmtree 清理的互斥校验前提已在 01-01 就位
- 真实数据基线更新:cam1/train 141/188(75.0%)、cam2/train 155/188(82.4%)、test 30/30;fail 80 张(措辞「SB 与 classic 均未检出」)待 01-03 原图拷贝供人工复核
- Phase 2/3:可用帧较 SB-only 多 9 帧;load_corners(allow_pickle=False)为标准读取入口

## Self-Check: PASSED

- src/stereo_calib/preprocess/parallel.py、tests/test_parallel.py、deferred-items.md 存在;detect.py/pipeline.py/test_corners.py/test_preprocess_pipeline.py 均已修改
- 提交 e3b743d、f7eb479 均在 git log;两个提交零文件删除
- `uv run pytest -q` 44 例全绿;grep -c 'cv2.cornerSubPix(' detect.py == 1
- 真实 406 张全量运行成功:检出 326(source: sb 317 / classic 9),detected == usable CSV 行数,data/ md5 前后一致

---
*Phase: 01-1*
*Completed: 2026-10-23*
