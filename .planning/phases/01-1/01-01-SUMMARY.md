---
phase: 01-1
plan: 1
subsystem: infra
tags: [opencv, chessboard-detection, corner-detection, numpy-npz, typer-cli, uv, yaml-config]

# Dependency graph
requires:
  - phase: none (greenfield 首计划)
    provides: 无——本计划即工程骨架起点
provides:
  - uv 工程(src layout + hatchling + stereo-calib 命令入口,uv.lock 入库)
  - 单一 config.yaml 配置层(load_config 中文校验 + output_root/data 互斥校验)
  - CLI 三子命令骨架(preprocess 真实入口,intrinsic/stereo 占位)与可选 --config 选项
  - SB 主路线角点检测(detect_sb,输出必过 canonicalize,窗口断言防线)
  - 数据扫描(io/dataset.py:两层嵌套、cam 号解析、韧性警告)
  - npz 带掩码稠密契约(cam{n}_{split}.npz 八数组)+ usable/fail CSV + meta.json
  - 合成棋盘测试夹具(make_chessboard_image,真值角点 < 0.5px)
affects: [01-2 classic 兜底与并行, 01-3 可视化与公共视场对, Phase 2 内参标定, Phase 3 双目外参标定]

# Tech tracking
tech-stack:
  added: [opencv-python>=4.14,<5.0, numpy==2.5.*, scipy==1.18.*, typer>=0.15, pyyaml>=6.0, joblib>=1.4, pytest>=8, ruff>=0.8, uv + hatchling]
  patterns: [配置文件驱动(命令行零算法参数), frozen dataclass 配置载体, 中文报错收口, 带掩码稠密 npz, 源码防线测试(count 锁定), data 只读 + 输出路径派生]

key-files:
  created:
    - pyproject.toml
    - config.yaml
    - uv.lock
    - src/stereo_calib/cli/main.py
    - src/stereo_calib/io/config.py
    - src/stereo_calib/io/dataset.py
    - src/stereo_calib/corners/window.py
    - src/stereo_calib/corners/canonical.py
    - src/stereo_calib/corners/detect.py
    - src/stereo_calib/preprocess/pipeline.py
    - src/stereo_calib/preprocess/artifacts.py
    - tests/conftest.py
    - tests/test_io_config.py
    - tests/test_corners.py
    - tests/test_io_dataset.py
    - tests/test_preprocess_pipeline.py
  modified: [.gitignore]

key-decisions:
  - "CLI 配置路径为 callback 上的可选 --config 选项(默认 config.yaml)而非固定常量——D-10 字面「命令行只传配置文件路径」"
  - "npz schema 定为八数组带掩码稠密(image_ids/paths/uv/objp/source/detected/sq_px/win_size),未检出整行 NaN——即 Phase 2/3 模块间契约,后续只增内容不改 schema"
  - "image_id = 叶子目录名 + _cam{n}(stem 全库只有 cam1/cam2 必碰撞)且过 _SAFE_IMAGE_ID 白名单正则"
  - "CSV 用 utf-8-sig 编码(Excel 双击直开中文不乱码);fail_{split}.csv 空时也写表头,口径稳定"
  - "SB 未检出 reason 暂记『SB 未检出』,01-02 classic 兜底落地后升级为双路线措辞"

patterns-established:
  - "移植 + 不变量测试锁定:算法路径照参考工程移植,三条硬不变量(SB 严禁二次精化/窗口 2w+1<sq_px 断言/必过 canonicalize)由测试与源码防线锁定"
  - "中文报错统一格式:库层 ValueError 带字段名与实际值,CLI 入口 except (ValueError, OSError) 收口为中文一句 + typer.Exit(1)"
  - "全部写路径从 cfg.output_root 派生,data/ 只 imread;e2e 测试锁定 data 树 mtime 不变"

requirements-completed: [PREP-01, PREP-02, PREP-03, PREP-05, INFRA-01, INFRA-02, INFRA-03]

# Metrics
duration: 37min
completed: 2026-10-23
---

# Phase 1 Plan 1: Walking Skeleton(工程骨架 + SB 角点预处理端到端)Summary

**uv + typer 工程骨架与配置层(单一 config.yaml、中文校验、output_root/data 互斥防线),`stereo-calib preprocess` 一条命令对真实 406 张双目图像端到端产出 Phase 2/3 契约的角点 npz(八数组带掩码稠密)+ usable/fail CSV + meta.json,data/ 只读经 mtime 快照双重验证**

## Performance

- **Duration:** 37 min
- **Started:** 2026-10-23T13:42:45Z
- **Completed:** 2026-10-23T14:19:48Z
- **Tasks:** 3/3
- **Files modified:** 23(22 新建 + .gitignore 追加)

## Accomplishments
- `uv run stereo-calib preprocess` 只依赖 config.yaml(可选 `--config`)端到端贯通:扫描 → SB 角点检测(必过 canonicalize)→ 每相机每数据集落盘 npz + CSV + meta(D-10 零算法参数)
- 真实数据全量运行成功(406 张,串行 SB 约 2 分钟):cam1/train 135/188(71.8%)、cam2/train 152/188(80.9%)、test 30/30(100%),fail 89 张全部记原因;窗口断言 2w+1<sq_px 全帧成立(实测最小方格 8.5px 仍安全)
- 配置层中文校验完备:max_workers=0、square_size_mm=0、未知 classic_flags、bool 冒充 int 均在加载层报错;output_root 与 data 目录相等/互相包含三种形态当场拒绝(01-03 运行前 rmtree 的数据安全前提)
- 三条硬不变量测试锁定:四变体规范化 allclose、窗口不等式参数化 + min_win 下限 AssertionError、SB 精度 < 0.5px(合成棋盘)、纯噪声返回 None、源码防线 `count("cv2.cornerSubPix(") == 0`
- 35 个测试全绿;data/ 只读双重验证(合成树 mtime 快照测试 + 真实数据 `find -printf | md5sum` 运行前后一致)

## Task Commits

Each task was committed atomically:

1. **Task 1: 工程骨架 + 配置层 + CLI 三子命令与 --config 选项** - `29b1ba5` (feat)
2. **Task 2: 角点核心三件套移植(SB + 窗口断言 + 规范化)与不变量测试** - `b71b4e9` (feat)
3. **Task 3: 扫描 + 串行流水线 + npz/CSV 落盘,端到端贯通** - `2a2637e` (feat)

## Files Created/Modified
- `pyproject.toml` / `uv.lock` - uv 构建清单与锁定依赖(照参考工程锁 opencv<5.0 / numpy==2.5.* / scipy==1.18.*)
- `config.yaml` - 唯一参数来源:board/data/output_root/cv2_threads/preprocess 全分段,每键中文注释
- `src/stereo_calib/cli/main.py` - typer 三子命令;callback 可选 --config;preprocess 坏输入中文收口 + 退出码 1
- `src/stereo_calib/io/config.py` - frozen dataclass 六件 + load_config 中文校验 + output_root/data 双向互斥校验
- `src/stereo_calib/io/dataset.py` - 两层通配扫描、cam 号解析、_SAFE_IMAGE_ID、单边/空目录/缺目录警告、imread_checked
- `src/stereo_calib/corners/window.py` - estimate_sq_px / derive_window_frame(2w+1<sq_px 断言)/ derive_window_per_corner
- `src/stereo_calib/corners/canonical.py` - canonicalize(纯几何唯一形态)
- `src/stereo_calib/corners/detect.py` - detect_sb(本文件无二次精化调用,源码防线锁定)
- `src/stereo_calib/preprocess/pipeline.py` - run_preprocess 编排:预分配 NaN 数组 → 串行 SB → 落盘 → 中文汇总
- `src/stereo_calib/preprocess/artifacts.py` - build_object_points / save_corners_npz / load_corners(allow_pickle=False)/ usable+fail CSV(utf-8-sig)
- `tests/*` - conftest(合成棋盘夹具)+ 4 个测试文件共 35 例
- `.gitignore` - 追加 output/ 与 .venv/(data/ 仍忽略,uv.lock 入库)

## Decisions Made
- 检测失败与读图失败统一记 fail 行(reason 区分),单帧失败不中断全批;检出即可用,无额外质量闸门(D-01)
- 空 (split, cam) 组不产出文件(缺 test 目录等合法输入记警告即可);fail CSV 即使 0 行也写表头,下游口径稳定
- 行号由 enumerate 直接给出,结构性规避"空列表 → float64 索引数组"坑(参考工程 L594-598 注释的知识以代码注释保留)
- detect.py docstring 与测试注释中不写出二次精化调用串原文,避免源码防线 count 自伤

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] 数据安全参数化测试未 chdir,相对路径按进程 cwd 解析**
- **Found during:** Task 1(test_io_config.py 首跑)
- **Issue:** 三个目录重叠形态用例断言消息含 tmp_path 的绝对路径,但 load_config 对相对路径(data/train、".")按进程 cwd(worktree 根)resolve,校验逻辑本身正确、断言却失败
- **Fix:** 测试加 monkeypatch.chdir(tmp_path),路径解析与运行目录解耦
- **Files modified:** tests/test_io_config.py
- **Verification:** `uv run pytest tests/test_io_config.py -q` 11 例全绿
- **Committed in:** 29b1ba5(Task 1 commit)

---

**Total deviations:** 1 auto-fixed(bug 修复 1)
**Impact on plan:** 测试侧修复,不影响产品代码与契约。无范围蔓延。

## Known Stubs(intentional,已在计划中定义后续归属)
- classic 兜底与 joblib 并行:detect.py 仅 detect_sb,fail reason 暂记「SB 未检出」——**01-02 落地**(源码防线断言届时改为 == 1)
- 失败帧原图拷贝(copy_fail_originals)与可视化标注图(jpeg_quality):配置键已解析校验,**01-03 消费**
- 公共视场对清单:npz schema 已含 group_id 可推导信息(paths/image_ids),**01-03 落地**
- intrinsic / stereo 子命令:输出「由后续阶段实现」提示,Phase 2/3 实现(D-09 三模块骨架预留)

## Issues Encountered
- uv sync 首次运行约 10 分钟后停滞(到 pypi CDN 的 TCP 连接无数据流动);kill 后重试,全部依赖已在本地 uv 缓存(参考工程同版本组合),仅 ruff wheel 需网络下载,数秒完成。未改任何依赖版本(T-01-05 包审计不受影响)
- 真实数据 sq_px 中位数 18.7px(范围 8.5~66.7),低于研究阶段的 43.9px——排查确认研究值来自 12 张抽样图(恰为近距簇),全量分布双峰(37 帧 <15px、58 帧 15~30px、32 帧 30~50px、8 帧 >50px),非处理缺陷;窗口断言在最小 8.5px 方格上仍成立(w=3 → 7 < 8.5)
- worktree 内无 data/(gitignored):真实数据冒烟以 symlink 接入主仓 data/,symlink 本身被 /data 规则忽略,不入库

## Threat Mitigations Implemented
- T-01-01:load_corners 强制 allow_pickle=False;npz 字符串一律定长 unicode(测试锁定)
- T-01-02:仅 yaml.safe_load,yaml 语法错转中文 ValueError(测试锁定)
- T-01-03:_SAFE_IMAGE_ID 白名单;全部写路径从 cfg.output_root 派生(pipeline 源码无 data/ 硬编码)
- T-01-04:data/ 只 imread;output_root/data 互斥校验三种形态参数化测试;e2e mtime 快照 + 真实数据 md5sum 双重验证
- T-01-05/T-01-SC:依赖仅经 uv 安装,版本组合与参考工程一致,uv.lock 入库;零新增审计表之外的包

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 01-02(classic 兜底 + 并行):detect.py 预留结构明确(classic 半边照参考移植,源码防线断言改 == 1);pipeline 串行循环即待替换为 run_parallel + detect_one worker 的组装点;npz/CSV 契约不变,仅 source 列新增 "classic"
- 01-03(可视化 + 公共视场对):copy_fail_originals/jpeg_quality 配置键已解析;output_root/data 互斥校验已为运行前 rmtree 清理就位
- 真实数据基线已建立:SB-only 检出率 cam1/train 71.8%、cam2/train 80.9%,classic 兜底的增量空间明确
- Phase 2/3:npz 八数组 schema(含 objp 契约 objp[87]=[250,175,0])即输入接口,load_corners 为标准读取入口

## Self-Check: PASSED

- pyproject.toml / config.yaml / src/stereo_calib/(9 文件)/ tests/(5 文件)/ uv.lock 全部存在
- 提交 29b1ba5、b71b4e9、2a2637e 均在 git log 中
- `uv run pytest -q` 35 例全绿;真实数据产物 output/{train,test}/ 四 npz + 四 usable CSV + 两 fail CSV + meta 齐全

---
*Phase: 01-1*
*Completed: 2026-10-23*
