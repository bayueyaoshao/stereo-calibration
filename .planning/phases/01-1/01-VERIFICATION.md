---
phase: 01-1
verified: 2026-10-23T15:15:50Z
status: human_needed
score: 5/5 must-haves verified
overrides_applied: 0
human_verification:
  - test: "人工抽查角点标注图:打开 output/train/visualization/ 与 output/test/visualization/ 抽 10-20 张(建议混入 common_fov_pairs 中 route 为 classic 的帧,如 npz source 列为 classic 的 9 帧)"
    expected: "角点圆点落在棋盘内角点上;连线顺序一致且 SB 帧与 classic 帧连线走向相同(canonicalize 生效,无 180° 错序);烧录的路径与 [sb/classic] 文字可读"
    why_human: "图像内容与角点位置正确性是视觉判断,自动化只能验证文件存在、可解码、张数与 detected 计数一致(SC4 的人工核验半边)"
  - test: "失败帧真假复核(D-02):逐张查看 output/train/fail_originals/ 的 80 张 fail_ 前缀原图"
    expected: "区分『真失败』(棋盘出画/严重遮挡/过曝)与『假失败』(棋盘在画面内但未检出);假失败帧应记录下来供后续处理决策"
    why_human: "棋盘是否在画面内、是否可检出属语义视觉判断;grep/脚本无法代判"
  - test: "CSV 可用性抽查:用 Excel/LibreOffice 双击打开 usable_cam1_train.csv、fail_train.csv、common_fov_pairs_train.csv"
    expected: "中文列头与内容不乱码(utf-8-sig BOM),筛选/排序可用"
    why_human: "Excel 渲染行为与桌面环境相关,无法在验证进程内代验"
---

# Phase 1: 工程骨架 + 模块1 数据预处理 Verification Report

**Phase Goal:** 建立配置文件驱动的工程骨架,数据预处理模块可独立运行:对 data/train 与 data/test 全部图像完成角点检测与筛选,产出两路相机可用清单、公共视场对清单与角点数据文件
**Verified:** 2026-10-23T15:15:50Z
**Status:** human_needed
**Re-verification:** No — initial verification

## MVP Mode Notes

Phase 1 has `Mode: mvp` in ROADMAP.md. The ROADMAP goal text is **not** in canonical
user-story format (`As a …, I want to …, so that …`) — a format discrepancy per MVP-mode
rules. However, `01-01-PLAN.md` (objective section) contains a proper user story for this
phase which matches the goal's outcome clause verbatim in substance, so verification
proceeded using it for the User Flow Coverage below. **Recommendation:** run
`/gsd mvp-phase 1` to reformat the ROADMAP goal, or accept as-is (human decision; does not
affect verification substance).

## User Flow Coverage

User story (from 01-01-PLAN.md): «As a 标定产线工程师, I want to 用一条只传配置文件路径的命令,
对 data/train(188 组)与 data/test(15 组)全部 cam1/cam2 图像完成棋盘格角点检测与筛选,
so that 我直接拿到两路相机可用清单、公共视场对清单、角点数据文件与可视化核验图,供后续标定与人工复核使用。»

| Step | Expected | Evidence | Status |
|------|----------|----------|--------|
| 运行一条命令 | `uv run stereo-calib preprocess` 仅凭 config.yaml(可选 --config)端到端运行,多进程并行 | cli/main.py:62-72(callback --config 默认 config.yaml,preprocess 零参数);parallel.py:100-103(loky);orchestrator 全量运行 107s/12 进程/406 张;`preprocess --help` 无任何算法选项 | ✓ |
| 全量角点检测 | 188+15 组全部 cam1/cam2 图像检测,SB 主路线 + classic 兜底 | 4 个 npz n=188/188/15/15(共 406 行,无一消失);detected 326(sb 317/classic 9);fail 80 逐张记因 | ✓ |
| 拿到可用清单 | 每路相机 usable CSV,行数 == 检出数 | 4 个 usable CSV 行数与对应 npz detected.sum() 逐一相等(程序核验) | ✓ |
| 拿到公共视场对清单 | 同组两路均检出的图像对 | common_fov_pairs_{train,test}.csv(112/15 对);每对 cam1_path/cam2_path 均在对应 usable 清单内(程序核验);对数 <= min(两路检出) | ✓ |
| 拿到角点数据文件 | 2D 角点坐标 + 3D 板系坐标 + 图像索引 | npz 八数组:image_ids/paths/uv(N,88,2)/objp(88,3)…;objp[3]=[75,0,0]、objp[87]=[250,175,0](11×8/25mm 契约);allow_pickle=False 可读 | ✓ |
| 拿到可视化核验图 | 每张检出图像一张角点标注图 | visualization/ 296(train)+30(test)=326 张,文件名与 detected image_ids 集合完全一致;抽 5 张均可解码(2560×1440,~500KB) | ✓(角点位置/顺序的人工核验 → human_verification) |
| Outcome | 可直接供后续标定与人工复核使用 | 全部产物在 output/{train,test}/ 固定目录且计数自洽;npz 即 Phase 2/3 契约;data/ 指纹 a183699da79875fc63f7cbfe75874398 跨全部运行不变 | ✓ |

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria — the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | 一条命令(仅传配置文件路径)对 data/train(188 组)+ data/test(15 组)全部 cam1/cam2 完成角点检测,全程多进程并行,不向 data/ 写入任何文件 | ✓ VERIFIED | CLI 仅 --config 选项(default config.yaml);orchestrator 生产运行 406 张/12 进程/107s;`find data/ -type f -printf '%T@ %s %p\n' \| sort \| md5sum` = a183699d…(与 01-02/01-03 SUMMARY 记录同值,且在本验证所有运行后复测不变);data/ 无任何非 cam1.jpg/cam2.jpg 文件;test_artifacts_under_output_and_data_untouched 以 st_mtime_ns 快照锁定 |
| 2 | 输出目录存在:每路相机可用清单、公共视场对清单(同组两路均检出)、角点数据文件(2D+3D+索引) | ✓ VERIFIED | 磁盘核验:usable_cam{1,2}_{train,test}.csv(141/155/15/15 行 == 各 npz detected);common_fov_pairs_{train,test}.csv(112/15 对,全部对路径 ∈ 两路 usable);cam{1,2}_{split}.npz 八数组含 uv/objp/image_ids,np.load(allow_pickle=False) 可读 |
| 3 | SB 主路线(不做二次亚像素)+ classic 兜底(窗口从本帧实测方格像素边长推导),策略由配置开关控制 | ✓ VERIFIED | detect.py:detect_sb 无任何精化调用;`grep -c 'cv2\.cornerSubPix(' detect.py` == 1(唯一调用在 classic 分支 _subpix_refine,源码防线测试锁定);窗口链 estimate_sq_px→derive_window_frame(断言 2w+1<sq_px);开关 sb_fallback_classic 由 config.yaml 解析校验(io/config.py:212-217)并在 detect_one 消费(pipeline.py:74);生产数据 source 列 sb 317/classic 9 双路线实际生效;全部检出帧 2*win+1<sq_px 零违反(326 帧程序核验) |
| 4 | 输出角点可视化抽查图,人工核验角点位置与顺序正确 | ✓ VERIFIED(产物半边) | 326 张标注图与 detected image_ids 集合精确相等(296+30);drawChessboardCorners 标准叠加 + putText 烧录路径/[路线](overlay.py:56-67);抽 5 张可解码、~500KB 非空图。**角点位置与顺序的人工核验 → human_verification 第 1 项**(按 orchestrator 指示路由,不自动判失败) |
| 5 | 棋盘格规格(11×8 内角点、25mm)、路径、算法参数均来自 YAML,命令行无算法参数 | ✓ VERIFIED | config.yaml 含 board(cols:11/rows:8/square_size_mm:25.0)/data/output_root/cv2_threads/preprocess(max_workers、corners、subpix、copy_fail_originals、visualization)全分段;`stereo-calib preprocess --help` 仅 --help 一项;npz meta board 规格与配置一致;非法值(max_workers=0)实测中文报错 + 退出码 1 不吐堆栈 |

**Score:** 5/5 truths verified

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|-------------|----------|
| 1 | INFRA-01 字面全量满足(intrinsic/stereo 模块可独立运行调试)——Phase 1 交付部分为三子命令骨架 + preprocess 独立运行 + 模块间文件契约(npz/CSV schema),intrinsic/stereo 占位提示已实测(退出码 0) | Phase 2 / Phase 3 | Phase 2 Goal:「内参标定模块可独立运行」;Phase 3 Goal:「双目标定模块可独立运行」 |

### PLAN must_haves 覆盖(补充核验,未计入 SC 分数)

| Plan | Truth 摘要 | Status |
|------|-----------|--------|
| 01-01 | CLI 只依赖 config 端到端、--config 唯一选项零算法参数 | ✓ VERIFIED(--help 实测) |
| 01-01 | 每 {cam,split} 一个八数组带掩码稠密 npz,未检出整行 NaN | ✓ VERIFIED(4 npz 键集/shape/dtype/NaN 程序核验:undetNaN=True detClean=True) |
| 01-01 | usable 行数 == detected;fail CSV 记失败帧与原因 | ✓ VERIFIED(逐文件相等;fail 80 行 reason 均非空「SB 与 classic 均未检出」) |
| 01-01 | intrinsic/stereo 提示后续实现不报错 | ✓ VERIFIED(实测 RC=0,输出占位提示) |
| 01-01 | 非法配置中文报错 + 退出码 1 不吐堆栈 | ✓ VERIFIED(max_workers=0 实测;50 例中含参数化非法用例) |
| 01-01 | output_root 与 data 重叠被 load_config 拒绝 | ✓ VERIFIED(相等/互含两形态实测 ValueError 含实际路径;三形态参数化测试) |
| 01-01 | 全程不向 data/ 写入 | ✓ VERIFIED(指纹三态一致 + mtime 快照测试) |
| 01-02 | SB 未检出自动 classic 兜底、开关控制、source 记 "classic" | ✓ VERIFIED(生产 npz classic 9 帧) |
| 01-02 | classic 精化后无条件过缺陷闸门 + 窗口阶梯重提 + recovery 留痕 | ✓ VERIFIED(detect.py:213-225;缺陷注入测试锁定) |
| 01-02 | cornerSubPix( 全文件恰好 1 次 | ✓ VERIFIED(grep == 1;源码防线测试) |
| 01-02 | max_workers>1 loky 并行;=1 串行回退同果 | ✓ VERIFIED(代码 + test_serial_and_parallel_consistent + test_preprocess_serial_matches_parallel 端到端 uv 逐帧 allclose)。注:全量 98-107s 未达计划 30-60s 估计——SUMMARY 已如实记录根因(classic 兜底在 89 张硬帧上 ~10 CPU-s/张,1039 CPU-s 总量决定墙钟下限;10.6x CPU 效率证明并行层达标);SC1 无时间硬指标,不构成缺口 |
| 01-02 | TaskFailed 信封不杀整批(结果数 == 任务数) | ✓ VERIFIED(_guarded_worker 全捕获 + 消费端 consumed==total 断言 pipeline.py:187-193 + 坏 JPEG 测试) |
| 01-02 | 全部角点 2*win+1<sq_px 并必过 canonicalize | ✓ VERIFIED(326 检出帧程序核验零违反;四变体规范化 allclose 测试 + SB/classic 同图 allclose 测试) |
| 01-03 | 每张检出图 visualization/{image_id}.jpg(烧录路径+路线) | ✓ VERIFIED(326 张文件名集合与 detected 精确相等) |
| 01-03 | common_fov_pairs 两路均检出成对、对数 <= min | ✓ VERIFIED(112 <= min(141,155)、15 <= 15;配对规则测试) |
| 01-03 | 失败帧原图 fail_ 前缀拷贝、开关控制、fail 只记检测失败 | ✓ VERIFIED(80 文件名与 fail 行精确对应;copy2 字节一致测试;开关关闭测试) |
| 01-03 | 运行前清理 output_root/{train,test} 重建无残留 | ✓ VERIFIED(pipeline.py:123-124 rmtree;孤儿文件清理测试) |
| 01-03 | 全量运行后 data/ mtime+size 指纹不变 | ✓ VERIFIED(指纹一致) |
| 01-03 | 无自动剔除/排除机制(D-03) | ✓ VERIFIED(grep exclude 零命中于 src/ 与 config.yaml) |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `pyproject.toml` | uv 构建 + stereo-calib 入口 | ✓ VERIFIED | 入口/依赖约束齐(opencv>=4.14,<5.0 等);50 例测试经 uv 运行 |
| `config.yaml` | 唯一参数来源全分段 | ✓ VERIFIED | board/data/output_root/cv2_threads/preprocess 全分段,含 sb_fallback_classic/copy_fail_originals/jpeg_quality |
| `src/stereo_calib/io/config.py` | 配置层 + 中文校验 + 互斥校验 | ✓ VERIFIED | 300 行,frozen dataclass 六件,yaml.safe_load,双向互斥校验(实测) |
| `src/stereo_calib/io/dataset.py` | 扫描/记录/imread_checked | ✓ VERIFIED | 两层通配 + _SAFE_IMAGE_ID + 韧性警告;源码无写死张数 |
| `src/stereo_calib/corners/window.py` | 实测边长 + 窗口断言 | ✓ VERIFIED | 含 "2 * w + 1 < sq_px" 断言;帧级/逐角点两粒度 |
| `src/stereo_calib/corners/canonical.py` | canonicalize 唯一形态 | ✓ VERIFIED | 纯几何翻转;文件内无 burst |
| `src/stereo_calib/corners/detect.py` | detect_sb + detect_classic + grid_defects | ✓ VERIFIED | 234 行双检测器;_RECOVERY_MAX_ITER=100/_RECOVERY_EPSILON=1e-6;cornerSubPix == 1 |
| `src/stereo_calib/preprocess/pipeline.py` | 编排:扫描→并行→配对→落盘→可视化→汇总 | ✓ VERIFIED | 320 行全链路;写路径均从 output_root 派生 |
| `src/stereo_calib/preprocess/artifacts.py` | npz + CSV + meta + load_corners | ✓ VERIFIED | 八数组契约 + allow_pickle=False + 三个 CSV 写出(utf-8-sig) |
| `src/stereo_calib/preprocess/parallel.py` | loky 并行 + TaskFailed 信封 | ✓ VERIFIED | parallel_config(backend="loky", inner_max_num_threads) + return_as="generator" + 串行回退 |
| `src/stereo_calib/visualization/overlay.py` | render_marked + copy_fail_original | ✓ VERIFIED | drawChessboardCorners + putText(ASCII)+ 字号按 min(h,w) 推导 + shutil.copy2 |
| `tests/`(6 文件) | 不变量与端到端锁定 | ✓ VERIFIED | 42 个测试函数/50 例全绿(实测 14.9s) |

所有 artifact 均通过四级检验:存在 → 实质内容(无桩/无占位)→ 接线(见下)→ 真实数据流(npz 内为真实检测数据,非空/静态)。

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| cli/main.py | io/config.py | load_config + cv2.setNumThreads | ✓ WIRED(main.py:18,48-49) |
| pipeline.py | corners/detect.py | detect_sb → detect_classic(开关) | ✓ WIRED(pipeline.py:32,73-75) |
| pipeline.py | preprocess/artifacts.py | save_corners_npz / write_usable_csv / write_fail_csv / write_common_pairs_csv | ✓ WIRED(:35-41,229-252) |
| detect.py | corners/canonical.py | 两路输出必过 canonicalize | ✓ WIRED(detect.py:32,61,228) |
| detect.py | corners/window.py | derive_window_frame/per_corner + 断言 | ✓ WIRED(detect.py:33-37,58,198-201) |
| pipeline.py | preprocess/parallel.py | run_parallel(生成器 + zip 对回) | ✓ WIRED(pipeline.py:42,164-165) |
| pipeline.py | visualization/overlay.py | render_marked / copy_fail_original(开关门控) | ✓ WIRED(pipeline.py:43,267-282) |
| overlay.py | cv2.drawChessboardCorners | 标准叠加 | ✓ WIRED(overlay.py:56) |
| dataset.py | io/config.py DataDirs | scan_dataset(cfg.data) | ✓ WIRED(dataset.py:32) |

无孤儿 artifact;无 PARTIAL 链路。

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| cam{n}_{split}.npz | uv/detected/source | detect_sb/detect_classic on 406 真实图像 | 是:326 检出帧、sb 317/classic 9、sq_px 中位 ~18.9px 范围 8.5~69.9 | ✓ FLOWING |
| usable_*.csv | rows | npz detected 填槽(usable_rows) | 是:行数与 detected 精确相等 | ✓ FLOWING |
| common_fov_pairs_*.csv | rows | 按 group_id 两路 detected 配对 | 是:112/15 对,路径可回溯到真实文件 | ✓ FLOWING |
| visualization/*.jpg | 标注像素 | npz uv 行 → drawChessboardCorners | 是:326 张可解码非空图 | ✓ FLOWING |
| fail_*.csv + fail_originals/ | rows + 拷贝 | result None / TaskFailed 信封 | 是:80 行 + 80 文件精确对应 | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 全量测试套件 | `uv run pytest -q` | 50 passed in 14.93s | ✓ PASS |
| CLI 三子命令 + --config | `uv run stereo-calib --help` | preprocess/intrinsic/stereo 列出,--config 默认 config.yaml,RC=0 | ✓ PASS |
| preprocess 零算法参数 | `uv run stereo-calib preprocess --help` | 仅 --help 一项 | ✓ PASS |
| 源码防线 | `grep -c 'cv2\.cornerSubPix(' src/stereo_calib/corners/detect.py` | 1 | ✓ PASS |
| npz 八数组/schema/NaN/窗口断言/objp 契约 | python 逐 npz 断言 | 4/4 通过,winViol=0,usable==detected | ✓ PASS |
| 非法配置中文报错 + 退出码 1 | `--config bad1.yaml preprocess`(max_workers: 0) | 中文一句 + RC=1,无堆栈 | ✓ PASS |
| output_root/data 互斥(相等/互含) | load_config 直调两形态 | 均 ValueError 含实际路径 | ✓ PASS |
| data/ 只读指纹 | `find data/ -type f -printf … \| sort \| md5sum` | a183699da79875fc63f7cbfe75874398(与生产运行记录一致,复测不变) | ✓ PASS |
| 占位子命令 | `intrinsic` / `stereo` | 「由后续阶段实现」提示,RC=0 | ✓ PASS |

注:全量 preprocess 生产运行由 orchestrator 在验证前即时执行(107s/406 张),其产物为上述磁盘核验对象;本验证未重复 107s 全量跑(产物计数与内部一致性已全查)。

### Probe Execution

SKIPPED — 仓库无 scripts/*/tests/probe-*.sh,PLAN/SUMMARY 亦未声明任何 probe 脚本;运行时验证以上述 Behavioral Spot-Checks 替代。

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PREP-01 | 01-01 | YAML 配置 + 命令行零算法参数 | ✓ SATISFIED | config.yaml 全分段 + CLI 实测零算法参数 |
| PREP-02 | 01-01, 01-02 | SB 主路线(严禁二次精化)+ classic 兜底(窗口实测推导)+ 多进程并行 | ✓ SATISFIED | 防线 == 1 + 测试锁定 + loky 并行 + 生产 sb/classic 双路线生效 |
| PREP-03 | 01-01 | 每路相机可用图像清单 | ✓ SATISFIED | 4 个 usable CSV,行数 == detected |
| PREP-04 | 01-03 | 两路公共视场对清单 | ✓ SATISFIED | common_fov_pairs CSV(112/15 对),配对规则程序核验 |
| PREP-05 | 01-01 | 角点数据文件(2D + 3D 板系 + 图像索引) | ✓ SATISFIED | npz uv/objp/image_ids,objp 契约逐点核验 |
| PREP-06 | 01-03 | 角点可视化抽查图 | ✓ SATISFIED | 326 张标注图(角点正确性的人工核验 → human_verification) |
| INFRA-01 | 01-01 | 三模块独立 CLI 入口、上游输出即下游输入 | ✓ SATISFIED(Phase 1 范围) | 三子命令注册、preprocess 独立可跑、npz/CSV 契约即 Phase 2/3 接口;intrinsic/stereo 全量落地见 deferred(Phase 2/3 Goal) |
| INFRA-02 | 01-01 | 全部传参由配置文件控制 | ✓ SATISFIED | 所有算法参数经 config.yaml 解析校验;源码无算法常量 |
| INFRA-03 | 01-01, 01-03 | data/ 只读、产物写独立输出目录 | ✓ SATISFIED | 指纹级证明 + 写路径全从 output_root 派生 + 互斥校验 + mtime 测试锁定 |

无 ORPHANED 需求:REQUIREMENTS.md 映射到 Phase 1 的 9 个 ID 全部出现在三个 PLAN 的 requirements 字段。

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| src/stereo_calib/cli/main.py | 50-52 | `except (ValueError, OSError): pass` | ℹ️ Info | 有注释说明的有意为之(--help/配置未就位时不拦截用户),非静默吞错;命令函数内会再校验。非桩 |
| .planning/REQUIREMENTS.md | — | PREP-01/03/05/INFRA-01/02 已实现但复选框未勾 | ⚠️ Warning | 记录性缺口(deferred-items.md 已登记,01-03 明确留给 orchestrator/verifier 补记);建议补勾以保持 traceability 一致 |
| src/stereo_calib/io/dataset.py | 79-88 | CR-01(评审):同 split 叶子目录重名静默合并 | ⚠️ Warning | 潜在缺陷,仅影响未来含重名叶子目录的数据集;当前数据 406 任务全量对号(188/188/15/15),目标不受影响;建议以 /gsd-quick 跟进修复 + 补测试 |
| src/stereo_calib/corners/detect.py | 161-162 | WR-01(评审):偶数 cap 时阶梯不含 3 | ⚠️ Warning | 边界缺陷;生产数据 winViol=0 且恢复路径未误产;属后续加固项 |
| src/stereo_calib/preprocess/pipeline.py | 171 | WR-02(评审):信封异常统一标「读图失败」 | ⚠️ Warning | 误导性标签;生产 80 个 fail 全为「SB 与 classic 均未检出」不受影响;后续加固项 |
| src/stereo_calib/corners/detect.py | 58 | WR-03(评审):SB 路线经断言函数记录 win_size | ⚠️ Warning | 极小方格帧存在丢弃风险;当前数据最小 8.5px 未触发;后续加固项 |
| (其余) | — | WR-04/WR-05、IN-01~08(评审) | ℹ️ Info | corner 粒度未启用(cv2_threads=0 默认),均为不影响本数据集目标的加固建议;01-REVIEW.md 为 advisory |

债务标记扫描:TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER 在本阶段全部源码/测试/配置中零命中。ruff 9 处(C408×8 + B008×1)已在 deferred-items.md 登记(01-02/01-03 自有文件 ruff-clean)。

### Gaps Summary

**无功能性缺口。** 5/5 成功判据、19/19 计划级 must-haves、9/9 需求全部以代码与磁盘证据核验通过;全链路(命令→配置→扫描→并行检测→配对→落盘→可视化)四级检验(存在/实质/接线/数据流)无桩、无孤儿、无断链;data/ 只读达指纹级证明。

非阻塞事项:
1. **需人工完成**(本报告状态为 human_needed 的原因):SC4 的角点位置/顺序人工核验、80 张失败帧真假复核、CSV Excel 打开抽查 —— 三项均已列入 frontmatter human_verification。
2. **MVP 目标格式**:ROADMAP 目标非规范 User Story 格式(mode: mvp 下);建议 `/gsd mvp-phase 1` 重排或人工接受(见 MVP Mode Notes)。
3. **REQUIREMENTS.md 补勾**:PREP-01/03/05/INFRA-01/02(deferred-items.md 已登记,本次验证已给出全部 SATISFIED 证据,可直接补记 Complete)。
4. **评审遗留加固项**(advisory,均不影响本数据集目标):CR-01、WR-01~05、IN-01~08;建议在 Phase 2 前用 /gsd-quick 择要处理(优先 CR-01 与 WR-02)。
5. **01-02 时间估计偏差**:全量 98-107s vs 计划估计 30-60s —— 根因已如实记录(classic 兜底 CPU 总量),SC1 无时间硬指标,两根提速杠杆(fast_check/降分辨率)均须用户授权,不宜擅自变更检测行为。

---

_Verified: 2026-10-23T15:15:50Z_
_Verifier: Claude (gsd-verifier)_
