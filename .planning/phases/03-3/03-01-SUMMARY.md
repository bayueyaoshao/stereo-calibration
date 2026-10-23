---
phase: 03-3
plan: 01
subsystem: stereo
tags: [stereo-calibration, pair-loading, config-layer, synthetic-fixture, connection-key, allow_pickle]

# Dependency graph
requires:
  - phase: 01-preprocess (Phase 1)
    provides: 八数组角点 npz + common_fov_pairs_{split}.csv(对装载的输入契约;load_corners allow_pickle=False 防线、build_object_points 直接复用)
  - phase: 02-intrinsic (Phase 2)
    provides: intrinsics.json 15 字段契约(params 九名 + 冗余 K/D,固定内参唯一读取口)+ intrinsic.models.INTRINSIC_NAMES 布局常量
provides:
  - StereoConfig/StereoRefineConfig/StereoOutlierConfig(frozen dataclass,决策 3/4/5 锁定默认值,_parse_stereo 全键中文校验接入 load_config)
  - config.yaml stereo 段(Phase 3 全部算法参数/开关,每键中文注释)
  - load_pairs(对装载唯一途径:连接键 + 三重防线 + 行序锁 + 路径交叉核对 + objp 三方 allclose)
  - select_pairs(幸存子集相对序重建,供 03-03 剔除循环)
  - load_intrinsics(固定内参唯一读取口:按名读 params + D.shape==(5,) 布局防护)
  - make_stereo_synth/write_stereo_synth_files 合成双目夹具(真值 R_rel/t_rel 两视角合成,03-02/03-03 全部测试的数据源)
affects: [03-02 solve, 03-03 evaluate/pipeline, 03-04 e2e/CLI]

# Tech tracking
tech-stack:
  added: []  # 零新增依赖(uv.lock 不变,威胁项 T-03-SC 兑现)
  patterns: [连接键对齐装载, 三重防线(exclude/detected/finite), 行序锁, 按名读内参+布局断言, 真值几何合成夹具]

key-files:
  created:
    - src/stereo_calib/stereo/__init__.py
    - src/stereo_calib/stereo/observation.py
    - tests/test_stereo_config.py
    - tests/test_stereo_pipeline.py
  modified:
    - src/stereo_calib/io/config.py
    - config.yaml
    - tests/conftest.py

key-decisions:
  - "决策 3/4/5 默认值逐字落位:flip_ratio 0.5(证据充分才翻)、median_spread_max_deg 1.0°、baseline_band_mm [20,2000] 硬带"
  - "flip_ratio 校验上限 1.0:超过即『证据不足也翻』,与消歧设计相反,配置层当场拒收"
  - "合成夹具 K/D 取 _PARAMS_TRUE 同源量级(focal ~1850、k1≈-0.4 强畸变域),让夹具几何与生产数据同性质"
  - "exclude_pairs 校验收 (list, tuple) 双序列形态:缺 stereo 段的默认值 () 是元组,list-only 会把向后兼容路径炸掉"
  - "write_stereo_synth_files 复用 Phase 1 write_common_pairs_csv 落盘(契约零漂移),不手写 CSV"

patterns-established:
  - "对装载行序锁:image_ids 按 pairs CSV 行序定死,下游绝不重排(03-02/03-03 消费 Pairs 时继承)"
  - "源码防线自伤规避:被 grep 计数的调用串原文只出现在断言处,observation.py 注释措辞避开"
  - "stereo 段配置四步体例:frozen dataclass + _fail 中文报错 + 白名单 + bool 冒充拦截(与 intrinsic 段同构)"

requirements-completed: [STER-01]

# Metrics
duration: 15min
completed: 2026-10-24
---

# Phase 3 Plan 01: 配置层 + 对装载层 + 合成双目夹具 Summary

**stereo 段配置解析(flip_ratio 0.5 / spread 1.0° / baseline 带 [20,2000] 全键中文校验)+ 对装载层(连接键 group_id+'_cam{n}'、三重防线、行序锁、按名读内参带 D 布局防护)+ 固定种子真值双目合成夹具;真实 Phase 1/2 产物冒烟 112/15 对全过**

## Performance

- **Duration:** 15 min
- **Started:** 2026-10-23T19:25:35Z
- **Completed:** 2026-10-23T19:40:51Z
- **Tasks:** 3/3
- **Files modified:** 7(4 created + 3 modified)

## Accomplishments

- 配置层 stereo 段:三只 frozen dataclass + `_parse_stereo` 接入 `load_config`,全部键有默认、非法值中文报错('lm' 白名单拒收、bool 冒充拦截、baseline 带反序拒收、flip_ratio 上限 1.0);config.yaml 追加段头注释块 + 逐键行尾中文注释;缺 stereo 段的旧配置向后兼容走默认值
- 对装载层 `stereo/observation.py`:`load_pairs` 按连接键 `group_id+"_cam{n}"` 对齐两路 npz(utf-8-sig 读 CSV),exclude→detected→finite 三重防线 + 路径交叉核对 + objp 三方 allclose + min_pairs 下限;`select_pairs` 相对序子集重建;`load_intrinsics` 按名读 params(INTRINSIC_NAMES 导航)+ 直读 K/D + D.shape==(5,) 断言(Pitfall 13)
- 合成双目夹具:`make_stereo_synth` 固定种子真值几何(cam2 位姿由 R_rel@R1 / R_rel@t1+t_rel 组合导出,方向推导注释防 Pitfall 6;Z>0 自检;flip_pair_indices 注入 uv2[::-1])+ `write_stereo_synth_files` 八数组 npz / pairs CSV 工厂;零渲染、零读图
- 真实数据冒烟:train 112 对 / test 15 对全装载(连接键零缺失、uv 形状 (V,88,2)、行序与 CSV 一致),两路 intrinsics fx 读出 2265.1 / 2021.5;全量 124→150 测试零回归

## Task Commits

Each task was committed atomically:

1. **Task 1: 配置层 stereo 段** - `f73ae4f` (feat)
2. **Task 2: conftest 双目合成夹具** - `2495f37` (feat)
3. **Task 3: stereo/observation.py 对装载层 + 真实数据冒烟** - `9fdcfe4` (feat)

## Files Created/Modified

- `src/stereo_calib/io/config.py` — StereoRefineConfig/StereoOutlierConfig/StereoConfig + `_parse_stereo`(+`_parse_stereo_refine`/`_parse_stereo_outlier`)接入 load_config
- `config.yaml` — 末尾追加 stereo 段(段头注释块 + 逐键行尾中文注释,含决策 3/4/5 依据)
- `src/stereo_calib/stereo/__init__.py` — 空 init(照 intrinsic 包体例)
- `src/stereo_calib/stereo/observation.py` — Pairs/load_pairs/select_pairs/IntrinsicsView/load_intrinsics(模块 docstring 声明输入契约与零读图纪律)
- `tests/conftest.py` — SynthStereo dataclass + make_stereo_synth + write_stereo_synth_files
- `tests/test_stereo_config.py` — 14 例:合法加载全字段断言 + 12 例非法值参数化 + 缺段走默认值
- `tests/test_stereo_pipeline.py` — 12 例:装载契约(计数/行序/形状/未检出组过滤)、三重防线(连接键/detected/路径/objp/NaN/min_pairs)、select_pairs 相对序、load_intrinsics 契约与 D 布局防护、源码防线

## Decisions Made

- **决策 3/4/5 默认值逐字落位**(planner 锁定):flip_ratio=0.5、median_spread_max_deg=1.0、baseline_band_mm=(20.0, 2000.0),distance_rms_max_mm=5.0;校验规则按 plan 逐条实现
- **合成夹具占位路径**:write_stereo_synth_files 的 paths 造 `data/{split}/{gid}/{gid}/cam{n}.jpg` 占位字符串——双目链路零读图,paths 仅作交叉核对(plan 明示的设计,非 stub)
- **observation.py 的 load_intrinsics 把 INTRINSIC_NAMES import 提到模块顶**(plan 未规定位置;与 scoring.py 体例一致,避免函数内局部 import)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] exclude_pairs 的 list-only 校验把缺段默认值 () 拒掉**
- **Found during:** Task 1(既有 test_intrinsic_config 回归失败暴露)
- **Issue:** 计划原文"exclude_pairs 必须 list 且每项 str",但缺 stereo 段时 `section.get(key, default)` 返回元组 `()`,严格 list 校验使"缺段走默认值"验收不可达
- **Fix:** 放宽为 `(list, tuple)` 双序列形态(照 intrinsic.exclude_images 同款体例),裸字符串仍拒收;yaml.safe_load 只产 list,行为面无差
- **Files modified:** src/stereo_calib/io/config.py
- **Verification:** test_stereo_config + test_io_config + test_intrinsic_config 38 例全绿
- **Committed in:** f73ae4f(Task 1 commit 内)

**2. [Rule 1 - Bug] conftest 写 npz 的 paths 手写 dtype="<U64" 截断 65 字符路径**
- **Found during:** Task 2(写盘 roundtrip 冒烟暴露)
- **Issue:** 占位路径最长 71 字符,定长 64 截断后路径交叉核对必假失败
- **Fix:** 改 `numpy.asarray(list[str])` 自动按最长串取宽(照 save_corners_npz 同款),注释写明禁手写宽度
- **Files modified:** tests/conftest.py
- **Verification:** roundtrip 冒烟断言 paths 逐字符相等
- **Committed in:** 2495f37(Task 2 commit 内)

**3. [Rule 1 - Bug] 装载契约用例被 min_pairs 默认下限误拦**
- **Found during:** Task 3(测试首跑 4 failed 暴露)
- **Issue:** 合成树 8 对 < load_pairs 默认 min_pairs=10,契约类用例先撞下限防线
- **Fix:** 契约用例显式传 min_pairs=2(下限用例单独显式 min_pairs=100)
- **Files modified:** tests/test_stereo_pipeline.py
- **Verification:** tests/test_stereo_pipeline 12 例全绿
- **Committed in:** 9fdcfe4(Task 3 commit 内)

**4. [Rule 1 - Bug] observation.py 模块 docstring 含 "imread" 字样触发源码防线 count==0**
- **Found during:** Task 3(源码防线用例失败暴露)
- **Issue:** 计划要求 docstring 声明「绝不 imread」又要求 count("imread")==0,字面冲突;仓库体例(自伤规避)规定被计数串原文只出现在断言处
- **Fix:** docstring 措辞改为「绝不打开任何图像文件(零读图)」,语义不变
- **Files modified:** src/stereo_calib/stereo/observation.py
- **Verification:** test_observation_source_guards 绿 + grep count 0 复核
- **Committed in:** 9fdcfe4(Task 3 commit 内)

---

**Total deviations:** 4 auto-fixed(4 × Rule 1 bug 修正,均为本计划自身代码/测试的缺陷,非范围蔓延)
**Impact on plan:** 四处均为验收不可达的阻断性缺陷,修正后全部验收项达成;无 scope creep。

## Issues Encountered

- 全仓 `ruff check src/ tests/` 报 5 处预存在告警(0.17.2 新规则 RUF007/C408,位于 Phase 1 落地的 test_corners.py / test_preprocess_pipeline.py)——与本计划改动无关,按 Scope Boundary 不修,已登记 `.planning/phases/03-3/deferred-items.md` 建议后续 wave 统一处理
- 计划冒烟注释里 fx2 预估 "~2230 量级",实际 Phase 2 产物为 2021.5——冒烟断言口径是"与 Phase 2 产物一致"(直读同值),满足验收;数字差异仅是计划期的估算误差

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **03-02(solve)可直接消费的三个接口已锁死:** StereoConfig(全部阈值/开关)、load_pairs(对装载唯一途径,Pairs 行序契约)、load_intrinsics(固定内参唯一读取口);make_stereo_synth 的 flip_pair_indices 注入通道已为消歧测试备好
- **夹具消费约定:** 03-02/03-03 测试请经 `from conftest import make_stereo_synth, write_stereo_synth_files`(pytest rootdir 模式);合成树 8 对,装载用例记得 min_pairs 显式低于 8
- **无阻塞项**;真实产物冒烟证明 Phase 1/2 输出与本装载层契约严丝合缝

## Self-Check: PASSED

- 7 个代码/测试文件 + deferred-items.md 全部在位(FOUND)
- 三个任务 commit(f73ae4f / 2495f37 / 9fdcfe4)全部在 git log 中(FOUND)
- 临时 output symlink 已移除;最终 `uv run pytest -q` 150 passed 零回归

---
*Phase: 03-3*
*Completed: 2026-10-24*
