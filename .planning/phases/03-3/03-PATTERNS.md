# Phase 3: 模块3 双目外参标定 - Pattern Map

**Mapped:** 2026-10-24
**Files analyzed:** 15(11 新建 + 4 修改)
**Analogs found:** 15 / 15(Phase 2 的 `intrinsic/` 包已落地,模块体例/契约体例全部有仓库同源体;算法内容照 03-RESEARCH.md Port Map 从只读参考工程 2 移植)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog (本仓库) | Match Quality |
|-------------------|------|-----------|------------------------|---------------|
| `src/stereo_calib/stereo/__init__.py` | package-init | — | `src/stereo_calib/intrinsic/__init__.py`(空文件) | exact |
| `src/stereo_calib/stereo/observation.py` | loader (service) | file-I/O | `src/stereo_calib/intrinsic/observation.py`(三重防线/行序锁/objp allclose);pairs CSV 列契约见 `preprocess/artifacts.py` 写侧 | exact |
| `src/stereo_calib/stereo/solve.py` | solver (service) | transform/batch | `src/stereo_calib/intrinsic/solve.py`(θ 装箱/row_index/解析雅可比/trf 调用)+ `intrinsic/scoring.py`(solvePnP 调用形态) | role-match(算法源=参考2 core/stereo.py + multibundle.py) |
| `src/stereo_calib/stereo/evaluate.py` | evaluator (service) | transform | `src/stereo_calib/intrinsic/scoring.py`(per-frame 6 参数 BA + 失败契约)+ `intrinsic/metrics.py`(float64 指标唯一出口) | role-match(算法源=参考2 06_evaluate.py + triangulation.py) |
| `src/stereo_calib/stereo/pipeline.py` | pipeline (orchestration) | batch | `src/stereo_calib/intrinsic/pipeline.py`(rmtree/剔除循环/单路降级/中文汇总) | exact |
| `src/stereo_calib/stereo/artifacts.py` | artifact-writer | file-I/O | `src/stereo_calib/intrinsic/artifacts.py`(JSON 契约断言/npz 三件套/utf-8-sig CSV/f-string 报告) | exact |
| `tests/test_stereo_solve.py` | test | — | `tests/test_intrinsic_solve.py`(合成真值恢复/θ 往返/雅可比数值基准) | exact |
| `tests/test_stereo_evaluate.py` | test | — | `tests/test_intrinsic_scoring.py`(合成真值+失败不抛)+ `tests/test_intrinsic_models.py`(数值锁定层) | role-match |
| `tests/test_stereo_pipeline.py` | test | — | `tests/test_intrinsic_pipeline.py`(合成 npz e2e/_tree_snapshot/重跑清残留) | exact |
| `tests/test_stereo_config.py` | test | — | `tests/test_intrinsic_config.py`(MINIMAL_YAML/_mutate/parametrize) | exact |
| `tests/test_stereo_guards.py` | test | — | `tests/test_intrinsic_scoring.py` L114-118 + `tests/test_intrinsic_models.py` L172-183 | exact |
| `tests/conftest.py`(修改) | fixture | — | 自身:`make_chessboard_image` 的 projectPoints 合成思路,新增双目几何合成夹具 | exact(自身) |
| `src/stereo_calib/io/config.py`(修改) | config | — | 自身:`_parse_refine`(L300-335)/`_parse_outlier`(L338-373)四步模式延伸 | exact(自身) |
| `src/stereo_calib/cli/main.py`(修改) | controller | request-response | 自身:`intrinsic` 命令 L77-95 逐行对照,替换 `stereo` 占位 L98-101 | exact(自身) |
| `config.yaml`(修改) | config | — | 自身:`intrinsic:` 段(L51-67)注释体例延伸 | exact(自身) |

**复用不改:** `src/stereo_calib/intrinsic/models.py` 是唯一数学源(投影/畸变/雅可比/单应变换全在此),stereo/ 只 import——**本阶段零修改**;`preprocess.artifacts.load_corners`/`build_object_points` 同样直接 import。

## 算法源(只读参考工程 2,只引用不修改)

`/home/lx/Works/src/bolight_robot/vvd_daq/multi_cameras/data_acquire/calibration/`(行号经 03-RESEARCH.md 实核):

```
core/stereo.py        L36-49 相对位姿 / L82-91 solvePnP 形态 / L113-121 中位聚合 / L137-163 逐帧 RMS
core/multibundle.py   L61-205 JointBundleAdjustment(pack/unpack 94-124、残差 133-150、稀疏 152-170)
                      L423-494 multiview_pose_ba(外参换算 447-457、残差 462-478)
core/triangulation.py L24-29 projection_matrix / L32-40 undist_pix / L59-85 DLT 保号 / L88-100 farthest
cli/05_extrinsic_multi.py L102-131 消歧(阈值 0.5)、L265-294 外参 BA 调用形态
cli/06_evaluate.py    L180-203 消歧(锚=当帧 cam1 PnP)、L245-269 undist+DLT+板系+对角距离
```

**规则(照 02-PATTERNS.md):** 算法与公式抄参考工程 2;文件结构、docstring 体例、错误收口、落盘契约、测试体例全部照本仓库 `intrinsic/` 既有模式。每个新文件模块 docstring 注明「移植自参考工程 …(只读引用,绝不修改)」。

---

## Pattern Assignments

### `src/stereo_calib/stereo/observation.py`(loader,file-I/O)— 对装载层

**Analog:** `src/stereo_calib/intrinsic/observation.py`(171 行,全读)

照抄四件事,改数据源:(1) frozen dataclass 载体 `Pairs`(照 `Views` L42-56:元组 ids + (V,88,2) uv 数组 + 行序契约 docstring);(2) 三重防线(exclude 对清单 → detected 断言 → `np.isfinite` 中文 ValueError,L138-159);(3) objp 与 `build_object_points(cfg.board)` 的 `np.allclose` 交叉核对(L125-131);(4) 行序锁——装载顺序按 pairs CSV 行序定死、下游绝不重排(docstring L17-19 的写法照搬)。**新增:连接键对齐**——`group_id + "_cam{n}"` 在两路 npz `image_ids` 里定位行(L102 的 `str(x)` 归一写法照用),不命中即中文报错;两路 objp 互相 allclose;路径交叉核对(csv cam{n}_path vs npz paths[row])。`load_corners` 直接 import(allow_pickle=False 防线不新写)。双目链路零读图——不搬 `_image_size_from_paths`。

### `src/stereo_calib/stereo/solve.py`(solver,transform/batch)

**Analog:** `src/stereo_calib/intrinsic/solve.py`(422 行,全读)

**θ 装箱/解箱 + 行号契约**照 `pack_theta`/`unpack_theta`(L174-221)与 `row_index`(L159-171)的体例:θ = [6 外参 | 每对 (ω,t) 6 个](cam1 恒 (I,0) 不进 θ),互逆断言 + 长度对不上中文 AssertionError。

**least_squares 调用形态**(L384-395 逐项对应):

```python
    sol = least_squares(
        traced_residual, theta0,
        jac=make_jacobian_fn(objp_arr, uvs_arr),
        # 不传稀疏结构矩阵参数:jac 为 callable 时 scipy 对它静默忽略(WR-02 教训)
        method=refine_cfg.method, tr_solver="lsmr", x_scale="jac",
        loss="linear",   # 离群由剔除机制处理,不引入鲁棒核
        max_nfev=..., ftol=..., xtol=..., gtol=...)
```

**解析稀疏雅可比**照 `make_jacobian_fn`(L272-308):`transform_to_camera(want_jacobian=True)` + `project_point_jacobian` einsum 链式 + COO→csr;严禁手写 −[Xc]×。**solvePnP 调用与异常家族**照 `intrinsic/scoring.py` L86-120(try/except 数值家族收口);PnP 失败帧跳过不入堆叠(防 NaN 传播,Pitfall 9)。**新增:** 相对位姿/中位聚合/消歧按 Port Map 移植;`_relative_pose` docstring 照参考原样带推导;离散度诊断超 `median_spread_max_deg` 中文报错。

### `src/stereo_calib/stereo/evaluate.py`(evaluator,transform)

**Analog:** `src/stereo_calib/intrinsic/scoring.py`(120 行,全读)+ `intrinsic/metrics.py`

**单帧失败契约**照 `PoseScore`/`_failure`(L36-52):任何失败路径返回 (NaN, 中文备注) 绝不抛异常中断整批;**AssertionError-safe 边界**(quick 261024-cr02 定案,scoring.py L114-120)照抄——数值异常家族收口为单帧失败,形状/参数契约仍在入口防线拦:

```python
    except (ValueError, ArithmeticError, RuntimeError, AssertionError, cv2.error) as e:
        return _failure(f"...:{type(e).__name__}——{e}")
```

**per-frame BA 初值路线**照 scoring L87-96:`models.unproject` 反投影 → 单位阵 K + 零畸变 solvePnP(强畸变防错误盆地,Phase 2 实测教训原文可引);6 参数 trf 残差只经 models。**指标口径**照 metrics.py:逐帧 RMS 与 pooled(平方和均值开根,不是逐帧简写平均)全 float64 单一出口;DLT/undist_pix/板系反变换/对角距离按 Port Map 移植,三角化只许住本文件(源码防线)。

### `src/stereo_calib/stereo/pipeline.py`(pipeline,batch)— 编排层

**Analog:** `src/stereo_calib/intrinsic/pipeline.py`(310 行,全读)

照抄五件事:(1) **运行前清理**(L252-258,目标改 `output_root/stereo`,安全前提注释照写);(2) **剔除循环** `_drop_outliers`(L59-109)整段平移:阈值 = `rms_multiplier × 幸存中位`、双保险丝(max_drop_fraction/max_iterations,加 min_pairs)、循环说明中文留痕、精化不进循环(L80 `no_refine = RefineConfig(enabled=False)` 同款——剔除循环用 pnp_median 快解);(3) **test 永不剔除只标记**(L162 注释「考卷不可改」照搬);(4) **单路降级**(L266-281:某环节失败降级留痕、绝不炸整批;except 家族 L269 五类);(5) **中文汇总体例**(L290-310:逐项指标 + PASS/FAIL + 剔除留痕 + 产物绝对路径清单)。双目把「两路相机循环」换成「train 求解 → test 评估」两段,pnp_median 阶段逐对 RMS 进 `typer.echo` 播报(L87 体例)。

### `src/stereo_calib/stereo/artifacts.py`(artifact-writer,file-I/O)

**Analog:** `src/stereo_calib/intrinsic/artifacts.py`(295 行,全读)

照抄四件:(1) **JSON 契约字段全集断言**(L43-47 常量 + L107-109)——`_EXTRINSICS_STEREO_JSON_KEYS` 照写,字段按 03-RESEARCH Pattern 7;(2) **npz 落盘** `write_residuals_npz`(L117-134):`_unicode_array` 定长 unicode、float64 显式、`allow_pickle=False` 可读;`train_residuals.npz` 照此;(3) **utf-8-sig CSV**(L155-168):`_PER_FRAME_FIELDS` 模块顶部常量 + `csv.DictWriter`,列改 group_id/route/reproj_rms/dist_err_mm/dropped/note;(4) **f-string markdown 报告**(L176-295):指标汇总表 + 剔除留痕表 + 判定列(PASS/FAIL 只作报告结论不作退出条件,L192-193 注释照搬)+ 配置快照附录。读侧 intrinsics.json 按 15 字段契约消费(优先直读 K/D 并断言 D.shape==(5,),Pitfall 13)。

### `src/stereo_calib/io/config.py`(修改,config)

**Analog:** 自身四步照抄:**1.** frozen dataclass 嵌套段(L89-132 体例)——新增 `StereoRefineConfig`/`StereoOutlierConfig`/`StereoConfig`(outlier 字段 `min_pairs` 代替 `min_views`,默认值按 RESEARCH Pattern 8);`StereoCalibConfig`(L134-143)追加 `stereo: StereoConfig = field(default_factory=StereoConfig)`。**2.** `_fail` 统一中文报错(L146-151)直接复用。**3.** 白名单校验:`_ALLOWED_REFINE_METHODS`(L28)复用,method 只收 "trf";容器型值先判 str 再 in set(L310-316 WR-01 注释照搬)。**4.** bool 冒充拦截(L267-269 注释原样)。接入点:`load_config` L441-442 之后加 `stereo = _parse_stereo(raw)` 并传入构造器;解析函数放 `_parse_intrinsic` 之后,保持自上而下现序。

### `src/stereo_calib/cli/main.py`(修改,controller,request-response)

**Analog:** 自身 `intrinsic` 命令(L77-95)逐行对照,替换占位 L98-101(`_not_implemented` 调用删除,函数名 `stereo` 保留):

```python
    try:
        from stereo_calib.stereo.pipeline import run_stereo   # 局部导入体例照 L83
        run_stereo(load_config(_config_path))
    except (ValueError, OSError, ArithmeticError, RuntimeError,
            AssertionError, cv2.error) as e:
        typer.echo(f"双目标定失败:{e}")
        typer.echo("  请检查 config.yaml 配置与 Phase 1/2 产物(output/ 下的角点 npz 与 intrinsics.json),"
                   "修正后重跑 stereo-calib stereo。")
        raise typer.Exit(code=1) from e
```

help 文案写「模块3 双目外参标定:消费 Phase 1 对清单/角点 npz + Phase 2 intrinsics.json → …」。`--config` callback 不动。

### `config.yaml`(修改,config)

**Analog:** 自身 `intrinsic:` 段(L51-67):段头注释块 + 逐键行尾中文注释(含义 + 默认值理由)。追加 `stereo:` 段,键位按 03-RESEARCH Pattern 8(refine.{enabled,method,max_nfev,ftol,xtol,gtol} / outlier.{enabled,rms_multiplier,max_drop_fraction,max_iterations,min_pairs} / flip_ratio / median_spread_max_deg / distance_rms_max_mm / baseline_band_mm / exclude_pairs)。

### 测试五件套 + conftest

| Test | Analog | 要点 |
|---|---|---|
| `test_stereo_config.py` | `test_intrinsic_config.py`(144 行,全读) | MINIMAL_YAML 加 stereo 段全键显式;`_mutate` 复用;参数化:rms_multiplier=1.0、max_drop_fraction=0、enabled=1、method='"lm"'、flip_ratio=0、baseline_band_mm 反序;缺 stereo 段走默认值(L114-120 体例) |
| `test_stereo_pipeline.py` | `test_intrinsic_pipeline.py` | `_synth_npz`(L404-443)合成思路升级:真值双目几何(R_rel/t_rel 已知)两视角 projectPoints + 0.05px 噪声 → 八数组 npz + pairs CSV,不读图;`_tree_snapshot`(L396-402)照抄;e2e 断言 JSON 契约/CSV 列序/报告节/剔除留痕/重跑清残留(L647-665 体例)/data 只读 |
| `test_stereo_solve.py` | `test_intrinsic_solve.py` | 合成真值恢复(转角误差 <0.1°、t 误差 mm 级,L79-102 断言体例);θ pack/unpack 往返(L195);解析雅可比对 `approx_derivative` 逐列 <1e-6(L210);精化开关关返回初值(L161);row_index 契约(L179) |
| `test_stereo_evaluate.py` | `test_intrinsic_scoring.py` + `test_intrinsic_models.py` | 合成板 DLT 往返恢复对角距离(<0.5mm);保号除法(负齐次分量用例,Pitfall 5);undist P=K 契约(对 `K @ undistortPoints(...,None,None)` 逐位);消歧翻转(arr[::-1] 注入→翻回,scoring L48 帧级断言体例);mid-BA Z<=0 收口不抛(scoring L94 体例) |
| `test_stereo_guards.py` | scoring L114-118 + models L172-183 | grep 计数:`cv2.stereoCalibrate` 全 src/ == 0;`_distort` 在 stereo/ 计数 == 0;`triangulate`(SVD 零空间)只在 evaluate.py;精化 jac 路线与 sparsity 组合断言(Pitfall 14)。**docstring/断言消息不写被锁调用串原文**(自伤规避,models L175 注释照搬) |
| `tests/conftest.py`(修改) | 自身 `make_chessboard_image` L69-165 | 新增双目合成夹具:K/D 取 `_PARAMS_TRUE` 量级(focal ~1850、k1≈-0.4)+ 假想 R_rel/t_rel 真值 → 两视角角点合成器(纯 projectPoints,零渲染);随机位姿种子固定(rng default_rng 体例) |

---

## Shared Patterns

1. **单一数学源** — `intrinsic/models.py` 复用零修改;stereo/ 残差只经 `models.project`/`transform_to_camera`/`unproject`;`_distort` 在 stereo/ 计数 0(源码防线)。
2. **float64 指标唯一出口** — metrics.py 口径延续:OpenCV 返回值只进 solver_meta,报告指标全 float64 回代(pooled = sqrt 全残差平方均值,非逐帧平均)。
3. **单帧降级边界** — scoring.py L114-120 except 家族(AssertError-safe);evaluate 侧单对失败 NaN+备注,pipeline 侧单段失败降级留痕。
4. **Pattern 7 JSON 契约** — artifacts.py L43-47+L107-109 字段全集断言模式;`extrinsics_stereo.json` 字段按 03-RESEARCH Pattern 7。
5. **utf-8-sig CSV + npz 安全** — artifacts.py L164-167(CSV)、`load_corners`/`_unicode_array`(allow_pickle=False、定长 unicode)。
6. **中文进度播报 + 留痕** — pipeline.py L87/L290-310 体例:每轮剔除/每段求解 `typer.echo`,flip_log/dropped_pairs 进 JSON+报告。
7. **源码防线测试** — grep count 锁定(禁 stereoCalibrate、三角化单文件、models 单源)。
8. **config frozen dataclass 四步** — parse/validate/defaults/中文 `_fail`;bool 先判;容器值先判 str。
9. **CLI 零算法参数** — `--config` 唯一选项;stereo 全部开关/阈值进 config.yaml。

## No Analog Found

文件体例层面无(全部有仓库同源体)。算法内容层面(PnP-median/消歧/DLT/相对位姿)仓库无先例,须按 03-RESEARCH.md Port Map 从参考工程 2 移植——与 02-PATTERNS.md「No Analog Found」同性质,算法源已在上文列全。

## Conventions

- **提交前缀:** `feat(03-0N):` / `test(03-0N):` / `fix(03):` / `docs(03):`(照 git log 既有 `feat(02-03)` 体例);GSD 流程产物由 orchestrator 提交。
- **测试:** `uv run pytest tests/test_stereo_solve.py -x`(task 级);`uv run pytest -q`(wave 合并,当前 124 tests 全绿基线);无 skip 无放宽阈值。
- **生产验收:** `uv run stereo-calib stereo`(真实数据,test 距离 RMS < 5mm)。
- **环境注意:** worktree 无 data/,真实数据冒烟需 symlink 主仓 data/(Phase 1 既有做法)。

## Metadata

**Analog search scope:** `src/stereo_calib/**`(24 文件)、`tests/**`(11 文件)、`config.yaml`;参考工程 2 仅作算法源引用(03-RESEARCH.md Port Map 已逐行核实行号,不重复展开)
**Files read in full:** 11(models/observation/scoring/metrics/solve/pipeline/artifacts.py、io/config.py、cli/main.py、config.yaml、conftest.py、test_intrinsic_models.py、test_intrinsic_config.py)+ 3 处定向 grep(test_intrinsic_pipeline/solve/scoring 的用例名与防线行号)
**Pattern extraction date:** 2026-10-24
