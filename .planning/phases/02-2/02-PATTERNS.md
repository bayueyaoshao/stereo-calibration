# Phase 2: 模块2 内参标定 - Pattern Map

**Mapped:** 2026-10-24
**Files analyzed:** 16(13 新建 + 3 修改)
**Analogs found:** 16 / 16(数学层 4 个文件仓库内无算法同源体,但模块体例/契约体例有直接 analog,算法源在只读参考工程)

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog (本仓库) | Match Quality |
|-------------------|------|-----------|------------------------|---------------|
| `src/stereo_calib/intrinsic/__init__.py` | package-init | — | `src/stereo_calib/preprocess/__init__.py`(空文件) | exact |
| `src/stereo_calib/intrinsic/models.py` | math-model (utility) | transform | `src/stereo_calib/corners/canonical.py`(纯函数模块体例) | role-match(算法源=参考工程 pinhole_brown.py) |
| `src/stereo_calib/intrinsic/observation.py` | loader (service) | file-I/O | `src/stereo_calib/preprocess/artifacts.py` `load_corners`/`build_object_points` | exact(契约读取入口已存在,直接复用) |
| `src/stereo_calib/intrinsic/solve.py` | solver (service) | transform/batch | `src/stereo_calib/corners/detect.py`(cv2 调用封装 + float32 纪律 + 失败契约体例) | role-match(算法源=参考工程 solve/{opencv,scipy_solver,residuals}.py) |
| `src/stereo_calib/intrinsic/metrics.py` | utility | transform | `src/stereo_calib/corners/window.py`(纯 numpy 指标函数体例) | role-match(算法源=参考工程 solve/result.py) |
| `src/stereo_calib/intrinsic/scoring.py` | service | transform | `src/stereo_calib/intrinsic/solve.py`(同为 least_squares 封装,同期新写) | role-match(算法源=参考工程 solve/scoring.py) |
| `src/stereo_calib/intrinsic/pipeline.py` | pipeline (orchestration) | batch | `src/stereo_calib/preprocess/pipeline.py` `run_preprocess` | exact(编排体例逐段对应) |
| `src/stereo_calib/intrinsic/artifacts.py` | artifact-writer | file-I/O | `src/stereo_calib/preprocess/artifacts.py`(npz/meta.json/CSV 三件套) | exact |
| `tests/test_intrinsic_models.py` | test | — | `tests/test_corners.py`(数值精度断言 + 源码防线) | role-match |
| `tests/test_intrinsic_solve.py` | test | — | `tests/test_corners.py` 源码防线 + `tests/test_preprocess_pipeline.py` 合成数据 | role-match |
| `tests/test_intrinsic_scoring.py` | test | — | `tests/test_corners.py`(合成真值回归用例体例) | role-match |
| `tests/test_intrinsic_pipeline.py` | test | — | `tests/test_preprocess_pipeline.py`(合成 npz → 进程内跑命令 → 契约断言) | exact |
| `tests/test_intrinsic_config.py` | test | — | `tests/test_io_config.py`(MINIMAL_YAML + _mutate + parametrize) | exact |
| `src/stereo_calib/io/config.py`(修改) | config | — | 自身:`_parse_preprocess`/`_parse_corners` 既有模式延伸 | exact(自身) |
| `src/stereo_calib/cli/main.py`(修改) | controller | request-response | 自身:`preprocess` 命令体例;`intrinsic` 占位 L75-78 替换 | exact(自身) |
| `config.yaml`(修改) | config | — | 自身:`preprocess:` 段注释体例延伸 | exact(自身) |

## 算法源(只读参考工程,只引用不修改)

`/home/lx/Works/src/lx/github/intrinsic-calibration/src/calib/` 布局(移植映射):

```
src/calib/
├── models/pinhole_brown.py   → intrinsic/models.py(裁剪为 M2 单模型,去注册机制)
├── models/base.py            → intrinsic/models.py 的 transform_to_camera(L185-191 Rodrigues 雅可比修复版)
├── solve/opencv.py L122-151  → intrinsic/solve.py OpenCV 初值路径
├── solve/scipy_solver.py     → intrinsic/solve.py scipy 精化路径(装箱/实用收敛)
├── solve/residuals.py        → intrinsic/solve.py(row_index 行序契约 + 解析稀疏雅可比)
├── solve/result.py L114-155  → intrinsic/metrics.py(compute_result_metrics)
├── solve/observation.py      → intrinsic/observation.py(过滤/断言/行序锁;npz 读取复用本仓库 load_corners)
├── solve/scoring.py L79-158  → intrinsic/scoring.py(solve_pose_only)
├── solve/result.py(三件套)  → intrinsic/artifacts.py(落盘体例照本仓库 artifacts.py)
└── config/calibration.yaml   → config.yaml intrinsic: 段(exclude_images/scipy 白名单防呆)
```

**规则:** 算法与公式抄参考工程;文件结构、docstring 体例、错误收口、落盘契约、测试体例全部照本仓库既有模式。每个新文件的模块 docstring 按 canonical.py/detect.py 体例注明「移植自参考工程 …(只读引用,绝不修改)」。

---

## Pattern Assignments

### `src/stereo_calib/intrinsic/pipeline.py`(pipeline,batch)— 编排层

**Analog:** `src/stereo_calib/preprocess/pipeline.py`

**模块头 + 依赖导入体例**(preprocess/pipeline.py L22-43):绝对导入 `stereo_calib.*`、frozen dataclass 配置整只传入、typer 只用于中文输出:

```python
from stereo_calib.io.config import StereoCalibConfig
from stereo_calib.preprocess.artifacts import build_object_points, ...
```

**运行前清理模式**(L130-138,INTR 阶段照抄,目标改为 `output_root/intrinsic`):

```python
    # ---- 运行前清理(D-11 固定覆盖 + Pitfall 7 防残留)----
    for split in _SPLITS:
        shutil.rmtree(out_root / split, ignore_errors=True)
```
注意 docstring 里的安全前提写法:明确「rmtree 只可能命中 output_root 之下的产物目录」,因为 `load_config` 已拒绝 output_root 与 data 重叠——intrinsic 的 rmtree 注释照写同一前提。

**单帧失败不中断 + 契约守卫**(L202-208):结果流不完整就整体报错、绝不静默少行——intrinsic 装载层断言(幸存视图数下限)与剔除保险丝触发时同样「如实报错/记录,绝不静默」:

```python
    if consumed != total:
        raise ValueError(
            f"并行任务流不完整:应到 {total} 个结果,实际收到 {consumed} 个。"
            f"这是异常信封防线被击穿的信号,请把本错误连同运行日志反馈维护者。"
        )
```

**中文汇总体例**(L300-334):逐相机统计、中文说明、末尾列全部产物绝对路径:

```python
    typer.echo("预处理完成,产物已写盘。检出统计(检出 / 总数):")
    ...
    typer.echo("  全部产物(绝对路径):")
    for p in artifacts:
        typer.echo(f"    {Path(p).resolve()}")
```
intrinsic 的 `run_intrinsic(cfg)` 照此:两路循环(cam1/cam2 串行)、每轮剔除留痕进报告、收尾打印 train/test RMS 与剔除清单。

**进度播报**(L48-49, L163-165):长耗时环节每 N 步 `typer.echo(f"  已处理 …")`——intrinsic 的 OpenCV 求解(~25-35s/路)与剔除循环轮次照此播报。

---

### `src/stereo_calib/intrinsic/artifacts.py`(artifact-writer,file-I/O)— 落盘层

**Analog:** `src/stereo_calib/preprocess/artifacts.py`(整文件,150 行,一次性读完照体例新写)

**npz 写盘 + meta.json 旁车**(L54-92):`mkdir(parents=True)` → `np.savez` → 显式 dtype → meta.json `ensure_ascii=False, indent=2`、`config_snapshot` 留痕:

```python
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{name}.npz"
    arrays = {
        "image_ids": np.asarray(image_ids),
        "uv": np.asarray(uv, dtype=np.float64),
        "source": np.asarray(source, dtype="<U16"),
        ...
    }
    np.savez(npz_path, **arrays)
    full_meta = {"file": npz_path.name, ..., **meta}
    with open(out_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
        json.dump(full_meta, f, ensure_ascii=False, indent=2)
```
intrinsic 的 `train_residuals.npz`/`test_residuals.npz`/`extrinsics.npz` 照此:字符串一律定长 unicode(不需要 pickle)、float64 显式、image_ids 列随行落盘供对账(Pitfall 10)。

**npz 读取安全防线**(L95-102,observation.py 直接 import 复用,不新写):

```python
def load_corners(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
```

**utf-8-sig CSV**(L110-121):`encoding="utf-8-sig", newline=""` + `csv.DictWriter` + 字段名常量在模块顶部:

```python
_USABLE_FIELDS = ["image_id", "path", "cam", "route", "sq_px"]

def write_usable_csv(path: Path | str, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_USABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path
```
intrinsic 的 `per_frame_cam{n}.csv`(INTR-04:列 = image_id / source / train_rms / test_rms / dropped / 中文说明)照此体例;`report.md` 用 f-string markdown(零模板引擎)。

**npz 契约文档体例**(L1-21 模块 docstring):逐数组列布局、dtype、NaN 语义、allow_pickle 理由——intrinsic/artifacts.py 的 docstring 照写自己的 npz 布局。

---

### `src/stereo_calib/intrinsic/observation.py`(loader,file-I/O)— 装载层

**Analog:** 复用 `preprocess/artifacts.py` 的 `load_corners` + `build_object_points`(L37-51, L95-102);断言/报错体例照 `io/dataset.py`

**objp 交叉核对所需函数已存在**(artifacts.py L37-51,直接 import,绝不重写):

```python
def build_object_points(board: BoardSpec) -> np.ndarray:
    xs = np.arange(board.cols, dtype=np.float64) * board.square_size_mm
    ys = np.arange(board.rows, dtype=np.float64) * board.square_size_mm
    objp = np.zeros((board.rows * board.cols, 3), dtype=np.float64)
    objp[:, 0] = np.tile(xs, board.rows)
    objp[:, 1] = np.repeat(ys, board.cols)
    return objp
```

**中文断言报错体例**(dataset.py L166-175):报错带病因定位与操作员指引:

```python
def imread_checked(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(
            f"照片读取失败:{path}。文件可能损坏、路径不对或没有读权限,"
            f"请人工检查这张图。"
        )
    return img
```
装载层的三重防线(detected 掩码过滤 → 有限值断言 → 幸存视图数下限)与 objp `np.allclose` 交叉核对(Pitfall 11)全部用这种带字段名/实际值/下一步指引的中文 ValueError。image_size 用 `imread_checked` 读首帧检出图像后 `height, width = img.shape[:2]` 派生(Pitfall 2)。

---

### `src/stereo_calib/intrinsic/models.py`(math-model,transform)— 数学层

**Analog(仓库体例):** `src/stereo_calib/corners/canonical.py`(52 行,全读)

**纯函数模块体例**(canonical.py L24-52):`from __future__ import annotations` + 仅 numpy 导入 + 类型标注齐全 + 输入 `np.asarray(..., dtype=np.float64)` 归一 + 返回新数组不改输入 + docstring 写「为什么」:

```python
def canonicalize(corners: np.ndarray, cols: int, rows: int) -> np.ndarray:
    """把行主序角点序列规范化为唯一形态:……

    corners: (rows*cols, 2) 任意变体的行主序角点;
    返回:  (rows*cols, 2) 规范化后的新数组(不修改输入)。
    """
    g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
```

**「移植裁剪」声明体例**(canonical.py L18-21):models.py 裁掉 M1/M3 与注册机制时照写:

```python
# 只移植 canonicalize;参考工程的 burst_consistency(同姿态连拍一致性
# 断言)不移植——本数据无同姿态连拍。
#
# 移植自参考工程 calib.corners.canonical(只读引用,绝不修改)。
```

**算法内容**:从参考工程 `models/pinhole_brown.py` 移植 `_distort`/`_radial`/project/project_jacobian/project_point_jacobian/unproject/intrinsics_matrix(RESEARCH.md Pattern 1 有核心摘录);`transform_to_camera` 从 `models/base.py` L185-191 移植(Rodrigues 雅可比修复版,严禁手写 −[Xc]×)。此文件是唯一数学源——solve/metrics/scoring 只准 import,不准出现公式(RESEARCH.md Anti-Patterns 第 1 条)。

**失败契约体例参考**(detect.py L41-66):`detect_sb` 展示「未检出返回 None、绝不返回半成品」的返回契约写法——models 的 unproject 对 Z≤0 之类按参考工程断言风格处理(数学层断言、编排层中文报错)。

---

### `src/stereo_calib/intrinsic/solve.py`(solver,transform/batch)

**Analog(仓库体例):** `src/stereo_calib/corners/detect.py`

**float32 装包纪律**——detect.py L96-98 与 L218 展示「cv2 绑定只收 32 位,进出显式转换」的既有写法,solve.py 的 calibrateCameraExtended 装包照同一纪律(RESEARCH.md Pattern 2):

```python
    refined = cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2).astype(np.float32),
                               (int(win), int(win)), (-1, -1), criteria)
    return refined.reshape(-1, 2).astype(np.float64)
```

**内部私有函数 + 模块级常量体例**(detect.py L72-84, L139-140):配置名→cv2 常量翻译、经验数值常量放模块顶部带来源注释:

```python
_CLASSIC_FLAG_VALUES = {
    "adaptive_thresh": cv2.CALIB_CB_ADAPTIVE_THRESH, ...
}
```

**算法内容**(参考工程只读移植,RESEARCH.md Pattern 2/3 有完整摘录):`solve/opencv.py` 的 flags=0 + float32 装包 + D 解包后 `assert D.shape == (5,)`;`solve/scipy_solver.py` + `solve/residuals.py` 的 θ 装箱、row_index 共用函数、COO→csr 稀疏雅可比、trf+lsmr+x_scale="jac"、实用收敛判定(status/nfev/final_cost/tail_improvement 全进 solver_meta)。OpenCV 返回 rms 只进 `solver_meta["opencv_rms"]`,绝不进指标(Pattern 4 源码防线)。

---

### `src/stereo_calib/intrinsic/metrics.py` 与 `scoring.py`(utility/service,transform)

**仓库 analog:** `src/stereo_calib/corners/window.py`(纯 numpy 小模块、单一职责、函数带中文 docstring 写清口径)。

**算法内容**(参考工程只读移植):
- metrics.py ← `solve/result.py` L114-155 `compute_result_metrics`:残差 (V,P,2) float64 回代 → per_view_rms / pooled rms_overall(RESEARCH.md Pattern 4 摘录);
- scoring.py ← `solve/scoring.py` L79-158 `solve_pose_only`:unproject 反投影初值 → 单位阵 K solvePnP → 6 参数 trf 精化;单帧失败记 (NaN, 中文备注) 不中断(RESEARCH.md Pattern 5 摘录,「不中断整批 + 留痕」体例同 preprocess/pipeline.py `_record_fail` L167-173)。

---

### `src/stereo_calib/io/config.py`(修改,config)

**Analog:** 自身既有模式——四步照抄:

**1. frozen dataclass 嵌套段**(L64-72 + L85-93):默认值用 `field(default_factory=...)`,顶层配置加字段:

```python
@dataclass(frozen=True)
class CornersConfig:
    classic_flags: tuple[str, ...] = ("adaptive_thresh", "normalize_image")
    subpix: SubpixConfig = field(default_factory=SubpixConfig)
```
新增 `RefineConfig` / `OutlierConfig` / `IntrinsicConfig` 三只 frozen dataclass(默认值取 RESEARCH.md Pattern 8),`StereoCalibConfig`(L85-93)追加 `intrinsic: IntrinsicConfig = field(default_factory=IntrinsicConfig)`。

**2. `_fail` 统一中文报错**(L96-101,直接复用):

```python
def _fail(field_name: str, actual, requirement: str) -> None:
    raise ValueError(
        f"配置字段 {field_name} 不合法:{requirement};实际读到的值是 {actual!r}。"
        f"请检查对应的 yaml 文件。"
    )
```

**3. 白名单校验**(L178-196 `_parse_subpix` + 模块级 `_ALLOWED_*` 常量 L20-24):method 白名单只收 "trf" 照此写;float 容差字段照 epsilon 的三连判:

```python
    epsilon = section.get("epsilon", cfg.epsilon)
    if not isinstance(epsilon, int | float) or isinstance(epsilon, bool) or epsilon <= 0:
        _fail("preprocess.corners.subpix.epsilon", epsilon, "必须是 > 0 的数(收敛容差,像素)")
```

**4. bool 冒充 int 拦截**(L212-217,先判 bool 再判其余;开关字段 enabled 照抄):

```python
    # 注意顺序:Python 里 True 同时也是 int(issubclass(bool, int)),但 1 不是 bool,
    # 所以先判 bool 再拒其余一切——整数 1/0 是 yaml 手写时最常见的笔误,必须拦下
    if not isinstance(fallback, bool):
        _fail("preprocess.corners.sb_fallback_classic", fallback, "必须是 true 或 false(…)")
```

接入点:`load_config`(L274-299)在 `_parse_preprocess(raw)` 之后加一行 `intrinsic = _parse_intrinsic(raw)` 并传入构造器。新增解析函数放在 `_parse_preprocess` 之后、`_check_output_data_disjoint` 之前,保持「顶层段解析函数自上而下」的现序。

---

### `src/stereo_calib/cli/main.py`(修改,controller,request-response)

**Analog:** 自身 `preprocess` 命令(L60-72,逐行对照替换占位 L75-78):

```python
@app.command(help="模块1 数据预处理:扫描 data/ 双目图像对 → 棋盘格角点检测(SB 主路线,输出必过"
                  "规范化)→ 每相机每数据集落盘角点 npz + 可用/失败清单 CSV。")
def preprocess() -> None:
    """模块1 命令入口:坏输入收口为中文一句 + 退出码 1,不吐堆栈。"""
    try:
        # 局部导入:让 --help / 占位命令不依赖 preprocess 模块,加快启动
        from stereo_calib.preprocess.pipeline import run_preprocess

        run_preprocess(load_config(_config_path))
    except (ValueError, OSError) as e:
        typer.echo(f"数据预处理失败:{e}")
        typer.echo("  请检查 config.yaml 配置与 data/ 数据,修正后重跑 stereo-calib preprocess。")
        raise typer.Exit(code=1) from e
```

`intrinsic` 版本(RESEARCH.md Code Examples 已给出成稿):help 写「模块2 单相机内参标定:…」;局部导入 `stereo_calib.intrinsic.pipeline.run_intrinsic`;except 消息「内参标定失败:{e}」+ 指引检查 config.yaml 与 Phase 1 产物(output/ 下的角点 npz);`raise typer.Exit(code=1) from e` 保留因果链。`stereo` 占位与 callback(`--config` 唯一选项,L32-52)不动。

---

### `config.yaml`(修改,config)

**Analog:** 自身 `preprocess:` 段(L28-49):段头注释块 + 逐键行尾中文注释(说明取值含义与默认值理由)。追加 `intrinsic:` 段,键位按 RESEARCH.md Pattern 8(refine.{enabled,max_nfev,ftol,xtol,gtol} / outlier.{enabled,rms_multiplier,max_drop_fraction,max_iterations} / rms_test_max / exclude_images),每键照 preprocess 体例写行尾注释。

---

### `tests/test_intrinsic_config.py`(test)

**Analog:** `tests/test_io_config.py`(135 行,全读,逐段照抄体例)

**最小合法配置常量 + 变异函数**(L15-58):`MINIMAL_YAML` 全键显式(需加 intrinsic 段,不依赖隐式默认)+ `_write_config`/`_mutate` 原样复用。

**非法值参数化中文报错**(L84-95):

```python
@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("max_workers", "0"),                                    # 并行数下限
        ("classic_flags", "[adaptive_thresh, bogus_flag]"),      # 白名单外的选项名
        ("max_workers", "true"),                                 # 布尔冒充 int 必须被拦
    ],
)
def test_invalid_values_raise_with_chinese_message(tmp_path, field, bad_value):
    with pytest.raises(ValueError, match="不合法"):
        load_config(_write_config(tmp_path, _mutate(field, bad_value)))
```
intrinsic 版参数化:multiplier=1.0(必须 >1)、max_drop_fraction=0、enabled=1(bool 拦截)、method="lm"(白名单拒收)、合法加载全字段断言(照 L61-81)。

---

### `tests/test_intrinsic_pipeline.py`(test)

**Analog:** `tests/test_preprocess_pipeline.py`(358 行,全读)

**数据只读快照防线**(L63-68,直接照抄):

```python
def _tree_snapshot(root: Path) -> dict:
    """INFRA-03:data/ 树逐文件 (mtime_ns, size) 快照,运行前后必须逐字节不变。"""
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(root.rglob("*")) if p.is_file()
    }
```

**进程内跑命令 + monkeypatch.chdir 夹具**(L102-108):

```python
@pytest.fixture
def synthetic_run(tmp_path, monkeypatch, make_chessboard_image):
    before = _build_synthetic_dataset(tmp_path, make_chessboard_image)
    monkeypatch.chdir(tmp_path)
    preprocess()  # 进程内直调命令函数;坏输入才会 typer.Exit(1),这里应成功
    return tmp_path / "output", before
```

**npz 契约断言体例**(L111-135):`np.load(..., allow_pickle=False)` → `set(data.files)` 全等 → shape/dtype 逐项 → NaN/掩码语义。intrinsic 版改测:合成 npz(直接按 Phase 1 八数组契约 `np.savez`,**不需要造图**——RESEARCH.md Code Examples 第 1 节:标定级测试用 cv2.projectPoints 合成 (V,88,2) 角点 + 0.05px 噪声,秒级)→ 进程内跑 `intrinsic()` → 断言 intrinsics.json 契约(params 按名/分辨率/K/D/rms/剔除清单)、CSV 列序、report.md 存在、剔除留痕可审计、data/ 快照不变、重跑 rmtree 清残留(照 L313-333 `test_rerun_clears_stale_artifacts`)。

---

### `tests/test_intrinsic_models.py` / `test_intrinsic_solve.py` / `test_intrinsic_scoring.py`(test)

**Analog:** `tests/test_corners.py`

**源码防线测试体例**(L118-126,Pitfall 3 的锁定手段):

```python
def test_detect_source_guard_single_refinement_call():
    source = pathlib.Path("src/stereo_calib/corners/detect.py").read_text(encoding="utf-8")
    assert source.count("cv2.cornerSubPix(") == 1, (...)
```
注意 docstring 警告(L6-7):**测试与注释里都不写出被锁定的调用串原文**,否则子串计数自伤。intrinsic 版:metrics/solve 的报告路径不引用 calibrateCameraExtended 返回 rms 的防线(锁定变量名出现在指标赋值处的次数)、残差只由 models.project 构成的防线(锁定 solve.py 中投影公式调用只经 models)。

**数值精度断言体例**(L131-139):`pytest.approx`/绝对阈值 + 中文失败消息写明「为什么这个数」:

```python
    assert result["uv"].shape == (88, 2)
    assert result["uv"].dtype == np.float64
```
intrinsic 版:models parity 对 `cv2.projectPoints`(float64 进,<1e-9px)、解析雅可比对 `scipy.optimize._numdiff.approx_derivative` 逐列校验(<1e-6)、project→unproject→project 往返(<1e-9px)、合成真值恢复(12 视图,|fx_est−fx_true|/fx_true < 1e-3)、大倾角错误盆地回归(scoring,合成 40° 倾角帧断言 RMS 不虚高)。

**夹具:** `tests/conftest.py` 的 `make_chessboard_image`(L69-165)已有——但标定级测试优先用直接合成 npz 路线(不经检测),夹具留给需要的角点级用例。

---

## Shared Patterns

### 1. CLI 中文错误收口
**Source:** `src/stereo_calib/cli/main.py` L63-72
**Apply to:** cli/main.py intrinsic 命令(唯一入口;pipeline 内层抛 ValueError/OSError 即可)
```python
    except (ValueError, OSError) as e:
        typer.echo(f"内参标定失败:{e}")
        raise typer.Exit(code=1) from e
```

### 2. 配置驱动,命令行零算法参数
**Source:** `config.yaml` 头注释(L3-5)+ `cli/main.py` callback(L32-52)
**Apply to:** intrinsic 全部参数(精化开关/容差/离群阈值/exclude_images)只进 config.yaml `intrinsic:` 段 → `IntrinsicConfig`;`--config` 仍是唯一 CLI 选项。

### 3. frozen dataclass + 统一中文校验
**Source:** `src/stereo_calib/io/config.py` L96-101(`_fail`)、L212-217(bool-int 陷阱)、L178-196(白名单)
**Apply to:** `_parse_intrinsic` 及其子段函数。

### 4. npz 安全契约
**Source:** `src/stereo_calib/preprocess/artifacts.py` L16-17(纪律)、L95-102(读取)、L68-77(写入)
**Apply to:** observation.py 读 Phase 1 npz(import `load_corners`,不新写);artifacts.py 写 residuals/extrinsics npz。全部 allow_pickle=False、字符串定长 unicode、dtype 显式。

### 5. 产物落盘三件套(utf-8-sig CSV / meta-json 留痕 / 固定覆盖)
**Source:** `preprocess/artifacts.py` L110-121(CSV)、L80-92(meta)、`preprocess/pipeline.py` L136-138(rmtree)
**Apply to:** intrinsic/artifacts.py 全部落盘;pipeline.py 运行前 `shutil.rmtree(out_root / "intrinsic", ignore_errors=True)`。

### 6. data/ 只读验证
**Source:** `tests/test_preprocess_pipeline.py` L63-68(`_tree_snapshot`)、L183-192(断言)
**Apply to:** test_intrinsic_pipeline.py 端到端用例照抄快照比对。

### 7. 源码防线测试(grep count 锁定)
**Source:** `tests/test_corners.py` L118-126
**Apply to:** test_intrinsic_solve.py(OpenCV rms 不进指标、残差只经 models.project)+ docstring 不写被锁调用串原文的自伤规避。

### 8. 重数学只在一个文件(单一数学源)
**Source:** 仓库先例是 canonical.py 之于 detect.py;本阶段强化为 RESEARCH.md Architecture Responsibility Map:投影/畸变/雅可比只准在 `intrinsic/models.py`,solve/metrics/scoring/artifacts 只 import。
**Apply to:** models.py + 全部消费方;测试双向锁定(parity + 源码防线)。

## No Analog Found

仓库内无任何非线性最小二乘求解器、稀疏雅可比组装或标定数学——以下 4 个文件的**算法内容**没有仓库同源体,必须按 RESEARCH.md Pattern 1-5 从只读参考工程移植;其**文件体例**(docstring/导入/错误契约)仍照上表仓库 analog:

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `src/stereo_calib/intrinsic/models.py` | math-model | transform | 仓库无投影/畸变数学;算法源 pinhole_brown.py + base.py |
| `src/stereo_calib/intrinsic/solve.py` | solver | transform/batch | 仓库无 least_squares 求解器;算法源 solve/{opencv,scipy_solver,residuals}.py |
| `src/stereo_calib/intrinsic/metrics.py` | utility | transform | 仓库无 RMS 回代;算法源 solve/result.py |
| `src/stereo_calib/intrinsic/scoring.py` | service | transform | 仓库无只估位姿评分;算法源 solve/scoring.py |

## Metadata

**Analog search scope:** `src/stereo_calib/**`(11 文件)、`tests/**`(6 文件)、`config.yaml`;参考工程布局仅作算法源引用
**Files read in full:** 10(cli/main.py、io/config.py、io/dataset.py、preprocess/pipeline.py、preprocess/artifacts.py、corners/detect.py、corners/canonical.py、config.yaml、tests/conftest.py、tests/test_preprocess_pipeline.py、tests/test_io_config.py)+ 2 处定向读(test_corners.py 源码防线段)
**Pattern extraction date:** 2026-10-24
