"""装载契约与 float64 指标尺测试:合成 npz 装载三重防线 + 真值噪声 RMS。

02-01 放装载契约与 metrics 用例;02-03 扩展落盘层 writer 单元契约与
端到端(合成 npz → 进程内调 run_intrinsic → 产物全契约 + 离群行为 +
清理/只读防线)。合成路线照 RESEARCH Code Examples:Phase 2 消费的是
角点数组——用真值参数 + 随机位姿 cv2.projectPoints 合成 (V,88,2) 角点,
按 Phase 1 八数组契约 np.savez(allow_pickle=False 可读),秒级完成
(图像文件仅 image_size 派生需要,渲染真实小图即可)。
"""

import csv
import json
import pathlib

import cv2
import numpy as np
import pytest
import typer

from stereo_calib.cli import main as cli_main
from stereo_calib.intrinsic.artifacts import (
    write_extrinsics_npz,
    write_intrinsics_json,
    write_per_frame_csv,
    write_report_md,
    write_residuals_npz,
)
from stereo_calib.intrinsic.metrics import compute_metrics
from stereo_calib.intrinsic.models import (
    INTRINSIC_NAMES,
    intrinsics_matrix,
    project,
    transform_to_camera,
)
from stereo_calib.intrinsic.observation import load_views
from stereo_calib.io.config import BoardSpec
from stereo_calib.preprocess.artifacts import build_object_points

# 真值参数(量级取自本数据锚点)+ 11×8 / 25mm 规格配置(与 config.yaml 同)
_TRUE_PARAMS = np.array([1850.0, 1840.0, 1280.0, 720.0, -0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])
_CFG = BoardSpec()


def _write_tiny_jpg(path: pathlib.Path, width: int = 640, height: int = 480) -> pathlib.Path:
    """最小噪声 JPEG(仅需"可读出分辨率",不需棋盘内容)。"""
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img), f"测试夹具写图失败:{path}"
    return path


def _synthetic_uv(n_views: int, seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """真值参数 + 随机位姿投影合成角点(+0.05px 高斯噪声),返回 (uv, rvecs, tvecs)。"""
    rng = np.random.default_rng(seed)
    objp = build_object_points(_CFG)
    uvs, rvecs, tvecs = [], [], []
    for _ in range(n_views):
        rvec = rng.normal(0.0, 0.3, 3)
        tvec = np.array([rng.normal(0.0, 40.0), rng.normal(0.0, 40.0),
                         rng.uniform(400.0, 900.0)])
        pts_cam, _ = transform_to_camera(objp, rvec, tvec)
        uvs.append(project(pts_cam, _TRUE_PARAMS) + rng.normal(0.0, 0.05, (88, 2)))
        rvecs.append(rvec)
        tvecs.append(tvec)
    return np.asarray(uvs), np.asarray(rvecs), np.asarray(tvecs)


def _write_npz(path: pathlib.Path, ids, paths, uv, objp, source, detected) -> pathlib.Path:
    """按 Phase 1 八数组契约写 npz(allow_pickle=False 可读:定长 unicode + 纯数值)。"""
    detected_arr = np.asarray(detected, dtype=bool)
    np.savez(
        path,
        image_ids=np.asarray(ids),
        paths=np.asarray(paths),
        uv=np.asarray(uv, dtype=np.float64),
        objp=np.asarray(objp, dtype=np.float64),
        source=np.asarray(source, dtype="<U16"),
        detected=detected_arr,
        sq_px=np.where(detected_arr, 20.0, np.nan).astype(np.float64),
        win_size=np.where(detected_arr, 9.0, np.nan).astype(np.float64),
    )
    return path


def _build_npz(tmp_path: pathlib.Path, n_detected: int, n_undetected: int = 1,
               seed: int = 3, image_size: tuple[int, int] = (640, 480)):
    """合成一份完整 npz:前 n_detected 帧检出,随后 n_undetected 帧整行 NaN。"""
    uv, _rvecs, _tvecs = _synthetic_uv(n_detected + n_undetected, seed)
    detected = np.array([True] * n_detected + [False] * n_undetected)
    uv = uv.copy()
    uv[~detected] = np.nan  # 未检出帧整行 NaN(带掩码稠密契约)
    ids = [f"{i:03d}_cam1" for i in range(len(detected))]
    w, h = image_size
    paths = [str(_write_tiny_jpg(tmp_path / "img" / f"{i:03d}.jpg", w, h))
             for i in range(len(detected))]
    source = ["sb"] * n_detected + [""] * n_undetected
    npz = _write_npz(tmp_path / "corners.npz", ids, paths, uv,
                     build_object_points(_CFG), source, detected)
    return npz, ids, detected


# ---------- 装载契约:幸存顺序 / 形状 / 掩码行 ----------

def test_load_contract_survivor_order_and_shapes(tmp_path):
    """检出帧全部幸存、未检出行不进装载;image_ids 顺序与幸存行一致(行序锁)。"""
    npz, ids, _detected = _build_npz(tmp_path, n_detected=12)
    v = load_views(_CFG, npz, (), min_views=10)
    assert v.image_ids == tuple(ids[:12]), "幸存视图必须按 npz 行序一一对应(行序锁)"
    assert ids[12] not in v.image_ids, "未检出帧不得进装载"
    assert v.uv.shape == (12, 88, 2)
    assert v.uv.dtype == np.float64
    assert v.objp.shape == (88, 3)
    assert v.source == ("sb",) * 12
    assert not np.isnan(v.uv).any(), "幸存视图里不得残留 NaN(三重防线)"
    assert v.image_size == (640, 480), "image_size 必须是 (宽, 高) 口径"


def test_exclude_images_skips_frames_keeps_order(tmp_path):
    """exclude_images 命中的帧被跳过(手动剔除通道),幸存顺序保持 npz 行序。"""
    npz, ids, _detected = _build_npz(tmp_path, n_detected=12)
    v = load_views(_CFG, npz, (ids[0], ids[5], "999_bogus_cam1"), min_views=10)
    expected = tuple(ids[i] for i in range(12) if i not in (0, 5))
    assert v.image_ids == expected
    assert v.uv.shape == (10, 88, 2)
    # 清单里不存在的编号不报错(逻辑剔除:命中才算)


# ---------- 三重防线:NaN / objp 交叉核对 / 视图数下限 ----------

def test_detected_row_with_nan_raises(tmp_path):
    """检出但角点含 NaN/Inf → 中文报错带帧号,绝不带毒进求解。"""
    uv, _r, _t = _synthetic_uv(12)
    uv[3, 7, 0] = np.nan  # 检出行埋一个 NaN
    ids = [f"{i:03d}_cam1" for i in range(12)]
    paths = [str(_write_tiny_jpg(tmp_path / "img" / f"{i:03d}.jpg")) for i in range(12)]
    npz = _write_npz(tmp_path / "corners.npz", ids, paths, uv,
                     build_object_points(_CFG), ["sb"] * 12, [True] * 12)
    with pytest.raises(ValueError, match="非数值"):
        load_views(_CFG, npz, (), min_views=2)


def test_objp_tampered_raises_rerun_preprocess(tmp_path):
    """npz 内 objp 被篡改/规格漂移 → 中文报错提示重跑 preprocess(Pitfall 11)。"""
    npz, _ids, _det = _build_npz(tmp_path, n_detected=12)
    data = dict(np.load(npz, allow_pickle=False))
    objp = data["objp"].copy()
    objp[0, 0] += 5.0
    npz2 = _write_npz(tmp_path / "tampered.npz", data["image_ids"], data["paths"],
                      data["uv"], objp, data["source"], data["detected"])
    with pytest.raises(ValueError, match="重跑 preprocess"):
        load_views(_CFG, npz2, (), min_views=2)


def test_min_views_underflow_raises(tmp_path):
    """幸存视图数低于下限 → 中文报错不硬算(两三个视角定不了内参)。"""
    npz, _ids, _det = _build_npz(tmp_path, n_detected=6)
    with pytest.raises(ValueError, match="可用视图"):
        load_views(_CFG, npz, (), min_views=10)


# ---------- image_size 派生:(宽, 高) 口径 + 抽查防线 ----------

def test_image_size_derivation_width_height_order(tmp_path, make_chessboard_image):
    """幸存首帧派生 image_size,返回 (宽, 高) 与直接 imread 的 shape 翻转一致。"""
    img, _corners, _sq = make_chessboard_image(image_size=(480, 640))  # numpy 形状 (高, 宽)
    target = _write_dir_img(tmp_path / "board" / "f000.jpg", img)
    uv, _r, _t = _synthetic_uv(2)
    npz = _write_npz(tmp_path / "corners.npz",
                     ["000_cam1", "001_cam1"], [str(target), str(target)],
                     uv, build_object_points(_CFG), ["sb", "sb"], [True, True])
    v = load_views(_CFG, npz, (), min_views=1)
    direct = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
    assert direct.shape == (480, 640)
    assert v.image_size == (640, 480), "严禁把 (高, 宽) 顺序传下去(Pitfall 2)"


def test_mixed_resolution_spot_check_raises(tmp_path):
    """首/中/末 3 帧抽查:混分辨率 → 中文报错(A2,单 image_size 语义防线)。"""
    uv, _r, _t = _synthetic_uv(3)
    paths = [str(_write_tiny_jpg(tmp_path / "img" / "a.jpg", 640, 480)),
             str(_write_tiny_jpg(tmp_path / "img" / "b.jpg", 320, 240)),   # 中间帧混分辨率
             str(_write_tiny_jpg(tmp_path / "img" / "c.jpg", 640, 480))]
    npz = _write_npz(tmp_path / "corners.npz", ["a", "b", "c"], paths,
                     uv, build_object_points(_CFG), ["sb"] * 3, [True] * 3)
    with pytest.raises(ValueError, match="分辨率不一致"):
        load_views(_CFG, npz, (), min_views=1)


def _write_dir_img(path: pathlib.Path, img: np.ndarray) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), img), f"测试夹具写图失败:{path}"
    return path


# ---------- metrics:真值噪声 RMS 落带 + 源码防线 ----------

def test_metrics_truth_noise_rms_in_band():
    """真值参数 + 随机位姿 + 0.05px 噪声 → pooled RMS ≈ 0.05(落 0.03–0.08 带)。

    数学预期:每坐标噪声 std 0.05px,平方欧氏残差期望 2×0.05²=0.005,
    pooled RMS ≈ sqrt(0.005) ≈ 0.0707(11×8=88 点 × 12 视图,估计涨落
    ~1.5%,带宽留足裕量)。这正是"float64 回代尺量出来就是注入的噪声水平"
    的端到端证明。
    """
    uvs, rvecs, tvecs = _synthetic_uv(12, seed=42)
    m = compute_metrics(build_object_points(_CFG), uvs, _TRUE_PARAMS, rvecs, tvecs)
    assert 0.03 < m.rms_overall < 0.08, f"pooled RMS {m.rms_overall:.4f}px 落带外"
    assert m.per_view_rms.shape == (12,)
    assert m.residuals.shape == (12, 88, 2)
    assert m.residuals.dtype == np.float64 and m.per_view_rms.dtype == np.float64
    # pooled 与逐帧的口径自洽:等角点帧下 pooled = sqrt(逐帧均方的平均)
    pooled_from_views = float(np.sqrt(np.mean(m.per_view_rms ** 2)))
    assert pooled_from_views == pytest.approx(m.rms_overall, rel=1e-12)


def test_metrics_and_observation_source_guards():
    """源码防线:metrics 不含畸变公式(数学只准 import models);
    observation 不直接调 numpy 加载接口(必须走 load_corners 的 pickle 防线)。"""
    metrics_src = pathlib.Path("src/stereo_calib/intrinsic/metrics.py").read_text(encoding="utf-8")
    assert "_distort" not in metrics_src, "指标层不得出现第二份畸变公式(唯一指标尺纪律)"
    assert "from stereo_calib.intrinsic import models" in metrics_src
    assert metrics_src.count("models.project(") == 1, "残差只准由 models.project 构成"
    obs_src = pathlib.Path("src/stereo_calib/intrinsic/observation.py").read_text(encoding="utf-8")
    assert "np.load(" not in obs_src, "装载必须经 load_corners(allow_pickle=False 防线)"
    assert "from stereo_calib.preprocess.artifacts import" in obs_src, "必须复用 Phase 1 读取入口"


# ===========================================================================
# 02-03 任务 1:落盘层 writer 单元契约(tmp_path roundtrip)
# ===========================================================================

# intrinsics.json 顶层字段全集(Pattern 7 契约;Phase 3 按名读取的唯一接口)
# 2026-10-24 增补 "diagnostics"(模型充分性体检),契约只增不改既有字段
_JSON_KEYS = {
    "cam", "image_size", "params", "K", "D", "solver", "refine_enabled",
    "rms_train_overall", "rms_test_overall", "n_views_train", "n_views_test",
    "dropped_frames", "solver_meta", "board", "config_snapshot", "diagnostics",
}


def test_write_intrinsics_json_roundtrip(tmp_path):
    """写 → 读回逐字段断言:字段全集、params 按名、K/D 形状与数值冗余一致。"""
    dropped = [{"image_id": "003_cam1", "rms": 15.2, "round": 1, "threshold": 0.35}]
    diagnostics = {"enabled": True, "verdict": "可靠", "reliable": True,
                   "red_count": 0, "red_names": [], "notes": [], "checks": []}
    out = write_intrinsics_json(
        tmp_path / "cam1" / "intrinsics.json", cam=1, image_size=(2560, 1440),
        params=_TRUE_PARAMS, K=intrinsics_matrix(_TRUE_PARAMS), D=_TRUE_PARAMS[4:9],
        solver="scipy", refine_enabled=True, rms_train_overall=0.3316,
        rms_test_overall=0.2389, n_views_train=141, n_views_test=15,
        dropped_frames=dropped, solver_meta={"status": 0, "opencv_rms": 2.5782},
        board=BoardSpec(), config_snapshot={"intrinsic": {"rms_test_max": 1.0}},
        diagnostics=diagnostics)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) == _JSON_KEYS, "intrinsics.json 字段集必须与 Pattern 7 契约全等"
    assert list(payload["params"]) == list(INTRINSIC_NAMES), "params 必须按名(模块 3 的固定接口)"
    for name, v in zip(INTRINSIC_NAMES, _TRUE_PARAMS):
        assert payload["params"][name] == pytest.approx(float(v))
    K = np.asarray(payload["K"])
    assert K.shape == (3, 3)
    assert K[0, 0] == pytest.approx(1850.0) and K[0, 2] == pytest.approx(1280.0)
    assert K[1, 1] == pytest.approx(1840.0) and K[1, 2] == pytest.approx(720.0)
    assert payload["D"] == pytest.approx([-0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])
    assert payload["image_size"] == [2560, 1440], "image_size 必须是 [宽, 高]"
    assert payload["cam"] == 1
    assert payload["solver"] == "scipy" and payload["refine_enabled"] is True
    assert payload["n_views_train"] == 141 and payload["n_views_test"] == 15
    assert payload["dropped_frames"] == [
        {"image_id": "003_cam1", "rms": 15.2, "round": 1, "threshold": 0.35}]
    assert payload["solver_meta"]["opencv_rms"] == pytest.approx(2.5782)
    assert payload["board"]["cols"] == 11 and payload["board"]["square_size_mm"] == 25.0
    assert payload["diagnostics"] == diagnostics, "diagnostics 键必须原样落盘(只增契约)"


def test_write_residuals_npz_roundtrip(tmp_path):
    """残差 npz:allow_pickle=False 可读,dtype/形状/对账列一致,NaN 原样通过。"""
    residuals = np.arange(2 * 4 * 2, dtype=np.float64).reshape(2, 4, 2)
    residuals[1, 0, 0] = np.nan  # test 份失败帧留痕语义:NaN 必须原样落盘读回
    per_view = np.array([0.5, np.nan])
    out = write_residuals_npz(tmp_path / "train_residuals.npz",
                              ["003_cam1", "004_cam1"], residuals, per_view)
    data = np.load(out, allow_pickle=False)
    assert data["residuals"].shape == (2, 4, 2) and data["residuals"].dtype == np.float64
    assert data["per_view_rms"].shape == (2,) and data["per_view_rms"].dtype == np.float64
    assert [str(s) for s in data["image_ids"]] == ["003_cam1", "004_cam1"]
    assert data["residuals"][0].tolist() == residuals[0].tolist()
    assert np.isnan(data["residuals"][1, 0, 0]) and np.isnan(data["per_view_rms"][1])


def test_write_extrinsics_npz_roundtrip(tmp_path):
    """外参 npz(诊断用):allow_pickle=False 可读,(V,3) float64 + 对账列。"""
    rvecs = np.linspace(-0.3, 0.3, 6, dtype=np.float64).reshape(2, 3)
    tvecs = np.stack([np.arange(3, dtype=np.float64),
                      np.arange(3, dtype=np.float64) + 10.0])
    out = write_extrinsics_npz(tmp_path / "extrinsics.npz",
                               ["003_cam1", "004_cam1"], rvecs, tvecs)
    data = np.load(out, allow_pickle=False)
    assert data["rvecs"].shape == (2, 3) and data["rvecs"].dtype == np.float64
    assert data["tvecs"].shape == (2, 3) and data["tvecs"].dtype == np.float64
    assert [str(s) for s in data["image_ids"]] == ["003_cam1", "004_cam1"]
    assert data["rvecs"].tolist() == rvecs.tolist()


def test_write_per_frame_csv_header_and_bom(tmp_path):
    """逐帧 CSV:表头列序恰为六列契约 + utf-8-sig BOM(Excel 双击直开)。"""
    rows = [{"image_id": "003_cam1", "split": "train", "source": "sb",
             "rms": "0.1234", "dropped": "True",
             "note": "第1轮剔除:RMS 15.200 超阈值 0.350"}]
    out = write_per_frame_csv(tmp_path / "per_frame_cam1.csv", rows)
    assert out.read_bytes().startswith(b"\xef\xbb\xbf"), "必须 utf-8-sig(带 BOM)"
    with open(out, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["image_id", "split", "source", "rms", "dropped", "note"]
        got = list(reader)
    assert got[0]["image_id"] == "003_cam1"
    assert got[0]["dropped"] == "True"
    assert "剔除" in got[0]["note"], "剔除行的中文说明必须原样落盘"


def test_write_report_md_sections(tmp_path):
    """报告:含指标对比(PASS/FAIL 判定)、按名参数表、离群留痕段与「无」占位。"""
    params = {n: float(v) for n, v in zip(INTRINSIC_NAMES, _TRUE_PARAMS)}

    def _summary(cam, rms_test, dropped, loop):
        return {"cam": cam, "image_size": (2560, 1440), "params": params,
                "solver": "scipy", "refine_enabled": True,
                "rms_train_overall": 0.33, "rms_test_overall": rms_test,
                "n_views_train": 10, "n_views_train_initial": 12, "n_views_test": 4,
                "dropped_frames": dropped, "outlier_loop": loop,
                "train_per_view_rms": [0.07] * 10,
                "test_per_view_rms": [0.08, 0.06, float("nan"), 0.09]}

    summaries = [
        _summary(1, 0.24, [], "第 1 轮无超阈帧,剔除循环收敛(共剔除 0 帧)"),
        _summary(2, 1.5, [], "第 1 轮无超阈帧,剔除循环收敛(共剔除 0 帧)"),
    ]
    out = write_report_md(tmp_path / "report.md", summaries, 1.0, {"intrinsic": {}})
    text = out.read_text(encoding="utf-8")
    assert "离群" in text, "报告必须有离群留痕段(INTR-04)"
    assert "PASS" in text and "FAIL" in text, "按 rms_test_max 给两路判定"
    assert "cam1" in text and "cam2" in text
    assert "fx" in text and "p2" in text, "参数表必须按名列全 9 项"
    assert "无(本轮运行没有帧超过离群阈值)" in text, "无剔除时离群表写「无」"
    assert "中位" in text and "per_frame_cam1.csv" in text, "逐帧摘要指向 CSV 明细"
    assert "config_snapshot" in text, "配置快照留痕进附录"


# ===========================================================================
# 02-03 任务 3:合成 npz e2e(全契约 + 离群行为 + 清理/只读防线)
# ===========================================================================

# 合成真值(与 02-02 真值恢复测试同源量级:fx/fy 差 10、强径向 + 弱切向)
_K_TRUE = np.array([[1850.0, 0.0, 1280.0], [0.0, 1840.0, 720.0], [0.0, 0.0, 1.0]])
_D_TRUE = np.array([-0.40, 0.23, 1.1e-4, -3.8e-5, -0.08])
_IMG_SHAPE = (1440, 2560)  # numpy (高, 宽);imread 派生的 image_size 应为 (2560, 1440)

# e2e 配置:intrinsic 段全键显式(与 config.yaml 同布局)。min_views 显式取 2:
# 本计划 e2e 的 test 划分只有 4 帧(装载下限按 10 会把考卷挡在门外),
# 剔除保险丝余量另由 max_drop_fraction 兜底(12 帧下限 9.6,可剔 2 帧)。
_E2E_CONFIG_YAML = """\
board:
  board_type: "chessboard"
  cols: 11
  rows: 8
  square_size_mm: 25.0
  unit: "mm"
data:
  train: "data/train"
  test: "data/test"
output_root: "output"
cv2_threads: 0
preprocess:
  max_workers: 1
  corners:
    classic_flags: [adaptive_thresh, normalize_image]
    sb_flags: 0
    sb_fallback_classic: true
    subpix: {window_granularity: frame, min_win: 3, max_iter: 30, epsilon: 0.01}
  copy_fail_originals: false
  visualization: {jpeg_quality: 90}
intrinsic:
  refine:
    enabled: true
    method: "trf"
    max_nfev: 300
    ftol: 1.0e-10
    xtol: 1.0e-12
    gtol: 1.0e-10
  outlier:
    enabled: true
    rms_multiplier: 5.0
    max_drop_fraction: 0.2
    max_iterations: 3
    min_views: 2
  rms_test_max: 1.0
  exclude_images: []
"""


def _tree_snapshot(root: pathlib.Path) -> dict:
    """INFRA-03:data/ 树逐文件 (mtime_ns, size) 快照,运行前后必须逐字节不变。"""
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size)
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _synth_npz(path: pathlib.Path, cam: int, split: str, n_views: int, seed: int,
               img_path: pathlib.Path, board: BoardSpec,
               bad_idx: int | None = None) -> list[str]:
    """按 Phase 1 八数组契约合成一份角点 npz(真值投影 + 0.05px 噪声)。

    detected 全 True、source 全 "sb"、sq_px 全 18.0、win_size 全 5.0;
    paths 指向渲染出的真实小图(image_size 派生要 imread,Plan 指定)。
    bad_idx 非 None 时给该帧 uv 加恒定 +15px 偏移(离群行为用例的坏帧)。
    """
    rng = np.random.default_rng(seed)
    objp_flat = build_object_points(board).reshape(-1, 1, 3)
    uvs = []
    for _ in range(n_views):
        rvec = rng.normal(0.0, 0.3, 3)
        tvec = np.array([rng.normal(0.0, 50.0), rng.normal(0.0, 50.0),
                         rng.uniform(400.0, 900.0)])
        proj, _ = cv2.projectPoints(objp_flat, rvec, tvec, _K_TRUE, _D_TRUE)
        uvs.append(proj.reshape(-1, 2) + rng.normal(0.0, 0.05, (88, 2)))
    uv = np.asarray(uvs, dtype=np.float64)
    if bad_idx is not None:
        # 坏帧注入用 15px 量级的逐角点散布,而非恒定 +15px 偏移:恒定偏移等价于
        # 棋盘的小平移,会被该帧自己的位姿解几乎完美吸收(实测注入后该帧逐帧
        # RMS 仅 0.2px,永远到不了剔除阈值)——plan 原文的恒定偏移注入模型无法
        # 触发剔除机制,按同量级"该帧角点整体劣化"修正(口径记录 SUMMARY)。
        uv[bad_idx] += np.random.default_rng(99).normal(0.0, 15.0, (88, 2))
    ids = [f"{split}{k:03d}_cam{cam}" for k in range(n_views)]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        image_ids=np.asarray(ids, dtype=np.str_),
        paths=np.asarray([str(img_path)] * n_views, dtype=np.str_),
        uv=uv,
        objp=build_object_points(board),
        source=np.asarray(["sb"] * n_views, dtype="<U16"),
        detected=np.ones(n_views, dtype=bool),
        sq_px=np.full(n_views, 18.0, dtype=np.float64),
        win_size=np.full(n_views, 5.0, dtype=np.float64),
    )
    return ids


def _build_run_tree(tmp_path: pathlib.Path, make_chessboard_image, *,
                    bad_train_idx: int | None = None,
                    bad_test_idx: int | None = None) -> dict:
    """搭最小运行树:渲染真实小图进 data/ + 合成 npz 进 output/ + 写 config.yaml。

    返回 {f"cam{cam}_{split}": image_ids} 供断言对账;两路 train 种子取
    5 种子探测里精度居中的 7 与 3(见 test_e2e_full_contract 的阈值注释)。
    """
    board = BoardSpec()
    img_paths: dict[int, pathlib.Path] = {}
    for cam in (1, 2):
        img, _corners, _sq = make_chessboard_image(image_size=_IMG_SHAPE)
        p = tmp_path / "data" / "train" / "ts1" / "leaf1" / f"cam{cam}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(p), img), f"渲染小图写盘失败:{p}"
        img_paths[cam] = p

    ids: dict[str, list[str]] = {}
    for cam in (1, 2):
        train_seed = (7, 3)[cam - 1]
        ids[f"cam{cam}_train"] = _synth_npz(
            tmp_path / "output" / "train" / f"cam{cam}_train.npz",
            cam, "train", 12, train_seed, img_paths[cam], board,
            bad_idx=bad_train_idx if cam == 1 else None)
        ids[f"cam{cam}_test"] = _synth_npz(
            tmp_path / "output" / "test" / f"cam{cam}_test.npz",
            cam, "test", 4, 200 + cam, img_paths[cam], board,
            bad_idx=bad_test_idx if cam == 1 else None)
    (tmp_path / "config.yaml").write_text(_E2E_CONFIG_YAML, encoding="utf-8")
    return ids


def _run_intrinsic_in(tmp_path: pathlib.Path) -> None:
    """进程内直调 run_intrinsic(不是 typer 命令,CLI 留 02-04)。"""
    from stereo_calib.intrinsic.pipeline import run_intrinsic
    from stereo_calib.io.config import load_config
    run_intrinsic(load_config(tmp_path / "config.yaml"))


@pytest.fixture
def intrinsic_run(tmp_path, monkeypatch, make_chessboard_image):
    """干净合成运行:12+4 视图 × 两路,无坏帧 → 全契约断言的公共夹具。"""
    ids = _build_run_tree(tmp_path, make_chessboard_image)
    monkeypatch.chdir(tmp_path)
    before = _tree_snapshot(tmp_path / "data")
    _run_intrinsic_in(tmp_path)
    return tmp_path / "output", before, ids


# 精化后参数对真值的判别口径(12 视图 / 0.05px 噪声的噪声限,5 个种子
# 实测:seed 3/7/42/101/202 下 fx..cy 相对误差最大 8.2e-4;k1/k2/k3 的
# 高阶畸变项弱约束,相对误差可到 1.3e-3 ~ 1.6e-2,绝对误差最大 1.3e-3;
# p1/p2 真值 ~1e-4 量级、像素效应远低于检测噪声(02-02 同款结论),
# 相对误差无判别力,按绝对误差 1e-4 判)。plan 原文的"逐项相对 < 1e-3"
# 对高阶畸变项物理不可达,口径修正记录在 SUMMARY 偏差节。
_REL_TIGHT = ("fx", "fy", "cx", "cy")
_ABS_DIST = {"k1": 1e-3, "k2": 2e-3, "k3": 2e-3, "p1": 1e-4, "p2": 1e-4}


def _assert_params_recover_truth(params: dict) -> None:
    for name in _REL_TIGHT:
        err = abs(params[name] - {"fx": 1850.0, "fy": 1840.0, "cx": 1280.0, "cy": 720.0}[name])
        assert err / {"fx": 1850.0, "fy": 1840.0, "cx": 1280.0, "cy": 720.0}[name] < 1e-3, \
            f"{name} 相对误差 {err:.3e} 超出噪声限带(精化未生效?)"
    truth = dict(zip(["k1", "k2", "p1", "p2", "k3"], _D_TRUE))
    for name, tol in _ABS_DIST.items():
        err = abs(params[name] - truth[name])
        assert err < tol, f"{name} 绝对误差 {err:.3e} 超出噪声限带 {tol}(精化未生效?)"


def test_e2e_full_contract_clean_run(intrinsic_run):
    """e2e 全契约:intrinsics.json 字段全集 == Pattern 7 清单,params 对真值
    恢复到噪声限带,image_size 与渲染图一致,RMS 有限正,dropped_frames 空。"""
    out, _before, _ids = intrinsic_run
    for cam in (1, 2):
        payload = json.loads(
            (out / "intrinsic" / f"cam{cam}" / "intrinsics.json").read_text(encoding="utf-8"))
        assert set(payload) == _JSON_KEYS, "字段全集必须与 Pattern 7 契约全等(Phase 3 固定接口)"
        assert list(payload["params"]) == list(INTRINSIC_NAMES)
        _assert_params_recover_truth(payload["params"])
        assert payload["cam"] == cam
        assert payload["image_size"] == [2560, 1440], "image_size 必须与渲染图一致"
        for key in ("rms_train_overall", "rms_test_overall"):
            v = payload[key]
            assert np.isfinite(v) and v > 0, f"{key} 必须是有限正数,实际 {v}"
        assert 0.03 < payload["rms_train_overall"] < 0.2, "干净合成数据 train RMS 应落在噪声带"
        assert payload["dropped_frames"] == [], "无坏帧时不得误剔"
        assert payload["n_views_train"] == 12 and payload["n_views_test"] == 4
        assert payload["solver"] == "scipy" and payload["refine_enabled"] is True
        assert payload["config_snapshot"]["intrinsic"]["rms_test_max"] == 1.0


def test_e2e_artifacts_layout_and_report(intrinsic_run):
    """产物布局:每路 cam{n}/ 四件 + 根下逐帧 CSV(16 行)与 report.md;报告
    含两路、PASS 判定与离群段;CSV 列序/utf-8-sig/split 分布正确。"""
    out, _before, _ids = intrinsic_run
    root = out / "intrinsic"
    for cam in (1, 2):
        cam_dir = root / f"cam{cam}"
        for name in ("intrinsics.json", "train_residuals.npz",
                     "test_residuals.npz", "extrinsics.npz"):
            assert (cam_dir / name).is_file(), f"cam{cam} 缺产物 {name}"
        csv_path = root / f"per_frame_cam{cam}.csv"
        assert csv_path.read_bytes().startswith(b"\xef\xbb\xbf"), "逐帧 CSV 必须 utf-8-sig"
        with open(csv_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["image_id", "split", "source", "rms", "dropped", "note"]
            rows = list(reader)
        assert len(rows) == 16, f"逐帧行数必须 == 12 train + 4 test,实际 {len(rows)}"
        assert [r["split"] for r in rows] == ["train"] * 12 + ["test"] * 4
        assert all(r["dropped"] == "False" for r in rows), "干净数据无人被剔"
        assert all(r["source"] == "sb" for r in rows)
        assert all(r["rms"] for r in rows), "干净数据所有帧都有有限 RMS"
    report = (root / "report.md").read_text(encoding="utf-8")
    assert "cam1" in report and "cam2" in report, "报告必须两路并排"
    assert "PASS" in report, "干净运行两路都应判 PASS(阈值 1.0,RMS ~0.07)"
    assert "FAIL" not in report
    assert "离群" in report and "无(本轮运行没有帧超过离群阈值)" in report
    assert "config_snapshot" in report


def test_e2e_residuals_npz_alignment(intrinsic_run):
    """残差/外参 npz:allow_pickle=False 可读,形状/dtype 正确,image_ids 与
    JSON 的 n_views 对齐(行序契约对账列)。"""
    out, _before, ids = intrinsic_run
    for cam in (1, 2):
        payload = json.loads(
            (out / "intrinsic" / f"cam{cam}" / "intrinsics.json").read_text(encoding="utf-8"))
        train_res = np.load(out / "intrinsic" / f"cam{cam}" / "train_residuals.npz",
                            allow_pickle=False)
        test_res = np.load(out / "intrinsic" / f"cam{cam}" / "test_residuals.npz",
                           allow_pickle=False)
        assert set(train_res.files) == {"image_ids", "residuals", "per_view_rms"}
        assert train_res["residuals"].shape == (12, 88, 2)
        assert train_res["residuals"].dtype == np.float64
        assert train_res["per_view_rms"].shape == (12,)
        assert [str(s) for s in train_res["image_ids"]] == ids[f"cam{cam}_train"]
        assert len(train_res["image_ids"]) == payload["n_views_train"]
        assert not np.isnan(train_res["residuals"]).any(), "train 残差恒全有限"
        assert test_res["residuals"].shape == (4, 88, 2)
        assert [str(s) for s in test_res["image_ids"]] == ids[f"cam{cam}_test"]
        assert len(test_res["image_ids"]) == payload["n_views_test"]
        extr = np.load(out / "intrinsic" / f"cam{cam}" / "extrinsics.npz", allow_pickle=False)
        assert extr["rvecs"].shape == (12, 3) and extr["tvecs"].shape == (12, 3)
        assert [str(s) for s in extr["image_ids"]] == ids[f"cam{cam}_train"]


def test_e2e_solver_and_refine_contract(intrinsic_run):
    """solver 留痕:solver=scipy、refine_enabled=True、solver_meta 收敛判定
    依据齐全(status/nfev/final_cost/tail_improvement/converged,不静默)。"""
    out, _before, _ids = intrinsic_run
    for cam in (1, 2):
        payload = json.loads(
            (out / "intrinsic" / f"cam{cam}" / "intrinsics.json").read_text(encoding="utf-8"))
        assert payload["solver"] == "scipy"
        assert payload["refine_enabled"] is True
        meta = payload["solver_meta"]
        for key in ("status", "nfev", "final_cost", "tail_improvement", "converged"):
            assert key in meta, f"solver_meta 缺 {key}(收敛判定必须留痕不静默)"
        assert meta["path"] == "scipy_refine"
        assert isinstance(meta["converged"], bool)


def test_e2e_outlier_train_dropped_and_test_never(tmp_path, monkeypatch,
                                                  make_chessboard_image):
    """离群行为:train 注入 15px 量级坏帧(角点整体劣化,见 _synth_npz 注释)
    → 自动剔除且留痕(image_id/轮次/阈值);test 注入同样坏帧 → 永不剔除
    只标记(dropped 恒 False,高 RMS 如实记录)。"""
    bad_train, bad_test = 7, 2
    ids = _build_run_tree(tmp_path, make_chessboard_image,
                          bad_train_idx=bad_train, bad_test_idx=bad_test)
    monkeypatch.chdir(tmp_path)
    _run_intrinsic_in(tmp_path)
    out = tmp_path / "output"

    payload = json.loads(
        (out / "intrinsic" / "cam1" / "intrinsics.json").read_text(encoding="utf-8"))
    dropped = payload["dropped_frames"]
    bad_id = ids["cam1_train"][bad_train]
    assert [d["image_id"] for d in dropped] == [bad_id], "恰剔除注入的坏帧,不误伤好帧"
    assert dropped[0]["round"] == 1 and dropped[0]["threshold"] > 0, "剔除留痕必须含轮次与阈值"
    assert dropped[0]["rms"] > dropped[0]["threshold"], "被剔帧的 RMS 必须超阈值"
    assert all(d["image_id"] not in ids["cam1_test"] for d in dropped), "test 帧永不剔除"
    assert payload["n_views_train"] == 11, "幸存数 = 12 - 1"
    assert payload["rms_train_overall"] < 0.2, "剔掉坏帧后 train RMS 应回噪声带"

    with open(out / "intrinsic" / "per_frame_cam1.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_id = {r["image_id"]: r for r in rows}
    assert by_id[bad_id]["dropped"] == "True", "CSV 剔除行必须标记 dropped=True"
    assert "剔除" in by_id[bad_id]["note"] and by_id[bad_id]["note"], "剔除行必须有中文说明"
    bad_test_id = ids["cam1_test"][bad_test]
    assert by_id[bad_test_id]["dropped"] == "False", "坏 test 帧只标记,绝不剔除"
    assert float(by_id[bad_test_id]["rms"]) > 1.0, "坏 test 帧的高 RMS 必须如实记录(标记语义)"
    assert all(r["dropped"] == "False" for r in rows if r["split"] == "test")

    # cam2 未注入坏帧:不受牵连,零剔除
    payload2 = json.loads(
        (out / "intrinsic" / "cam2" / "intrinsics.json").read_text(encoding="utf-8"))
    assert payload2["dropped_frames"] == []


def test_e2e_rerun_clears_stale_and_data_readonly(tmp_path, monkeypatch,
                                                  make_chessboard_image):
    """清理与只读防线:重跑前 output/intrinsic/ 下的 stale 残留随运行消失
    (rmtree 固定覆盖);data/ 树快照运行前后逐字节一致(数据只读)。"""
    _build_run_tree(tmp_path, make_chessboard_image)
    monkeypatch.chdir(tmp_path)
    before = _tree_snapshot(tmp_path / "data")
    _run_intrinsic_in(tmp_path)

    intrinsic_root = tmp_path / "output" / "intrinsic"
    (intrinsic_root / "stale_leftover.json").write_text("{}", encoding="utf-8")
    (intrinsic_root / "cam1" / "stale.npz").write_bytes(b"stale from previous run")
    _run_intrinsic_in(tmp_path)  # 重跑:rmtree 应把整个 intrinsic 目录重建

    assert not (intrinsic_root / "stale_leftover.json").exists(), "固定覆盖后残留必须消失"
    assert not (intrinsic_root / "cam1" / "stale.npz").exists(), "子目录残留也必须消失"
    assert (intrinsic_root / "cam1" / "intrinsics.json").is_file(), "重建后产物仍齐全"
    assert (intrinsic_root / "report.md").is_file()
    assert _tree_snapshot(tmp_path / "data") == before, "intrinsic 流程绝不允许改动 data/"


def test_run_intrinsic_single_camera_failure_degrades_not_aborts(tmp_path, monkeypatch,
                                                                 make_chessboard_image):
    """编排层单路失败收口:cam2 定稿求解抛 AssertionError(模拟精化期
    models.project 的 Z>0 数据防卫)→ 该路降级为失败记录,cam1 照常跑完
    落盘,report.md 如实留痕失败段,run_intrinsic 绝不向外抛(CR-01 回归)。"""
    _build_run_tree(tmp_path, make_chessboard_image)
    monkeypatch.chdir(tmp_path)

    import stereo_calib.intrinsic.pipeline as pipeline_mod

    real_solve = pipeline_mod.solve_calibration
    refined_calls = {"n": 0}

    def flaky_solve(uvs, objp, image_size, refine_cfg):
        # 剔除循环走 no_refine(enabled=False);带精化的定稿求解每路恰一次、
        # 按cam 序串行——第 2 次精化调用即 cam2,精准注入单路失败。
        if refine_cfg.enabled:
            refined_calls["n"] += 1
            if refined_calls["n"] == 2:
                raise AssertionError("project 收到 Z<=0 的点(测试注入:模拟数据防卫触发)")
        return real_solve(uvs, objp, image_size, refine_cfg)

    monkeypatch.setattr(pipeline_mod, "solve_calibration", flaky_solve)
    _run_intrinsic_in(tmp_path)  # 不抛即第一契约:单路失败不中断整批

    root = tmp_path / "output" / "intrinsic"
    assert (root / "cam1" / "intrinsics.json").is_file(), "成功路必须照常跑完并落盘"
    assert not (root / "cam2").exists(), "失败路不得留下半套产物"
    report = (root / "report.md").read_text(encoding="utf-8")
    assert "## 失败留痕" in report and "AssertionError" in report, (
        "报告必须如实留痕失败路与其病因")
    assert "cam1" in report, "成功路必须照常出现在报告中(不受失败路影响)"


def test_pipeline_and_artifacts_source_guards():
    """源码防线:pipeline 零数学(投影/求解内部符号不得出现,只准编排);
    artifacts 零数学且 JSON 契约字段表存在(字段全集测试已单独锁定)。"""
    pipeline_src = pathlib.Path("src/stereo_calib/intrinsic/pipeline.py").read_text(encoding="utf-8")
    for banned in ("_distort", "least_squares", "calibrateCameraExtended"):
        assert banned not in pipeline_src, \
            f"编排层不得出现 {banned}(数学只准经 models/solve 入口,零数学防线)"
    for required in ("rmtree", "load_views", "solve_calibration",
                     "score_pose_only", "compute_metrics", "median", "rms_multiplier"):
        assert required in pipeline_src, f"编排层必须经 {required} 入口(编排只调模块入口)"
    artifacts_src = pathlib.Path("src/stereo_calib/intrinsic/artifacts.py").read_text(encoding="utf-8")
    for banned in ("_distort", "least_squares", "calibrateCameraExtended"):
        assert banned not in artifacts_src, f"落盘层不得出现 {banned}(落盘零数学)"


# ===========================================================================
# 02-04 任务 1:CLI intrinsic 子命令行为(真实入口,进程内直调)
# ===========================================================================

def test_cli_intrinsic_bad_input_chinese_exit_one(tmp_path, monkeypatch):
    """坏环境(config 在而 Phase 1 角点 npz 缺失)→ CLI 收口为中文一句 +
    typer.Exit code 1,坏输入绝不吐堆栈(照 preprocess 体例)。"""
    (tmp_path / "config.yaml").write_text(_E2E_CONFIG_YAML, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_main, "_config_path", pathlib.Path("config.yaml"))
    with pytest.raises(typer.Exit) as ei:
        cli_main.intrinsic()
    assert ei.value.exit_code == 1, "坏输入必须退出码 1(中文收口契约)"
    assert not (tmp_path / "output" / "intrinsic").exists(), "失败运行不得留下半套产物"


def test_cli_intrinsic_happy_path_end_to_end(tmp_path, monkeypatch, make_chessboard_image):
    """正常合成环境:CLI 一条命令(仅 --config,零算法参数)端到端不抛,
    五类产物齐全在位——真实命令入口锁定(02-03 e2e 夹具复用)。"""
    _build_run_tree(tmp_path, make_chessboard_image)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_main, "_config_path", pathlib.Path("config.yaml"))
    cli_main.intrinsic()  # 不抛即通过:中文收口层没有误伤正常路径
    root = tmp_path / "output" / "intrinsic"
    for cam in (1, 2):
        assert (root / f"cam{cam}" / "intrinsics.json").is_file()
        assert (root / f"per_frame_cam{cam}.csv").is_file()
    assert (root / "report.md").is_file()


def test_cli_intrinsic_numeric_exception_family_chinese_exit_one(tmp_path, monkeypatch):
    """数值路径异常家族(AssertionError 等,CR-01)逃出编排层时,CLI 仍收口为
    退出码 1,绝不吐堆栈(最外层中文收口防线对异常家族的扩展锁定)。"""
    (tmp_path / "config.yaml").write_text(_E2E_CONFIG_YAML, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_main, "_config_path", pathlib.Path("config.yaml"))

    import stereo_calib.intrinsic.pipeline as pipeline_mod

    def boom(cfg):
        raise AssertionError("project 收到 Z<=0 的点(测试注入)")

    monkeypatch.setattr(pipeline_mod, "run_intrinsic", boom)
    with pytest.raises(typer.Exit) as ei:
        cli_main.intrinsic()
    assert ei.value.exit_code == 1, "数值路径异常家族必须同样收口为退出码 1"
