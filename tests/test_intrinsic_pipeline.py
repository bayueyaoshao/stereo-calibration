"""装载契约与 float64 指标尺测试:合成 npz 装载三重防线 + 真值噪声 RMS。

本计划(02-01)只放装载契约与 metrics 用例;端到端(合成 npz → 进程内跑
intrinsic 命令 → 产物契约)由 02-03 扩展。合成路线照 RESEARCH Code Examples:
Phase 2 消费的是角点数组,不需要渲染图像——用真值参数 + 随机位姿投影合成
(V,88,2) 角点,按 Phase 1 八数组契约 np.savez(allow_pickle=False 可读),
秒级完成(图像文件仅 image_size 派生需要,用最小噪声图即可)。
"""

import csv
import json
import pathlib

import cv2
import numpy as np
import pytest

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
_JSON_KEYS = {
    "cam", "image_size", "params", "K", "D", "solver", "refine_enabled",
    "rms_train_overall", "rms_test_overall", "n_views_train", "n_views_test",
    "dropped_frames", "solver_meta", "board", "config_snapshot",
}


def test_write_intrinsics_json_roundtrip(tmp_path):
    """写 → 读回逐字段断言:字段全集、params 按名、K/D 形状与数值冗余一致。"""
    dropped = [{"image_id": "003_cam1", "rms": 15.2, "round": 1, "threshold": 0.35}]
    out = write_intrinsics_json(
        tmp_path / "cam1" / "intrinsics.json", cam=1, image_size=(2560, 1440),
        params=_TRUE_PARAMS, K=intrinsics_matrix(_TRUE_PARAMS), D=_TRUE_PARAMS[4:9],
        solver="scipy", refine_enabled=True, rms_train_overall=0.3316,
        rms_test_overall=0.2389, n_views_train=141, n_views_test=15,
        dropped_frames=dropped, solver_meta={"status": 0, "opencv_rms": 2.5782},
        board=BoardSpec(), config_snapshot={"intrinsic": {"rms_test_max": 1.0}})
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
