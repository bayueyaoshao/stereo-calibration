"""角点核心三件套的不变量锁定测试:窗口断言、排序规范化、SB 检测精度、源码防线。

移植自参考工程 tests/test_corners.py(只读引用,绝不修改),按本计划
(01-01 仅 SB 主路线)裁剪:classic 精度/缺陷闸门测试随 01-02 一并落地。

源码防线注意:测试与注释里都不写出二次精化的调用串原文,否则子串计数
防线自伤(计数对象是 detect.py 的源文本,不是本文件)。
"""

import pathlib

import numpy as np
import pytest

from stereo_calib.corners.canonical import canonicalize
from stereo_calib.corners.detect import detect_sb
from stereo_calib.corners.window import derive_window_frame, estimate_sq_px
from stereo_calib.io.config import CornersConfig

# 检测器测试用配置:直接用数据类默认值(与仓库 config.yaml 一致)
_CFG = CornersConfig()


# ---------- 帧级窗口推导(不变量 2)----------

@pytest.mark.parametrize("sq_px", [15.0, 20.0, 25.0, 40.0])
def test_derive_window_frame_inequality(sq_px):
    """四个典型方格边长逐一验证:搜索区 2w+1 必须严格小于方格边长。

    例:sq=20 → w=9,搜索区 19 < 20;sq=15 → w=6,搜索区 13 < 15。
    """
    w = derive_window_frame(sq_px)
    assert w == int(sq_px // 2) - 1
    assert 2 * w + 1 < sq_px


def test_derive_window_frame_min_win_floor():
    """min_win=3 下限生效:极小方格不给小于 3 的窗口。

    sq=7.5:floor(3.75)−1 = 2,被下限托到 3,且 7 < 7.5 仍成立(合法);
    sq=5:下限托到 3 后 7 ≥ 5,不等式违反——按设计必须当场抛断言错误
    (这是正确行为,用测试锁定,绝不允许带病窗口往下流)。
    """
    assert derive_window_frame(7.5, min_win=3) == 3
    with pytest.raises(AssertionError):
        derive_window_frame(5.0, min_win=3)  # 方格过小必须当场报错


def test_estimate_sq_px_conservative(make_chessboard_image):
    """合成图上估计的方格边长接近真值且保守取小(不应超过真值太多)。"""
    _img, corners, sq_truth = make_chessboard_image(
        square_px=24.0, rvec=np.array([0.0, 0.35, 0.0]))
    sq_est = estimate_sq_px(corners, cols=11, rows=8)
    assert 0.9 * sq_truth < sq_est <= sq_truth * 1.001


# ---------- 排序规范化(不变量 3)----------

def _four_variants(corners: np.ndarray, cols: int, rows: int) -> dict[str, np.ndarray]:
    """把同一份规范角点序列变换成四种排列变体(恒等/水平/垂直/旋转180°)。"""
    g = np.asarray(corners, dtype=np.float64).reshape(rows, cols, 2)
    return {
        "恒等": g.copy(),
        "水平翻转": g[:, ::-1, :],
        "垂直翻转": g[::-1, :, :],
        "旋转180": g[::-1, ::-1, :],
    }


def test_canonicalize_maps_all_variants_to_same_sequence(make_chessboard_image):
    """四种翻转变体经规范化后逐点相等:输出唯一形态(合成图行朝右、列朝下,
    夹具真值本身就是规范形态)。"""
    _img, corners, _ = make_chessboard_image()
    base = canonicalize(corners, cols=11, rows=8)
    assert np.allclose(base, corners)  # 规范输入原样通过(不动手)
    for name, variant in _four_variants(corners, 11, 8).items():
        out = canonicalize(variant.reshape(-1, 2), cols=11, rows=8)
        assert np.allclose(out, base), f"变体「{name}」没有被规范化到唯一形态"


# ---------- SB 主路线检测(不变量 1/2/3 的行为面)----------

def test_detect_sb_accuracy_within_half_pixel(make_chessboard_image):
    """合成图上 SB 检测器检出,角点对真值的逐点最大定位误差 < 0.5px。"""
    img, corners, _ = make_chessboard_image(degrade="blur")
    result = detect_sb(img, 11, 8, _CFG)
    assert result is not None, "SB 检测器未检出合成棋盘"
    assert result["detector"] == "sb"
    assert result["uv"].shape == (88, 2)
    assert result["uv"].dtype == np.float64
    assert result["recovery"] is None
    truth = canonicalize(corners, 11, 8)
    max_err = float(np.linalg.norm(result["uv"] - truth, axis=1).max())
    assert max_err < 0.5, f"SB 最大定位误差 {max_err:.3f}px 超过 0.5px"


def test_detect_sb_window_rule_on_synthetic(make_chessboard_image):
    """SB 返回的窗口记录也满足 2*win+1 < sq_px(结构防线延伸)。"""
    img, _, _ = make_chessboard_image(square_px=24.0, degrade="noise")
    result = detect_sb(img, 11, 8, _CFG)
    assert result is not None
    assert 2 * result["win_size"] + 1 < result["sq_px"]


def test_detect_sb_returns_none_on_pure_noise():
    """纯噪声图:SB 未检出,统一返回 None(不返回半成品)。"""
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 255, size=(480, 640), dtype=np.uint8)
    assert detect_sb(noise, 11, 8, _CFG) is None


def test_detect_sb_source_guard_no_refinement_call():
    """源码防线:01-01 阶段 detect.py 里二次精化的调用形式出现次数为 0
    (SB 路线严禁二次精化;01-02 落地 classic 后本断言改为 == 1)。"""
    source = pathlib.Path("src/stereo_calib/corners/detect.py").read_text(encoding="utf-8")
    assert source.count("cv2.cornerSubPix(") == 0, (
        "二次亚像素精化的调用形式在本阶段必须出现 0 次(classic 未落地);"
        "SB 路线再精化属于头号污染错误"
    )
