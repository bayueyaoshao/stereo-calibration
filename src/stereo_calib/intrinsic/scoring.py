"""test 帧只估位姿评分:内参当尺子,只挪棋盘位姿(INTR-03 的评分器)。

移植自参考工程 calib/solve/scoring.py 的 solve_pose_only(只读引用,
绝不修改);单帧失败不中断整批(参考 T-02-06-04 模式)——任何失败路径
返回 (NaN, None, 中文备注),绝不抛异常,调用方(02-03 编排)循环整批
test 帧,一帧失败只留痕。

白话:标定把内参算出来之后,"一张没参与标定的照片拍得准不准"怎么量?
把内参**当尺子用**,只挪棋盘的位姿(6 个数),让棋盘投影尽量贴近检测
角点——挪完剩下的偏差就是这张照片的重投影误差。内参固定不动,解的是
6 参数小最小二乘,每帧毫秒级。

初值路线(Pitfall 6 修复,错误盆地真实案例:40° 倾角 + 强畸变 + 偏心
帧被老路线误评 4.82px、正确 0.07px):
1. 角点先经 models.unproject 反投影成单位射线(牛顿迭代反投影天然完成
   去畸变),取归一化平面坐标——初值不带畸变偏差;
2. cv2.solvePnP(物点, 平面坐标, 单位阵 K, 零畸变)解位姿初值;
3. 6 参数 trf 精化(内参固定不进 theta),残差只经 models(单一数学源,
   源码防线锁定)。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from stereo_calib.intrinsic import models

# solvePnP 迭代解法要求的最少点数(4 点以下方程组欠定,硬算必错)
_MIN_PNP_POINTS = 4


@dataclass(frozen=True)
class PoseScore:
    """单帧只估位姿评分结果(不可变)。

    rms:像素单位的该帧 RMS,NaN 表示该帧失败;
    residuals:(P,2) 精化后残差,失败为 None;
    note:中文备注(成功为空串,失败写明病因与指引)。
    """

    rms: float
    residuals: np.ndarray | None
    note: str


def _failure(note: str) -> PoseScore:
    """失败帧统一出口:(NaN, None, 中文备注),绝不抛异常中断整批。"""
    return PoseScore(rms=float("nan"), residuals=None, note=note)


def score_pose_only(uv, objp, params) -> PoseScore:
    """内参当尺子用、只挪棋盘的位姿,给没参与标定的照片打分。

    参数:
        uv:     (P,2) 单视角实测角点(像素),P >= 4;
        objp:   (P,3) 物点网格(与 uv 逐行对齐,Z=0 棋盘平面);
        params: (9,) 标定得到的内参参数向量(布局 INTRINSIC_NAMES,固定不动)。
    返回:
        PoseScore(rms, residuals, note)——任何失败路径都不抛异常。
    """
    uv_arr = np.asarray(uv, dtype=np.float64)
    if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
        return _failure(
            f"角点形状不合:期望 (P,2) 单视角像素坐标,实际是 {uv_arr.shape}。"
            f"只估位姿一次处理一张照片,请检查调用方的数据排布。")
    objp_arr = np.asarray(objp, dtype=np.float64)
    if objp_arr.shape != (uv_arr.shape[0], 3):
        return _failure(
            f"物点形状不合:期望 ({uv_arr.shape[0]},3) 与角点逐行对齐,"
            f"实际是 {objp_arr.shape}。物点网格与检测角点必须一一对齐,请检查行序。")
    if uv_arr.shape[0] < _MIN_PNP_POINTS:
        return _failure(
            f"角点数 {uv_arr.shape[0]} 少于位姿求解下限 {_MIN_PNP_POINTS} 个:"
            f"方程组欠定,位姿解不出来,请检查该帧的角点检出质量。")
    if not np.isfinite(uv_arr).all():
        bad = int((~np.isfinite(uv_arr)).sum())
        return _failure(
            f"角点里混入 {bad} 个非数值(NaN/无穷),该帧几何可疑:"
            f"按单帧失败记录,不影响其余帧。")

    p = np.asarray(params, dtype=np.float64).ravel()
    try:
        # ---- 初值:反投影到归一化平面坐标后按单位阵 K 解 PnP ----
        # 零畸变直接按像素解 PnP 的老路线会把精化带进错误盆地(见模块
        # docstring 的真实案例);反投影路线天然去畸变,只经模型层接口。
        rays = models.unproject(uv_arr, p)
        if not bool((rays[:, 2] > 0).all()):
            return _failure(
                "反投影射线出现 Z 分量非正的角点,归一化平面坐标不存在,"
                "该帧几何可疑:按单帧失败记录,不影响其余帧。")
        ok, rvec0, tvec0 = cv2.solvePnP(
            objp_arr, rays[:, :2] / rays[:, 2:3], np.eye(3), np.zeros(5))
        if not ok:
            return _failure(
                "位姿初值求解失败:该帧角点可能退化(近共线/分布过窄),"
                "按单帧失败记录,不影响其余帧。")

        # ---- 精化:6 参数最小二乘,内参加锁(尺子不是被量对象) ----
        def residual(theta: np.ndarray) -> np.ndarray:
            pts_cam, _pose_block = models.transform_to_camera(
                objp_arr, theta[:3], theta[3:])
            return (models.project(pts_cam, p) - uv_arr).ravel()

        theta0 = np.concatenate([np.asarray(rvec0, dtype=np.float64).ravel(),
                                 np.asarray(tvec0, dtype=np.float64).ravel()])
        sol = least_squares(residual, theta0, method="trf", x_scale="jac")
        residuals = np.asarray(sol.fun, dtype=np.float64).reshape(-1, 2)
        rms = float(np.sqrt(np.mean(np.sum(residuals * residuals, axis=1))))
        return PoseScore(rms=rms, residuals=residuals, note="")
    except (ValueError, ArithmeticError, RuntimeError, cv2.error) as e:
        # 数值路径的失败(np.linalg.LinAlgError 是 ValueError 子类,一并覆盖)
        # 收口成单帧失败记录;形状/参数错误已在入口防线拦下,不吞代码契约。
        return _failure(f"只估位姿求解失败:{type(e).__name__}——{e}")
