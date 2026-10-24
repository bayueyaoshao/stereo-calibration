"""float64 回代重投影指标(唯一指标尺):逐帧 RMS 与 pooled 整体 RMS。

移植自参考工程 calib.solve/result.py 的 compute_result_metrics(只读引用,
绝不修改)。残差只由 intrinsic.models.transform_to_camera + models.project
构成(单一数学源,RESEARCH Pattern 4 口径)——本模块与任何文件里都不准
出现第二份投影或畸变公式(源码防线测试锁定:本模块源文不含畸变核符号)。

两把尺的差异是实测的(Pitfall 3):OpenCV 返回的整体 RMS 基于 float32
装包角点,与 float64 回代差约 1e-4 像素。成功标准明确禁止把 OpenCV 返回
值当报告指标——本模块是全部指标(train/test、逐帧、整体)的唯一出口;
OpenCV 的 rms 只准进 solver_meta 留对账参考(02-02 接入)。

口径:逐帧 RMS = sqrt(该帧全部角点平方欧氏残差的均值);pooled 整体
RMS = sqrt(全部视图全部角点平方残差的均值)。它不是"逐帧 RMS 再平均"
的简写——pooled 公式唯一,等角点帧下数值相同也要按 pooled 算。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stereo_calib.intrinsic import models


@dataclass(frozen=True)
class ResidualMetrics:
    """回代指标集(不可变):per_view_rms/residuals 行序与输入视图逐行对应。

    per_view_rms (V,) float64;residuals (V, P, 2) float64;
    rms_overall 为 pooled 口径的整体 RMS(float 标量)。
    """

    per_view_rms: np.ndarray
    rms_overall: float
    residuals: np.ndarray


def compute_metrics(objp: np.ndarray, uvs: np.ndarray, params: np.ndarray,
                    rvecs: np.ndarray, tvecs: np.ndarray) -> ResidualMetrics:
    """按视图循环回代计算残差与 RMS(视图内全 numpy 向量化,Pitfall 5)。

    参数形状:objp (P,3);uvs (V,P,2);params (9,)/(12,)(布局
    intrinsic_names(模型):M2 九参 / M8 十二参);
    rvecs/tvecs (V,3)——位姿必须与 uvs 视图逐行对齐(行序契约,Pitfall 4/10),
    错位不报错、只是悄悄算错,装载层的行序锁是它的上游防线。
    """
    objp_arr = np.asarray(objp, dtype=np.float64)
    uvs_arr = np.asarray(uvs, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64).ravel()
    rvecs_arr = np.asarray(rvecs, dtype=np.float64)
    tvecs_arr = np.asarray(tvecs, dtype=np.float64)

    if uvs_arr.ndim != 3 or uvs_arr.shape[2] != 2:
        raise ValueError(
            f"uvs 形状不合:期望 (V,P,2),实际是 {uvs_arr.shape}。"
            f"请检查调用方传入的是不是逐视图逐角点的像素坐标。"
        )
    n_views = uvs_arr.shape[0]
    n_points = objp_arr.shape[0]
    if rvecs_arr.shape != (n_views, 3) or tvecs_arr.shape != (n_views, 3):
        raise ValueError(
            f"回代位姿形状不合:期望 rvecs/tvecs 各 ({n_views},3),"
            f"实际是 {rvecs_arr.shape} / {tvecs_arr.shape}。"
            f"位姿必须与 {n_views} 个视图逐行对齐(行序契约)。"
        )

    residuals = np.empty((n_views, n_points, 2), dtype=np.float64)
    for i in range(n_views):
        pts_cam, _pose_deriv = models.transform_to_camera(objp_arr, rvecs_arr[i], tvecs_arr[i])
        residuals[i] = models.project(pts_cam, p) - uvs_arr[i]

    squared = np.sum(residuals * residuals, axis=2)   # 每角点的平方欧氏残差 (V,P)
    per_view_rms = np.sqrt(squared.mean(axis=1))      # (V,)
    rms_overall = float(np.sqrt(squared.mean()))      # pooled
    return ResidualMetrics(per_view_rms=np.asarray(per_view_rms, dtype=np.float64),
                           rms_overall=rms_overall, residuals=residuals)
