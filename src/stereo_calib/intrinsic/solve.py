"""标定求解引擎:OpenCV 初值 + scipy 精化(模块2 引擎切片,INTR-01)。

移植自参考工程 calib/solve/opencv.py、calib/solve/scipy_solver.py、
calib/solve/residuals.py(只读引用,绝不修改)。裁剪声明:本工程模型
固定 M2 九参数,参考工程的多模型分派/注册机制一律不移植;算法与调用
形态照搬参考工程已验证版本。

结构(单一数学源纪律,RESEARCH Pattern 1/3):
- OpenCV 路径 solve_opencv:标定只当黑盒初始化器,float32 装包规则隔离
  在本文件调用处(绑定层只收 32 位,进 64 位会当场拒收);
- scipy 路径 refine_scipy:theta 联合装箱 + 解析稀疏雅可比 + trf/lsmr;
  残差与雅可比只准 import intrinsic.models 的投影/变换函数,本文件不
  出现第二份投影或畸变公式(源码防线测试锁定);
- 编排入口 solve_calibration:初值 →(配置开关)精化,02-03 唯一消费它。

指标口径(Pitfall 12):OpenCV 返回的整体 RMS 基于 float32 装包角点,
与 float64 回代差约 1e-4 像素——只进 solver_meta["opencv_rms"] 留对账
参考;一切报告指标只准走 intrinsic.metrics 的 float64 回代尺。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# 求解器自身的视图数底线:两三个视角定不了内参(装载层 min_views 默认 10,
# 是编排层的配置下限;这里是硬算底线,低于它直接中文报错不硬算)
_MIN_SOLVE_VIEWS = 2


@dataclass(frozen=True)
class SolveResult:
    """一次求解的参数与逐视图位姿(不可变):全部数组 float64。

    params (9,) 布局见 models.INTRINSIC_NAMES;rvecs/tvecs (V,3) 与输入
    视图逐行对齐(行序契约,装载层定死,下游绝不重排);solver 如实标注
    解的来源("opencv" 初值 | "scipy" 精化);solver_meta 留对账与收敛
    判定依据(status/nfev/final_cost/tail_improvement/opencv_rms),
    不静默(T-02-05)。
    """

    params: np.ndarray
    rvecs: np.ndarray
    tvecs: np.ndarray
    solver: str
    solver_meta: dict


def _as_view_array(uvs) -> np.ndarray:
    """把 (V,P,2) 数组或逐视图 (P,2) 列表统一成 (V,P,2) float64 并断言形状。"""
    arr = np.asarray(uvs, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 2 or arr.shape[0] < 1:
        raise ValueError(
            f"角点观测形状不合:期望 (视图数,角点数,2),实际是 {arr.shape}。"
            f"请检查调用方传入的是不是逐视图逐角点的像素坐标(或这类数组的列表)。"
        )
    return arr


def _check_common_inputs(uvs, objp, image_size) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """求解入口的公共形状/取值防线(T-02-04:装载层三重防线之后的再断言)。"""
    uvs_arr = _as_view_array(uvs)
    objp_arr = np.asarray(objp, dtype=np.float64)
    if objp_arr.ndim != 2 or objp_arr.shape[1] != 3:
        raise ValueError(
            f"物点形状不合:期望 (角点数,3),实际是 {objp_arr.shape}。"
            f"板系物点必须逐点给 (x,y,0) 三个数,请检查棋盘物点网格的排布。"
        )
    if uvs_arr.shape[1] != objp_arr.shape[0]:
        raise ValueError(
            f"角点与物点数不一致:角点 {uvs_arr.shape[1]} 个,物点 {objp_arr.shape[0]} 个。"
            f"两者必须逐点对齐,请检查角点文件与棋盘规格是否同一套。"
        )
    size = tuple(image_size)
    if (len(size) != 2 or not all(isinstance(v, (int, np.integer)) or float(v).is_integer()
                                  for v in size) or any(v <= 0 for v in size)):
        raise ValueError(
            f"image_size 不合法:期望 (宽, 高) 两个正整数,实际是 {image_size!r}。"
            f"注意 OpenCV 口径是 (宽, 高),与 numpy 图像形状的 (高, 宽) 相反。"
        )
    return uvs_arr, objp_arr, (int(size[0]), int(size[1]))


def solve_opencv(uvs, objp, image_size) -> SolveResult:
    """OpenCV 初值路径:全 5 畸变一起估,返回"初值版"标定结果。

    白话:把角点观测交标定黑盒拿一份能用的起点(内参 + 每视图位姿),
    精度到初值口径为止——后续要么直接消费(精化开关关),要么交给
    refine_scipy 精化。flags=0 即按 M2 布局一次估全 5 个畸变系数。

    参数:
        uvs:        (V,P,2) 或逐视图 (P,2) 列表,角点像素坐标(行序契约);
        objp:       (P,3) 板系物点(Z=0 平面);
        image_size: (宽, 高) 元组——OpenCV 口径,与 numpy 图像形状的
                    (高, 宽) 相反(Pitfall 2,写反会让 RMS 高得离谱)。
    """
    uvs_arr, objp_arr, size = _check_common_inputs(uvs, objp, image_size)
    n_views = uvs_arr.shape[0]
    n_points = uvs_arr.shape[1]
    if n_views < _MIN_SOLVE_VIEWS:
        raise ValueError(
            f"视图数不足以标定:只给了 {n_views} 个视图(下限 {_MIN_SOLVE_VIEWS}),"
            f"image_size={size}。两三个视角定不了内参,硬算必错——请检查角点"
            f"检出率与 exclude_images 清单。"
        )

    # OpenCV 绑定只收 32 位浮点(Point2f/Point3f),传 64 位报"Unsupported
    # format"——显式降精度装包:像素坐标 float32 表达精度 ~1e-4 像素,远
    # 小于 0.5px 断言线,可接受(参考工程同款纪律);回代指标走 float64。
    objs32 = [objp_arr.astype(np.float32).reshape(n_points, 1, 3) for _ in range(n_views)]
    imgs32 = [uv.astype(np.float32).reshape(n_points, 1, 2) for uv in uvs_arr]

    try:
        # 第 8 个返回值 per_view 是 float32 口径的逐帧表,剔除判据与报告
        # 一律不用它(Pitfall 12),下划线接收不读。
        rms, K, D, rvecs, tvecs, _std_int, _std_ext, _per_view = cv2.calibrateCameraExtended(
            objs32, imgs32, size, None, None, flags=0)
    except cv2.error as e:
        raise ValueError(
            f"OpenCV 标定求解失败:{n_views} 个视图,image_size={size},"
            f"OpenCV 报错——{e}。多半是数据退化(角点近共线/位姿过近),"
            f"请检查该批视图的分布。"
        ) from e

    # 解包断言:畸变向量必须恰 5 长(防 flags 布局陷阱;M2 布局 [k1,k2,p1,p2,k3])。
    # OpenCV 把 D 装成 (1,5) 矩阵返回,先拉平再断言。
    D = np.asarray(D, dtype=np.float64).ravel()
    assert D.shape == (5,), (
        f"OpenCV 解包的畸变向量长度是 {D.shape},期望 (5,)。本工程固定全 5 畸变,"
        f"出现此错说明求解调用被改动,请对照参考工程核查。")
    params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D], dtype=np.float64)
    rvecs_arr = np.asarray([np.asarray(r, dtype=np.float64).ravel() for r in rvecs],
                           dtype=np.float64).reshape(n_views, 3)
    tvecs_arr = np.asarray([np.asarray(t, dtype=np.float64).ravel() for t in tvecs],
                           dtype=np.float64).reshape(n_views, 3)
    return SolveResult(
        params=params, rvecs=rvecs_arr, tvecs=tvecs_arr, solver="opencv",
        solver_meta={"path": "calibrateCameraExtended", "flags_used": 0,
                     "opencv_rms": float(rms)})
