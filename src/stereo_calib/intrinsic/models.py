"""M2/M8 相机模型(针孔 + Brown-Conrady 五系数 / rational 八系数畸变)——内参链路的唯一数学源。

移植裁剪声明(体例照 corners/canonical.py):移植自参考工程
calib.models/pinhole_brown.py 与 calib.models/base.py(只读引用,绝不修改)。
参考工程是多模型族 + 注册机制;本工程不做注册表——M2(9 参:fx,fy,cx,cy,
k1,k2,p1,p2,k3)与 M8(12 参:上 9 参 + k4,k5,k6 rational 分母项)双模型
最小支持,公式体只有一份:M2 是 M8 的「den≡1 特例」,径向公式在全部代码
里只有一处。2026-10-24 用户指令 supersession:此前「扩展模型时按参考工程
方式扩回」的裁剪声明由「按本文件的最小双模型方式扩展」取代(根因:M2 对
本相机 8mm C-mount 4MP 欠拟合,rational8 是唯一堵点);注册机制依旧不移植。

白话口径:三维点先除以深度(Z)得归一化坐标,经径向/有理畸变弄弯,再乘
焦距加主点落到像素。M2 径向缩放 rad = 1+k1·r²+k2·r⁴+k3·r⁶;M8 为
ratio = num/den(num = 1+k1·r²+k2·r⁴+k3·r⁶,den = 1+k4·r²+k5·r⁴+k6·r⁶,
即 OpenCV 布局 D = [k1,k2,p1,p2,k3,k4,k5,k6],den 项的文档名叫 k4,k5,k6)。
切向项 (p1,p2) 加在径向缩放之外——嵌套方式经参考工程实测对齐 cv2 4.14.0
的 projectPoints(M2/M8 parity 均 < 1e-9 像素,测试锁定)。

单一数学源纪律(D-06 同款):残差、精化、评分、报告回代只准 import 本
模块的投影/变换函数,任何文件里都不许出现第二份畸变或旋转公式
(metrics/solve 的源码防线测试锁定)。

dtype 与形状硬规则:全函数 float64 进出(入口 np.asarray 归一);返回
新数组不改输入;非法输入抛中文 AssertionError(数学层断言,绝不返回
半成品)。OpenCV 侧的 float32 装包规则属于求解器(02-02),不在本文件。
"""

from __future__ import annotations

import cv2
import numpy as np

# 参数向量布局:任何代码不得按位置猜参数,统一引用本常量(或 intrinsic_names 出口)。
# INTRINSIC_NAMES 保留为 M2 别名(存量测试与旧契约的向后兼容锚点)。
INTRINSIC_NAMES_M2: tuple[str, ...] = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
INTRINSIC_NAMES_M8: tuple[str, ...] = (
    "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
INTRINSIC_NAMES: tuple[str, ...] = INTRINSIC_NAMES_M2

_NAMES_BY_MODEL = {"M2": INTRINSIC_NAMES_M2, "M8": INTRINSIC_NAMES_M8}
_NAMES_BY_LEN = {len(v): v for v in _NAMES_BY_MODEL.values()}

# rational 分母的近零防线:工作域 |den−1| 最大约 0.5,远离 0;低于它即病态
_DEN_MIN = 1.0e-6


def intrinsic_names(model: str) -> tuple[str, ...]:
    """模型名 → 参数布局名元组(全工程唯一的按名取布局出口)。

    "M2" → 九名(fx…k3,D 长 5);"M8" → 十二名(fx…k6,D 长 8,k4,k5,k6
    是 rational 分母的 r²,r⁴,r⁶ 项)。其余值中文 ValueError——模型名只
    出自 config.yaml 与 intrinsics.json,拼错必须在入口拦下。
    """
    if model in _NAMES_BY_MODEL:
        return _NAMES_BY_MODEL[model]
    raise ValueError(
        f"畸变模型名不合法:{model!r}。只支持 'M2'(五参 Brown-Conrady,"
        f"D 长 5)或 'M8'(rational 8,D 长 8)。"
        f"请检查 config.yaml 的 intrinsic.distortion_model 或内参产物。")


def _as_points3d(pts: np.ndarray, what: str = "pts_cam") -> np.ndarray:
    """校验并统一三维点数组为 (N,3) float64;形状不合抛中文 AssertionError。"""
    arr = np.asarray(pts, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise AssertionError(
            f"{what} 形状不合:期望 (N,3),实际是 {np.asarray(pts).shape}。"
            f"相机坐标系三维点必须逐点给 (x,y,z) 三个数,请检查调用方的数据排布。"
        )
    return arr


def _as_params(params: np.ndarray) -> np.ndarray:
    """校验参数向量并统一为 float64;布局按长度分派(9 → M2、12 → M8)。"""
    p = np.asarray(params, dtype=np.float64).ravel()
    names = _NAMES_BY_LEN.get(p.shape[0])
    if names is None:
        raise AssertionError(
            f"参数向量形状不合:期望 (9,)(布局 {INTRINSIC_NAMES_M2})或 (12,)"
            f"(布局 {INTRINSIC_NAMES_M8}),实际拍平后是 {p.shape}。"
            f"请按 intrinsic_names(模型名) 给出的顺序传参,不要按位置猜。"
        )
    return p


def intrinsics_matrix(params: np.ndarray) -> np.ndarray:
    """参数向量 → 标准 3×3 内参矩阵 K([[fx,0,cx],[0,fy,cy],[0,0,1]]),float64。

    skew 恒 0(参数表里本来就没有 skew 项);供 OpenCV 调用装包与
    Phase 3 按名读取,冗余出口只此一份。
    """
    p = _as_params(params)
    return np.array([[p[0], 0.0, p[2]],
                     [0.0, p[1], p[3]],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def _radial_and_derivs(r2: np.ndarray,
                       dist: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """rational 径向缩放 ratio = num/den 及其全部导数原料(投影雅可比与畸变逆解共用)。

    dist 布局按长度分派(公式体只有这一份,M2 是 den≡1 特例):
      5 长(M2):num = 1+k1·r²+k2·r⁴+k3·r⁶(下标 0,1,4),den ≡ 1;
      8 长(M8):num 同上,den = 1+k4·r²+k5·r⁴+k6·r⁶(下标 5,6,7)。
    返回 (ratio (N,), d_ratio_dr2 (N,), d_ratio_dk (N,K), k_dist_slots):
      ratio         径向缩放系数(M2 时即旧 rad,数值逐位一致);
      d_ratio_dr2   对 r² 的导数(商法则 (num'·den − num·den')/den²);
      d_ratio_dk    对 k 槽位的导数列:分子槽 r^(2j)/den,分母槽 −num·r^(2j)/den²;
      k_dist_slots  k 槽位在 dist 里的下标(M2:(0,1,4);M8:(0,1,4,5,6,7));
                    参数向量下标 = 4 + 槽位(dist 从 params[4:] 开始)。
    den 在工作域不取零(本机 |den−1| 最大约 0.5),不做正则化保护;|den|
    低于 _DEN_MIN 时中文 AssertionError 防静默垃圾(除零/翻折前拦下)。
    """
    d = np.asarray(dist, dtype=np.float64).ravel()
    r4 = r2 * r2
    r6 = r4 * r2
    if d.shape == (5,):
        k4 = k5 = k6 = 0.0                     # den ≡ 1(M2 特例,数值逐位退化)
        k_slots = (0, 1, 4)
    elif d.shape == (8,):
        k4, k5, k6 = float(d[5]), float(d[6]), float(d[7])
        k_slots = (0, 1, 4, 5, 6, 7)
    else:
        raise AssertionError(
            f"畸变向量长度不合:期望 5(M2)或 8(M8),实际是 {d.shape[0]}。"
            f"布局 [k1,k2,p1,p2,k3(,k4,k5,k6)],请检查参数向量的排布。")
    num = 1.0 + d[0] * r2 + d[1] * r4 + d[4] * r6
    den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    if bool(np.any(np.abs(den) < _DEN_MIN)):
        raise AssertionError(
            f"rational 分母 den 取到近零值(min |den| = {float(np.abs(den).min()):.3e},"
            f"下限 {_DEN_MIN})——畸变映射在该处不可逆,继续算只会产出静默垃圾。"
            f"请检查畸变系数与点的工作域。")
    den2 = den * den
    ratio = num / den
    dnum_dr2 = d[0] + 2.0 * d[1] * r2 + 3.0 * d[4] * r4
    dden_dr2 = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4
    d_ratio_dr2 = (dnum_dr2 * den - num * dden_dr2) / den2
    cols_num = np.stack([r2 / den, r4 / den, r6 / den], axis=1)
    if d.shape == (8,):
        cols_den = np.stack([-num * r2 / den2, -num * r4 / den2, -num * r6 / den2], axis=1)
        d_ratio_dk = np.concatenate([cols_num, cols_den], axis=1)
    else:
        d_ratio_dk = cols_num
    return ratio, d_ratio_dr2, d_ratio_dk, k_slots


def _distort(x: np.ndarray, y: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """归一化无畸变坐标 (x,y) → 归一化畸变坐标 (xd,yd)(畸变核本体)。

    切向项加在径向缩放之外(先算径向缩放,再加切向偏移)——嵌套方式经
    参考工程实测对齐 cv2 4.14.0 的 projectPoints(M2/M8 双模型),parity
    测试锁死在 1e-9 像素内。本函数是畸变公式在全部代码里的唯一存在,
    严禁抄写;M8 即把径向缩放从 rad 换成 num/den,其余一字不改。
    """
    r2 = x * x + y * y
    ratio, _dratio_dr2, _dratio_dk, _slots = _radial_and_derivs(r2, dist)
    p1, p2 = dist[2], dist[3]
    xd = x * ratio + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * ratio + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def transform_to_camera(objp: np.ndarray,
                        rvec: np.ndarray,
                        tvec: np.ndarray,
                        want_jacobian: bool = False
                        ) -> tuple[np.ndarray, np.ndarray | None]:
    """棋盘平面点 → 相机坐标系点;want_jacobian 时一并给出对 6 外参的解析导数块。

    旋转必须走 cv2 的 Rodrigues(与 OpenCV 全兼容),雅可比用其第二返回值
    rot_jac(布局 rot_jac[k, 3i+j] = ∂R_ij/∂ω_k,对中心差分实测 ~1e-10,
    任意大小转角都成立)。严禁手写 ∂Xc/∂ω = −[Xc]×——那只是 ω=0 处的
    极限形式,0.1 rad 的小转角就已偏 70%(参考工程真实修复教训),本文件
    一律用 cv2 同源的现成雅可比。

    参数:
        objp: (N,3) float64 棋盘平面物点(Z=0 平面);
        rvec: (3,) float64 旋转向量(Rodrigues 语义:方向=转轴、模长=转角);
        tvec: (3,) float64 平移向量;
        want_jacobian: 是否需要外参导数块(求解器精化用,评分/回代传 False)。
    返回:
        (pts_cam (N,3) float64, dxc_dpose 或 None);
        dxc_dpose (N,3,6) float64 的 6 列依序是 ∂Xc/∂ω(3 列)与 ∂Xc/∂t(3 列)。
    """
    pts = _as_points3d(objp, "objp")
    r = np.asarray(rvec, dtype=np.float64).ravel()
    t = np.asarray(tvec, dtype=np.float64).ravel()
    if r.shape != (3,) or t.shape != (3,):
        raise AssertionError(
            f"rvec/tvec 形状不合:期望各 (3,),实际是 {r.shape} / {t.shape}。"
            f"旋转向量与平移向量必须各给 3 个数(Rodrigues 语义)。"
        )
    rot, rot_jac = cv2.Rodrigues(r)  # 旋转不自己推公式:cv2 现成,含雅可比
    pts_cam = pts @ rot.T + t
    if not want_jacobian:
        return pts_cam, None

    dxc_dpose = np.empty((pts.shape[0], 3, 6), dtype=np.float64)
    # ∂Xc_i/∂ω_k = Σ_j rot_jac[k,i,j] · Xb_j(平移不随 ω 动,逐点 einsum 一次算完)
    dxc_dpose[:, :, 0:3] = np.einsum("kij,nj->nik", rot_jac.reshape(3, 3, 3), pts)
    dxc_dpose[:, :, 3:6] = np.eye(3, dtype=np.float64)  # ∂Xc/∂t = I(平移逐分量直接搬)
    return pts_cam, dxc_dpose


def project(pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
    """投影:(N,3) 相机坐标系点 + (9,)/(12,) 参数 → (N,2) 像素,float64。

    Z 轴朝前:先断言 Z 全 > 0(针孔要除以 Z),再除 Z → 畸变 → 乘焦距加主点。
    参数向量按长度分派 M2(9)/M8(12);畸变核只此一份(_distort)。
    """
    pts = _as_points3d(pts_cam)
    p = _as_params(params)
    if not np.all(pts[:, 2] > 0.0):
        raise AssertionError(
            "project 收到 Z<=0 的点:针孔投影要除以 Z,Z 必须全为正"
            "(相机坐标系 Z 轴朝前)。请检查点的坐标系。"
        )
    z = pts[:, 2]
    x = pts[:, 0] / z
    y = pts[:, 1] / z
    xd, yd = _distort(x, y, p[4:])
    return np.stack([p[0] * xd + p[2], p[1] * yd + p[3]], axis=1)


def project_jacobian(pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
    """投影对内参的解析雅可比:(N,3)+(9,)/(12,) → (N,2,9)/(N,2,12),float64。

    只含内参列;对位姿的链式块(d像素/d相机点 × ∂Xc/∂外参)由
    project_point_jacobian 与 transform_to_camera 组装。解析式对
    scipy 数值差分基准逐列校验 < 1e-6(测试锁定,M8 含 k4,k5,k6 列)。
    """
    pts = _as_points3d(pts_cam)
    p = _as_params(params)
    fx, fy = p[0], p[1]
    z = pts[:, 2]
    x = pts[:, 0] / z
    y = pts[:, 1] / z
    r2 = x * x + y * y
    dist = p[4:]
    ratio, _dratio_dr2, dratio_dk, k_slots = _radial_and_derivs(r2, dist)
    p1, p2 = dist[2], dist[3]
    xd = x * ratio + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * ratio + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    n = pts.shape[0]
    jac = np.zeros((n, 2, p.shape[0]), dtype=np.float64)
    # 焦距与主点:线性项,直接是畸变坐标/常数 1
    jac[:, 0, 0] = xd
    jac[:, 1, 1] = yd
    jac[:, 0, 2] = 1.0
    jac[:, 1, 3] = 1.0
    # 径向 k 系数:d u/d k = fx·x·d(ratio)/dk(分子槽 r^(2j)/den;分母槽 −num·r^(2j)/den²)
    # k 槽位的 dist 下标 (0,1,4[,5,6,7]) → 参数向量下标 4+槽位(M2: 4,5,8;M8: 4,5,8,9,10,11)
    for j, col in enumerate(4 + s for s in k_slots):
        jac[:, 0, col] = fx * x * dratio_dk[:, j]
        jac[:, 1, col] = fy * y * dratio_dk[:, j]
    # 切向 p1,p2(对归一化坐标是多项式,对像素再乘焦距;与径向模型无关,两模型同式)
    jac[:, 0, 6] = fx * 2.0 * x * y
    jac[:, 1, 6] = fy * (r2 + 2.0 * y * y)
    jac[:, 0, 7] = fx * (r2 + 2.0 * x * x)
    jac[:, 1, 7] = fy * 2.0 * x * y
    return jac


def project_point_jacobian(pts_cam: np.ndarray, params: np.ndarray) -> np.ndarray:
    """投影对相机坐标系点的解析雅可比:(N,3)+(9,)/(12,) → (N,2,3),float64。

    链式 = 畸变核对归一化坐标的 2×2 导数(与 unproject 牛顿迭代里的那份
    2×2 公式逐字相同——同一份公式两处用,保证正逆解导数一致)乘上归一化
    坐标对相机点的导数(∂x/∂X=1/z、∂x/∂Z=−x/z 这组除法链)。
    """
    pts = _as_points3d(pts_cam)
    p = _as_params(params)
    fx, fy = p[0], p[1]
    z = pts[:, 2]
    if not np.all(z > 0.0):
        raise AssertionError(
            "project_point_jacobian 收到 Z<=0 的点:针孔投影要除以 Z,"
            "Z 必须全为正。请检查点的坐标系。"
        )
    x = pts[:, 0] / z
    y = pts[:, 1] / z
    r2 = x * x + y * y
    dist = p[4:]
    ratio, dratio_dr2, _dratio_dk, _slots = _radial_and_derivs(r2, dist)
    p1, p2 = dist[2], dist[3]

    # 2×2 畸变核导数(与 unproject 牛顿迭代的雅可比同式):a12 对称共用
    a11 = ratio + 2.0 * x * x * dratio_dr2 + 2.0 * p1 * y + 6.0 * p2 * x
    a12 = 2.0 * x * y * dratio_dr2 + 2.0 * p1 * x + 2.0 * p2 * y
    a22 = ratio + 2.0 * y * y * dratio_dr2 + 6.0 * p1 * y + 2.0 * p2 * x

    jac = np.empty((pts.shape[0], 2, 3), dtype=np.float64)
    inv_z = 1.0 / z
    jac[:, 0, 0] = fx * a11 * inv_z
    jac[:, 0, 1] = fx * a12 * inv_z
    jac[:, 0, 2] = -fx * (a11 * x + a12 * y) * inv_z
    jac[:, 1, 0] = fy * a12 * inv_z
    jac[:, 1, 1] = fy * a22 * inv_z
    jac[:, 1, 2] = -fy * (a12 * x + a22 * y) * inv_z
    return jac


def unproject(uv: np.ndarray, params: np.ndarray) -> np.ndarray:
    """反投影:像素 → 归一化射线方向 (N,3),真逆解(牛顿迭代解畸变正映射的逆)。

    做法:像素减主点除焦距得归一化畸变坐标 (xd,yd);解方程组
    "xd = x·ratio(r2)+切向、yd = y·ratio(r2)+切向"(畸变核正向公式)反求
    无畸变坐标 (x,y)——牛顿迭代:初值取 (xd,yd)(畸变小则几乎一步收敛),
    每步解 2×2 线性方程修正,迭代到修正量 < 1e-15(最多 50 步)。
    返回 (x,y,1)/‖(x,y,1)‖ 单位射线;project 对射线缩放不敏感,
    往返契约 project → unproject → project 误差 < 1e-9 像素(测试锁定,
    M2/M8 双模型同阈值)。它是 test 帧评分正确性的关键(反投影天然完成
    去畸变,防错误盆地);强 rational 畸变下 cv2 迭代逆误差可达 1.9px,
    绝不允许换 cv2 逆。
    """
    uv_arr = np.asarray(uv, dtype=np.float64)
    if uv_arr.ndim != 2 or uv_arr.shape[1] != 2:
        raise AssertionError(
            f"unproject 收到形状 {uv_arr.shape} 的像素坐标,期望 (N,2)。"
            f"请检查调用方传的是不是逐点 (u,v)。"
        )
    p = _as_params(params)
    fx, fy, cx, cy = p[:4]
    dist = p[4:]
    xd = (uv_arr[:, 0] - cx) / fx
    yd = (uv_arr[:, 1] - cy) / fy

    # 牛顿迭代解畸变正映射的逆:残差 = 当前猜测的畸变结果 − 观测畸变坐标
    x, y = xd.copy(), yd.copy()
    p1, p2 = dist[2], dist[3]
    for _ in range(50):
        r2 = x * x + y * y
        ratio, dratio_dr2, _dratio_dk, _slots = _radial_and_derivs(r2, dist)
        x_pred = x * ratio + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        y_pred = y * ratio + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        res_x = x_pred - xd
        res_y = y_pred - yd
        # 2×2 雅可比(畸变坐标对无畸变坐标),解析式:
        a11 = ratio + 2.0 * x * x * dratio_dr2 + 2.0 * p1 * y + 6.0 * p2 * x
        a12 = 2.0 * x * y * dratio_dr2 + 2.0 * p1 * x + 2.0 * p2 * y
        a22 = ratio + 2.0 * y * y * dratio_dr2 + 6.0 * p1 * y + 2.0 * p2 * x
        det = a11 * a22 - a12 * a12
        dx = (a22 * res_x - a12 * res_y) / det
        dy = (a11 * res_y - a12 * res_x) / det
        x = x - dx
        y = y - dy
        if max(float(np.abs(dx).max()), float(np.abs(dy).max())) < 1e-15:
            break

    rays = np.stack([x, y, np.ones_like(x)], axis=1)
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)
