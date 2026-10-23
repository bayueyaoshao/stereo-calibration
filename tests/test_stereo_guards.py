"""双目模块全树源码防线:内建例程禁用 / 单一数学源 / 三角化单文件 /
雅可比组合 / 零读图。

体例照 tests/test_intrinsic_models.py L172-183(源文计数防线 + 自伤规避:
被锁调用串原文不写进本文件的 docstring 与断言消息——内建例程串用字符串
拼接规避;计数范围只圈 src/,tests/ 自身不在扫描列)。

这五条防线对应研究文档的五条红线:双目外参必须出自自研求解链
(03-02 的 solve_stereo),绝不许内建双目标定例程回潮;投影/畸变公式
只准经 intrinsic.models;SVD 三角化只准住 evaluate.py;精化雅可比走
解析路线(绝不传稀疏性组合);双目链路纯角点级,一行读图代码都没有。
"""

import pathlib

# src/ 根(从仓库根出发;pytest rootdir = 仓库根)
_SRC = pathlib.Path("src/stereo_calib")
_STEREO = _SRC / "stereo"


def _all_src_source() -> str:
    """src/ 全部 .py 源文聚合成一份文本(计数用)。"""
    return "\n".join(
        p.read_text(encoding="utf-8") for p in sorted(_SRC.rglob("*.py")))


def test_no_builtin_stereo_routine_anywhere_in_src():
    """内建双目标定例程在全 src/ 零出现(拼接规避自伤,体例照 models 防线):
    外参只准出自 03-02 自研求解链(pnp 中位初值 + 解析雅可比精化),
    内建例程一旦回潮即静默绕过全部自研防线(消歧/剔除/留痕),当场抓住。"""
    token = "cv2.stereo" + "Calibrate"
    source = _all_src_source()
    n = source.count(token)
    assert n == 0, (
        f"src/ 出现 {n} 处内建双目例程调用:外参必须出自自研求解链,"
        f"内建例程会静默绕过消歧/剔除/留痕全部防线")


def test_single_math_source_no_private_distort_in_stereo_subtree():
    """单一数学源:stereo/ 子树零私写畸变公式符号——投影/去畸变只准经
    intrinsic.models(含 undist_pix 的精确逆路线),子树内出现该私有
    符号即第二份数学源回潮,当场抓住。"""
    for path in sorted(_STEREO.rglob("*.py")):
        n = path.read_text(encoding="utf-8").count("_distort")
        assert n == 0, (
            f"{path.name} 出现 {n} 处私写畸变核符号:投影/畸变公式只准经 "
            f"intrinsic.models(单一数学源)")


def test_svd_triangulation_lives_only_in_evaluate():
    """三角化单文件:SVD 零空间解法只准住 evaluate.py,stereo/ 子树内
    其他任何文件出现即第二份三角化住址(两处实现漂移的温床),当场抓住。"""
    svd_token = "np.linalg." + "svd"
    for path in sorted(_STEREO.rglob("*.py")):
        n = path.read_text(encoding="utf-8").count(svd_token)
        if path.name == "evaluate.py":
            assert n >= 1, "evaluate.py 丢失 SVD 三角化(三角化的唯一住址)"
        else:
            assert n == 0, (
                f"{path.name} 出现 SVD 零空间解法:三角化只准住 evaluate.py")


def test_solve_refinement_uses_analytic_jac_no_sparsity():
    """雅可比组合(决策 1 / Pitfall 14):solve.py 精化走解析雅可比 +
    稀疏子问题解法器,绝不传稀疏性组合参数(黑盒估计路线);两个关键字
    在位 + 被禁组合零出现,双向锁死。"""
    source = (_STEREO / "solve.py").read_text(encoding="utf-8")
    assert source.count("jac_" + "sparsity") == 0, (
        "solve.py 出现稀疏性组合参数:精化雅可比必须解析路线(决策 1),"
        "黑盒稀疏估计不得回潮")
    assert "tr_solver" in source and "x_scale" in source, (
        "solve.py 精化缺少稀疏子问题解法器与列缩放关键字(解析雅可比"
        "路线的配套在位断言)")


def test_stereo_subtree_never_reads_images():
    """零读图:stereo/ 子树一行读图调用都没有——上游 preprocess 落盘
    角点 npz,双目链路纯角点级(paths 只作交叉核对),读图回潮即绕过
    Phase 1 检测/筛选产物,当场抓住。"""
    for path in sorted(_STEREO.rglob("*.py")):
        n = path.read_text(encoding="utf-8").count("imread")
        assert n == 0, (
            f"{path.name} 出现 {n} 处读图调用:双目链路零读图,"
            f"角点一律来自 preprocess 产物 npz")


def test_joint_refinement_uses_analytic_jac_no_sparsity():
    """joint.py 雅可比组合(WR-02,jac 为 callable 时 scipy 静默忽略稀疏性
    结构参数):解析路线 + 稀疏子问题解法器在位,被禁组合零出现;内参列必须
    经 models.project_jacobian(单一数学源)。体例照 solve.py 既有用例。"""
    source = (_STEREO / "joint.py").read_text(encoding="utf-8")
    assert source.count("jac_" + "sparsity") == 0, (
        "joint.py 出现稀疏性组合参数:联合精化雅可比必须解析路线(WR-02),"
        "黑盒稀疏估计不得回潮")
    assert "tr_solver" in source and "x_scale" in source, (
        "joint.py 缺少稀疏子问题解法器与列缩放关键字(解析雅可比路线的配套在位断言)")
    assert "project_jacobian" in source, (
        "joint.py 内参列必须经 models.project_jacobian(单一数学源,"
        "禁第二份投影导数)")
