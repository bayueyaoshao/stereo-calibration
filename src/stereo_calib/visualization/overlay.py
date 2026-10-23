"""角点标注图与失败帧原图副本(D-07/D-08/D-02):人工复核的证据本体。

为什么要有它:"角点是否可信"最终要靠人眼核查——汇总统计只是索引,这里
的全量标注图才是逐张核查的证据本体(D-07:全部成功图像一张不落、绝不
抽样,复核目标是把"成功检出但角点明显偏移"的帧挑出来,抽样会漏)。

标注内容(D-08):cv2.drawChessboardCorners 一行出标准叠加(交替色圆点
+ 顺序连线,自带角点序信息——Don't Hand-Roll,绝不自绘圆圈连线),外加
图上烧录图像路径与检测路线 [sb/classic] 文字,翻图时无需对照清单。

图内文字只能 ASCII(Pitfall 9:cv2.putText 不支持 CJK 字形,中文画出来是
问号);图像路径与 sb/classic 恰为 ASCII。字号/线宽按画面尺寸比例推导,
不写死像素(换分辨率不失效;体例移植自参考工程 calib.visualization.overlay
的 MISS 分支,只读引用)。

真实性承诺(INFRA-03):原图只读进内存(imread),标注画在内存副本上,
源文件自始至终不被改写;失败帧"原图副本"用 shutil.copy2 逐字节复制,
fail 标记只体现在输出目录的文件名(fail_ 前缀),绝不碰 data/。
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np


def render_marked(image_path: Path | str,
                  uv: np.ndarray,
                  cols: int, rows: int,
                  route: str,
                  out_path: Path | str,
                  jpeg_quality: int = 90) -> None:
    """把一张检出帧画上角点标注 + 路径/路线烧录文字后另存(D-07/D-08)。

    参数:
        image_path:   原图路径(只读进内存,绝不修改)
        uv:           规范化序角点 (cols*rows, 2);渲染时转 float32 供 OpenCV 用
        cols / rows:  内角点列数 / 行数(11×8 = 88)
        route:        检测路线("sb" / "classic"),与图上路径一起烧录(ASCII)
        out_path:     标注图输出路径(JPEG,写到 output_root 派生目录)
        jpeg_quality: JPEG 质量 1-100(来自 config 的 preprocess.visualization.jpeg_quality)

    读图或写盘失败抛中文 ValueError(带路径,把报错位置钉在病因上);
    正常返回 None(调用方按"全量渲染"计数,不需要返回值)。
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            f"照片读取失败:{image_path},无法生成角点标注图,请人工检查这张图。")

    # OpenCV 的标准角点叠加:格子交替着色 + 角点顺序连线,一行代码出图
    corners = np.asarray(uv, dtype=np.float32).reshape(-1, 1, 2)
    cv2.drawChessboardCorners(img, (int(cols), int(rows)), corners, True)

    # 烧录文字(D-08):内容 = 图像路径 + [路线],仅 ASCII(Pitfall 9);
    # 字号/线宽按画面尺寸比例推导,不写死像素(换分辨率不失效)
    height, width = img.shape[:2]
    thickness = max(4, min(height, width) // 80)
    font_scale = max(1.0, min(height, width) / 480)
    cv2.putText(img, f"{image_path} [{route}]",
                org=(thickness * 2, thickness * 2 + int(font_scale * 22)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                color=(0, 0, 255), thickness=max(2, thickness // 2),
                lineType=cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), img,
                       [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]):
        raise ValueError(f"角点标注图写盘失败:{out_path}")


def copy_fail_original(src_path: Path | str, image_id: str,
                       out_dir: Path | str) -> Path:
    """把失败帧原图逐字节拷贝到输出目录,返回副本路径(D-02 假失败复核)。

    shutil.copy2 连元数据一起复制(逐字节一致,测试锁定);源文件只读,
    绝不改写。目标名 fail_{image_id}.jpg——image_id 已在扫描层过
    _SAFE_IMAGE_ID 白名单正则(01-01 锁定),不会把副本写出输出目录。
    fail 标记只体现在这个文件名前缀上,data/ 里的源文件一个字节不动。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"fail_{image_id}.jpg"
    shutil.copy2(src_path, dst)
    return dst
