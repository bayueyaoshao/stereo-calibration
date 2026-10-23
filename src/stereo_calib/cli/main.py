"""stereo-calib 命令行入口:三个子命令对应三个独立模块(D-09/D-10)。

体例移植自参考工程 calib.cli.main(只读引用,绝不修改):typer app +
callback 统一准备、命令入口把坏输入收口为中文一句 + 退出码 1(不吐堆栈)。

本工程差异(参考用固定常量 CONFIG_PATH,Deviations #10):配置路径是
callback 上的可选 --config 选项(默认 config.yaml),满足 D-10 字面
「命令行只传配置文件路径」——除该选项外零参数,零算法参数不变。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import typer

from stereo_calib.io.config import load_config

app = typer.Typer(
    name="stereo-calib",
    help="双目相机标定生产工具:preprocess 数据预处理 / intrinsic 单相机内参标定 / "
         "stereo 双目外参标定。三个模块相互独立,可单独运行调试。",
    no_args_is_help=True,
)

# 配置文件路径(D-10:唯一命令行选项,默认当前工作目录下的 config.yaml;
# callback 的 --config 选项会更新它,命令函数统一从这里读)
_config_path = Path("config.yaml")


@app.callback()
def _setup_runtime(
    # typer 的声明式选项必须写在参数默认值里(其官方文档对 B008 的既定豁免);
    # Phase 1 遗留告警,02-04 触达本文件时顺手消音(不改任何行为)
    config: Path = typer.Option(  # noqa: B008
        Path("config.yaml"),
        "--config", "-c",
        help="配置文件路径,默认 config.yaml(唯一命令行选项,算法参数一律在配置文件里改)。",
    ),
) -> None:
    """每个命令执行前的统一准备:记下配置路径并按配置限制 OpenCV 线程数。

    OpenCV 默认会占满所有核,批量处理(几百张图)时会把机器拖垮;
    线程数从 config.yaml 的 cv2_threads 读(0 = 关闭 OpenCV 自带多线程)。
    """
    global _config_path
    _config_path = config
    try:
        cfg = load_config(_config_path)
        cv2.setNumThreads(cfg.cv2_threads)
    except (ValueError, OSError):
        # --help / 配置还没就位时不拦着用户:真正需要配置的命令自己会再校验
        pass


def _not_implemented(name: str) -> None:
    """未实现模块的占位输出。"""
    typer.echo(f"命令「{name}」由后续阶段实现。")


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


@app.command(help="模块2 单相机内参标定:消费 Phase 1 角点 npz → 两路相机分别 M2 标定"
                  "(calibrateCameraExtended 初值,精化开关配置控制)→ train 离群自动剔除留痕"
                  "→ test 集量化验证 → intrinsics.json + 报告落盘。")
def intrinsic() -> None:
    """模块2 命令入口:坏输入收口为中文一句 + 退出码 1,不吐堆栈。"""
    try:
        # 局部导入:让 --help / 占位命令不依赖 intrinsic 模块,加快启动
        from stereo_calib.intrinsic.pipeline import run_intrinsic

        run_intrinsic(load_config(_config_path))
    except (ValueError, OSError) as e:
        typer.echo(f"内参标定失败:{e}")
        typer.echo("  请检查 config.yaml 配置与 Phase 1 产物(output/ 下的角点 npz),"
                   "修正后重跑 stereo-calib intrinsic。")
        raise typer.Exit(code=1) from e


@app.command(help="模块3 双目外参标定(后续阶段实现)")
def stereo() -> None:
    """模块3 占位。"""
    _not_implemented("stereo")


def main() -> None:
    """[project.scripts] 的入口函数。"""
    app()


if __name__ == "__main__":
    main()
