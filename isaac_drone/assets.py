"""无人机资产路径：优先用本仓库 assets/ 下的本地副本，没有时回退到 Isaac 云端资产库。

在 Isaac Sim 的 Script Editor 或 Isaac Lab 配置里用：
    from isaac_drone.assets import ARL_ROBOT_1_USD
    usd_path = ARL_ROBOT_1_USD
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
CLOUD_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac"


def asset_path(rel: str) -> str:
    """返回资产的 USD 路径，rel 是相对资产库根目录的路径，例如 "Robots/NTNU/ARL-Robot-1/arl_robot_1.usd"。"""
    local = ASSETS_DIR / rel
    if local.is_file():
        return str(local)
    return f"{CLOUD_ROOT}/{rel}"


ARL_ROBOT_1_USD = asset_path("Robots/NTNU/ARL-Robot-1/arl_robot_1.usd")
CRAZYFLIE_USD = asset_path("Robots/Bitcraze/Crazyflie/cf2x.usd")
