# isaac-drone-sim

基于 Isaac Sim 6.1 + Isaac Lab 的无人机仿真：模型编辑、运动控制、空气动力学扩展。

## 目录

| 路径 | 内容 |
|---|---|
| `assets/Robots/NTNU/ARL-Robot-1/` | ARL-Robot-1 四旋翼（Isaac Lab 推进器模型用的机型） |
| `assets/Robots/Bitcraze/Crazyflie/` | Crazyflie cf2x 小型四旋翼 |
| `download_assets.sh` | 从 Isaac 云端资产库重新下载原版资产 |
| `isaac_drone/assets.py` | 资产路径：优先用本地 `assets/`，没有就回退到云端地址 |
| `isaac_drone/control/` | 控制器：PID、几何控制器等 |
| `isaac_drone/aero/` | 空气动力学补充模型：阻力、地面效应、风 |
| `scripts/script_editor/` | 粘贴到 Isaac Sim 界面 Script Editor 里运行的脚本 |
| `scripts/standalone/` | 独立运行的脚本（`env_isaacsim/bin/python xxx.py` 或 Isaac Lab） |
| `configs/` | 无人机、控制器、仿真参数配置（每次测试一版参数） |
| `tests/` | 测试 |

`assets/` 保留了资产库原有的目录结构，文件之间的相对引用不会断。
原版地址：`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/`

资产各自带 NVIDIA 资产库的 LICENSE，本仓库保持私有。

## 工作流

```
本地编辑 → git commit → git push → 服务器 git pull → Isaac Sim 里测试
```

服务器（`uc1`）首次拉取：

```bash
sudo apt install git-lfs && git lfs install
git clone https://github.com/Desmond-Yuka/isaac-drone-sim.git ~/isaac-drone-sim
```

在 Isaac Sim 里用本仓库的模型：把场景里的引用、或 Isaac Lab 配置里的 `usd_path`，
从云端地址改成 `~/isaac-drone-sim/assets/...` 下的本地路径。代码里直接用：

```python
import sys; sys.path.insert(0, "/home/ubuntu/isaac-drone-sim")
from isaac_drone.assets import ARL_ROBOT_1_USD, CRAZYFLIE_USD
```

## 不在仓库里的东西

- `IsaacLab/`：官方仓库，要改请 fork `isaac-sim/IsaacLab`
- `isaacsim-6.1/`、日志、`isaacsim_spiral_recording/`：服务器运行产物
- Isaac Sim 的 Python 环境 `env_isaacsim`（29GB）：可重新安装
