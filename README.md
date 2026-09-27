# isaac-drone-sim

基于 Isaac Sim 6.1 + 本地 Isaac Lab 3.0 / PhysX 的 ARL-Robot-1 无人机运动控制工程。

已实现通用配置、质心运动控制、有界推力分配、原生电机接入、扰动/气动与电池扩展。螺旋上升保留接口，尚未实现轨迹。当前默认只使用 ARL-Robot-1。

## 目录

| 路径 | 内容 |
|---|---|
| `assets/Robots/NTNU/ARL-Robot-1/` | ARL-Robot-1 四旋翼（Isaac Lab 推进器模型用的机型） |
| `assets/Robots/Bitcraze/Crazyflie/` | 历史资产保留，当前配置与运行入口不使用 |
| `download_assets.sh` | 从 Isaac 云端资产库重新下载原版资产 |
| `isaac_drone/assets.py` | 资产路径：优先用本地 `assets/`，没有就回退到云端地址 |
| `isaac_drone/control/` | 位置 PID、几何姿态控制、有界控制分配 |
| `isaac_drone/aero/` | 风场、阵风、相对气流阻力；地面效应尚未实现 |
| `isaac_drone/disturbances/` | 外力/力矩、作用点、时间窗与插件组合 |
| `isaac_drone/power/` | 电池等效电路、已标定负载与推力包线接口 |
| `isaac_drone/backends/` | 真实 PhysX 质量/惯量/几何、原生电机和合力提交 |
| `isaac_drone/trajectories/` | hold 联调目标、起飞—三维 helix—终点悬停的解析轨迹 |
| `isaac_drone/runtime.py` | 控制与物理步调度、复位、模型耦合 |
| `scripts/script_editor/` | 粘贴到 Isaac Sim 界面 Script Editor 里运行的脚本 |
| `scripts/standalone/` | 独立运行的脚本（`env_isaacsim/bin/python xxx.py` 或 Isaac Lab） |
| `configs/` | 无人机、控制器、仿真参数配置（每次测试一版参数） |
| `tests/` | 测试 |

`assets/` 保留了资产库原有的目录结构，文件之间的相对引用不会断。
原版地址：`https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/`

资产各自带 NVIDIA 资产库的 LICENSE，本仓库保持私有。

## 配置与运行

主配置：[configs/arl_robot_1.yaml](configs/arl_robot_1.yaml)。架构、参数来源、参考系、扩展方法及验证边界见 [docs/architecture.md](docs/architecture.md)。

```bash
# 普通 Python：配置校验与无仿真依赖测试
python3 scripts/standalone/run_arl.py --validate
python3 -m pytest -q tests

# 已安装 Isaac Lab / Isaac Sim 的环境，从 IsaacLab/ 运行
uv run python ../scripts/standalone/run_arl.py --inspect --headless
uv run python ../scripts/standalone/run_arl.py --headless --duration 10
```

依赖是 NumPy、PyYAML；pytest 用于测试。可用 `python -m pip install -e '.[test]'` 安装。离线 USD 检查工具 `scripts/standalone/inspect_arl_asset.py` 另需 `usd-core`，不需启动仿真。

当前采用用户明确选择的**对角同向仿真旋向**，不宣称经过实机测量。实际旋翼位置从 USD/PhysX 读取；上游参考分配矩阵与资产几何存在差异，日志保留对比。质量、质心、完整惯量不在控制器中写死。

未知气动和电池参数保留为 null；相关模型默认关闭，开启必须补齐标定数据。当前开发环境已完成数值/契约测试和真实 USD 离线检查，**尚未在完整 Isaac Sim 环境中完成验飞与增益调参**。

## 三维 helix 螺旋上升

[完整任务配置](configs/arl_robot_1_helix.yaml) 从地面、零电机转速开始：2 秒起转，4 秒原地垂直升高 1 m，24 秒沿半径 1 m 的圆柱螺旋上升 2 圈并增高 3 m，最后保持终点直到 40 秒实验结束。这些是可改的任务参数，不是实机标定值。圆柱轴线自动偏置，使螺旋起点与垂直起飞终点重合；轨迹不是二维盘旋或半径不断扩大的平面 spiral。

```bash
# 普通 Python 先验证配置
python3 scripts/standalone/helix_ascent.py --validate
# 已安装 Isaac Lab / Isaac Sim 的环境，从 IsaacLab/ 运行
uv run python ../scripts/standalone/helix_ascent.py --headless
```

`spiral_ascent.py` 保留为同一 helix 任务的兼容入口。修改 `trajectory.spiral` 参数段即可配置半径、圈数、上升高度、阶段时长、偏航策略。九次时间多项式使参考在各阶段连接处达到 C4 连续，并在螺旋后半程逐步减速。数学推导见 [docs/spiral_math.md](docs/spiral_math.md)。

控制链是 **几何控制 → 每桨分配 → RPS 命令 → 电机响应 → 四个旋翼各自的推力/反扭矩 → PhysX**。初始化以后不写整机位姿或速度；总力矩只用于分配和记录。推进模型保留各电机实际采样的系数、非对称响应、限幅与积分设置；不需要 CFD，但也不代表已模拟完整空气流动。

终点判定看真实位置、速度、偏航和角速度，连续满足配置容差 2 秒才成功。成功后仍继续闭环悬停；超时、用户提前停止或结束时不满足悬停条件，日志不会标记成功，入口返回非零状态。

可重复运行普通 Python 的独立闭环回归：

```bash
python3 scripts/standalone/validate_helix_numerics.py --output runs/helix_cpu_reference.json
```

这个合成刚体测试保留非对角惯量、非零质心以及各电机不同的推力系数和响应时间，但仅有简化的测试接触约束，未使用 ARL/PhysX 物理引擎。当前 40 秒 / 8000 步回归通过，最大位置误差约 0.062 m，并完成持续悬停；它不能作为实机或 Isaac Sim 的精度保证。

## 基础数据记录

默认每个物理步记录一次（当前200 Hz），同时输出 `basic.csv`、`telemetry.jsonl` 和 `telemetry_schema.json`。包含实际质心位置/速度/加速度、四电机转速、不同阶段的力与力矩、目标位置和三轴/三维位置误差。列名包含单位和分量，可直接用于 Excel、MATLAB 或 Python。

加速度同时保留每步速度差分值与可用的 PhysX 求解器值；电机转速来自直接积分的 RPS 电机状态，不冒称编码器测量。力/力矩区分控制需求、电机输出、提交值和由运动推导的净值。

字段定义、时间对齐、误差符号及记录频率见 [docs/telemetry.md](docs/telemetry.md)。文件在实际运行后写入 `runs/<时间_编号>/`。

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
