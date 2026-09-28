# isaac-drone-sim

基于 Isaac Sim 6.1 + 本地 Isaac Lab 3.0 / PhysX 的 ARL-Robot-1 无人机运动控制工程。

已实现通用配置、质心运动控制、有界推力分配、原生电机接入、扰动/气动与电池扩展，以及地面起飞—三维 helix 上升—终点悬停任务。当前默认只使用 ARL-Robot-1。

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

# 服务器：激活 env_isaacsim 后在本仓库根目录运行。Isaac Lab 3.0 用 --visualizer none 代替 --headless
python scripts/standalone/run_arl.py --inspect --visualizer none
python scripts/standalone/run_arl.py --visualizer none --duration 10
```

WebRTC 观看：加 `--livestream 1`，并用 `PUBLIC_IP` 指定服务器公网 IP。脚本自己启动 Isaac Sim 并推流，不需要另开 Isaac Sim；同一时间只能有一个进程占用 49100 端口。加 `--wait-for-start` 后，场景加载完先停住，客户端连上看到画面后在终端按回车才开始。

有画面时默认按真实时间 1× 播放（`--playback-speed 2` 为两倍速，`0` 为不限速），并在视口中画出绿色虚线目标轨迹、红色实际质心轨迹和绿色小球表示的当前目标点（`--no-path-overlay` 关闭）。这两项只影响显示，不改变物理步进和日志。

依赖是 NumPy、PyYAML；pytest 用于测试。可用 `python -m pip install -e '.[test]'` 安装。离线 USD 检查工具 `scripts/standalone/inspect_arl_asset.py` 另需 `usd-core`，不需启动仿真。

当前采用用户明确选择的**对角同向仿真旋向**，不宣称经过实机测量。实际旋翼位置从 USD/PhysX 读取；上游参考分配矩阵与资产几何存在差异，日志保留对比。质量、质心、完整惯量不在控制器中写死。

未知气动和电池参数保留为 null；相关模型默认关闭，开启必须补齐标定数据。当前开发环境已完成数值/契约测试和真实 USD 离线检查，**尚未在完整 Isaac Sim 环境中完成验飞与增益调参**。

## 三维 helix 螺旋上升

[完整任务配置](configs/arl_robot_1_helix.yaml) 从地面、零电机转速开始：2 秒起转，原地垂直升高 1 m，再沿半径 1 m 的圆柱螺旋上升 5 圈并增高 5 m，最后保持终点直到 20 秒实验结束。这些是可改的任务参数，不是实机标定值。圆柱轴线自动偏置，使螺旋起点与垂直起飞终点重合；轨迹不是二维盘旋或半径不断扩大的平面 spiral。

**最快飞行（最短时间）**：起飞和螺旋的时长填 `null` 时，启动时按实际质量、惯量、分配矩阵和每桨推力上下限规划“加速—匀速巡航—减速”，在下列约束内使该段用时最短：名义每桨推力落在推力范围的 `1-thrust_utilization` ~ `thrust_utilization` 之间（默认 15%~85%，即转速最高约 92%，两端余量留给反馈）；为克服电机一阶滞后所需的超前指令 `f + τ·df/dt` 也在该范围内；倾角、偏航速率和加速度不超过控制配置。用 uc1 上读到的 ARL 质量/惯量和参考分配矩阵离线规划：起飞约 1.2 s、螺旋约 7.5 s，巡航约 5.0 m/s、倾角约 68°，运动约 10.7 s 结束（实际数值以运行时打印为准）。机头用 `tangent`（沿切线）：高速绕圈时四桨推力均衡，才能用满推力；`fixed` 机头要持续的姿态力矩，四桨推力拉开，只能到约 4.1 m/s。启动时终端打印 `Helix plan: ...` 给出各段时长、峰值速度和峰值推力/转速占比；运行元数据的 `trajectory_feasibility` 里有完整规划。时长填数字则仍使用原来的固定时长九次多项式。

```bash
# 普通 Python 先验证配置
python3 scripts/standalone/helix_ascent.py --validate
# 服务器无画面运行
python scripts/standalone/helix_ascent.py --visualizer none
# 服务器串流观看：先连 WebRTC 客户端，看到画面后在终端按回车开始
PUBLIC_IP=<服务器公网IP> python scripts/standalone/helix_ascent.py --livestream 1 --wait-for-start
```

`spiral_ascent.py` 保留为同一 helix 任务的兼容入口。修改 `trajectory.spiral` 参数段即可配置半径、圈数、上升高度、阶段时长（或 `null` 最短时间）、推力利用率、偏航策略。两种时间律在各阶段连接处都达到 C4 连续。同样的推力下线速度约与 `sqrt(半径)` 成正比，想更快可加大半径。数学推导见 [docs/spiral_math.md](docs/spiral_math.md)。

控制链是 **几何控制 → 每桨分配 → RPS 命令 → 电机响应 → 四个旋翼各自的推力/反扭矩 → PhysX**。初始化以后不写整机位姿或速度；总力矩只用于分配和记录。推进模型保留各电机实际采样的系数、非对称响应、限幅与积分设置；不需要 CFD，但也不代表已模拟完整空气流动。

终点判定看真实位置、速度、偏航和角速度，连续满足配置容差 2 秒才成功。成功后仍继续闭环悬停；超时、用户提前停止或结束时不满足悬停条件，日志不会标记成功，入口返回非零状态。

可重复运行普通 Python 的独立闭环回归：

```bash
python3 scripts/standalone/validate_helix_numerics.py --output runs/helix_cpu_reference.json
```

这个合成刚体测试保留非对角惯量、非零质心以及各电机不同的推力系数和响应时间，但仅有简化的测试接触约束，未使用 ARL/PhysX 物理引擎。当前最快 helix 配置 20 秒 / 4000 步回归通过：最高速度约 4.7 m/s，最大位置误差约 0.16 m（RMS 0.04 m），13.3 s 完成悬停判定；它不能作为实机或 Isaac Sim 的精度保证。误差主要来自电机响应滞后（上升时间常数 50–80 ms），因此该配置把位置增益提高到 kp=[12,12,8]、kd=[7,7,5]。

## 基础数据记录

默认每个物理步记录一次（当前200 Hz），同时输出 `basic.csv`、`telemetry.jsonl` 和 `telemetry_schema.json`。位置、速度、加速度、姿态、角速度、电机转速、单桨推力、力和力矩都同时记录**理想值、实际值和误差（理想 − 实际）**；力/力矩还保留分配值、扰动、提交值和由运动推导的净值。列名包含单位和分量，可直接用于 Excel、MATLAB 或 Python。

飞行结束后自动在 `runs/<时间_编号>/plots/` 生成英文 PNG 图（理想 vs 实际，末行为误差）；也可用 `python scripts/standalone/plot_run.py [运行目录]` 对任意一次运行重新生成，只需 NumPy 和 matplotlib（`pip install -e '.[plot]'`）。

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

## 服务器试飞清单：开机 → WebRTC → 回车起飞

服务器当前公网 IP 为 `117.50.205.33`；实例重启后若 IP 变化，下面命令和本地 `~/.ssh/config` 中 `uc1` 的 HostName 都要同步修改。

**1. 开机**：在云服务商控制台启动实例。

**2. 本地终端连服务器**

```bash
tmux attach -t isaac || tmux new -s isaac    # 进入共享终端（有就接入，没有就新建）
ssh uc1                                      # 已配置密钥登录
```

**3. 服务器准备环境**

```bash
nvidia-smi                                   # 确认 GPU 空闲，没有残留的 Isaac Sim
cd ~/isaac-drone-sim
git pull
source ~/isaacsim-6.1/env_isaacsim/bin/activate
python scripts/standalone/helix_ascent.py --validate   # 可选：几秒内检查配置和资产
```

**4. 启动 Isaac Sim，加载场景和无人机**

```bash
PYTHONUNBUFFERED=1 PUBLIC_IP=117.50.205.33 python scripts/standalone/helix_ascent.py --livestream 1 --wait-for-start 2>&1 | tee runs/helix_$(date +%m%d-%H%M).log
```

等终端出现 `Scene ready: connect the stream client, then press Enter here to start.`（通常约 1 分钟；开机后首次运行需编译着色器，可能 2–3 分钟）。此时无人机停在地上，物理不推进。

**5. 连 WebRTC**：客户端填 Server `117.50.205.33`、Signal `49100`、Stream `47998`，点 Connect，看到操作页面和无人机。

**6. 开始运行**：点一下运行脚本的终端，确保输入焦点在它上面，按**回车**。任务为 20 s 仿真时间，默认按真实时间播放；物理和记录本身慢于实时时会如实变慢（uc1 串流实测约 0.3×，约 70 s，见输出里的 `playback`）。结束后打印 `{"success": ...}`，程序退出，串流随之断开。再飞一次需重新执行第 4 步。

**7. 看结果、关机**

```bash
python scripts/standalone/summarize_run.py   # 最新一次运行的分阶段误差、倾角、电机转速、饱和比例
ls runs/<运行目录名>/plots/                    # 飞完自动生成的英文图；重画：python scripts/standalone/plot_run.py
```

拷回本地（在本地仓库根目录运行，增量同步全部运行记录到本地 `runs/`）：`rsync -az --info=progress2 uc1:isaac-drone-sim/runs/ runs/`。不用时在控制台关机。

**注意**

- 不要同时运行独立的 Isaac Sim 串流程序（`isaacsim.exp.full.streaming`），两者会抢 49100 端口。
- 停止程序用 `Ctrl+C`；`Ctrl+Z` 只是挂起，进程仍占着显存和端口。
- 终端粘贴用 `Ctrl+Shift+V`。
- WebRTC 同时只能连一个客户端；连不上时先关闭所有旧客户端窗口再连。

## 不在仓库里的东西

- `IsaacLab/`：官方仓库，要改请 fork `isaac-sim/IsaacLab`
- `isaacsim-6.1/`、日志、`isaacsim_spiral_recording/`：服务器运行产物
- Isaac Sim 的 Python 环境 `env_isaacsim`（29GB）：可重新安装
