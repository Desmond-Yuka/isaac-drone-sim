# isaac-drone-sim

基于 Isaac Sim 6.1 + 本地 Isaac Lab 3.0 / PhysX 的 ARL-Robot-1 无人机运动控制工程。

控制链：**轨迹 → 控制器 → 有界推力分配 → RPS 电机指令 → 电机动态 → 四个旋翼各自的推力/反扭矩 → 物理引擎**。
初始化以后不写整机位姿或速度。轨迹、控制器、后端都是可替换的插件，通过 YAML 的 `kind:` 选择：

| 可替换部分 | 现有实现 |
|---|---|
| 后端 `--backend` | `isaaclab`（Isaac Sim / PhysX，服务器）· `synthetic`（CPU 合成刚体，秒级，本地和 CI 可跑） |
| 控制器 `controller.kind` | `geometric`（Lee SO(3) + jerk/snap 前馈）· `cascaded_pid`（位置→速度→姿态→角速度） |
| 轨迹 `trajectory.kind` | `hold` · `helix`（原地起飞 + 三维圆柱螺旋上升 + 终点悬停，可最短时间规划） |

两个后端共用同一套控制循环、日志、指标和画图。合成后端**不是** Isaac Sim，也不是 ARL 模型：它用刻意不对称的合成质量/惯量/电机参数，适合快速调参、回归和对比，结论需要回到 Isaac Sim 验证。

## 快速开始

```bash
python -m pip install -e '.[dev]'           # NumPy、PyYAML、pytest、ruff、matplotlib
python -m pytest -q                          # 约 1 分钟；-m "not slow" 跳过长闭环回归
python -m isaac_drone validate --config configs/helix.yaml
python -m isaac_drone run --backend synthetic --config configs/helix.yaml   # 约 12 s
python -m isaac_drone summarize              # 最新一次运行的分阶段误差、倾角、转速、饱和比例
```

服务器（激活 `env_isaacsim` 后，在仓库根目录）：

```bash
python -m isaac_drone inspect --visualizer none                     # 读取 PhysX 实际质量/几何后退出
python -m isaac_drone run --backend isaaclab --config configs/helix.yaml --visualizer none
PUBLIC_IP=<服务器公网IP> python -m isaac_drone run --backend isaaclab --config configs/helix.yaml \
    --livestream 1 --wait-for-start                                  # WebRTC 串流，按回车开始
```

开机、串流、拷回结果的完整步骤见 [docs/server_runbook.md](docs/server_runbook.md)。

## 命令

`python -m isaac_drone <命令>`，`-h` 查看全部参数：

| 命令 | 作用 | 需要 Isaac Sim |
|---|---|---|
| `validate` | 校验配置与本地资产，打印关键参数 | 否 |
| `run --backend synthetic\|isaaclab` | 飞一次任务，写入运行目录 | 仅 isaaclab |
| `inspect` | 加载 PhysX 实际属性并打印分配矩阵/质量/惯量 | 是 |
| `audit-usd` | 离线读取 USD 原始数据（需 `usd-core`） | 否 |
| `sweep` | 合成后端网格扫参，多进程并行，按指标排序 | 否 |
| `compare` | 多次运行的指标表和叠加图 | 否 |
| `summarize` / `plot` | 单次运行的分阶段表 / 重画 PNG | 否 |

所有与配置有关的命令都接受 `--config`、`--set 键路径=值`（可重复）和 `--duration`。例如：

```bash
python -m isaac_drone run --backend synthetic --config configs/helix.yaml \
    --set controller.position_kp=[10,10,6] --set trajectory.turns=3
```

## 目录

| 路径 | 内容 |
|---|---|
| `configs/` | 实验配置：`arl_robot_1.yaml` 为基础，其它文件用 `extends` 继承并只写差异，见 [docs/configuration.md](docs/configuration.md) |
| `isaac_drone/core/` | 状态/力/质量等数据契约、SO(3) 数学、数值校验、插件参数解析与注册表 |
| `isaac_drone/config/` | YAML 加载（`extends`、`--set`）与 schema 校验 |
| `isaac_drone/trajectories/` | 分段轨迹框架、hold、helix、时间律、最短时间规划、可行性筛查、完成判据 |
| `isaac_drone/control/` | 控制器接口与注册表、几何控制、级联 PID、有界推力分配 |
| `isaac_drone/effects/`、`power/` | 风/气动/外力扰动；电池等效电路（默认关闭，需标定） |
| `isaac_drone/runtime/` | 物理步控制循环 `MotionControlLoop`、与后端无关的运行器和钩子（叠加显示、录像、节拍、指标、画图） |
| `isaac_drone/sim/` | 后端接口；`synthetic.py` CPU 合成对象；`isaaclab/` 场景、PhysX 后端、相机、应用入口；原生电机积分的复用 |
| `isaac_drone/telemetry/` | 每步基础数据、JSONL/CSV 记录器与字段 schema |
| `isaac_drone/analysis/` | 指标、分阶段汇总、对比、扫参、画图 |
| `isaac_drone/viz/` | 轨迹叠加几何、墙钟节拍、机位取景、视频编码 |
| `assets/Robots/` | ARL-Robot-1（使用中）与 Crazyflie（保留）USD 资产，保持原资产库目录结构 |
| `scripts/download_assets.sh` | 从 Isaac 云端资产库重新下载原版资产 |
| `tests/` | 单元、契约和闭环回归测试（`tests/data/golden_synthetic.npz` 锁定重构前的飞行轨迹） |
| `docs/` | [配置与扩展](docs/configuration.md)、[架构与物理约定](docs/architecture.md)、[遥测字段](docs/telemetry.md)、[helix 数学](docs/spiral_math.md)、[服务器操作](docs/server_runbook.md) |

## 典型实验流程

1. **写配置**：复制最接近的 `configs/*.yaml`，用 `extends` 继承，只改要测试的部分（轨迹参数、控制器类型/增益、限制等）。
2. **CPU 上迭代**：`run --backend synthetic` 检查能否飞通；`sweep` 扫增益或轨迹参数；`compare` 对比控制器或参数组。
3. **Isaac Sim 验证**：把选定配置在服务器上 `run --backend isaaclab`；拷回运行目录后同样可以 `summarize` / `compare`。
4. **新增轨迹或控制器**：一个参数 dataclass + 一个注册的工厂函数即可，见 [docs/configuration.md 第 6 节](docs/configuration.md#6-新增控制器或轨迹)。控制器可以直接输出每桨推力（`output = "rotor_thrust"`），为接入学习策略预留。

对照示例（同一中速 helix，几何控制 vs 级联 PID）：

```bash
python -m isaac_drone run --backend synthetic --config configs/helix_moderate.yaml
python -m isaac_drone run --backend synthetic --config configs/helix_cascaded_pid.yaml
python -m isaac_drone compare runs/<几何控制运行> runs/<PID运行> --labels geometric pid --phase helix
```

CPU 合成对象上 helix 段最大位置误差：几何控制约 0.03 m，级联 PID 约 0.21 m。

## 运行目录

每次运行写入 `runs/<UTC时间_编号>/`，两个后端格式相同：

| 文件 | 内容 |
|---|---|
| `config.json` | 实际使用的完整配置（已展开 `extends`、`--set`） |
| `metadata.json` | 后端描述、资产层哈希、任务阶段时间表、可行性/最短时间规划报告、git commit 与命令行 |
| `telemetry.jsonl` / `basic.csv` / `telemetry_schema.json` | 每物理步的完整记录、便于 Excel/MATLAB 的基础数据表、字段单位与含义 |
| `metrics.json` | 整体和分阶段指标（位置/速度/姿态误差、倾角、速度、电机转速范围、饱和比例、推力） |
| `plots/*.png` | 理想 vs 实际及误差曲线（需 matplotlib） |
| `*.mp4`、`video_frames.jsonl` | 仅在 `--record-video` 时生成 |

位置、速度、加速度、姿态、角速度、电机转速、单桨推力、力和力矩都同时记录**理想值、实际值和误差（理想 − 实际）**。字段定义、时间对齐和误差符号见 [docs/telemetry.md](docs/telemetry.md)。

## 三维 helix 任务

[configs/helix.yaml](configs/helix.yaml)：从地面、零电机转速开始，2 s 起转，原地垂直升高 1 m，再沿半径 1 m 的圆柱螺旋上升 5 圈并增高 5 m，最后保持终点直到实验结束。圆柱轴线自动偏置，使螺旋起点与起飞终点重合。

起飞和螺旋时长填 `null` 时按实际质量、惯量、分配矩阵、每桨推力上下限和电机滞后规划最短时间：名义每桨推力保持在推力范围中间 `thrust_utilization`（默认 85%）内，两端余量留给反馈，并满足 `limits` 的倾角、偏航速率和加速度。当前配置规划结果约为起飞 1.3 s、螺旋 7.8 s、峰值 4.7 m/s、名义倾角约 66°（启动时打印 `Trajectory plan: ...`，实际数值以运行时为准）。机头用 `tangent`（沿切线）时高速绕圈四桨推力均衡，才能用满推力。时长填数字则使用固定时长的九次多项式。

终点判定看真实位置、速度、偏航和角速度，连续满足容差 2 s 才成功；成功后仍继续闭环悬停。超时、用户提前停止或结束时不满足条件，运行都不会标记成功，命令返回非零状态（2）。

当前采用用户明确选择的**对角同向仿真旋向**，不宣称经过实机测量。实际旋翼位置、质量、质心和完整惯量都从 USD/PhysX 读取，不在控制器里写死。未知的气动和电池参数保留为 null，相关模型默认关闭。**尚未在完整 Isaac Sim 环境中完成增益调参验飞**；CPU 合成对象上的结果不能作为实机或 Isaac Sim 的精度保证。

## 可选多机位视频录制（仅 isaaclab 后端）

默认不录制。加 `--record-video` 输出每个机位的 MP4 和同步拼屏 `combined.mp4`，可配合 WebRTC，也可在 `--visualizer none` 下离屏录制。

```bash
python -m pip install -e '.[video]'   # 在运行 Isaac Sim 的 Python 环境中安装一次
python -m isaac_drone run --backend isaaclab --config configs/helix.yaml --visualizer none --record-video
python -m isaac_drone run --backend isaaclab --visualizer none --record-video --video-fps 120 \
    --video-cameras overview follow --video-width 1920 --video-height 1080
```

| 文件 | 画面 |
|---|---|
| `overview.mp4` | 固定全景，按计划轨迹范围自动取景 |
| `follow.mp4` | 斜后上方跟随实际质心，保持稳定地平线 |
| `top.mp4` | 动态俯视，始终在实际位置正上方 |
| `combined.mp4` | 同步拼屏：三路时全景在上、跟随与俯视在下；两路左右并排 |

YAML 的 `recording` 段设置默认值（`enabled/fps/width/height/cameras`），命令行只覆盖当次运行；只改帧率、尺寸或机位不会自动开启录制。帧率不超过物理频率，尺寸为正偶数。各机位在同一物理时刻统一渲染，按仿真时间轴编码：`--playback-speed`、显示跳帧、机器快慢都不改变视频时间轴。结束或 `Ctrl+C` 时会关闭编码器并保存已录片段。

有画面时默认按真实时间播放（`--playback-speed 2` 两倍速，`0` 不限速），视口中绿色虚线为目标轨迹、红色为实际质心轨迹、绿色小球为当前目标点（`--no-path-overlay` 关闭）。这些只影响显示，不改变物理步进和日志。

## 工作流与仓库外内容

```
本地编辑 → 合成后端验证 → git commit/push → 服务器 git pull → Isaac Sim 运行 → rsync 拷回 runs/ → 本地分析
```

不在仓库里：`IsaacLab/` 官方仓库（契约测试会通过 `ISAACLAB_PATH` 或同级目录自动找到它，找不到时跳过对比测试）、`isaacsim-6.1/` 与服务器运行产物、Isaac Sim 的 Python 环境 `env_isaacsim`（29GB，可重新安装）。资产各自带 NVIDIA 资产库的 LICENSE，本仓库保持私有。
