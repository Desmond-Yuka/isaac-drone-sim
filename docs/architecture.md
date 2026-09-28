# ARL-Robot-1 运动控制架构

本项目以本地 ARL USD 和当前 `IsaacLab/` 源码为依据，面向 **单机、固定连接的刚体总成、PhysX**。通用运动控制基础设施已经接入原地起飞、三维圆柱 helix 上升和终点悬停任务。接口、配置校验与数值测试可在普通 Python 下运行；实际物理步进必须使用安装了当前 Isaac Lab / Isaac Sim 的环境。

## 1. 数据流与目录

```text
configs/arl_robot_1.yaml
        │
        ├── backends/isaaclab.py ── USD + PhysX 实际质量/质心/惯量/几何
        │                              │
        └── runtime.MotionControlLoop  │
              ├── trajectories：位置、速度、加速度、jerk、snap、偏航及其导数
              ├── StateProvider：当前为仿真真值，未来可注入传感器/估计器
              ├── control.GeometricController：期望质心力/力矩
              ├── control.BoundedAllocator：每桨有界推力目标 [N]
              ├── power.ActuatorEnvelope：电气状态下的已标定指令包线
              ├── disturbances + aero：每物理步的附加质心力/力矩
              └── backend：RPS 电机动态 → 每个旋翼独立施力 → PhysX
```

| 目录/文件 | 责任 |
|---|---|
| `isaac_drone/config.py` | 严格 YAML 解析、重复/未知字段检查、物理量与跨字段验证、本地资产检查 |
| `isaac_drone/types.py` | 统一状态、质心惯量、轨迹、力/力矩契约 |
| `isaac_drone/control/` | 位置 PID、SO(3) 姿态控制、完整惯量前馈、有界控制分配 |
| `isaac_drone/backends/` | Isaac Lab API、四元数转换、真实几何/惯量提取、原生电机与施力 |
| `isaac_drone/trajectories/` | 运动目标；`hold.py` 可用于联调，`spiral.py` 实现 C4 三维圆柱 helix（历史命名） |
| `isaac_drone/aero/` | 风场、阵风、随机风、相对气流阻力 |
| `isaac_drone/disturbances/` | 扰动组合、作用点与参考系转换、时间窗 |
| `isaac_drone/power/` | 电池等效电路、实测负载与执行器包线接口 |
| `isaac_drone/runtime.py` | 控制降频、物理步调度、复位及异常状态处理 |
| `isaac_drone/measurements.py` | 每步真实运动数据、完整角动量差分、位置误差及力/力矩阶段区分 |
| `isaac_drone/telemetry.py` | 配置快照、资产层哈希、JSONL/CSV 运行日志与字段 schema |
| `isaac_drone/recording.py`、`recording_views.py` | 可选多机位编码、仿真时间采样、同步拼屏与跟随取景 |
| `isaac_drone/backends/video.py` | 独立 USD 相机与 RTX RGB 同步采集、资源清理 |
| `scripts/standalone/run_arl.py` | 配置校验、真实物理属性检查、独立仿真入口 |

## 2. 已确认的资产事实与来源

参考源码：

- `IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/arl_robot_1.py`
- `IsaacLab/source/isaaclab_contrib/isaaclab_contrib/actuators/thruster.py` 和 `thruster_cfg.py`
- `IsaacLab/source/isaaclab_contrib/isaaclab_contrib/assets/multirotor/`
- `IsaacLab/source/isaaclab_contrib/isaaclab_contrib/controllers/`
- `assets/Robots/NTNU/ARL-Robot-1/` 的全部 USD 层

对当前资产进行了真实 USD 读取，而非从文件名推测：

- 一个 `base_link`、四个旋翼刚体、四个 **固定关节**。
- USD 显式质量约为机身 1.20 kg、各旋翼 0.01 kg，总和约 1.24 kg。此数值仅用于资产审计；控制器每次从 PhysX 加载结果取值，不在控制器里写死。
- 旋翼原点为 BL=(-0.10,+0.10,0)、BR=(-0.10,-0.10,0)、FL=(+0.10,+0.10,0)、FR=(+0.10,-0.10,0)，单位 m，相对根链接。
- USD 中质心为未显式填写的默认值，不能把它替换成零。运行时使用 PhysX 求值后的质心。
- 上游参考分配矩阵使用 ±0.13 m，符号/排列也与实际几何不一致；其旋向 `[-1,+1,-1,+1]` 与实际位置组合后，集体推力/滚转/俯仰/偏航矩阵秩为 3。
- 用户明确选择**对角同向仿真配置**。当前动作顺序固定为 `[back_left, back_right, front_left, front_right]`，方向系数为 `[-1,+1,+1,-1]`，来源标记 `simulation_diagonal_pairs`。这不是实机测量，也不根据静态桨叶网格猜测 CW/CCW。

`rotor_directions` 沿用原生字段名；后端使用它决定 `torque_to_thrust_ratio * thrust` 的反扭矩符号。将来接实机时必须同时核对转子物理旋向、电机编号、反扭矩正号与坐标系。

分配矩阵依据实际模型构建：每列由推力单位轴和 `(旋翼位置 - 整机质心) × 推力轴 + 反扭矩` 构成。YAML 的上游矩阵仅保留作差异诊断。改动 USD 几何后无需手工假设相同臂长。

## 3. 坐标、质量与惯量

- 世界系 Z 向上，所有数值使用 SI。
- 项目四元数一律 **wxyz**，表示机体系到世界系。当前本地 Isaac Lab 3.0 状态/写入接口使用 **xyzw**，只在接入边界转换。
- `VehicleState.position_w/linear_velocity_w` 为**整机质心**的位置/速度；`angular_velocity_b` 是机体系角速度。
- `Wrench` 为绕整机质心、以机体系表示的 `[Fx,Fy,Fz,Tx,Ty,Tz]`。
- YAML `initial_state.pos` 保留原生根链接原点位置语义；`lin_vel` 是原生根刚体质心的世界系线速度，`ang_vel` 是世界系角速度。它们与共享状态的定义刻意分开，由后端转换。
- 整机惯量由全部刚体的质量、实际质心、惯量主轴姿态和完整惯量张量，通过旋转和平行轴定理求和。不会假定质心为零、惯量对角或各旋翼质量忽略不计。
- 四个旋翼推力和反扭矩分别施加在对应旋翼刚体的原点，经固定关节传递；不会向根刚体提交控制器的整机目标 wrench。只有显式配置的外部扰动力作用在根刚体，其力矩按整机质心与根刚体质心偏移转换。整机合力/力矩仅用于控制分配和日志诊断。

OpenUSD 的 `diagonalInertia` 是主轴坐标中的惯量，`principalAxes` 是该主轴相对 prim 的姿态；`centerOfMass` 位于 prim 局部坐标。参见 [OpenUSD MassAPI](https://openusd.org/25.11/api/class_usd_physics_mass_a_p_i.html)。PhysX `get_inertias` 的 COM local frame 语义参见 [Omni Physics Tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/108.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)。

## 4. 配置的可写范围

主入口是 `configs/arl_robot_1.yaml`，每次实验可复制为独立文件并传 `--config`。未识别字段、重复 YAML key、NaN/Inf、错误单位形状、负时间常数都会报错。

- `simulation`：物理步长、控制降频、渲染降频、重力、设备、随机种子、时长。
- `recording`：默认关闭；帧率默认 60、每机位 1280×720，可选 `overview/follow/top` 的非空无重复列表。旧配置省略整段时补默认值；启用时帧率不超过物理频率，图像尺寸为正偶数。CLI 可单次覆盖开关、帧率、尺寸和机位。
- `vehicle.thrusters`：上游全部动力学字段，保留推力上下限、推力系数范围、上升/下降时间常数、反扭矩系数、速率限制、积分方式、离散近似。
- `vehicle.initial_state`：完整姿态、位置、速度、每个旋翼初始 rps。
- `vehicle.launch`：地面高度、初始间隙及是否按真实碰撞体计算初始 root Z；启用时要求零初速度和零电机转速。计算包含 instance proxy 内的启用碰撞体，并重算形状 extent，避免当前资产的过期 authored extent 造成穿地。只在 reset 初始化时写入该位置。
- `vehicle.native_overrides`：递归覆盖当前原生机器人配置，含刚体、关节、碰撞、质量、材质等。未知原生字段在实例化原生配置时拒绝。
- `simulation.native_overrides`：原生 `SimulationCfg` / `PhysxCfg` 的其余参数。无须为了暴露某个 PhysX 字段重写整个配置类。
- 主字段如 dt、资产路径、动作顺序使用单一来源，禁止在 native overrides 中重复覆盖，避免控制器与物理模型各用一套值。
- `vehicle.usd_overrides`：启动场景后、物理初始化前修改已有 prim 的质量、质心、主惯量和主轴属性，不回写原始 USD 文件。例如：

```yaml
usd_overrides:
  - prim_path: base_link
    attribute: physics:centerOfMass
    value: [0.01, 0.0, 0.0]  # 示例格式，不是本项目默认物理参数
```

不能用 `null` 假装实测零值。原生覆盖中的 null 保留原生值；未填写的电池/气动标定值表示未知，开启后必须补齐。原生配置本身已有的 `linear_damping=0` 等数值注明其上游来源，不将它们表述为真实飞行器无阻力。

可写不代表每种物理设置适用于当前控制器。自由飞行要求重力与控制配置一致、刚体动态启用、无固定基座，且四个推力轴均沿机体 +Z。倾转旋翼、可动内部关节、系留约束、运动学机体需要相应控制器/后端扩展；检查模式允许审计，飞行前会拒绝不兼容组合。

## 5. 时序、执行器与控制

默认物理步长 0.005 s、控制降频 2，即控制 100 Hz、电机和扰动 200 Hz。这是可修改的数值实验选择，未宣称它是硬件采样率或已经收敛的最佳步长。

每个物理步：

1. 读状态；若到控制时刻，采样轨迹并计算期望 wrench。
2. 在当步有效推力包线内做有界加权分配。四旋翼枚举至多 81 个约束组合，求解加权最小二乘，记录实际分配残差，而不是简单伪逆后截断。
3. 计算当步风/气动/外力。
4. 使用每个电机自己的采样 kf 将分配推力转为 **RPS** 命令；起转阶段有平滑启动包络。直接积分 RPS，按 kf·RPS² 计算实际推力，每个旋翼独立受力。向 PhysX 提交一次逐刚体数组，根刚体只有显式外部扰动。
5. `sim.step()`，`robot.update(dt)`，记录真实推进器输出及状态，更新可选电池。

原生电机保留 `kf` 和 tau 的复位随机采样、非对称动态、rps² 推力关系、RK4/Euler、饱和。`max_thrust_rate` 虽沿用原生字段名/配置值，源码在 rps 误差积分中限幅，不能按名字误读成严格 N/s；日志记录真实采样的 kf/tau。

当前上游实现还有两个已局部兼容处理的初始化问题：推进器创建时可能拿到尚未填充的初始 rps 副本；reset 将 rps 写进 N-valued 目标。后端复位时重新绑定初始 rps，并在第一次动力学更新前以真实初始推力覆盖目标。没有修改 `IsaacLab/` 官方副本。

位置环输出加速度指令；姿态环使用直接力矩增益，带完整 `ω×Jω` 和期望角加速度前馈。轨迹可以给速度、加速度、jerk、snap 及偏航导数。提供 jerk/snap 且未限幅时，参考力的导数采用解析值，反馈部分仍为数值差分；限幅后退回整个力向量的差分，避免对投影后的力使用错误解析导数。位置积分默认关闭，可显式配置，启用时包含积分限制和分配反馈防饱和。

适用范围必须明确：当前姿态控制器面向直立飞行；SO(3) 平滑误差在精确 180° 存在平衡点。没有提供 jerk/snap 时，推力方向导数由历史差分得到，首帧尚无历史，突变参考与噪声会造成导数瞬态。`max_yaw_rate_rad_s` 限制角速度前馈，不会自动把偏航角阶跃规划成平滑转弯。

`prepare_step/finish_step` 必须成对调用。任意插件或推进失败后禁止继续重试同一帧，以免重复积分电量或控制器状态；需要完整 reset。仿真暂停不会推进模型，停止后结束本次运行。

## 6. 扰动和气动扩展

已实现但默认关闭：

- 世界系或机体系的常值力/力矩、起止时间窗、相对质心或世界绝对作用点。
- 均匀风、空间梯度、平滑阵风、可复现的 OU 随机风。
- 相对风速的线性矩阵阻力、逐轴二次阻力、角速度阻力、压力中心力臂。

气动开启时要求明确风场，即使选择静风，也要写出 `[0,0,0]`。阻力计算包含作用点的 `ω×r` 速度。矩阵阻尼检查耗散性；风可以向飞行器输入能量，因此不会错误地强制世界系总做功永远为负。

例如额外横风只能先配置风速，还必须填入已辨识的阻力参数，不能自动以某个“典型无人机系数”代替。地面效应、桨间干扰、诱导流、旋翼惯性陀螺效应、叶片拍振等尚未建立专用物理模型；未来通过 `EffectModel` 或执行器插件接入，不填虚构参数。

## 7. 电池与电机电气模型

电池当前默认未启用，所有 ARL 特定参数为 null。`power/` 已提供可独立测试的表格 OCV + 串联电阻 + 可选多个 RC 极化支路模型，包含 SOC、充放电效率、端口能量积分、欠压/过压/电流/温度有效域检查。

这是一种**显式选择的等效电路模型**，不是完整电化学模型。温度依赖、老化、容量衰减、热动力学需更多已标定数据及模型。温度缺失会标记未知，不默认室温。

启用需同时提供：

1. 完整电池参数、OCV-SOC 表和标定编号。
2. 每步实测电流，或实现 `LoadModel.current_a` 的已标定负载模型。电流正值放电、负值充电；不使用“推力 × 空速”猜电功率。
3. `ActuatorEnvelope.thrust_bounds_n`，由标定给出当前电压/状态下每桨指令上下限；不假设推力按电压平方变化。
4. 初始有载电流工作点与每步温度来源。

`MotionControlLoop` 支持注入 `PowerSystem`。通用 CLI 不含未标定的插件，开启电池后会要求使用自定义入口接线。当前包线约束**推力命令**，原生动态仍可能产生滞后输出；这不等价于已经仿真了 ESC、反电动势、铜耗和母线耦合。若包线低于原生最小推力，明确拒绝继续；当前已支持零 RPS 命令按原生下降动态停桨，停转输出为零；但电池失电的策略、母线耦合和原生下限不兼容时的运行决策仍需专门插件，不能用残留的 0.1 N 假装已经关机。

## 8. 运行与验证

普通 Python 环境需要 NumPy、PyYAML；测试需要 pytest。可在自己的环境中以 editable 方式安装项目：

```bash
python -m pip install -e '.[test]'
python -m pytest -q tests
python scripts/standalone/run_arl.py --validate
```

离线 USD 审计还需 `usd-core`（`pip install -e '.[usd]'`）；在不启动 Isaac Sim 的情况下可读取真实资产并查看未求值属性：

```bash
python scripts/standalone/inspect_arl_asset.py
```

在已经正确安装依赖的 Isaac Lab 环境中（服务器激活 `env_isaacsim` 后），从本仓库根目录运行。Isaac Lab 3.0 已去掉 `--headless`，无画面运行用 `--visualizer none`：

```bash
python scripts/standalone/run_arl.py --inspect --visualizer none
python scripts/standalone/run_arl.py --visualizer none --duration 10
```

WebRTC 串流用 `--livestream 1`（公网，`PUBLIC_IP` 指定服务器 IP）。Isaac Lab 3.0 把串流主机视为 headless，步进时不会调用 `app.update()`，所以串流时入口脚本在每个渲染步自行刷新 Kit 界面，并在刷新期间关闭 `playSimulations`，不额外推进物理。`--wait-for-start` 在场景加载后保持界面刷新、不推进物理，终端按回车后才开始任务。

有画面（串流或本地窗口）时，`isaac_drone/playback.py` 按墙钟对齐仿真时间，默认 `--playback-speed 1` 即真实时间；`0` 关闭对齐。每个渲染步：仿真超前就等待；刷新界面比帧间隔慢、仿真落后超过 50 ms 时跳过该帧，让物理追上，但至少每 0.25 s 画一帧。结束事件 `finished.playback` 记录墙钟时长、实际倍速、已画/跳过帧数和平均刷新耗时；若物理和记录本身就慢于实时，这里会如实显示达不到的倍速。`isaac_drone/visualization.py` 在 `/World/Visuals` 下创建纯显示用的 USD 几何（不带任何物理/碰撞 API）：绿色虚线为参考质心轨迹，红色实线为实际质心轨迹（每移动 5 mm 记一点，在渲染帧写入），绿色小球为当前时刻的参考点；`--no-path-overlay` 关闭。无画面且不录制时不创建这些几何，也不做墙钟对齐；离屏录制时也可保留轨迹叠加。

视频录制位于独立入口的展示/输出层，不接入 `MotionControlLoop` 或动力学后端。`--record-video` 在 AppLauncher 启动前检查可选 `imageio-ffmpeg` 及编码器，并启用 `enable_cameras`（不启动 Isaac Lab 自带的 gym 录像链路）。`IsaacCameraRig` 为选定机位创建独立相机和 render product；全景按参考路径取景，跟随/俯视按每步实际质心定位。每次采集统一更新相机，调用 `sim.forward()` 同步 Fabric，再临时关闭 `playSimulations` 刷新一次 Kit，读取所有 RGB；不调用物理 step。非采集时暂停这些 render product，退出释放自身资源。

`MultiViewRecorder` 以整数帧索引计算 `index/fps` 采样时刻，与 `render_interval`、墙钟显示节流分离。初始化相机/等待输入不编码；从任务 t=0 起采样，在到达采样时刻的物理步取图，终点处仅补齐早于终点的采样时刻。每次图像同时送入独立 MP4 与 `combined.mp4`，同步拼屏不经过事后时间匹配。编码器采用流式 FFmpeg/H.264（[imageio-ffmpeg 提供可执行文件](https://github.com/imageio/imageio-ffmpeg)），不会将整个实验的原始帧堆积在内存中。`video_frames.jsonl` 记录编码帧索引/视频时间/实际仿真采样时间；结束/中止事件记录各文件帧数及编码状态。`ExitStack` 在正常完成、仿真停止及异常（含 Ctrl+C）时关闭全部采集和编码资源。用法见 [README 的录制章节](../README.md#可选多机位视频录制)。

也可以直接使用服务器现有 Isaac Lab Python 环境运行同一个脚本。代码不会自动下载 29GB 运行环境，也不会切换到另一个机型。`--inspect` 会加载 PhysX 实际属性并打印分配矩阵/质量/惯量/采样参数，然后退出，不执行飞行控制。

运行结果默认写到 `runs/<UTC时间_唯一编号>/`，包含配置快照、所有本地 USD 层 SHA256、实际物理参数、`telemetry.jsonl`、`basic.csv` 和 `telemetry_schema.json`。默认每物理步记录一次（当前200 Hz），包含位置、速度、加速度、四电机转速、力/力矩及位置误差；日志区分期望、分配结果、原生电机实际输出、附加力与运动推导的净作用。禁用电池记录 null，不记录虚构 SOC。完整字段和时间语义见 [telemetry.md](telemetry.md)。

三维 helix 入口：

```bash
python scripts/standalone/helix_ascent.py --validate
# 然后在完整仿真 Python 环境中运行（无画面加 --visualizer none）
python scripts/standalone/helix_ascent.py
```

默认使用 `configs/arl_robot_1_helix.yaml`；`spiral_ascent.py` 兼容相同任务。`trajectory.kind: helix` 和旧名 `spiral` 都表示三维圆柱螺旋，参数段名仍为 `trajectory.spiral`。起点由 reset 后真实质心确定，参考经过原地垂直起飞、恒定半径螺旋上升、平滑减速、无限期终点 hold。参考 C4 连续并提供解析四阶导数，数学推导和约束说明见 [spiral_math.md](spiral_math.md)。

运行前检查实际推力上下限下的悬停可行性、解析最大偏航速率及离散参考采样点的力矩/倾角可行性；这些检查不能替代有限电机响应、接触、扰动下的闭环验飞。`mission.py` 每物理步用真实状态判定连续悬停时间，超时失败不会被迟到的状态重写。正常达到终点以后继续运行反馈控制。JSONL 的 `mission` 给出容差、停留时间和成功/超时状态，`finished.success` 是整个运行的最终判定。

## 9. 本次验证的边界

已执行普通 Python 数值/契约测试、配置校验，并使用实际 USD 解析检查结构和参数。测试覆盖非零质心、非对角惯量、旋翼顺序、力矩作用点、分配饱和、复位与时间配对、扰动和电池能量等。

当前开发机器没有安装 Isaac Sim、Torch、Warp 完整运行栈，因此**尚未完成真实 PhysX 端到端悬停验飞**。假对象后端测试不替代该验证；提供 `--inspect` 和 `hold` 入口用于服务器后续实测。现有控制增益来自上游范围，尚未针对实际资产重新调参；没有声称这些增益已保证稳定飞行。
