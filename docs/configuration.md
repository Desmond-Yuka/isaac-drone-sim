# 实验配置与扩展（schema_version 2）

一次实验 = 一个 YAML 文件（可继承）+ 可选的命令行覆盖。所有配置在飞行前严格校验：未知字段、重复 key、NaN/Inf、错误向量长度、布尔值冒充数字都会报错，不会悄悄取默认值。

## 1. 文件组织与继承

| 文件 | 内容 |
|---|---|
| `configs/arl_robot_1.yaml` | 基础配置：仿真、机型、推进器、场景、日志、录制、默认几何控制器、空中初态定点悬停（hold） |
| `configs/helix.yaml` | 继承基础配置：地面零转速起飞 + 最短时间 helix，完成判据 |
| `configs/helix_moderate.yaml` | 继承 helix：同一几何、固定时长（约 3.3 m/s 峰值），给无前馈的控制器留余量 |
| `configs/helix_cascaded_pid.yaml` | 继承 helix_moderate：只把控制器换成级联 PID |

子文件用 `extends: <相对路径>` 继承父文件，只写差异：

- 映射（mapping）逐层合并；列表和标量整体替换。
- 若子文件某段的 `kind` 与父文件不同（例如 `trajectory.kind: hold → helix`，`controller.kind: geometric → cascaded_pid`），这一段**整段替换**，不会继承旧类型的参数。
- 支持多级继承，循环继承会报错。

新实验的推荐做法：复制一个最接近的文件，改名，只保留要改的字段。

## 2. 命令行覆盖

所有命令都接受 `--set 键路径=值`（可重复），值按 YAML 解析：

```bash
python -m isaac_drone run --backend synthetic --config configs/helix.yaml \
    --set controller.position_kp=[10,10,6] --set trajectory.turns=3 --set simulation.seed=1
# 整段替换：换成定点悬停，并关闭完成判据
python -m isaac_drone run --backend synthetic --config configs/helix.yaml \
    --set 'trajectory={kind: hold}' --set completion=null
```

覆盖后的完整配置会写进运行目录的 `config.json`，命令行本身记在 `metadata.json` 的 `provenance.argv`（另有 git commit 与是否有未提交修改）。

## 3. 各段含义

| 段 | 作用 |
|---|---|
| `simulation` | 物理步长 `dt`、控制降频、渲染间隔、设备、重力、随机种子、时长、原生 SimulationCfg 覆盖 |
| `recording` | 可选多机位视频（仅 isaaclab 后端），见 README |
| `vehicle` | 资产、旋翼方向及其来源、推进器动力学、初态、`launch`（地面起飞、离地间隙、`spin_up_s` 起转时长）、原生/USD 覆盖 |
| `limits` | 控制器和轨迹规划**共用**的飞行包线：`max_tilt_rad`、`max_yaw_rate_rad_s`、`max_acceleration_m_s2` |
| `controller` | `kind` + 该控制器的参数（见第 4 节） |
| `allocation` | 有界加权分配的权重和正则化 |
| `trajectory` | `kind` + 该轨迹的参数（见第 5 节） |
| `completion` | 终点悬停完成判据；`null` 表示不设判据（跑完所有步即成功） |
| `effects` / `power` | 风、气动阻力、外力扰动；电池模型。默认关闭，开启必须给出标定参数 |
| `scene` / `logging` | 地面与灯光；日志目录、记录间隔、刷新间隔 |

`vehicle.launch.spin_up_s`：起转阶段参考保持起点，电机指令乘以 0→1 的平滑包络；轨迹在起转结束时刻才开始。地面起飞必须 > 0。地面起飞时轨迹锚定在后端真值（物理放置位置）；空中起始锚定在控制器看到的状态（可以是估计器输出）。

## 4. 控制器

### `geometric`：几何控制（Lee SO(3)）

位置 PID + SO(3) 姿态控制，含 jerk/snap 与完整惯量前馈。增益：`position_kp` [s⁻²]、`velocity_kd` [s⁻¹]、`position_ki` [s⁻³]、`integral_limit_m_s`、`attitude_kp` [N·m/rad]、`angular_rate_kd` [N·m·s/rad]。

### `cascaded_pid`：级联 PID

位置 → 速度 → 姿态 → 机体角速度四级串联。增益单位都是 1/s（积分 1/s²），角速度环按实际完整惯量归一化，所以增益可以直接理解为各环带宽，与机体尺寸无关。

| 参数 | 含义 |
|---|---|
| `position_kp` | 位置误差 → 速度修正 [1/s] |
| `velocity_kp`、`velocity_ki`、`velocity_integral_limit_m_s2` | 速度误差 → 加速度；积分项上限 [m/s²] |
| `attitude_kp` | 姿态误差 → 角速度设定 [1/s] |
| `rate_kp`、`rate_ki`、`rate_integral_limit_rad_s2` | 角速度误差 → 角加速度；积分项上限 [rad/s²] |
| `max_velocity_correction_m_s`、`max_body_rate_rad_s` | 可选限幅，`null` 为不限 |

它没有 jerk/snap 或姿态角速度前馈，是对照组而不是跟踪最优。最短时间 helix 按“完美前馈跟踪”规划、贴着推力包线（名义倾角约 66°），在 CPU 合成对象上测试过的所有 PID 增益都飞不了，所以对照实验使用 `helix_moderate.yaml` 的轨迹。

## 5. 轨迹

轨迹由若干**几何段**组成，每段按一个从静止到静止的时间律走完，最后在终点无限期悬停（阶段名 `hold`）。时长填数字 → 九次多项式（C4）；填 `null` → 按实际质量、惯量、分配矩阵、推力上下限和电机滞后规划“加速—巡航—减速”最短时间（需要 `thrust_utilization`）。所有段在飞行前都做名义可行性筛查（推力、倾角、偏航速率、加速度）。

### `hold`

`position_w_m`（`null` = reset 后真实质心）、`yaw_rad`（`null` = 实际朝向）。

### `helix`

阶段 `takeoff`（原地垂直升高）→ `helix`（恒定半径圆柱螺旋上升）→ `hold`。参数：`takeoff_height_m`、`takeoff_duration_s`、`radius_m`、`turns`（正数从 +Z 看逆时针，可为负数或小数）、`climb_height_m`、`helix_duration_s`、`initial_phase_rad`、`yaw_mode`（`fixed`/`tangent`）、`yaw_offset_rad`、`thrust_utilization`、`rest_speed_tolerance_m_s`、`rest_angular_speed_tolerance_rad_s`（reset 时要求静止）。数学推导见 [spiral_math.md](spiral_math.md)。

## 6. 新增控制器或轨迹

两类插件都是“参数 dataclass + 注册的工厂函数”，参数由 `isaac_drone.core.params` 严格解析（支持 `float`、`int`、`bool`、`str`、`Literal`、`Vec3`、`X | None` 等），范围检查写在 `__post_init__` 里。

**控制器**（`isaac_drone/control/my_controller.py`，并在 `control/__init__.py` 里 import 一次以完成注册）：

```python
@dataclass(frozen=True)
class MyGains:
    kp: Vec3

class MyController:
    output = "wrench"            # 或 "rotor_thrust"：直接输出每桨推力（如学习策略），只做上下限裁剪
    def __init__(self, gains, vehicle, limits): ...   # vehicle: 实际质量/惯量/分配矩阵/推力上下限/控制周期
    def reset(self): ...
    def compute(self, state, setpoint, dt_s): ...      # 返回 Wrench（机体系、绕质心）或每桨推力
    def notify_allocation(self, result): ...           # 分配结果，用于抗积分饱和；可忽略
    def diagnostics(self): ...                         # 期望姿态/角速度（写入遥测）及任意明细

@CONTROLLERS.register("my_controller", params=MyGains)
def build(params, *, vehicle, limits):
    return MyController(params, vehicle, limits)
```

也可以不注册，直接把控制器实例传给 `MotionControlLoop(config, backend, controller=...)`。

**轨迹**：继承 `SegmentedTrajectory`，实现 `_build(position_w, yaw_rad)`，返回 `(段列表, 终点位置, 终点偏航)`。每个 `Segment` 给出 `setpoint(progress)`（进度及其 1–4 阶导数 → 位置到 snap 与偏航）、路径长度、偏航变化量、时长（`None` 为最短时间）。这样就自动获得最短时间规划、可行性筛查、阶段划分、完成判据、轨迹叠加显示和按阶段统计的指标。参考 `trajectories/helix.py`。

## 7. 批量实验

```bash
# 网格扫参（CPU 合成后端，多进程并行）：每个点都是完整运行目录
python -m isaac_drone sweep --config configs/helix_cascaded_pid.yaml \
    --grid 'controller.attitude_kp=[[8,8,3],[10,10,3],[12,12,3]]' \
    --grid 'controller.rate_kp=[[20,20,8],[24,24,8]]' \
    --rank helix.position_error_max_m
# 对比几次运行（可只比某个阶段）
python -m isaac_drone compare runs/<A> runs/<B> --labels geometric pid --phase helix
```

扫参开始前会先校验所有点的配置；单个点发散只记为失败，不会中断整个扫参。结果写在 `runs/sweep_<时间>/summary.{csv,json,md}`，按 `--rank` 指标升序排列；指标名见 `isaac_drone/analysis/metrics.py`（整体指标直接写名字，分阶段写 `阶段名.指标名`）。

## 8. 从 schema_version 1 迁移

| v1 | v2 |
|---|---|
| `control:`（增益 + 限制混在一起） | `limits:`（`max_tilt_rad`、`max_yaw_rate_rad_s`、`max_acceleration_m_s2`）+ `controller: {kind: geometric, 增益}` |
| `trajectory: {kind, hold: {...}, spiral: {...}, completion: {...}}` | `trajectory: {kind: hold/helix, 该类参数}` + 顶层 `completion:` |
| `trajectory.spiral.start_delay_s` | `vehicle.launch.spin_up_s` |
| `spiral_duration_s` | `helix_duration_s` |
| `initial_speed_tolerance_m_s`、`initial_angular_speed_tolerance_rad_s` | `rest_speed_tolerance_m_s`、`rest_angular_speed_tolerance_rad_s` |
| `kind: spiral` 别名 | 已删除，用 `helix` |
| `configs/arl_robot_1_helix.yaml` | `configs/helix.yaml`（`extends: arl_robot_1.yaml`） |
| 遥测 `mission_phase`：`delay/takeoff/spiral/hold` | `spin_up/takeoff/helix/hold` |
