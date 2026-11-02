# 实验配置与扩展（schema_version 2）

一次实验 = 一个 YAML 文件（可继承）+ 可选的命令行覆盖。所有配置在飞行前严格校验：未知字段、重复 key、NaN/Inf、错误向量长度、布尔值冒充数字都会报错，不会悄悄取默认值。

## 1. 文件组织与继承

| 文件 | 内容 |
|---|---|
| `configs/arl_robot_1.yaml` | 基础配置：仿真、机型、推进器、场景、日志、录制、默认几何控制器、空中初态定点悬停（hold） |
| `configs/helix.yaml` | 继承基础配置：地面零转速起飞 + 最短时间 helix，完成判据 |
| `configs/helix_moderate.yaml` | 继承 helix：同一几何、固定时长（约 3.3 m/s 峰值）的慢速版，给反馈留余量 |
| `configs/helix_rotor_wind.yaml` | 继承 helix：关掉质心 OU 力，改用世界系风场作用到旋翼（只在 x 轴随机的 OU 示例，系数为注明来源的名义值） |

子文件用 `extends: <相对路径>` 继承父文件，只写差异：

- 映射（mapping）逐层合并；列表和标量整体替换。
- 若子文件某段的 `kind` 与父文件不同（例如 `trajectory.kind: hold → helix`），这一段**整段替换**，不会继承旧类型的参数。
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
| `vehicle` | 资产、旋翼方向及其来源、转子惯量 `rotor_inertia_kg_m2`（陀螺力矩，默认 null 关闭）、电机响应模型开关 `motor_model_enabled`（默认 false：转速当步等于指令，无响应延迟；true：原生一阶滞后，按 `tau_inc_range`/`tau_dec_range`）、推进器参数、初态、`launch`（地面起飞、离地间隙、`spin_up_s` 起转时长）、原生/USD 覆盖 |
| `limits` | 控制器和轨迹规划**共用**的飞行包线：`max_tilt_rad`、`max_yaw_rate_rad_s`、`max_acceleration_m_s2` |
| `controller` | `kind` + 该控制器的参数（见第 4 节） |
| `allocation` | 有界加权分配的权重和正则化 |
| `trajectory` | `kind` + 该轨迹的参数（见第 5 节） |
| `completion` | 终点悬停完成判据；`null` 表示不设判据（跑完所有步即成功） |
| `effects` | 常值外力/力矩、`ou_wrench` 随机外扰（力世界系、力矩机体系，均值 + 逐轴标准差 + 相关时间）、风场 `wind`、风作用到旋翼 `rotor_wind`、气动阻力；`ou_wrench` 默认开启（力均值 [1,0,0] N、各轴标准差 0.5 N、τ=1 s；`helix.yaml` 从起转结束 2 s 起作用），其余默认关闭。风场与风作用到旋翼见下文 |
| `model_mismatch` | 对象实际值 / 控制器标称值之比：`mass_ratio`、`inertia_ratio`、`thrust_coefficient_ratio`、`reaction_torque_ratio`；全为 1 时无失配，默认全为 1（关闭） |
| `scene` / `logging` | 地面与灯光；日志目录、运行目录名里的实验名 `run_name`（`null` = YAML 文件名，目录名规则见 README「运行目录」）、记录间隔、刷新间隔 |

`vehicle.launch.spin_up_s`：起转阶段参考保持起点，电机指令乘以 0→1 的平滑包络；轨迹在起转结束时刻才开始。地面起飞必须 > 0。地面起飞时轨迹锚定在后端真值（物理放置位置）；空中起始锚定在控制器看到的状态（可以是估计器输出）。

### 风场与风作用到旋翼（`effects.wind`、`effects.rotor_wind`）

`wind` 生成世界系风速（m/s，写的是空气往哪走），`rotor_wind` 再按机体系把它分成两路作用到飞机上。两者默认关闭。

风场 = 平均风 `velocity_w_m_s` + 阵风 `gusts` + OU 湍流 `turbulence`（另可加线性梯度 `spatial_gradient_per_s`）：

- **时间窗** `start_time_s` / `end_time_s`：窗口外是静风，到 `start_time_s` 时风直接以满强度出现。OU 在复位时就按平稳分布取值，之后每个物理步都在演化，所以开窗那一刻湍流已经是满强度，不会从 0 慢慢长起来。`helix.yaml` 设为 2 s，即起转结束时。
- **湍流** `turbulence.mode`：
  - `axes`（默认）：每个世界轴一个独立 OU，`time_constant_s`、`stationary_std_w_m_s` 各写 3 个数。只随机一个轴，就把另外两个轴的标准差设为 0。
  - `direction`：沿 `direction_w`（自动归一化）的单个 OU，`time_constant_s`、`stationary_std_w_m_s` 各写 1 个数。`[1, 0, 0]`、`[0, 1, 0]`、`[0, 0, 1]` 就是单轴，`[1, 1, 0]` 是水平 45°。
- **阵风** `gusts`：每项写 `start_time_s`、`duration_s`、`delta_velocity_w_m_s`、`shape`。`one_minus_cosine`（默认）平滑升起再回落；`step` 在 `start_time_s` 瞬间出现，到时瞬间消失，`duration_s: null` 表示一直持续到结束。

风作用到旋翼（`rotor_wind`，需要 `wind.enabled`）按机体系分工，两路各有开关：

- **沿桨轴**（`axial`）：把每个桨处的风转到机体系，取沿桨轴的分量 `u_axial,i`（上升气流为正），换算成等效转速偏移 `δΩ_i = c/2 · Ω_hover,i · u_axial,i`，在电机模型之后叠加：推力和反扭矩按 `k_f·(Ω_i + δΩ_i)²` 计算，电机状态本身不变。`c` 即 `thrust_gain_per_m_s`，是悬停时每 1 m/s 轴向风使每桨推力变化的比例，BEMT 线性化给出 `c = κ / v_h`；`Ω_hover,i` 是四桨均分重量时第 i 个桨的转速。偏移与当前转速无关，所以推力变化与转速成正比，而不是与推力成正比。
- **桨盘平面内**（`inplane_drag`）：转速产生不了水平力，这部分作为阻力 `k · u_inplane` 加在质心、顺着风，`k` 即 `drag_n_per_m_s`。
- 只算风本身，不算飞机自身运动带来的相对气流，所以静风时结果与不开风完全一样。线性模型在总风速约 3 m/s（约为悬停下洗速度的一半）以内可靠。
- 系数没有默认值，必须显式填写并注明来源；`configs/helix_rotor_wind.yaml` 给出名义值及其推导来源。

旧的质心 OU 力 `ou_wrench` 保留不变，可作对照组。

```bash
# 只在 z 轴随机的风，其余同 helix_rotor_wind.yaml
python -m isaac_drone run --backend synthetic --config configs/helix_rotor_wind.yaml \
    --set 'effects.wind.turbulence.direction_w=[0,0,1]'
# 沿同一方向扫风速标准差和种子
python -m isaac_drone sweep --config configs/helix_rotor_wind.yaml \
    --grid 'effects.wind.turbulence.stationary_std_w_m_s=[0.5,1.0,1.5,2.0]' --grid 'simulation.seed=[0,1,2,3,4]'
# 6 s 时突然出现 2 m/s 的下沉气流，持续 2 s
python -m isaac_drone run --backend synthetic --config configs/helix_rotor_wind.yaml \
    --set 'effects.wind.gusts=[{start_time_s: 6.0, duration_s: 2.0, delta_velocity_w_m_s: [0, 0, -2], shape: step}]'
```

## 4. 控制器

### `geometric`：几何控制（Lee SO(3)，唯一控制器）

位置环 PID + SO(3) 姿态 PD，含 jerk/snap 与完整惯量前馈。

| 参数 | 含义 |
|---|---|
| `position_kp` [s⁻²] | P：位置误差 → 加速度 |
| `velocity_kd` [s⁻¹] | D：速度误差 → 加速度 |
| `position_ki` [s⁻³] | I：位置误差积分 → 加速度；默认开启（基础配置 `[1.5, 1.5, 0.5]`，`helix.yaml` `[2, 2, 0.5]`），某轴填 0 即关闭该轴积分 |
| `integral_limit_m_s` [m·s] | 位置误差积分的逐轴上限；分配饱和或推力方向被限幅时积分不再增长 |
| `attitude_kp` [N·m/rad]、`angular_rate_kd` [N·m·s/rad] | 姿态环 PD 的直接力矩增益（姿态环没有积分项） |

`helix.yaml` 的位置环 PID（kp `[20, 20, 16]`、kd `[9, 9, 7]`、ki `[2, 2, 0.5]`）是 2026-10-20 在 Isaac 里按无扰动 helix 的同一时刻跟踪误差调的：误差随 D 单调下降，但 D 越大，OU 力扰动下对反馈力的数值微分越会让电机饱和（σ=2 N 时 kd 9 为 1.1%，kd 20 为 14%），所以取 kd 9；kp 对无扰动误差几乎无影响；ki 越小越好，取测试过的最小值。理想双积分器闭环 `s³ + kd s² + kp s + ki` 的稳定上限是 `ki < kp·kd`。

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

鲁棒性实验按 Kidambi（IEEE Access 2025）的方式扫参数失配与扰动强度，例如：

```bash
python -m isaac_drone sweep --config configs/helix_moderate.yaml --grid model_mismatch.mass_ratio=[0.75,0.9,1.0,1.15,1.3]
python -m isaac_drone sweep --config configs/helix_moderate.yaml --set effects.ou_wrench.enabled=true \
    --grid effects.ou_wrench.force_std_w_n=[[0.5,0,0],[1,0,0],[2,0,0]] --grid simulation.seed=[0,1,2,3,4]
```

```bash
# 网格扫参（CPU 合成后端，多进程并行）：每个点都是完整运行目录
python -m isaac_drone sweep --config configs/helix.yaml \
    --grid 'controller.position_ki=[[0,0,0],[3,3,2],[6,6,4]]' \
    --grid 'controller.velocity_kd=[[6,6,4],[7,7,5]]' \
    --rank helix.position_error_max_m
# 对比几次运行（可只比某个阶段）
python -m isaac_drone compare runs/<A> runs/<B> --labels no_integral integral --phase helix
```

扫参开始前会先校验所有点的配置；单个点发散只记为失败，不会中断整个扫参。结果写在 `runs/sweep_<月日-时分>/summary.{csv,json,md}`，按 `--rank` 指标升序排列；每个点的运行目录是 `points/<序号>/`，序号即 summary 里的 `index` 列；指标名见 `isaac_drone/analysis/metrics.py`（整体指标直接写名字，分阶段写 `阶段名.指标名`）。

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
