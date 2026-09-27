# 基础飞行数据记录

导师要求的 position、velocity、acceleration、motor speeds、force、torque、position error 均有对应记录。运行一次仿真后，在 `runs/<时间_编号>/` 保存：

- `basic.csv`：每行一个已完成物理步，列名包含分量和单位，可用于 Excel、MATLAB、Python 绘图。
- `telemetry.jsonl`：保留完整的步前/步后状态、控制输入、实际执行器状态和基础数据。
- `telemetry_schema.json`：CSV 列与 JSON 字段的对应关系、单位、参考系、采样时间与来源说明。
- `config.json` / `metadata.json`：实际配置、USD 层哈希、机型属性和模型边界。

## 字段对应

以下字段位于每条 JSON 记录的 `basic` 对象中。CSV 将向量按分量展开，具体列名以 `telemetry_schema.json` 为准。

| 要求 | JSON 字段 | 单位与含义 |
|---|---|---|
| Position | `position_w_m` | 世界系整机质心位置，m |
| Velocity | `velocity_w_m_s` | 世界系整机质心速度，m/s |
| Acceleration | `acceleration_w_m_s2` | 当前物理步的世界系平均运动加速度，m/s² |
| Solver acceleration | `acceleration_native_w_m_s2` | 可用时记录 PhysX 各刚体质心加速度的质量加权值，m/s² |
| Motor speeds | `motor_speed_rps` / `motor_speed_rpm` / `motor_speed_rad_s` | 四个旋翼转速幅值：rps、rpm、rad/s |
| Force | `force_command_b_n` / `force_motor_b_n` / `force_submitted_b_n` / `force_net_w_n` | 分别为控制需求、电机产生、提交物理引擎、由运动推导的总合力，N |
| Torque | `torque_command_b_nm` / `torque_motor_b_nm` / `torque_submitted_b_nm` / `torque_net_w_nm` | 对应上述不同阶段、关于整机质心的力矩，N·m |
| Position error | `position_error_w_m` / `position_error_norm_m` | 三轴目标减实际位置误差，以及三维欧氏距离，m |

同时保存目标位置/速度/加速度、四元数、角速度/角加速度、分配后的力/力矩、扰动力/力矩，以及每桨命令推力和实际推力。

## 时间对齐

每条 `basic` 记录有四个时间字段：

- `step_start_time_s`：本物理步开始时刻。
- `sample_time_s`：本物理步结束时刻；实际位置、姿态、速度取这个时刻。
- `interval_dt_s`：两者之差，即实际使用的物理步长。
- `reference_sample_time_s`：本步使用的目标最近一次由控制循环采样的时刻。

控制可以比物理仿真慢。例如默认物理 200 Hz、控制 100 Hz，第二个物理步继续使用第一个物理步采样的控制目标。记录的误差定义为：

```text
position_error_w_m = 本物理步持有的目标位置 - 物理步结束后的实际质心位置
position_error_norm_m = sqrt(error_x² + error_y² + error_z²)
```

这反映实际使用的目标，不额外调用轨迹生成器来生成一个从未交给控制器的未来目标。未来螺旋任务如果还需要“连续解析轨迹在当前时刻的误差”，应单独增加该参考量，不能混淆两种定义。

为兼容原日志，JSON 顶层 `time_s` 仍表示步首，`state` 仍是控制所用的步首状态，`post_step_state` 是步末仿真真值。绘制 `basic` 数据请使用它自己的 `sample_time_s`。

## 加速度来源

主记录加速度每个物理步都计算：

```text
acceleration_w = (velocity_w_end - velocity_w_start) / interval_dt
```

计算使用本步两端的实际仿真质心速度，不使用目标加速度、估计器输出或上一次写入文件的速度。因此即使改变日志降采样频率，加速度也不会误用错误时间间隔。复位后的第一条记录使用复位状态和第一次物理步结束状态，不补一个虚构零值。

这是**区间平均运动加速度**，包含重力在运动中造成的影响，不是 IMU 比力。若要模拟加速度计，还需要传感器坐标、重力处理、偏置和噪声模型。

`acceleration_native_w_m_s2` 独立记录 PhysX 提供的加速度，使用各刚体质量加权到整机质心；只有求解器数据可用且已经完成相应物理步时才记录。未初始化/不可用为 null，CSV 中为空，不把它当作零。`acceleration_native_source` 标明来源。两种加速度的采样/数值定义不同，不能保证逐点完全相等。

世界系角加速度由世界系角速度差分计算；若输出为机体系，则将该世界系平均量转到步末机体系。不能直接拿两个不同姿态下的机体系分量相减当作同一世界方向的变化。

## 电机转速来源

本机型 USD 中旋翼通过固定关节连接，没有可直接读取的真实旋转关节编码器。原生推进器内部使用：

```text
thrust_state = sampled_kf × rps²
```

日志直接读取积分后的 RPS 电机状态，再换算 rpm、rad/s。实际推力由每个电机**自己的当次采样系数**乘以实际 RPS² 得到；目标转速与实际转速分开记录。

旋翼顺序固定为后左 BL、后右 BR、前左 FL、前右 FR。记录值为非负幅值；配置里的反扭矩正负号不被当作已测量的机械转速方向。

`motor_speed_source` 明确说明这是使用原生积分公式演化的转速状态，不是实机编码器数据。`motor_thrust_command_n` 和 `motor_thrust_applied_n` 分开保留，便于观察电机响应滞后和饱和。

## 力和力矩要区分阶段

| 字段前缀 | 表示什么 |
|---|---|
| `force_command` / `torque_command` | 控制器希望实现的整机力/力矩 |
| `force_allocated` / `torque_allocated` | 考虑四桨推力约束后分配出的目标 |
| `force_motor` / `torque_motor` | 原生电机动态更新后实际输出推力对应的整机力/力矩 |
| `force_disturbance` / `torque_disturbance` | 显式扰动和气动模型产生的附加力/力矩 |
| `force_submitted` / `torque_submitted` | 逐旋翼提交量与显式附加作用关于整机质心的合计，仅为诊断合计，不作为整机控制量提交 |
| `force_net` / `torque_net` | 从本物理步实际运动推导的区间平均总合力/合力矩 |

提交值**不包含** PhysX 自行计算的重力、碰撞接触和原生阻尼，所以不把它叫作所有物理作用的总和。力/力矩命令和提交值在物理步开始时的机体系中表达；运动推导的净值在世界系表达。

对当前固定连接的刚体总成，净作用使用：

```text
force_net_world = mass × Δ(velocity_world) / Δt
angular_momentum_world = R_body_to_world × inertia_com_body × angular_velocity_body
torque_net_world = Δ(angular_momentum_world) / Δt
```

角动量使用完整惯量张量和两个端点的实际姿态，保留转动耦合。这些是由运动反推的量，不是独立力传感器测量；也不能作为独立证据验证同一运动数据是否正确。

## 记录频率与验证范围

当前 `logging.every_n_steps: 1`，默认每个物理步记录一次，配合 `simulation.dt: 0.005` 为每秒仿真时间 200 条。若以后增大该间隔，只降低写入频率，计算仍在每个物理步进行，不会变成跨多个日志行的差分。JSON 的结束/异常事件不会混进 CSV 数据行。

缺失读数写 null/空单元格；非有限数、错误向量长度会报错，不悄悄填零。字段来源、单位和时间说明同时写入机器可读 schema，便于将数据交给导师或后续分析程序。

当前验证覆盖数值计算、转速状态与单位换算、时间配对、复位、坐标系、JSON/CSV 输出契约；开发机器没有完整 Isaac Sim 运行栈，仍需在服务器实际运行后验证真实日志和飞行表现。

## Helix 任务新增 JSONL 诊断

`motor_command_rps` 是当步四电机命令（经过启动包络及运行上下限），`backend.commanded_motor_speed_rps` 为执行器接受的目标，`basic.motor_speed_rps` 为动态响应后的实际转速。`startup_fraction` 是 0..1 的起转包络，`motor_command_thrust_n` 对应接受转速的静态推力目标；`allocated_thrust_n` 则保留启动缩放前的闭环分配结果。

`setpoint.jerk_w`、`setpoint.snap_w` 记录解析参考导数；`controller_force_derivative_source` 说明解析参考/数值反馈或限幅后全数值差分路径。`mission_phase` 为 `delay / takeoff / spiral / hold`，其中 `spiral` 历史名称特指三维 helix。`mission` 保存 post-step 真值误差、连续达标时间、成功及超时标志。这些诊断位于 JSONL；基础 CSV 仍保留导师要求的运动与动力数据字段。
