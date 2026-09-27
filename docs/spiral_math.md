# 三维圆柱螺旋上升（Helix）的轨迹与控制推导

本实现生成的是**半径恒定、绕竖直轴旋转、同时持续爬升的三维圆柱螺旋线（helix）**。它的水平投影为圆，竖直投影持续升高；半径不会随时间扩大。文件名 `spiral.py` 和 `SpiralTrajectory` 保留旧入口，`HelixTrajectory` 是同一实现的清晰别名。配置 `kind: helix` 与兼容名称 `spiral` 使用同一条三维轨迹，参数继续放在 `trajectory.spiral` 段。

这里推导的是控制参考，不能据此声称真实无人机完全沿曲线飞行。实际运动还受完整刚体动力学、控制误差、电机滞后、推力饱和、地面接触及已启用扰动影响。下面的参数和示例是工程任务设定，不是 ARL 实机标定结果。

## 1. 从实际质心开始的四个阶段

`reset(initial_state)` 记录实际整机质心位置 $p_0$、实际朝向和仿真时刻 $t_0$。位置始终指**整机质心**，不是 USD 根链接原点。当前入口根据碰撞几何范围和显式离地间隙安置机体，并按配置复位速度；它尚未实现依据持续接触或静止时间判断的 ground-settling gate。轨迹会拒绝线速度或角速度超过显式阈值的初态，不会把移动状态偷偷改写为静止。

记延时为 $T_d$，起飞高度为 $h$，起飞时长为 $T_o$，三维螺旋阶段时长为 $T_h$：

| 阶段 | 相对复位时刻的区间 | 参考行为 |
|---|---|---|
| `delay` | $[0,T_d)$ | 保持 $p_0$ 和复位时实际偏航 |
| `takeoff` | $[T_d,T_d+T_o)$ | 原地竖直上升 $h$，平滑调整偏航 |
| `spiral` | $[T_d+T_o,T_d+T_o+T_h)$ | 半径恒定的三维圆柱螺旋上升 |
| `hold` | $[T_d+T_o+T_h,\infty)$ | 永久保持终点位置及最终偏航 |

`phase()` 的 `spiral` 返回值也是兼容名称，指的仍然是三维 helix。`mission_duration_s` 返回 $T_d+T_o+T_h$，不包含无限期保持时间。到达终点只意味着参考进入保持阶段，**不会关闭电机或停止控制器**。

`start_delay_s` 只是参考时间平移。当前运行入口在延时期间施加转速启动包络；接触稳定判定和自动起飞许可门控尚未实现。轨迹模块不操作电机、不推进物理时间，也不修改状态。

## 2. 九次平滑进度函数的来源

每个运动阶段使用归一化时间 $u=(t-t_{\mathrm{start}})/T$。在 $u\le0$ 时取进度 0，在 $u\ge1$ 时取进度 1；区间内取

$$
S(u)=126u^5-420u^6+540u^7-315u^8+70u^9.
$$

它满足十个端点条件：

$$
S(0)=0,\quad S(1)=1,\qquad
S^{(k)}(0)=S^{(k)}(1)=0,\quad k=1,2,3,4.
$$

因此位置及速度、加速度、jerk（加加速度）、snap（加加加速度）都能与静止段连续拼接。一个构造方法是让

$$
S'(u)=C u^4(1-u)^4.
$$

两端的四重零点保证一到四阶导数为零，积分归一化给出

$$
1=\int_0^1 S'(u)\,du=C\frac{4!4!}{9!},\qquad C=630.
$$

展开并积分便得到上面的九次多项式。它也是这些十个边界条件下唯一的至多九次多项式：如果存在两个解，其差在 0 和 1 都至少有五重零点，必须含因子 $u^5(u-1)^5$，与次数至多 9 矛盾。

**这是满足指定端点条件的九次 Hermite 插值，本文不把它称为 minimum-snap 最优解。** 没有求解或证明某个轨迹积分代价的全局最小值。

对归一化时间的前四阶导数为

$$
\begin{aligned}
S' &=630u^4(1-u)^4,\\
S''&=2520u^3(1-u)^3(1-2u),\\
S'''&=2520u^2(1-u)^2(3-14u+14u^2),\\
S''''&=15120u(1-u)(1-2u)(1-7u+7u^2).
\end{aligned}
$$

换成物理时间时，每阶都要除以相应的时间尺度：

$$
s_k(t)=\frac{1}{T^k}S^{(k)}(u),\quad k=1,2,3,4,\qquad s_0(t)=S(u).
$$

代码用导数的因式分解形式保持端点零值；位置在 $u>1/2$ 时利用 $S(u)=1-S(1-u)$ 求值，减少接近终点时多项式相减造成的数值误差。

## 3. 竖直起飞

起飞阶段的终点为

$$
p_o=p_0+h e_z,\qquad e_z=(0,0,1)^\mathsf T.
$$

使用该阶段的进度函数：

$$
p(t)=p_0+h s_0(t)e_z,\qquad p^{(k)}(t)=h s_k(t)e_z,\quad k=1,2,3,4.
$$

整个起飞阶段的水平位置保持 $p_{0,xy}$。首尾四阶运动导数均为零，且末位置恰好为下一阶段的起点。

## 4. 自动定位圆柱轴线，消除横向跳变

设半径 $R>0$、初始相位 $\theta_0$、有符号圈数 $N\ne0$、本阶段净上升高度 $H>0$。正圈数表示从世界 +Z 方向向下看逆时针，负圈数表示顺时针；圈数允许非整数。

圆柱轴线的水平位置由实际起点计算：

$$
c_{xy}=p_{0,xy}-R
\begin{bmatrix}\cos\theta_0\\\sin\theta_0\end{bmatrix}.
$$

因此 $p_0$ 位于圆周起点，而不是被误设成圆心后瞬间横移一个半径。令

$$
\Theta=2\pi N,\qquad
\theta(t)=\theta_0+\Theta s_0(t),
$$

则三维轨迹为

$$
p(t)=
\begin{bmatrix}
c_x+R\cos\theta(t)\\
c_y+R\sin\theta(t)\\
p_{o,z}+H s_0(t)
\end{bmatrix}.
$$

半径恒定，且

$$
\frac{dz}{d\theta}=\frac{H}{\Theta}
$$

在阶段内部恒定，说明它是一条几何上节距固定的圆柱螺旋线；改变时间进度只改变沿线的速度，不改变几何曲线。

终点为

$$
p_f=\begin{bmatrix}
c_x+R\cos(\theta_0+\Theta)\\
c_y+R\sin(\theta_0+\Theta)\\
p_{0,z}+h+H
\end{bmatrix}.
$$

若 $N$ 是整数，终点水平位置回到起点；非整数圈数则停在相应圆周相位。接口 `endpoint_position_w` 返回这个位置的副本。

## 5. 速度、加速度、jerk 和 snap 的完整解析式

定义随相位变化的单位径向和单位切向：

$$
e_r=(\cos\theta,\sin\theta,0)^\mathsf T,\qquad
e_t=(-\sin\theta,\cos\theta,0)^\mathsf T.
$$

令 $\theta_k=\Theta s_k$。由于 $\dot e_r=\theta_1e_t$、$\dot e_t=-\theta_1e_r$，逐次求导得到

$$
\begin{aligned}
v &= R\theta_1 e_t+Hs_1e_z,\\
a &= R(\theta_2e_t-\theta_1^2e_r)+Hs_2e_z,\\
j &= R\big[(\theta_3-\theta_1^3)e_t-3\theta_1\theta_2e_r\big]+Hs_3e_z,\\
s_{\mathrm{snap}} &= R\big[(\theta_4-6\theta_1^2\theta_2)e_t
 +(\theta_1^4-3\theta_2^2-4\theta_1\theta_3)e_r\big]+Hs_4e_z.
\end{aligned}
$$

这些式子包括全部三角函数链式求导项：不仅有切向加速，也有向心加速度及更高阶耦合项。代码将它们写入 `velocity_w`、`acceleration_w`、`jerk_w` 和 `snap_w`，不会用离散位置差分代替已知解析参考导数。

在 helix 起止处，$\theta_1\ldots\theta_4$ 和 $s_1\ldots s_4$ 全为零，因而四阶导数全部归零。与起飞及终点保持连接后，整段位置参考属于 $C^4$；五阶导数可以在连接处跳变，本文不宣称 $C^5$ 连续。

## 6. 为什么前半程加速、后半程减速

螺旋阶段的总路径长度为

$$
L=\sqrt{(R\Theta)^2+H^2}.
$$

因为径向切向与竖直方向相互正交，且 $s_1\ge0$：

$$
\|v(t)\|=L s_1(t)=\frac{630L}{T_h}u^4(1-u)^4.
$$

速度大小的时间导数是

$$
\frac{d\|v\|}{dt}=\frac{2520L}{T_h^2}u^3(1-u)^3(1-2u).
$$

所以 $0<u<1/2$ 时严格加速，$1/2<u<1$ 时严格减速，峰值在中点：

$$
v_{\max}=\frac{315L}{128T_h}.
$$

这一结论针对**速度大小**。中点虽然切向加速度为零，向心加速度仍然存在，不能因此把三维加速度设为零。增加阶段时长会降低速度峰值，并进一步降低各阶参考导数；它不是电机能力或跟踪误差的保证。

## 7. 连续偏航策略

初始偏航 $\psi_0$ 从实际姿态的机体 X 轴世界 XY 投影提取。若该投影退化为零，轨迹明确拒绝初始化；不会捏造一个朝向。

`fixed` 模式的目标偏航为 $\psi_h=\psi_0+\psi_{\mathrm{offset}}$。为了避免非零 offset 造成初始化跳变，延时阶段保持 $\psi_0$，起飞阶段使用

$$
\psi(t)=\psi_0+(\psi_h-\psi_0)s_0(t),\quad
\dot\psi=(\psi_h-\psi_0)s_1,\quad
\ddot\psi=(\psi_h-\psi_0)s_2.
$$

到达 helix 起点后偏航固定为 $\psi_h$。offset 按给出的有符号值执行；如果明确填写多圈角度，起飞阶段也会按该角度过渡。

`tangent` 模式在 helix 阶段沿**水平运动切线方向**加偏航偏置：

$$
\psi(t)\equiv\theta(t)+\operatorname{sgn}(N)\frac{\pi}{2}+\psi_{\mathrm{offset}}
\pmod{2\pi}.
$$

开始起飞前，先选择与 $\psi_0$ 最近的等价切向角作为 $\psi_h$，再用上述起飞偏航公式平滑过渡。螺旋阶段取连续、**不做 $[-\pi,\pi]$ 包裹**的偏航值：

$$
\psi(t)=\psi_h+\Theta s_0(t),\quad
\dot\psi=\Theta s_1,\quad \ddot\psi=\Theta s_2.
$$

终点保持最终累计偏航值，偏航速度和加速度都为零。零速端点虽然没有瞬时运动切线，但左右极限定义的目标朝向连续。

必须核对轨迹偏航峰值与控制器的前馈限制：tangent 螺旋阶段的峰值为 $315|\Theta|/(128T_h)$，起飞偏航峰值为 $315|\psi_h-\psi_0|/(128T_o)$。单纯缩短时间可能导致控制器限幅，不能仍宣称所有解析前馈被原样执行。

## 8. 与底层运动控制的关系

记实际质心位置和速度为 $p,v$，参考值为 $p_d,v_d,a_d$，真实整机质量为 $m$，世界重力向量为 $g$。未限幅时的位置反馈产生期望力

$$
F_d=m\big[a_d+K_p(p_d-p)+K_v(v_d-v)+K_i\eta-g\big].
$$

这里的 $K_p,K_v,K_i$ 按轴作用，积分状态 $\eta$ 有显式上限和防饱和反馈。质量来自实际载入的模型，不是假设每个旋翼无质量或质心位于根原点。

控制器先处理已配置的加速度和倾角约束。若未限幅且参考提供 jerk/snap，期望力导数拆为

$$
\dot F_d=m j_d+\frac{dF_{\mathrm{feedback}}}{dt},\qquad
\ddot F_d=m s_d+\frac{d^2F_{\mathrm{feedback}}}{dt^2},
$$

其中 $F_{\mathrm{feedback}}=F_d-ma_d$。参考贡献解析计算，反馈贡献利用状态历史作差分；不能因为有解析轨迹就把误差反馈的导数忽略掉。发生约束限幅时改为对实际受限的总期望力求数值导数。首次计算还没有反馈历史，这一部分导数未知，初始化为零；它不代表真实无人机反馈项的导数必然为零。

由期望力方向和偏航构造期望姿态 $R_d$：

$$
b_{3d}=\frac{F_d}{\|F_d\|},\quad
b_{2d}=\frac{b_{3d}\times(\cos\psi_d,\sin\psi_d,0)^\mathsf T}
{\|b_{3d}\times(\cos\psi_d,\sin\psi_d,0)^\mathsf T\|},\quad
b_{1d}=b_{2d}\times b_{3d},\quad R_d=[b_{1d}\ b_{2d}\ b_{3d}].
$$

零推力和方向退化时由控制模块的明确分支处理。此构造中的偏航是辅助水平航向；倾斜时不应把它误解为与实际姿态所有 Euler 角定义完全相同。

记完整质心惯量为 $J$、实际机体系角速度为 $\Omega$：

$$
e_R=\frac12(R_d^\mathsf TR-R^\mathsf TR_d)^\vee,\qquad
e_\Omega=\Omega-R^\mathsf TR_d\Omega_d.
$$

期望角速度和角加速度由期望旋转矩阵的一、二阶导数得到，控制力矩为

$$
M_d=-K_Re_R-K_\Omega e_\Omega
+\Omega\times(J\Omega)
+J\left(R^\mathsf TR_d\dot\Omega_d
-\Omega\times(R^\mathsf TR_d\Omega_d)\right).
$$

姿态增益遵循本地 Lee 控制器的**直接力矩增益**语义，不再额外乘惯量；惯量完整出现在陀螺项与角加速度前馈项中。总正向推力指令是

$$
f_d=\max(0,F_d^\mathsf TRe_3).
$$

随后有界分配器把期望力/力矩分配为每桨目标，转速/推进器层再按选定模型执行。以上公式不绕过电机动态、不直接写入物体位置，也没有将几何可行的曲线自动等同于动力学可跟踪的飞行任务。

## 9. 工程示例与接口

示例：延时 2 s；原地起飞 1 m、用时 4 s；半径 1 m、旋转 2 圈、额外上升 3 m、螺旋阶段 24 s。运动参考在 30 s 后进入永久保持，终点比初始实际质心高 4 m。螺旋路径长度约 12.92 m，速度峰值约 1.325 m/s；tangent 模式的螺旋偏航速率峰值约 1.289 rad/s。这些数值需要结合电机能力、倾角限制和验证结果进一步调整，不是已经完成的实机标定。

```python
from isaac_drone.trajectories.spiral import HelixTrajectory

trajectory = HelixTrajectory(config["trajectory"]["spiral"])
trajectory.reset(actual_initial_state)
reference = trajectory.sample(current_simulation_time_s)
phase = trajectory.phase(current_simulation_time_s)
endpoint = trajectory.endpoint_position_w
```

`sample()` 是不推进内部时间的纯采样接口，相同时间返回相同参考，允许回查记录。`time_s` 是绝对仿真时间，不能早于最近一次 reset 的时间；再次 reset 会建立新的实际起点和时间原点。配置检查入口为 `validate_spiral_config()`。独立数学回归位于 `tests/test_spiral.py`，包含四阶解析导数校验、端点连续性、正反方向、偏航连续性和速度峰值；这些测试不是 Isaac Sim 实际飞行验证的替代品。

## 10. 起飞前的必要可行性筛查

`validate_helix_feasibility()` 使用已经根据实际初态 reset 的轨迹、真实质量、完整质心惯量、实际分配矩阵和当前每桨推力上下限，拒绝明显无法执行的计划。它不会自动修改任务参数，也不会为了让检查通过而假设理想电机。

检查分为三部分：

1. **静态悬停条件。** 要求四个推力轴与当前控制器的机体 +Z 约定兼容，集体推力和三个姿态通道独立；在实际每桨上下限内，要求能够分配悬停所需力及零合力矩。质心偏移时，各桨悬停推力不一定相等；下限过高也可能使悬停不可行。
2. **解析偏航峰值。** 根据已知 $S'$ 的准确峰值，检查起飞偏航转换和 tangent 螺旋偏航的最大速度是否超过控制配置。这个检查覆盖两个阶段的全部连续时间，不是从有限采样中猜测峰值。
3. **有限样本的名义动力学检查。** 起飞阶段取 33 个时间点、螺旋阶段取 65 个时间点，包含各自端点和中点；合并共享边界并加入延时起点后，正常示例共 98 个不同时间点。检查参考加速度、所需倾角及名义前馈力矩能否在当前推力上下限内实现。

在每个名义检查点，设参考被准确跟踪，仅用于计算该参考所需的基本刚体力/力矩：

$$
F=m(a_d-g),\qquad f=\|F\|,\qquad
M=J\dot\Omega_d+\Omega_d\times(J\Omega_d).
$$

$R_d,\Omega_d,\dot\Omega_d$ 由解析 jerk/snap 和偏航导数计算；分配器检查目标机体系 wrench $(0,0,f,M_x,M_y,M_z)$ 是否在给定边界内可实现。姿态 PD 误差项此处为零只是**名义轨迹计算点的定义**，不是把实际跟踪误差假定为零，更不是省略真实仿真中的反馈控制。

这些输出刻意区分严格解析结果和样本统计：`exact_*yaw_rate_peak*` 是本时间律的解析峰值，`sampled_max_*` 和各桨推力余量只来自列出的检查点。数值残差容差用于浮点求解误差，不是人为增加的动力学余量。

**通过筛查不构成连续时间可行性证书，也不保证闭环稳定或无碰撞飞行。** 采样间隔中仍可能存在未采到的极值；当前筛查没有验证电机上升/下降动态、指令速率、电池瞬态、传感器噪声、扰动或反馈纠偏余量。报告明确返回 `continuous_time_certificate: false`、`closed_loop_guarantee: false`、`motor_dynamic_feasibility_checked: false`。真实电机限制继续在执行器模型中生效，后续必须结合实际仿真轨迹和日志判断跟踪效果。

## 11. 几何控制来源与理论适用范围

姿态误差、角速度坐标变换及包含完整惯量的几何控制结构参考 Lee、Leok、McClamroch 的 [Geometric Tracking Control of a Quadrotor UAV on SE(3), CDC 2010](https://mathweb.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf)，并核对了本地 Isaac Lab Lee 控制器源码。论文的机体第三轴指向推力反方向；本项目采用机体 +Z 为正推力方向，因此这里按本项目约定写出平移和推力公式，不能不转换坐标就照搬符号。

本文没有把该连续刚体控制理论的稳定性结果直接声明为本项目保证。本地实现包含离散控制周期、有限电机推力、实际电机动态、倾角及指令限幅、数值反馈导数、接触和可选扰动；这些实现条件需要单独验证。九次轨迹的端点光滑性、名义前馈可行性筛查、连续控制理论和实际离散仿真效果是不同层面的结论。

## 12. 从机体控制目标分配到四个旋翼

令 $r_i$ 为第 $i$ 个旋翼推力作用点在根机体系中的位置，$c$ 为真实整机质心在同一坐标系中的位置，$a_i$ 为该旋翼推力单位轴。令 $f_i\ge0$ 为其推力大小，$\sigma_i\in\{-1,+1\}$ 为沿该轴的反扭矩符号，$k_{\tau i}$ 为反扭矩与推力之比，单位 m。

该旋翼对整机质心的贡献为

$$
F_i=f_i a_i,\qquad
M_i=(r_i-c)\times(f_i a_i)+\sigma_i k_{\tau i}f_i a_i.
$$

因此分配矩阵的第 $i$ 列为

$$
A_i=\begin{bmatrix}
a_i\\
(r_i-c)\times a_i+\sigma_i k_{\tau i}a_i
\end{bmatrix},\qquad
w=Af,\quad f=(f_1,f_2,f_3,f_4)^\mathsf T.
$$

这里所有位置、轴线和质心来自实际加载结果；$k_{\tau i}$ 从相应原生执行器配置读取。没有假定四个力臂长度相等，也不把 YAML 上游参考矩阵的臂长直接代入。$\sigma_i$ 是反扭矩符号约定；它与转速大小分开记录，不能把一个负号直接解释成某个未经确认的实机接线方向。

分配器求解

$$
\min_{\underline f\le f\le\overline f}
\|W(Af-w_d)\|_2^2+\lambda\|f-f_{\mathrm{prev}}\|_2^2.
$$

$W=\mathrm{diag}(\mathrm{weights})$ 对力和力矩残差作数值缩放，$\lambda\ge0$ 是相对上一条有效推力指令的平滑惩罚。权重按残差乘数解释，因此二次代价中的系数是各权重的平方；把 N 与 N·m 混合进代价需要显式的尺度/优先级选择。非零正则化允许为了减小指令变化而保留少量 wrench 残差，不能再把它说成严格无误差分配。

四个变量各有“在下界、自由、在上界”三种活动状态。程序枚举至多 $3^4=81$ 种组合，在每个组合内解自由变量的最小二乘，保留满足边界且总代价最小的解。因此不会先求伪逆再粗略截断所有电机。实际分配矩阵、每桨边界、分配结果和未实现残差均保留在日志中。

## 13. 推力到转速的单位和转换

每个旋翼使用复位后真实采样的 $k_{f i}>0$：

$$
f_i=k_{f i}n_i^2,\qquad
n_{i,\mathrm{cmd}}=\sqrt{\frac{f_{i,\mathrm{cmd}}}{k_{f i}}}.
$$

这里 $n_i$ 的单位是 **rps（转/秒）**，不是 rpm，也不是机体角速度。转换关系为

$$
\mathrm{rpm}_i=60n_i,\qquad \omega_{r,i}=2\pi n_i\quad[\mathrm{rad/s}].
$$

若改用转子角速度写 $f_i=k_{\omega i}\omega_{r,i}^2$，相应系数必须改为

$$
k_{\omega i}=\frac{k_{f i}}{(2\pi)^2}.
$$

不能把 rps² 系数直接套入 rad²/s² 公式，否则会产生 $(2\pi)^2$ 的尺度错误。转子的 $n_i,\omega_{r,i}$ 与整机姿态角速度 $\Omega$ 也不能混用。

运行上下限转换为

$$
n_{i,\min}=\sqrt{\frac{f_{i,\min}}{k_{f i}}},\qquad
n_{i,\max}=\sqrt{\frac{f_{i,\max}}{k_{f i}}}.
$$

转速转换先保留原生正指令的运行下限/上限，明确的零指令则保持为零。不同电机采样到不同 $k_f$ 时，达到相同推力需要不同转速；不能用统一“典型电机系数”替代这些实际样本。

## 14. 非对称电机动态及离散积分

`NativeRpsActuator` 直接保存转速状态 $n_i$，复用本地原生推进器的 rate 函数及 RK4/Euler 积分函数。每个物理步根据受限目标与当前转速选择

$$
\tau_i=
\begin{cases}
\tau_{\mathrm{dec},i},&n_{i,\mathrm{target}}<n_i,\\
\tau_{\mathrm{inc},i},&n_{i,\mathrm{target}}\ge n_i.
\end{cases}
$$

该步的混合系数为

$$
\alpha_i=
\begin{cases}
1/(\Delta t+\tau_i),&\texttt{use\_discrete\_approximation=true},\\
1/\tau_i,&\texttt{use\_discrete\_approximation=false}.
\end{cases}
$$

原生 rate 函数对应

$$
G_i(e)=\operatorname{clip}(\alpha_i e,-q_i,q_i),\qquad
\dot n_i=G_i(n_{i,\mathrm{target}}-n_i).
$$

字段 `max_thrust_rate` 沿用上游名称，但这里的 $q_i$ 实际限制**转速变化率 rps/s**，不能按字段名字误当作 N/s。上述带 $\Delta t$ 的混合方式也是当前原生模型的明确选择，不冒称已辨识的电磁电机方程。

Euler 更新为 $n_i^+=n_i+\Delta t\,G_i(e_i)$。RK4 在本步选定的 $\tau_i,\alpha_i$ 和目标下计算

$$
\begin{aligned}
k_1&=G_i(e_i),\\
k_2&=G_i(e_i-\tfrac12\Delta t\,k_1),\\
k_3&=G_i(e_i-\tfrac12\Delta t\,k_2),\\
k_4&=G_i(e_i-\Delta t\,k_3),\\
n_i^+&=n_i+\frac{\Delta t}{6}(k_1+2k_2+2k_3+k_4).
\end{aligned}
$$

之后把数值状态限制在 $[0,n_{i,\max}]$，并由**实际状态**产生

$$
f_{i,\mathrm{actual}}=k_{f i}(n_i^+)^2.
$$

因此目标转速不等于实际转速，目标推力也不等于当前推力。启动/停转时，实际转速可以暂时低于正指令的最小运行转速；不能凭运行下限给静止转子硬加一个非零推力。零目标使电机按下降时间常数减速，不会瞬间抹掉已有转速。$k_f$ 和上升/下降时间常数的原生随机采样继续保留并记录。

这一层是可执行的转速—推力经验模型，不是 CFD，也不是已经包含 ESC、电流、电压、桨叶流场或自由转动关节的完整模型；新增气动和电气影响仍需相应标定数据与模型。

## 15. 启动包络和每个旋翼的实际施力

延时阶段使用同一个九次函数构造未限幅转速启动包络：

$$
\widetilde n_{i,\mathrm{cmd}}(t)=S((t-t_0)/T_d)\,n_{i,\mathrm{allocated}}(t).
$$

随后按真实运行约束处理：

$$
n_{i,\mathrm{target}}=
\begin{cases}
0,&\widetilde n_{i,\mathrm{cmd}}=0,\\
\operatorname{clip}(\widetilde n_{i,\mathrm{cmd}},n_{i,\min},n_{i,\max}),
&\widetilde n_{i,\mathrm{cmd}}>0.
\end{cases}
$$

$S$ 本身端点平滑，但存在正运行下限时，第一个正目标可能直接落到 $n_{\min}$；所以这里**不宣称受限后的转速目标仍严格 $C^4$**。实际转速继续由上一节的电机动态积分。包络之后实际可请求的推力及 wrench 会反馈给控制器的积分防饱和逻辑，避免把启动期间尚未施加的力当作已实现控制量。

后端把每个 $f_{i,\mathrm{actual}}a_i$ 和相应反扭矩施加到该旋翼刚体的局部推力原点，由 PhysX 通过固定关节传递至整机。电机输出不通过“向机体根链接直接施加控制合力”绕过旋翼作用点；根刚体只接收单独配置的外部扰动及其正确的质心力矩转换。

日志中的

$$
w_{\mathrm{motor}}=A f_{\mathrm{actual}},\qquad
w_{\mathrm{diagnostic}}=w_{\mathrm{motor}}+w_{\mathrm{external}}
$$

是围绕整机质心汇总的诊断值，**不会再次整体提交到根刚体**。否则会把已通过四个旋翼提交的力重复施加。真实净合力还包含重力、接触和原生阻尼等物理求解项，应与日志中的运动推导净力/净力矩区分。

本节实现依据为 `isaac_drone/actuation.py`、`isaac_drone/backends/isaaclab.py`、`isaac_drone/control/allocation.py` 以及本地 `IsaacLab/source/isaaclab_contrib/isaaclab_contrib/actuators/thruster.py`。这组公式描述当前实际执行路径，不以理想瞬时电机替换原生动态。
