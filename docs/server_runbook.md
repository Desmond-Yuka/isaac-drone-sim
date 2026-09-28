# 服务器试飞清单：开机 → WebRTC → 回车起飞

服务器 `uc1` 当前公网 IP 为 `117.50.205.33`；实例重启后若 IP 变化，下面命令和本地 `~/.ssh/config` 中 `uc1` 的 HostName 都要同步修改。

## 首次准备

```bash
sudo apt install git-lfs && git lfs install
git clone https://github.com/Desmond-Yuka/isaac-drone-sim.git ~/isaac-drone-sim
cd ~/isaac-drone-sim
source ~/isaacsim-6.1/env_isaacsim/bin/activate
python -m pip install -e '.[plot]'          # 录像另加 '.[video]'
```

在 Isaac Sim 的 Script Editor 里使用本仓库的模型时：

```python
import sys; sys.path.insert(0, "/home/ubuntu/isaac-drone-sim")
from isaac_drone.assets import ARL_ROBOT_1_USD
```

## 每次试飞

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
python -m isaac_drone validate --config configs/helix.yaml   # 可选：几秒内检查配置和资产
```

**4. 启动 Isaac Sim，加载场景和无人机**

```bash
mkdir -p runs && PYTHONUNBUFFERED=1 PUBLIC_IP=117.50.205.33 python -m isaac_drone run --backend isaaclab \
    --config configs/helix.yaml --livestream 1 --wait-for-start 2>&1 | tee runs/helix_$(date +%m%d-%H%M).log
```

等终端出现 `Scene ready: connect the stream client, then press Enter here to start.`（通常约 1 分钟；开机后首次运行需编译着色器，可能 2–3 分钟）。此时无人机停在地上，物理不推进。

**5. 连 WebRTC**：客户端填 Server `117.50.205.33`、Signal `49100`、Stream `47998`，点 Connect，看到操作页面和无人机。

**6. 开始运行**：点一下运行命令的终端，确保输入焦点在它上面，按**回车**。任务为 20 s 仿真时间，默认按真实时间播放；物理和记录本身慢于实时时会如实变慢（uc1 串流实测约 0.3×，约 70 s，见输出里的 `playback`）。结束后打印 `{"success": ...}`，程序退出，串流随之断开。再飞一次需重新执行第 4 步。

换控制器或轨迹只需换 `--config` 或加 `--set`，例如 `--config configs/helix_cascaded_pid.yaml`。无画面批量运行用 `--visualizer none`（不串流时不需要 `--wait-for-start`）。

**7. 看结果、关机**

```bash
python -m isaac_drone summarize              # 最新一次运行的分阶段误差、倾角、电机转速、饱和比例
ls runs/<运行目录名>/plots/                    # 飞完自动生成的英文图；重画：python -m isaac_drone plot
```

拷回本地（在本地仓库根目录运行，增量同步全部运行记录到本地 `runs/`）：`rsync -az --info=progress2 uc1:isaac-drone-sim/runs/ runs/`。本地同样可以 `summarize`、`plot`、`compare`。不用时在控制台关机。

## 注意

- 不要同时运行独立的 Isaac Sim 串流程序（`isaacsim.exp.full.streaming`），两者会抢 49100 端口；同一时间只能有一个进程占用它。
- 停止程序用 `Ctrl+C`；`Ctrl+Z` 只是挂起，进程仍占着显存和端口。
- 终端粘贴用 `Ctrl+Shift+V`。
- WebRTC 同时只能连一个客户端；连不上时先关闭所有旧客户端窗口再连。
- Isaac Lab 3.0 已去掉 `--headless`，无画面运行用 `--visualizer none`。
