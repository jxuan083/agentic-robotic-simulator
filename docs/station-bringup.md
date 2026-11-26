# 比賽機台上機檢查：Jetson Thor + 比賽用 SO-ARM101

給在比賽機台上跑的 Claude（或操作者）照做。目的：確認環境能跑我們的控制程式、做比賽手臂的 LeRobot 校正，並用尺量驗證關節映射。結果帶回實驗室，用來準備決賽。

先讀 repo 根目錄的 `AGENTS.md`。另外：
- 不推到 main。
- 原始校正檔和量測不進 git。
- 沒有操作者在旁確認，不開扭力。

本輪**沒有 global camera（D455）**。和 D455 相關的步驟（`d455`、`reach`、`d455-capture`）先跳過。

## 0. 取得程式

```bash
git clone -b feat/real-teleop-record-replay https://github.com/jxuan083/agentic-robotic-simulator.git
cd agentic-robotic-simulator
git log --oneline -1          # 記下 commit
```

## 1. 系統資訊（唯讀）

把下列輸出存進 `handoff/thor-<YYMMDD>/system.txt`：

```bash
uname -a; cat /etc/nv_tegra_release; dpkg -l | grep -E 'nvidia-jetpack|cuda-toolkit' | head
nvidia-smi || tegrastats --interval 1000 | head -2
python3 -V; df -h ~; free -h; nproc
ls -l /dev/serial/by-id/ /dev/v4l/by-path/ 2>&1; lsusb
ping -c1 pypi.org >/dev/null && echo "internet ok"
```

## 2. 環境

實驗室的版本是：Python 3.10、lerobot 0.4.4、feetech-servo-sdk 1.0.0、numpy 2.2.6、opencv-python-headless 4.12、pillow、torch 2.7.1。

**安裝**：用 conda/mamba（aarch64 版）或 uv 建立 Python 3.10 環境，裝 `lerobot[feetech]==0.4.4`。
- 裝不起來時，依序試：Python 3.12 → lerobot 最新版。記下實際裝成的版本和遇到的錯誤。
- torch 要用 Thor／JetPack 對應的 CUDA 版本（NVIDIA 或 jetson-ai-lab 的 wheel）。

**驗證**：

```bash
python -c "import lerobot, scservo_sdk, numpy, cv2, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"
python -c "import torch,time; x=torch.randn(4096,4096,device='cuda'); torch.cuda.synchronize(); t=time.time(); [x@x for _ in range(20)]; torch.cuda.synchronize(); print('matmul ms', (time.time()-t)/20*1000)"
```

**順便確認**：`pip install mujoco` 能不能裝（只測匯入，不是必要）。

**設定 `PY`**：之後的 `tools/lab_day.sh` 都要設 `PY=<這個環境的 python>`。

## 3. 裝置路徑

這輪只有 follower（和腕相機），不建 `outputs/single.local.yaml`。`joints`／`torque` 改用環境變數指定 follower：

```bash
export PY=<這個環境的 python>
export FOLLOWER_PORT=/dev/serial/by-id/<比賽手臂>   # ls -l /dev/serial/by-id/
export FOLLOWER_ID=wowrobo_follower
```

用 `v4l2-ctl --list-devices`、`v4l2-ctl -d <dev> --list-formats-ext` 查腕相機的型號、解析度和幀率，並記下 `/dev/v4l/by-path/...`。

## 4. 馬達（唯讀，扭力關）

用 lerobot 的 `FeetechMotorsBus`，或 `scservo_sdk` 直接讀，不寫入任何暫存器。逐顆 ID 1–6 讀：
- 型號（model number）
- 韌體版本
- `Present_Voltage`
- `Present_Temperature`
- `Torque_Enable`

記到 `motors.txt`。

**為什麼要讀**：實驗室 follower 是 STS3215，6 顆同型號。比賽手臂若混用不同減速比或電壓（7.4 V / 12 V），之後的速度上限要跟著調。

## 5. LeRobot 校正（扭力關，操作者手動擺）

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=<by-id> --robot.id=wowrobo_follower
```

流程：
1. 先擺到 LeRobot 文件的中間姿勢，按 Enter。
2. 逐軸轉滿全程，按 Enter。

有 leader 時，用 `--teleop.type=so101_leader --teleop.id=wowrobo_leader` 做同樣的事。

**完成後**：
- 校正檔在 `~/.cache/huggingface/lerobot/calibration/robots/so_follower/wowrobo_follower.json`。複製一份到 handoff 目錄。
- 印出偏移，和實驗室手臂比：

```bash
$PY -c "import sys,json; sys.path.insert(0,'src'); from agentic_sim.real_kinematics import calibration_offsets, find_calibration; print(calibration_offsets(json.load(open(find_calibration('wowrobo_follower')))))"
```

實驗室手臂的偏移是 `+5.10 −1.19 −6.07 −7.43 +0.04`（pan、lift、elbow、wrist_flex、roll，度）。差幾度屬正常。

## 6. 關節映射驗證：用尺量（扭力關，這輪最重要）

```bash
tools/lab_day.sh joints      # 用第 3 節的 PY / FOLLOWER_PORT / FOLLOWER_ID
```

`joints` 是唯讀即時讀數。每一行會印出：
- LeRobot 六軸角度；
- `EE(calib) mm`：用校正檔映射算出的夾取中心位置。

**座標系**：原點在 follower 基座底面中心，+X 正前方，+Y 左，+Z 上。

**量測方式**：
- 在桌上用膠帶標出 +X 軸，並標出各點。
- 量測點要是一顆約 30 mm 的方塊。手扶夾爪，讓它垂直朝下，兩指夾在方塊兩側，就像要夾起它。此時夾取中心約在方塊中心，z ≈ 15 mm。
- 量好後輸入標籤存檔。

| 標籤 | 方塊中心位置 (x, y) mm | 預期 EE(calib) |
|---|---|---|
| `c200_0` | (200, 0) | x≈200, y≈0, z≈15 |
| `c250_0` | (250, 0) | x≈250 |
| `c200_l80` | (200, +80) | y≈+80 |
| `c200_r80` | (200, −80) | y≈−80 |
| `c160_l120` | (160, +120) | |
| `park` | 停放姿勢（見下） | 實驗室手臂 ≈ (180, 2, 67) |
| `middle` | 校正時的中間姿勢 | |

**停放姿勢**：手臂收起靠在支撐上，就是實驗室的停放位。實驗室手臂在這個姿勢的讀數約為 `1.7 −109.1 97.3 66.1 3.0 0.6`（pan、lift、elbow、wrist_flex、roll、gripper）。
- 比賽現場的停放支撐若不同，拍照，並量支撐高度。

**通過標準**：EE(calib) 和尺量的 xy 誤差 ≤ 15 mm，z 的誤差 ≤ 20 mm。實驗室手臂因下垂約高 20–30 mm，這裡手扶著不會下垂。

若方向相反（例如 y 正負顛倒）或誤差超過 30 mm：
- 不要改程式；
- 記下讀數，拍照（俯視、側視），帶回實驗室分析。

樣本存在 `outputs/<date>-joints/samples.jsonl`。

## 7. 腕相機

拍一張腕相機畫面，存下解析度與實際幀率：
- 對著白桌面；
- 對著一顆方塊，夾爪張開。

並拍照記錄相機在夾爪上的安裝位置：正面、側面各一張。

## 8.（可選，需要操作者在旁）小幅扭力測試

確認以下條件，並得到操作者口頭同意後才做：
- 手臂在停放位，或已用手扶在安全姿勢；
- 工作區淨空；
- 操作者知道停止方式：Ctrl-C，扭力會全關。

**測試內容**：寫一個短腳本，逐軸從目前讀數往 ± 方向移 5°，速度 ≤ 20°/s，每軸回到原位，最後關扭力。在 `motion.txt` 記下：
- 是否順暢；
- 指令與實際的差；
- 夾爪開到 40% 和閉到 8% 時的讀數。

收尾一定要跑 `tools/lab_day.sh torque`，確認每軸扭力都是 0。

## 9. 帶回

`handoff/thor-<YYMMDD>/` 內放：

| 檔案 | 內容 |
|---|---|
| `system.txt` | 第 1 節的系統資訊 |
| `env.txt` | 安裝方式、版本、錯誤 |
| `motors.txt` | 第 4 節的馬達讀數 |
| 校正檔 JSON | follower，以及 leader（有帶時） |
| `offsets.txt` | 第 5 節印出的偏移 |
| `samples.jsonl` | 第 6 節的量測樣本 |
| 照片 | 停放支撐、相機安裝、尺量擺法 |
| `motion.txt` | 若有做第 8 節 |
| `README.md` | 一頁摘要：哪些通過、哪些不通過、待解問題 |

打包成 `thor-<YYMMDD>.zip`，帶回實驗室機器放在 `AllenHZL/`。
