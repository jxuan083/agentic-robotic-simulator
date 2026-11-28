# 比賽手臂上機檢查：Jetson Thor + 比賽用 SO-ARM101

在工作坊只用 Thor 和比賽手臂（沒有 global camera，也不是比賽場地），量決賽前需要的東西：
- Thor 能不能跑我們的程式；
- 比賽手臂的校正與關節映射；
- 馬達與腕相機規格。

場地相關的項目留到比賽現場再做，包括停放位、路線和 D455 位姿。

先讀 repo 根目錄的 `AGENTS.md`。另外：
- 不推到 main；
- 不改 `src/`；
- 全程扭力關，不開扭力。

## 0. 取得程式

```bash
git clone -b feat/real-teleop-record-replay https://github.com/jxuan083/agentic-robotic-simulator.git
cd agentic-robotic-simulator && git log --oneline -1
mkdir -p handoff/thor-$(date +%y%m%d)
```

## 1. Thor 系統（唯讀）→ `system.txt`

```bash
uname -a; cat /etc/nv_tegra_release; dpkg -l | grep -E 'nvidia-jetpack|cuda-toolkit' | head
python3 -V; df -h ~; free -h; nproc
ls -l /dev/serial/by-id/ /dev/v4l/by-path/ 2>&1; lsusb
```

## 2. 環境 → `env.txt`

**實驗室版本**（照這組裝）：

| 套件 | 版本 |
|---|---|
| Python | 3.10 |
| `lerobot[feetech]` | 0.4.4 |
| feetech-servo-sdk | 1.0.0 |
| numpy | 2.2 |
| opencv-python-headless | 4.12 |
| pillow | — |
| torch | 2.7 |

**安裝**：
- 用 aarch64 的 conda/mamba 或 uv 建立環境。
- torch 要用 Thor／JetPack 對應的 CUDA wheel（NVIDIA 或 jetson-ai-lab 的）。
- 裝不起來時，改用 Python 3.12 或 lerobot 最新版。記下實際版本和錯誤。

**驗證**：

```bash
export PY=$(which python)
$PY -c "import lerobot, scservo_sdk, numpy, cv2, torch; print(lerobot.__version__, torch.__version__, torch.cuda.is_available())"
$PY -c "import torch,time; x=torch.randn(4096,4096,device='cuda'); torch.cuda.synchronize(); t=time.time(); [x@x for _ in range(20)]; torch.cuda.synchronize(); print('matmul ms', (time.time()-t)/20*1000)"
```

## 3. 馬達（唯讀）→ `motors.txt`

```bash
export FOLLOWER_PORT=/dev/serial/by-id/<比賽手臂>   # 第 1 節的列表
export FOLLOWER_ID=wowrobo_follower
$PY - <<'PY'
import os, scservo_sdk as s
port = s.PortHandler(os.environ["FOLLOWER_PORT"]); port.openPort(); port.setBaudRate(1_000_000)
pk = s.PacketHandler(0)
for i in range(1, 7):
    model, res, _ = pk.ping(port, i)
    if res != 0:
        print(i, "no reply"); continue
    fw = [pk.read1ByteTxRx(port, i, a)[0] for a in (0, 1)]
    volt, temp, torque = (pk.read1ByteTxRx(port, i, a)[0] for a in (62, 63, 40))
    print(f"id {i}: model {model} fw {fw[0]}.{fw[1]} {volt / 10:.1f} V {temp} C torque {torque}")
port.closePort()
PY
```

實驗室 follower 是 6 顆同型號的 STS3215（model 777），12 V 供電。若比賽手臂型號或電壓不同，之後的速度上限要調。

## 4. LeRobot 校正 → 校正檔與 `offsets.txt`

**校正**：

```bash
lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=wowrobo_follower
```

1. 照 LeRobot 文件擺到中間姿勢，按 Enter。
2. 逐軸轉滿全程，按 Enter。

**複製校正檔**：把 `~/.cache/huggingface/lerobot/calibration/robots/so_follower/wowrobo_follower.json` 複製到 handoff 目錄。

**印出偏移**：

```bash
$PY -c "import sys,json; sys.path.insert(0,'src'); from agentic_sim.real_kinematics import calibration_offsets, find_calibration; print(calibration_offsets(json.load(open(find_calibration('wowrobo_follower')))))"
```

實驗室手臂的偏移是 `+5.10 −1.19 −6.07 −7.43 +0.04`（pan、lift、elbow、wrist_flex、roll，單位度）。差幾度屬正常。

## 5. 關節映射檢查 → `samples.jsonl`

```bash
tools/lab_day.sh joints      # 唯讀，用上面的 PY / FOLLOWER_PORT / FOLLOWER_ID
```

**讀數與座標系**：
- 每行印出 LeRobot 六軸角度，和 `EE(calib) mm`（用校正檔映射算出的夾取中心位置）。
- 座標系：+X 為手臂正前方，+Y 為它的左邊，+Z 為上。

**存檔方式**：手扶穩後輸入標籤，按 Enter 存一筆。

### 5a 方向（不存檔）

| 動作 | `EE(calib)` 應該 |
|---|---|
| 手臂往它的左邊轉 | y 變大 |
| 往前伸 | x 變大 |
| 抬高 | z 變大 |

### 5b 桌面點

**擺法**：夾爪閉合、垂直朝下，指尖輕觸桌面。依序在近、遠、左、右、左前、右前各存一筆。

**標籤**：`table_1` … `table_6`。

### 5c A4 四角（可選）

**擺法**：
1. 一張 A4 紙橫放在手臂前方，紙的中心約在手臂前方 20 cm。
2. 指尖依同樣擺法，繞一圈依序觸碰四個角。

**標籤**：`a4_1` … `a4_4`。

### 判讀

```bash
$PY - outputs/*-joints/samples.jsonl <<'PY'
import json, sys, itertools, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
p = {r["label"]: np.array(r["ee_calib_mm"]) for r in rows}
t = [v for k, v in p.items() if k.startswith("table")]
if t: print("table z: mean %.1f  spread %.1f mm" % (np.mean([v[2] for v in t]), np.ptp([v[2] for v in t])))
a = [p[k] for k in sorted(p) if k.startswith("a4_")]
for (i, u), (j, w) in itertools.combinations(enumerate(a, 1), 2):
    print(f"a4_{i}-a4_{j}: xy {np.hypot(*(u - w)[:2]):.0f} mm")
PY
```

**通過條件**：
- 方向全對；
- 桌面點 z 的變動範圍 ≤ 10 mm；
- A4 的邊長應為 297／210 mm、對角線 364 mm，誤差 ≤ 8 mm。

**不通過時**：不改程式，記下讀數並拍照（俯視、側視）帶回。

## 6. 腕相機（有裝才做）→ `wristcam.txt`

```bash
v4l2-ctl --list-devices; v4l2-ctl -d <dev> --list-formats-ext
```

記下：
- 型號與 USB ID；
- by-path 路徑；
- 640×480 MJPG 下可用的幀率。

拍照：
- 用腕相機拍一張畫面，夾爪張開；
- 用手機拍相機在夾爪上的安裝位置，正面、側面各一張。

## 7. 收尾

```bash
tools/lab_day.sh torque      # 每軸扭力都要是 0
```

在 handoff 目錄寫一頁 `README.md`：
- 通過的項目；
- 不通過的項目；
- Thor 上實際的版本。

把整個 handoff 目錄打包成 `thor-<YYMMDD>.zip`，帶回實驗室機器，放在 `AllenHZL/`。
