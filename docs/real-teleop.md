# SO-101 單臂與雙臂採集

`tools/real_teleop.py` 接上本機 LeRobot 的 SO-101 校正與 Feetech 驅動。設定檔所列的具名配對決定會開哪些 leader 和 follower。單臂設定只列出要操作的一組。雙臂命令按臂名寫入個別序列埠。關節以肩、肘、腕到夾爪的固定順序列出，五個身體關節是校正後的度數，夾爪是 0–100 的開啟百分比。

## 假硬體與離線讀回

用 Python 3.10 以上即可重現樣本、驗證磁碟資料並產生瀏覽器檢視頁。這條路徑只使用標準函式庫。

```bash
python3 -S tools/real_teleop.py fake --arms right --output outputs/fake-right
python3 -S tools/real_teleop.py read outputs/fake-right --frame 2 --html
python3 -S tools/real_teleop.py fake --arms left right --output outputs/fake-dual
```

開啟 episode 裡的 `replay.html`。拖曳控制幀可看兩路影像、配對時間差和每個關節的命令與實測曲線。時間軸以同一主機的單調時鐘配對最近的相機幀。移到另一台電腦時帶上整個 episode 目錄，資料內路徑都相對於該目錄。

讀回會各自核對每路相機的影格編號及 RGB／深度檔名。重複引用、檔名錯配或影像損毀會讓 `read` 回傳失敗，報告會指出相機與索引。檔名錯配會附上預期及實際檔名。

報告的 `control_timing` 包含設定控制頻率、預期週期，以及最大間隔前後的控制幀與時間。控制間隔大於 1.5 個設定週期時，會在 `warnings` 列出該次實際間隔，仍可產生離線檢視頁。時間重複或倒退會使讀回失敗。請搭配錄製紀錄判讀時間缺口的成因。

單臂範例在第 0、1 幀送出肩部轉向 2°、4° 的策略命令。第 2 幀切換人工，清空剩餘的 80°、85° 命令，拒絕舊控制權收到的延遲策略段，首筆人工指令取 follower 實測的 2.5°。下一幀以接管時的 leader 位置為基準開始追蹤位移。`fake_device_trace.json` 記錄真正打到假裝置的命令和開啟／關閉記錄。所有範例影像是合成資料，manifest 標記 `source: fake`。

## 現場採集

沿用具備 Feetech 的 LeRobot 環境。11/20 的單臂原版使用 conda `lerobot`（Python 3.10、LeRobot 0.4.4），雙臂原版使用 LeHome 的硬體 venv（Python 3.11、LeRobot 0.4.3）。本機 conda 環境同時具備 RealSense SDK 與 Feetech。額外環境的依賴列於 `requirements-real.txt`。

從 `configs/real_teleop.single.example.yaml` 或 `configs/real_teleop.dual.example.yaml` 複製私用設定至 `outputs/`，填入各組序列埠、校正 ID、相機和校正版本。腕相機外參待量時使用 `unmeasured-v1`。LeRobot 校正由原本的本機 cache 或設定中的 `calibration_dir` 載入。相機可使用 D455 的原生 RGB-D，或 V4L2 RGB。

```bash
conda activate lerobot
python tools/real_teleop.py check --config outputs/single.local.yaml
python tools/real_teleop.py record --config outputs/single.local.yaml \
  --layout lab-right --output outputs/lab-right-01 --allow-motion
python tools/real_teleop.py read outputs/lab-right-01 --html > outputs/lab-right-01/readback.json
```

`check` 只做檔案、依賴與裝置路徑檢查。實機採集需要現場操作員、互動終端和明確的 `--allow-motion`。將 follower 放在可卸力的支撐姿勢，保持淨空，確認 leader 扭力已關且 follower 在位置模式。採集器核對已載入的校正和馬達讀回，以 follower 的當前姿勢起步。第一幀對齊後，follower 跟隨 leader 位移。各路相機獨立寫檔，`control_hz` 設定迴圈目標。夾爪閉合指令最多比最新回讀少 3 個百分點。此值沿用 11/20 過載後的重錄設定。回報 `RECORDING` 後開始操作，結束前支撐 follower 再按 `q`。退出會逐軸讀回扭力。中斷會保留已寫資料並標記該 episode。

這次接管驗證以假硬體提供策略段。實機 `record` 使用人工 leader；政策伺服器與積木 ACT 的接合留待上機驗收。

## 檔案與時序

| 路徑 | 內容 |
| --- | --- |
| `manifest.json` | 資料來源（`real`、`fake` 或已轉換資料的 `converted`）、程式版本、臂名、關節名與單位、各裝置校正 ID／版本、幀數 |
| `control.jsonl` | 每個控制幀的來源、命令時間、送出的具名命令、送出後的具名狀態與回讀時間 |
| `<camera>/frames.jsonl` | 各相機自己的影格序號、收圖時間、影像路徑，可用時另保留裝置時間與序號 |
| `<camera>/*.png` | 相機原始解析度的 RGB，RealSense 啟用深度時另存原生 16-bit 深度 |

接管行帶 `event.type=takeover`、`switch_ns`、清空數與對齊姿態。人工修正可依 `task_is_policy=0` 切出。資料來源是 episode manifest 的欄位，和每個動作的命令來源各自追蹤。RealSense manifest 保留深度單位、內參和 depth-to-color 變換方向。`read` 逐路核對相機 index、檔案、PNG 完整性、序號與時間。發現缺幀會指出相機和幀號。實機臂校正版本以原始 JSON 的雜湊比對，episode 只保留識別資訊。

## 指定 follower 的實機回放

`tools/replay_real_follower.py` 是獨立的硬體入口。它先檢查實機 episode 完整性，核對選定臂名與 follower 校正版本。以下指令會移動選中的右臂，使用與採集相同的現場確認。回放會要求接近首幀，依原始時間間隔送出命令並檢查跟隨誤差。

```bash
python tools/replay_real_follower.py outputs/lab-right-01 \
  --config outputs/single.local.yaml --arm right --allow-motion
```

LeHome 舊的 `scripts/replay_real_robot.py` 讀取其 12 軸 LeRobot dataset。競賽 repo 的 `tools/read_episode.py` 繼續讀 Isaac 的模擬示範格式。
