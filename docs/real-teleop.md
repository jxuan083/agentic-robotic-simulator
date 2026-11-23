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

## 右臂任務卡與續收

沿用具備 Feetech 的 LeRobot 環境。11/20 的單臂原版使用 conda `lerobot`（Python 3.10、LeRobot 0.4.4），雙臂原版使用 LeHome 的硬體 venv（Python 3.11、LeRobot 0.4.3）。本機 conda 環境同時具備 RealSense SDK 與 Feetech。額外環境的依賴列於 `requirements-real.txt`。

設備設定沿用 `outputs/single.local.yaml`。先記下現場核對過的右臂停放支撐、標準起姿、通道點與工作區。停放位須有固定支撐。可用以下入口讀取扭力關閉時的實測教示值，`1` 是所需中途點數的範例。教示前由現場核對通道，手扶或支撐手臂；教示入口只讀位置與扭力。

```bash
conda activate lerobot
python tools/real_teleop.py check --config outputs/single.local.yaml
python tools/real_collection.py --config outputs/single.local.yaml --setup outputs/261124-right.setup.json --teach 1
python tools/real_collection.py --config outputs/single.local.yaml --setup outputs/261124-right.setup.json --session outputs/261124-right --allow-motion
```

教示後補齊 JSON 的實走關節範圍。盤位以同一套各盤局部座標填入 `cards.zones` 的近區及遠區，每點為 `[x_mm, y_mm, yaw_deg]`，只列現場可達位置與角度。系統依固定 seed 各留五個有間距的保留位置，剩餘位置用於 120 張任務卡。架位及校正改變時另建對應的 setup 與 session。可先參考 [設定範例](../configs/real_collection.setup.example.json)。

在桌面終端啟動視窗，先執行「空桌動作驗證」，確認只動右臂、方向、限位、開合、停止與支撐。開始任務前將 follower 放在已教示的位置附近。程式承接實測姿態，沿通道限速到起姿，停穩才開始寫任務列。正常或可控取消會先保持目前姿態，等操作者確認空爪與退場路徑。復位逆走實測路徑，保持空爪開度；停放到位後確認固定支撐，再次確認停穩，逐軸卸扭力並讀回。狀態不明時流程停送並保存原因與末次有效姿態，由現場依硬體急停或支撐程序處理。

任務完成後自動從磁碟讀回。首筆先看片標定接手、閉爪與抬穩，並對同筆按「實機重播」。此按鈕顯示開場圖，待盤面還原、人手離開後才啟動 follower。首筆合格後，有效示範自動換卡，待確認筆可在畫面循環抽查、修正標幀或保留重錄。歷史筆也能獨立抽播，重播台帳與配額各自更新。關閉再開須沿用同一個 `--session`。首筆報告帶 D455 深度摘要，有可辨識原生深度時可由讀回視窗點選台階兩側，保存方塊頂面及盤底沿視線深度。

本機保存 `session.json`、`held_out.json`、`held_out.html` 和 `dataset_groups.json`。每筆 `episodes/` 保留原始影像與 `control.jsonl`，以及 `motion.jsonl`、`readback.json`、可選的 `review.json` 和深度比對。重播結果及實測關節在 `replays/`，DAgger 新回合在 `dagger/`。時間列把對齊、任務與復位分段，訓練可用 `review.json` 的起訖幀。

從乾淨測試目錄執行 headless 假裝置流程。也可省略 `--demo` 從桌面開啟 fake 任務卡。假策略會在接管前排入多步，人工接手以當前 follower 實測為基準，插入切換標記並繼續錄影。實機 ACT 策略接合待上機驗收。

```bash
python3 -S tools/real_collection.py --fake --demo --session outputs/261123-fake-session
```

## 檔案與時序

| 路徑 | 內容 |
| --- | --- |
| `manifest.json` | 資料來源（`real`、`fake` 或已轉換資料的 `converted`）、程式版本、臂名、關節名與單位、各裝置校正 ID／版本、幀數 |
| `control.jsonl` | 每個控制幀的來源、命令時間、送出的具名命令、送出後的具名狀態與回讀時間 |
| `<camera>/frames.jsonl` | 各相機自己的影格序號、收圖時間、影像路徑，可用時另保留裝置時間與序號 |
| `<camera>/*.png` | 相機原始解析度的 RGB，RealSense 啟用深度時另存原生 16-bit 深度 |

接管行帶 `event.type=takeover`、`switch_ns`、清空數與對齊姿態。人工修正可依 `task_is_policy=0` 切出。資料來源是 episode manifest 的欄位，和每個動作的命令來源各自追蹤。RealSense manifest 保留深度單位、內參和 depth-to-color 變換方向。`read` 逐路核對相機 index、檔案、PNG 完整性、序號與時間。發現缺幀會指出相機和幀號。實機臂校正版本以原始 JSON 的雜湊比對，episode 只保留識別資訊。

## 指定 follower 的實機回放

`tools/replay_real_follower.py` 是獨立的硬體入口。它先檢查實機 episode、臂名、校正、起姿版本和首幀實測。回放和採集共用對齊、停放及扭力流程。舊版 episode 可用資料讀回檢查，實機重播需要有相符起姿的版本。

```bash
python tools/replay_real_follower.py outputs/261124-right/episodes/D001-01 \
  --config outputs/single.local.yaml --setup outputs/261124-right.setup.json --arm right --allow-motion
```

`tools/real_teleop.py record` 的單臂入口也使用 `--setup` 與同一動作流程，適用於任務卡外的手動短錄。雙臂保留原有入口與現場流程。

LeHome 舊的 `scripts/replay_real_robot.py` 讀取其 12 軸 LeRobot dataset。競賽 repo 的 `tools/read_episode.py` 繼續讀 Isaac 的模擬示範格式。

## 盤位參考與腕部標定

`tools/calibrate_real_cameras.py` 沿用正式相機設定。先執行 `measure` 輸入標定紙實測尺寸，再用 `global` 存 D455 的三標記 RGB-D 參考，或用 `wrist` 收 ChArUco 原圖。現場預覽使用含 OpenCV GUI 和 ArUco 的 Python 環境。預覽結束會釋放相機。也可用 `global --episode <目錄>` 從已錄下的標準起姿影格讀取盤位參考。

`fit-wrist` 在同一批原圖比較 Brown 和 fisheye，保存留出影格的全畫面與邊緣誤差。腕部手眼外參另需固定板與同期關節。D455 若有實測的 `base_from_marker40_m`，可用 `check-global --links <JSON> --holdout <點位 JSON>` 分報獨立近／遠點的 XY 和 Z 誤差。轉換以 `destination_from_source` 命名，平移以 m 表示。
