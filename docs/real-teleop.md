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

## 停放位與通道位路線

`tools/real_route.py` 管理單臂的兩個固定姿態。停放位是 follower 靠在固定支撐上、可以解扭力的姿勢；通道位是夾取任務的出入口。路線只錄一次：follower 直接跟隨 leader 的關節角（原版 LeRobot 的絕對映射），從停放位走到通道位後按 `c`。去程照錄下的命令播放，回程反向播放。

```bash
conda activate lerobot
python tools/real_route.py record --config outputs/single.local.yaml --route outputs/routes/right-park-channel.json \
  --layout lab-right-20261125 --support "前臂靠在固定墊塊上" --allow-motion
python tools/real_route.py run --config outputs/single.local.yaml --route outputs/routes/right-park-channel.json --allow-motion
python tools/real_route.py recover --config outputs/single.local.yaml --route outputs/routes/right-park-channel.json --allow-motion
python tools/real_route.py show --route outputs/routes/right-park-channel.json
```

- `record`：在扭力關閉時讀取停放位，然後顯示 leader 與 follower 各軸的差距。各軸差距都在門檻內（身體 12°、夾爪 40%）才能按 Enter 開扭力。開扭力後的 1.5 秒內，限速由 20°/s 漸增到 90°/s。到通道位按 `c` 存檔，接著按 `r` 反向播放回停放位。目視確認支撐後按 `y` 解扭力，路線檔會記下反向驗證通過。
- `run`：從停放位去程到通道位並保持。可選 `t` 用 leader 操作一段，按 `c` 沿實走路徑退回通道位；或選 `r` 反向回停放位並解扭力。`--rate` 可調整相對錄製時間的播放速度。
- `recover`：上一輪中斷、follower 扭力仍開時使用。程式接手後找出路線上最近的點（8° 以內），沿路線退回停放位再解扭力。
- 超出校正範圍只會截斷命令，畫面顯示「到範圍邊界」，不會停機。播放時，follower 落後超過 4° 會暫停播放時鐘，等跟上再走；同一處卡住約 1 秒會判定受阻。Esc 或 Ctrl-C 也一樣：原地保持扭力，選單提供 `g` 繼續、`b` 原路退回、`x` 緊急原地解扭力（須輸入 OFF）。
- 正常解扭力只發生在停放位 4° 以內、已停穩、操作者按 `y` 之後。
- leader 與 follower 的校正零點不一致、絕對映射無法對齊時，可改用 `--relative`。
- 每次執行都在 `<route>-runs/<時間>-<指令>/` 保存 `control.jsonl`（每幀命令、回讀與 leader）、`events.jsonl` 和 `status.json`。路線檔保存停放位取樣、通道位命令與回讀、簡化後的路徑點（0.35° 容差）、各段錄製時間及兩向驗證紀錄。校正雜湊改變時拒絕使用舊路線。
- `record-drop --tray pink|beige --out <drop.json>`：從停放位回放到通道位，再用 leader 從通道位帶到該盤上方的放開位置（方塊離盤面數公分）按 `c`。`--release` 是放開時的夾爪開度（預設 60%）。存檔後倒放回通道位、回停放位並解扭力。
- `setup --route <park-channel.json> --drop <pink.json> --drop <beige.json> --cards <cards.json> --version <起姿版本> --workspace <工作區> --out <setup.json>`：把路線合成採集用的 setup。停放位、途經點、通道位（標準起姿）和放開路線都取自實走路線，關節範圍取自校正。輸出前用採集端的同一套檢查驗證。`configs/real_collection.cards.template.json` 是由托盤尺寸產生的候選擺位，現場刪掉夾不到的點後當 `--cards`。座標原點是托盤內緣、離基座最近的右側角，+x 朝前、+y 朝左。
- 假裝置演練：`python tools/real_route.py record --fake --route /tmp/r.json --layout fake --support test`，之後用 `run --fake --no-leader` 跑同一檔。

### 採集日入口

`tools/lab_day.sh <步驟>` 依序包好當天的指令：`env`、`check`、`overlay`、`route`（需 `SUPPORT="…"`）、`route-test`、`drop-beige`、`drop-pink`、`try`、`cards`、`cards-drop`、`setup`、`collect`、`recover`、`status`、`pack`。日期預設今天（`LAB_DATE` 可改），檔名都帶日期：`outputs/routes/right-park-channel-<日期>.json`、`outputs/<日期>-cards.json`、`outputs/<日期>-right.setup.json`、session `outputs/<日期>-right/`。`FAKE=1` 以假手臂演練路線步驟，`REPLACE=1` 重教既有路線。

- 任務卡：每張卡一個目標位置（120 個），另外依卡片輪替三種盤面：只有目標、一顆鄰近方塊（中心距 55–75 mm，一半同色）、一到三顆遠處方塊（≥ 90 mm）。採集視窗列出其他方塊的大約位置，開場全局圖記錄實際擺位。
- `tools/real_cards.py`：列出、刪除任務卡擺位（依編號、`--max-x 盤=mm` 或 `--yaw`），每次刪完檢查是否仍能產生 120 張卡與 20 個保留擺位，舊檔另存。
- `tools/pack_handoff.py`：第一次執行產生交接索引 `handoff/<日期>_2080Ti_實機採集/README.md`，補完現場欄位後再執行一次：打包成 `AllenHZL/<日期>_2080Ti_實機採集交接.zip`（含 SHA256SUMS），做 CRC 與必要項目檢查，通過後把交接目錄移到 `handoff_origin/`。

### 採集流程的動作規則（11/25 起）

`real_collection.py` 的對位、復位與重播共用 `real_motion.Motion`，和路線工具採用相同的寬鬆規則：
- 路線插值在 follower 落後超過 4° 時暫停等待，同一處卡住約 1 秒才判定受阻，並先原地保持再報錯。
- 遙操作命令只截斷到核准範圍，每幀變化量受限；不會因單筆超出 5° 就停止送令。
- 停穩判定容許 2.5° 的負載下垂；起始走廊為 3°；讀回可超出核准範圍 3°（例如靠在支撐上的停放位）。

setup 有 `drops` 時，操作者夾起抬穩後按「結束」，確認對話框選「是」：程式直接移回通道位、回放到目的盤的放開位、張爪，再倒放回通道位，之後只走固定路線回停放位。放置不在 `control.jsonl` 裡，manifest 記 `placement`。實機重播在重放完夾取段後，以同樣方式放置。

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

托盤擺位用 `tools/tray_overlay.py` 對齊。以 `outputs/261125-camera/global/rgb.png` 為基準，`--serial 038122250316` 開 D455 即時疊圖，用滑桿調基準比例，把盤緣移到和基準重合；`s` 存疊圖與當前原圖，`q` 離開。離線比較用 `--image <新圖> --output <疊圖.png>`。兩張圖須是相同原生解析度、未裁切未鏡像的全局 RGB。

`fit-wrist` 在同一批原圖比較 Brown 和 fisheye，保存留出影格的全畫面與邊緣誤差。腕部手眼外參另需固定板與同期關節。D455 若有實測的 `base_from_marker40_m`，可用 `check-global --links <JSON> --holdout <點位 JSON>` 分報獨立近／遠點的 XY 和 Z 誤差。轉換以 `destination_from_source` 命名，平移以 m 表示。
