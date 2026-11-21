# 模擬示範採集與讀回（Isaac Sim，單臂）

操作者在 Isaac Sim 的 `lab_20261120` 場景，用既有的 keyboard／mouse EE 操作，從粉色儲存盤抓起指定顏色的積木，放進米色目標盤。這段過程會被錄成一筆 episode，可以從磁碟重新讀回檢查（#13）。範圍只到模擬採集與讀回；實機串接、轉成 LeRobot dataset、ACT 訓練與 DAgger 留給後續 issue。

場景見 [lab-scene-cameras.md](lab-scene-cameras.md)，操作方式見 [isaac-ee-teleop.md](isaac-ee-teleop.md)。

## 從共同入口錄製與讀回

```bash
export DISPLAY=:1
uv run tools/sim.py        # 選「錄示範」或「檢查上一筆示範」
```

不必手填 Docker、場景或相機參數，也不需要模型服務（Qwen）。

- **錄示範**：
  - 預設設定：Isaac、`lab_20261120`、keyboard／mouse、積木配置 `red,yellow,blue`（依槽位）、目標 `yellow`、20 Hz、場景預設的腕部鏡頭（`real`）、輸出到 `outputs/demos`。
  - 開始前顯示實際設定、輸出位置和採集命令。只需確認目標顏色；輸入 `layout` 可以改積木配置。
  - 目標顏色不在配置裡、或顏色名稱不認得時，留在選單提示修正，不會啟動 Isaac。
  - **錄製中**，終端機的狀態列依序顯示：
    - 等待相機；
    - 開始錄製，並顯示目標顏色與已錄幀數；
    - 已存檔，並顯示結果、幀數、時長與 episode 的實際路徑。
  - Isaac 的輸出另外寫在 `outputs/sim_entry/record_<時間>.log`。
  - 結束時請關閉 Isaac 視窗；按 Ctrl+C 中斷時，當下那一筆不會存成結果。
- **檢查上一筆示範**：
  - 讀這次選單裡錄製端回報存檔的 episode。這次存了好幾筆時，可以選一筆，預設是最後一筆成功的。
  - 這次還沒錄製時，只列出標成「既有」的 episode 或讓你輸入路徑，不會把舊資料夾當成剛錄的那一筆。
  - 讀回是離線工具，不會啟動 Isaac 或模型服務。結果和下一節的 `tools/read_episode.py` 完全相同，因為選單執行的就是畫面上顯示的那條命令。
  - 讀回失敗時，會分成三類：資料不完整（缺幀或錯檔，列出相機、幀號、原因）、任務未完成、抓放時刻缺少或順序不符。
  - 讀完可以選擇產生雙相機對照圖。
- **單獨除錯**：選單顯示的採集命令和讀回命令，都可以貼到另一個終端機直接執行。非互動用法：

  ```bash
  uv run tools/sim.py --mode record [--target-color yellow] [--block-colors red,yellow,blue] [--record-hz 20] \
      [--record-dir outputs/demos] [--wrist-lens real|pinhole] [--dry-run]
  uv run tools/sim.py --mode readback --episode outputs/demos/<episode> [--figures] [--dry-run]
  ```

- **狀態來源**：狀態和幀數取自錄製端寫的 `<輸出目錄>/recorder_events.jsonl`，每行一個事件：
  - `episode`：開新一筆、等待相機；
  - `recording`：寫入第一幀；
  - `progress`：每 20 幀一次；
  - `saved`：存檔，含結果、幀數、時長；
  - `closed`：錄製結束。

  入口只讀這次啟動後、同一個錄製程式寫的事件。單獨執行採集命令時，也會寫這個檔。
- **episode 名稱的時間**：資料夾名稱與 `started` 用的是 Isaac 容器的時鐘（UTC），入口 log 檔名用的是主機時間。

## 採集

```bash
DISPLAY=:1 docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py \
    --scene configs/scenes/lab_20261120.yaml --block_colors red,yellow,blue --target_color yellow \
    --record outputs/demos
```

- **參數**：
  - `--block_colors`：每顆積木的顏色，必填，用來辨認目標積木。
  - `--target_color`：要抓的顏色，預設是第一顆積木的顏色。
  - `--record_hz`：錄製頻率，預設 20 Hz，和實驗室錄製的控制頻率相同。
- **開始**：場景載入、兩台相機都送出第一張影像後開始錄。手臂會先從實驗室休息姿態移到操作起點（約 1.3 秒），這段也在資料裡。
- **成功時自動結束**：目標積木曾離開儲存盤，之後停在米色盤內、夾爪命令為打開、持續 1 秒。終端機顯示 `episode ...: success` 和資料夾路徑。
- **下一筆**：按 R（Reset）。積木回到原位，開始新的一筆。
- **未成功就結束**：按 R 或關閉視窗時，當下這筆照樣存檔，標記為未成功（`reset`／`closed`）。
- **關閉視窗**：直接關掉 Isaac 視窗即可，程式會存完檔再結束。
  - 錄製會載入 Replicator。它在關閉時會停止 timeline，而 Isaac Lab 的 STOP 處理會一直算繪、等待重新播放，視窗就關不掉。
  - 因此程式在關閉事件先關掉這個處理（先於 Replicator 執行）。
  - 用 Ctrl+C 中斷時，當下這筆不會寫入結果，`manifest.json` 停在 `status: recording`。
- **不錄的幀**：暫停（P）期間模擬不前進，也不錄。
- **末端標記**：錄製時會隱藏末端位置的藍色球，避免它出現在相機影像裡。目標環是 viewport 上的 UI，本來就不會進影像。
- **資料來源**：
  - 加上 `--script`（注入輸入事件的測試腳本）錄到的 episode，會標記 `operator: scripted_input`，不能當示範資料。
  - 示範資料必須是人工操作，標記為 `operator: human`。
  - `--trajectory`、`--task` 由程式驅動手臂，不能和 `--record` 一起用：啟動前就以錯誤結束（結束碼非零），不會產生資料。任務執行器在錄製中也不能接手。
- **效能**：錄製時操作畫面約 26–29 fps，不錄時約 38–41 fps（模擬端實測）。模擬時間每幀固定前進 1/60 秒，所以資料的時間軸不受影響。

## 資料格式

沿用實機執行端原始錄製的結構（`trc.so101.lab_capture.v1`），schema 為 `trc.so101.sim_capture.v1`：

```
outputs/demos/<YYYYmmdd_HHMMSS>_<顏色>/
  manifest.json          場景、目標顏色、來源 sim、操作者、提交版本、相機內參、單位、托盤範圍、結果、幀數、時長
  control.jsonl          每個錄製步一列：同一步的命令與狀態
  global/frames.jsonl    每張影像一列：{frame, rgb, step, t_sim, capture_ns}
  global/000000.png      D455 global，1280×720 RGB
  wrist/frames.jsonl
  wrist/000000.png       腕部，640×480 RGB；預設 real 鏡頭模式（有桶形變形，和實拍一致）
```

**相機內參**：`manifest.json` 的 `cameras.<相機>` 記的是該次渲染實際使用的投影，在錄製開始時從相機 prim 讀出，不是場景檔的數值。

| 相機 | `model` | 內參 | 說明 |
|---|---|---|---|
| global | `pinhole` | fx = fy = 639.287，cx 640，cy 360 | 由 prim 的焦距與光圈換算。Omniverse 不支援主點偏移，所以主點在影像中心，和場景檔的實測 D455 值（fx 639.53、fy 639.05、cx 640.10、cy 369.29）不同 |
| 腕部，`real` 模式 | `opencv_pinhole` | fx = fy = 400，cx 320，cy 240，`distortion_opencv` [−0.15, 0, 0, 0, 0] | RTX 的 OpenCV pinhole 鏡頭模型（`estimated`，由 11/20 實拍推算） |
| 腕部，`pinhole` 模式 | `pinhole` | fx = fy ≈ 407.439，cx 320，cy 240，沒有畸變 | 場景的理想針孔視角 |

**對齊方式**：

- `control.jsonl` 第 k 列，和每台相機的第 k 張影像，有相同的 `frame` = k 和模擬步號 `step`。
- `step` 是控制步計數，每步 1/60 秒；錄製步之間相隔 3 步（20 Hz）。
- `t_sim` 是模擬時間，`capture_ns` 是讀回影像當下的時鐘。
- 每一列的狀態和影像，都在同一步的物理更新與畫面算繪之後才讀取。

`control.jsonl` 每一列的欄位：

| 欄位 | 內容 | 單位與座標系 |
|---|---|---|
| `command.ee_pos` | EE 目標（操作者輸入經受限 IK 接受後的目標，夾爪中心） | m，mount frame |
| `command.ee_yaw_offset` | 夾爪 yaw，相對控制器的夾取方向 | rad |
| `command.gripper` | 夾爪關節位置目標 | rad |
| `command.joint_pos` | 同一步送給關節驅動的位置目標：5 個手臂關節加夾爪 | rad |
| `state.joint_pos`／`state.joint_vel` | 量測的關節位置與速度（順序同上） | rad、rad/s |
| `state.ee_pos` | 量測關節經正向運動學得到的夾爪中心 | m，mount frame |
| `target`／`readback` | 上面命令與量測換成 LeRobot 單位（手臂度數，夾爪 % 開度），和實驗室錄製的欄位同名 | 換算是 provisional 的 Workshop 線性對應 |
| `blocks` | 每顆積木的顏色、位置、姿態（wxyz） | 模擬真值，m |
| `source`、`paused`、`focused`、`ik_status` | 這一步的輸入來源（keyboard／mouse）、暫停、視窗焦點、IK 狀態 | — |
| `images` | 這一步各相機的影像檔名 | — |

- **命令的種類**：命令同時記 EE 目標和關節位置目標。操作者控制的是 EE；關節目標是控制器在同一步實際送給模擬關節驅動的值。
- **mount frame**：原點在手臂底座下方的桌面，+X 朝前、+Y 朝左、+Z 朝上，和 Isaac world 相同。

## 讀回

```bash
uv run tools/read_episode.py outputs/demos/<episode>                          # 中文摘要（--json 改印完整 JSON）
uv run --with matplotlib tools/read_episode.py outputs/demos/<episode> --figures
```

- **完整性**：
  - 每一列都有上表欄位。
  - `frame` 連續，`step` 等間距。
  - 每台相機的每一幀，逐幀核對控制列和 `frames.jsonl`：
    - 兩邊都有這一幀；
    - 檔名相同（指向相鄰幀的既有影像也算錯）；
    - 步號相同；
    - 影像檔存在。
  - 有錯時列出相機、幀號和原因（`check.errors`），讀回失敗。
- **三個時刻**：只從錄到的資料推出。

  | 時刻 | 判斷方式 |
  |---|---|
  | close（夾爪閉合） | 抓起前，最後一次開始關閉夾爪命令的那一幀 |
  | lift（積木離盤） | 目標積木第一次比初始高度高 1 cm |
  | release（目標盤放開） | 抓起後，第一次開始打開夾爪命令的那一幀 |

- **先後檢查**：
  - close < lift < release。
  - close 時積木還在盤上，量測的夾爪在命令之後才停在積木上。
  - release 時積木在米色盤正上方，結束時停在米色盤內。
- **輸出**：
  - `readback/readback.json`：上述報告。
  - `--figures` 另外產生：
    - `moments.png`：三個時刻前 0.25 秒、當下、後 0.25 秒的 global 與腕部畫面，附當幀命令與狀態。
    - `timeline.png`：夾爪命令與量測、積木高度、EE 高度，並標出三個時刻。
- 結束碼：全部通過為 0，否則為 1。

## 驗收

- **測試**：`tests/test_capture.py`。合成一次抓放，走真的錄製與讀回程式。
  - 完整性與三個時刻的順序。
  - 缺一張影像會被報出。
  - 控制列與相機索引逐幀不一致時會失敗：檔名不存在、指向相鄰幀的既有影像、步號不同、影像檔缺失。
  - Reset、關閉時的結果標記。
  - 等兩台相機都有影像才開始錄。
  - LeRobot 單位換算來回一致。
- **管線測試**：用注入鍵盤事件的腳本 `configs/capture/test_keyboard_yellow.yaml`，在 Isaac GUI 抓起黃色積木放進米色盤。
  - 結果：171 幀、8.5 秒，自動判定成功。讀回完整，三個時刻先後正確。
  - 這筆標記為 `scripted_input`，只用來檢查管線，不是示範。
- **人工示範**：操作者在 Isaac GUI 用鍵盤操作，把黃色積木從粉色盤放進米色盤（2026-11-21，模擬端）。同一次操作錄到 2 筆成功的 episode，讀回都通過（結束碼 0）。

  | episode（`outputs/demos/`） | 幀數 | 時長 | close（幀／步／秒） | lift | release |
  |---|---|---|---|---|---|
  | `20261121_155631_yellow` | 299 | 14.90 s | 176／530／8.83 | 195／587／9.78 | 264／794／13.23 |
  | `20261121_154822_yellow` | 248 | 12.35 s | 165／2545／42.42 | 180／2590／43.17 | 226／2728／45.47 |

  - 「秒」是 `t_sim`，從該次啟動或 Reset 起算的模擬時間。
  - 兩筆都沒有缺欄位、缺幀或缺圖，兩台相機的步號都和控制列一致。
  - 先後檢查全部成立：
    - close < lift < release；
    - close 時積木在盤上，量測的夾爪在命令之後停在積木上（約 0.22 rad）；
    - release 時積木在米色盤正上方，結束時在米色盤內。
  - 其餘幾筆是按 R 重來或中斷的未成功紀錄，只留在本機，不算示範。
  - **內參補正**（2026-11-22）：
    - 這兩筆錄製時，程式記的是場景檔的內參，不是實際渲染用的值。
    - 已改成用修正後程式從 prim 讀出的值（同場景、同 `real` 模式），原值留在 `intrinsics_correction.previous`。
    - 影像、控制列、提交版本與時間不變，重新讀回結果同上。
- **內參與鏡頭模式**：同一場景用 `real` 和 `pinhole` 各錄一筆短的 episode。
  - manifest 分別記 400 px 加畸變係數，以及 407.439 px、沒有畸變。
  - 腕部影像也對得上：`real` 的托盤邊緣有桶形彎曲，`pinhole` 是直線。
- **自動模式**：`--record` 搭配 `--trajectory` 或 `--task` 都在啟動前被拒絕，結束碼非零，沒有產生資料夾。

### 共同入口（#16）

模擬端實測，2026-11-22：

| 案例 | 結果 |
|---|---|
| 從選單錄一筆 | 操作者在 `uv run tools/sim.py` 選「錄示範」，用預設配置與黃色目標，以鍵鼠完成抓放。<br>• 狀態列依序顯示等待相機、錄製中、已存檔。<br>• 存成 `outputs/demos/20261121_173908_yellow`：成功，329 幀，16.4 s。<br>• 同一次啟動中，另外有 3 筆按 R 重來或關窗的未完成紀錄。 |
| 從磁碟讀回 | 「檢查上一筆」讀的就是上面那筆：完整性通過，兩台相機都沒有缺幀或錯配。三個時刻（幀／模擬步／t_sim）：<br>• 夾爪閉合 211／2194／36.57 s；<br>• 積木離盤 228／2245／37.42 s；<br>• 目標盤放開 294／2443／40.72 s。<br><br>另外在沒有錄製的情況下，選既有的 `20261121_155631_yellow` 讀回，沒有啟動 Isaac 或模型服務。 |
| CLI 單獨除錯 | 把入口顯示的採集命令拿到另一個終端機執行，錄到的那筆和入口錄的那筆設定完全相同：場景、目標、積木配置、鏡頭模式（`real`）、20 Hz、輸出根目錄。讀回命令單獨執行的輸出，和選單的輸出逐字相同。 |
| 錯誤可定位 | 在 `20261121_155631_yellow` 的副本裡，把第 5 幀的 wrist 檔名改錯。選單與 CLI 都回報讀回失敗（結束碼 1），列出「wrist 第 5 幀：control row names ...」與缺影像檔。原始那筆仍然通過。 |
| 原有入口相容 | 在只裝 PyYAML 的環境（沒有 NumPy），以下三條都成功：<br>• `python tools/sim.py --help`<br>• `--backend isaac --experiment lab_20261120 --mode teleop --dry-run`<br>• `--mode record --dry-run` |

## 限制

- 資料留在模擬端的 `outputs/`，不進 repo。
- 影像存成逐幀 PNG；轉成 LeRobot dataset（影片、parquet）留給 ACT 的 issue。
- 積木位置與姿態是模擬真值，實機沒有對應欄位。
- LeRobot 單位的換算沿用 provisional 的 Workshop 線性對應，要等逐關節實機核對。
