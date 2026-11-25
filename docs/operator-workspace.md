# 操作工作區與時鐘（Isaac、MuJoCo）

在模擬器裡按 M 進入全螢幕的操作工作區，再按一次回到原本的開發版面（#16）：
- **中間**：連續的 3D 操作區，可以拖曳 EE 目標環。
- **右欄**：目標顏色、目前任務、夾爪、畫面錄影與示範資料的狀態、時鐘；面板上有「Record screen」按鈕。
- **下方**：global 和腕部兩台相機並排，每台可以在原圖和標註之間切換。

兩個引擎各用原生的繪圖方式，畫面上的文字都是英文，因為 Kit 和 MuJoCo 的字型沒有中文字。錄示範的入口見 [demo-capture.md](demo-capture.md)，操作細節見 [isaac-ee-teleop.md](isaac-ee-teleop.md) 和 [mujoco-backend.md](mujoco-backend.md)。

## 版面

| | Isaac | MuJoCo |
|---|---|---|
| 進出 | M 或面板的 Focus view；`--focus` 直接進入 | M；`--focus` 直接進入 |
| 全螢幕 | app 視窗全螢幕，收起主選單列、狀態列、工具列 | GLFW 視窗切到主螢幕全螢幕 |
| 中間 | 3D viewport，收起它自己的選單列與座標軸 | 3D 畫面 |
| 右欄 | `Operator status` 停靠在 EE Teleop 面板上方 | 原本的操作面板 |
| 下方 | global、wrist 兩個相機 viewport 並排 | global、wrist 並排，最右邊是狀態欄 |
| 收起的視窗 | Stage、Layer、Render Settings、Property、Semantics Schema Editor、Content、Console、Main ToolBar、Status Bar、各視窗的分頁標籤、感知預覽浮窗 | — |
| 離開時 | 恢復原本開著的視窗、前景分頁、浮動面板位置、視窗大小 | 恢復原本的視窗位置與大小 |

- **不影響的操作**：EE 拖曳、鍵盤、夾爪與 yaw 按鈕、暫停、Reset 都照常可用。切換版面不改相機和 teleop 狀態。
- **Isaac 改積木數量時**：場景會重建，新的面板和相機 viewport 會在幾幀內放回同樣的版面，不會留下重複的視窗，鍵盤焦點仍在 3D 畫面。
- **MuJoCo 的焦點**：感知第一次取圖時，會建立隱藏的離屏 GL 視窗，在 X11 上會搶走主視窗的焦點。原本要再點一下視窗，按鍵和拖曳才會恢復；現在建立後會自動把焦點拿回來。

## 相機格：原圖／標註

- **切換**：按 1 切 global、按 2 切 wrist；也可以點相機格（Isaac 是格子底部的按鈕，MuJoCo 是整格）。只影響顯示，感知照常執行。
- **標註**：顯示「產生這次估測的那張影格」加上框和頂點的快照，並標出是幾秒前的結果。框只畫在原本那張影格上，不會套到新的畫面。
- **沒有結果或結果過舊**（超過 2 秒）：格子顯示即時原圖，並註明原因（`no result yet`／`annotation too old`）。
- **取像或估測出錯**：
  - 出錯那台相機的格子立刻改回即時原圖，並顯示錯誤原因（`perception error: ...`）；另一台不受影響。
  - 要等這台相機下一次估測成功，才恢復標註；出錯前的結果不會再顯示。
  - 狀態欄也會列出錯誤。
- **錄製不受影響**：錄製器用自己的 render product，以相機原解析度讀原始 RGB。影像裡沒有任何 UI 或標註（見 [demo-capture.md](demo-capture.md)）。
- **感知開關與標註顯示分開**：
  - `--live-perception` 決定要不要跑感知；平常錄鍵鼠示範預設關閉。
  - `--annotations on|off` 決定相機格一開始顯示標註還是原圖；預設 on，執行中用 1／2 切換。
  - `--perception-stage capture` 只取圖、不估測，給效能對照用。
  - `--perception-delay 秒數` 讓每次估測故意晚一點完成，只用於測試，用來確認舊結果會被認出來。
  - `--perception-fault 相機:秒數:秒數`（例如 `global:0.5:4`）：只用於測試。這台相機第一次有效結果之後過 0.5 秒，接下來 4 秒內的估測全部失敗，之後恢復；用來確認出錯時會退回原圖。

## 時鐘

狀態欄分開顯示下列數值，彼此不混用：

| 顯示 | 定義 |
|---|---|
| UI FPS | 每秒主機時間畫了幾幀（`time.monotonic()`，最近 1 秒） |
| RTF | 同一區間的 Δ模擬時間 ÷ Δ主機單調時間。暫停或 Reset 會清掉前一段，不會跨段計算；暫停時不顯示，暫停中模擬時間停住 |
| 感知結果年齡 | 本機單調時間 − 取像時刻。估測晚完成時，仍以原本的取像時刻計算。沒有結果時顯示 `no result yet` |
| 估測延遲 | 估測完成時刻 − 取像時刻 |

- **錄製資料的時間軸**：episode 的 `step`、`t_sim`、原始影像與相機姿態的對應方式不變（見 [demo-capture.md](demo-capture.md)）。
- **自動注入腳本的時鐘**：
  - **注入的按鍵**（移動、轉向、夾爪）依模擬時間推進：每幀前進一個控制步（1/60 模擬秒），不論畫面多快，同一段腳本命令的動作都相同。
  - **人工鍵鼠**仍依主機時間計算，操作手感不變。
  - **暫停**時不推進任何位移；恢復後也不會補上暫停期間的時間。
  - **按住按鍵的 `hold`**，逾時（`timeout`）也以模擬秒數計算，所以不論多慢，按鍵命令的移動量都相同。
  - **不按鍵的 `hold: []`**（等 UI 或感知），逾時仍以主機秒數計算。
  - 物理動作之後的等待，改用模擬時間：
    - `{settle: 秒數}`：模擬時間經過這麼久才進下一步；
    - `hold` 的 `sim_timeout`：以模擬秒數計算逾時。
  - UI 等待（`wait`、暫停期間）和服務逾時，仍用主機單調時鐘。
  - **原本的問題**：腳本只靠主機時間等待。開了感知後模擬只跑真實時間的約 16%，夾爪還沒合上就已經檢查，抓取因此失敗。
  - **改法**：`configs/capture/test_keyboard_yellow.yaml` 改成 `settle` 和 `sim_timeout`。
  - **測試**：`tests/test_capture.py` 模擬「0.16 倍速」的情況，`settle` 版本通過、`wait` 版本如預期失敗。
- **每秒紀錄**：`--metrics_log 路徑` 每秒寫一行 JSON（UI FPS、RTF、各相機的結果年齡與估測延遲），給效能對照用；`--metrics_run_id` 會寫進每一行。

## 畫面錄影（右欄）

從 `tools/sim.py` 進入「模擬」（Isaac、MuJoCo）或 Isaac 的「錄示範」後，按右欄面板的 **Record screen** 開始錄模擬器視窗的 MP4，按 **Stop screen video** 停止。

- **錄什麼**：
  - 只錄這次的模擬器視窗（以 X 視窗 id 指定），包括中央操作區、狀態欄和雙相機，不錄桌面或其他視窗。
  - 錄影中切換全螢幕工作區會分段錄製，停止後合成同一支 MP4。
- **存在哪裡**：
  - 每次開始都存成新檔：`outputs/screen_video/<日期_時間>_<引擎>.mp4`。
  - 右欄分開顯示「Screen video」（畫面錄影）和「Demo data」（示範資料）的狀態。
  - 停止後，要等檔案封裝完成才顯示 `saved <路徑>`。
- **和示範資料分開**：示範錄製器照樣保存原始影像與關節資料，畫面錄影不影響 episode。
- **關閉模擬器**：錄影中關閉視窗時，這段錄影會正常收尾存檔，入口結束時會列出這次的畫面錄影。
- **失敗時**：右欄顯示原因（例如 `helper not running`、`could not start: ...`），不會顯示不存在的檔案。
- **運作方式**：
  - Isaac 容器裡沒有 ffmpeg，所以由主機上的小程式負責錄影（`tools/screen_video.py agent`，內部用 `tools/record_isaac_window.sh`）。
  - 模擬器和它透過 repo 內的控制目錄（`outputs/screen_video/control/<id>/`）溝通，兩個引擎共用同一套啟停邏輯（`agentic_sim/screen_video.py`）。
  - `tools/sim.py` 啟動模擬器時會一併啟動它；MuJoCo 單獨執行時，會自己在程序內啟動。
  - Isaac 單獨執行採集命令時，要另開終端機執行：

    ```bash
    uv run tools/screen_video.py agent <--screen_video_dir 的目錄>
    ```
- **平台**：
  - 視窗錄影需要 X11 的視窗 id 和 shell 工具，所以只在有 X 顯示的 Linux 提供。
  - 在 Windows、macOS 或沒有 `DISPLAY` 的環境：
    - 按鈕停用，顯示 `Screen video unavailable`；
    - 右欄寫明原因（`Screen video: not available on this platform (Windows)`）；
    - `tools/sim.py` 在終端機顯示「此平台未提供視窗錄影」，不會啟動錄影 helper；
    - 其他操作不受影響。
  - `ARS_SCREEN_VIDEO=off` 可以在 Linux 上模擬這個狀態。
  - Windows 原生 MP4 錄影另案處理。
- **獨立 CLI**：`tools/record_isaac_window.sh video 檔名.mp4` 照舊可用。另外可用 `ISAAC_WINDOW_ID` 指定視窗；收到 SIGINT／SIGTERM 時會結束目前片段並寫出檔案。

## 效能對照

```bash
uv run tools/bench_workspace.py run                       # 依序開兩個引擎的 GUI，約 30 分鐘
uv run --with matplotlib tools/bench_workspace.py report  # outputs/bench/summary.{json,md}、bench.png
```

- **條件相同**：
  - 每一組都重播 `configs/trajectories/lab_pick_place.yaml`。這條軌跡依模擬時間驅動，所以不論跑多快，命令的動作都相同。
  - 場景 `lab_20261120`，積木 `red,yellow,blue`，場景預設的腕部鏡頭，在工作區版面（`--focus`）。
  - 錄製都關閉；模擬時間前 3 秒當暖機，不計入。
- **四種設定**：
  - `raw`：只有雙相機原圖；
  - `capture`：感知只取圖；
  - `estimate_hidden`：取圖加估測，不顯示標註；
  - `estimate_shown`：取圖加估測，並顯示標註。
- **錄製共存**：MuJoCo 沒有錄製器，所以只在 Isaac 另外測。用上面的模擬時間注入腳本加 `--record`，分別在不開感知、開感知時跑。
- **只讀本次執行**：
  - 每次執行都會產生一個 run id，寫在 `<out>/<run>.run.json`，並傳給模擬器：每一行 metrics 都帶著它，錄製的事件和 manifest 也帶著它。
  - 錄製寫到同一個 `--out` 底下（`<out>/record_<設定>/`）。
  - 報告只讀帶有最新一次 run id 的資料。同一個檔案裡舊的 metrics、同一個目錄裡舊的 episode 都不會算進來。
  - **只採認能核對的 episode**：本次的 `saved` 事件，要對應到存在、讀得到、而且 `capture_run_id` 相同的 manifest，才算有效 episode。
    - 缺少 manifest、讀不到、或 run id 不符時，報告列出路徑與原因（`rejected episode`），不算進有效 episode 和成功數。
    - 同一次執行裡其他有效的 episode 照樣保留。
    - 幀數和成功與否都以 manifest 為準。
    - 沒有有效 episode 時，報告寫「本次無可驗證 episode」，成功數為 0。
  - `--runs isaac_record_raw` 可以只跑指定的組合。

結果見下節。

## 驗收（模擬端，2026-11-22）

以下都在模擬端的 GUI 實際執行。全部 242 項測試通過。

| 項目 | 結果 |
|---|---|
| Isaac 工作區（`configs/acceptance/operator_workspace.yaml`，lab 場景加感知） | 16/16 通過，全程錄影 59 秒，涵蓋：<br>• 全螢幕、收起編輯器、面板與狀態欄在前景、沒有分頁標籤、沒有重複視窗；<br>• EE 拖曳；1 鍵只切換 global；<br>• 暫停時模擬時間停住、RTF 不顯示，恢復後重新量測；<br>• Reset 後版面和鍵盤焦點都還在；<br>• 改積木數量重建場景後，版面、面板、兩個相機格、焦點都還在，鍵盤可以驅動手臂；<br>• 退出後回到視窗，編輯器版面還原。 |
| Isaac 原本的專注模式（`focus_mode.yaml`，minimal 場景） | 10/10 |
| MuJoCo 工作區（`operator_workspace_mujoco.yaml`） | 10/10：全螢幕、EE 拖曳、2 鍵只切換 wrist、暫停與 RTF、Reset 與焦點、鍵盤、退出 |
| 故意延遲估測（`perception_delay.yaml`，`--perception-delay 3`） | 兩個引擎都 2/2：結果年齡 3.4–6.2 秒時，兩個相機格都顯示即時原圖並標示 `annotation too old`，沒有舊的框疊在新畫面上；狀態欄另外列出結果年齡和估測延遲 |

### 效能對照

`tools/bench_workspace.py`，同一台機器，條件見上一節。表中先列中位數，括號內是 p5；延遲和年齡是中位數／p95，單位秒。

| 引擎 | 設定 | UI FPS | RTF | wrist 估測延遲 | wrist 結果年齡 |
|---|---|---|---|---|---|
| Isaac | 雙相機原圖 | 35.1 (32.4) | 0.58 (0.54) | — | — |
| Isaac | 只取圖 | 33.7 (30.9) | 0.56 (0.52) | 0.06／0.08 | 0.21／0.41 |
| Isaac | 估測，不顯示標註 | 19.7 (12.2) | 0.33 (0.20) | 0.27／0.82 | 0.57／1.15 |
| Isaac | 估測，顯示標註 | 18.1 (12.1) | 0.30 (0.20) | 0.21／0.78 | 0.59／1.16 |
| MuJoCo | 雙相機原圖 | 58.2 (56.6) | 0.97 (0.94) | — | — |
| MuJoCo | 只取圖 | 56.8 (55.6) | 0.95 (0.93) | 0.02／0.04 | 0.38／0.45 |
| MuJoCo | 估測，不顯示標註 | 56.9 (53.8) | 0.95 (0.90) | 0.21／0.54 | 0.49／1.02 |
| MuJoCo | 估測，顯示標註 | 51.9 (46.5) | 0.86 (0.77) | 0.27／0.56 | 0.44／0.91 |
| Isaac 錄製 | 不開感知 | 27.4 (24.4) | 0.46 (0.41) | — | — |
| Isaac 錄製 | 開感知並顯示標註 | 13.7 (12.6) | 0.23 (0.21) | 0.19／0.89 | 0.32／1.20 |

- **Isaac**：主要成本在估測本身。只取圖幾乎不影響；開始估測後 UI FPS 和 RTF 約降一半；顯示標註只多一點。
- **MuJoCo**：
  - 修正前開感知只有 11–13 fps。原因是 `WristLens.render_size()` 每次取圖都在主執行緒重算全畫面反畸變，每次約 0.8 秒；只取圖時甚至只有 1.4 fps。
  - 已加上快取（`agentic_sim/wrist_lens.py`）。修正前的數據留在證據的 `first_run/`。
- **錄製共存**：錄製本身沒有出錯，兩筆 episode 讀回都完整。
  - 上表是第二組（2026-11-22）的數據。當時開感知時抓取失敗，原因是注入的按鍵依主機時間移動：13 fps 時每幀約 6 mm，`until` 門檻被衝過頭，夾取時 y 偏了 4.6 mm。
  - 第三輪改成注入按鍵依模擬時間移動後，開關感知的夾取位置完全相同，見下節。
- **偶發**：Isaac 第一次跑錄製共存時，在開頭遇到一次 GPU crash（Vulkan）。同一組合重跑兩次都正常。

## 驗收（第三輪，模擬端，2026-11-22）

以下都在模擬端的 GUI 實際執行。證據（JSON、log、影片、截圖）保留在模擬端。

| 項目 | 結果 |
|---|---|
| **注入按鍵依模擬時間**：<br>兩回都開示範錄製，同一場景、初始擺位與黃色抓放腳本（`test_keyboard_yellow.yaml`）；一回關感知，一回開估測與標註 | 兩回都抓起黃塊並放進米盤（episode 345／337 幀，讀回成功）：<br>• 關感知：UI 23.4 fps、RTF 0.39；<br>• 開感知：13.4 fps、0.22；<br>• 閉爪時的目標位置都是 (0.2327, 0.1573, 0.0273) m，實際 EE 都是 (0.2282, 0.1546, 0.0231) m，兩回差距小於 0.1 mm |
| **暫停**（`script_sim_time.yaml`）：<br>Isaac 開感知 12.7 fps、不開 34.9 fps；MuJoCo 開感知 51.3 fps | 三組都 6/6：<br>• 0.5 模擬秒的按鍵在不同 fps 下移動距離相同（Isaac 41.3 mm，因為按鍵晚一幀生效；MuJoCo 40.0 mm）；<br>• 暫停 2 秒期間目標和模擬時間完全不動；<br>• 恢復後每幀最多一個控制步，沒有跳步 |
| **感知出錯**（`perception_fault.yaml`，`--perception-fault global:10:4`） | Isaac、MuJoCo 都 7/7：<br>• 出錯前兩格都顯示標註；<br>• global 的估測出錯的那一幀，最後一個有效結果還不到 2 秒，但格子已改回即時原圖並顯示 `perception error: ...`；<br>• wrist 維持標註；<br>• 2 秒後仍是原圖；<br>• 收到新的有效結果（較新的影格編號）才恢復標註 |
| **效能對照只讀本次執行**：<br>`--out outputs/r3/bench`，同一目錄跑兩次 `isaac_record_raw` | • 第一次：報告指向本次的成功 episode（345 幀，manifest 的 run id 相符）；<br>• 第二次在存檔前中止（40 秒逾時）：報告寫「本次無可驗證 episode」，成功數 0，沒有沿用第一次的路徑、幀數或成功結果；metrics 也只算本次的行 |
| **畫面錄影**（`screen_video.yaml`，點右欄按鈕），Isaac、MuJoCo 各一回 | 兩個引擎都 6/6：<br>• 錄影中移動 EE、切全螢幕再切回，停止後得到同一支 MP4，從視窗大小切到全螢幕再切回都有畫面，只有模擬器視窗（Isaac 12.3 秒，MuJoCo 9.3 秒）；<br>• 停止後手臂仍可操作；<br>• 再錄一段是不同的檔案；<br>• 錄影中關閉模擬器，第三段也正常存檔 |
| **錄示範加畫面錄影**（`test_keyboard_yellow_screen.yaml`） | 5/5：<br>• episode 先開始，再開始畫面錄影；<br>• 抓起後停止，MP4 的最後一幀正是剛抓起的時候；<br>• episode 繼續錄到放進米盤，成功，363 幀，放開在 MP4 停止後約 6.7 模擬秒；<br>• 入口的「檢查上一筆」和 `tools/read_episode.py` 讀回結果逐字相同；<br>• 錄到的影像是原始 RGB |
| **錄影無法啟動** | 兩個引擎都 3/3：<br>• Isaac 沒有啟動 agent 時，右欄顯示 `helper not running`；<br>• MuJoCo 的錄影工具不存在時，右欄顯示 `failed - could not start: ...`；<br>• 兩者都沒有列出檔案，模擬器照常運作 |
| **既有驗收回歸**（注入按鍵與按鍵 `hold` 逾時改用模擬時間之後） | `minimal_pick_place` 的全部情境（Isaac 11 個、MuJoCo 11 個）和兩個引擎的 `operator_workspace` 全部通過。<br>• Isaac 的 `robustness_keyboard` 原本用主機 5 秒逾時去推到可達邊界，模擬較慢時推不到；改成按鍵 `hold` 以模擬秒數逾時後，14/14 通過。<br>• Isaac 的 `keyboard_pick_place` 有一次因為視窗失焦丟掉按鍵；重跑時沒有失焦，8/8 通過。 |
