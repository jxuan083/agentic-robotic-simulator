# 操作工作區與時鐘（Isaac、MuJoCo）

在模擬器裡按 M 進入全螢幕的操作工作區，再按一次回到原本的開發版面（#16）：
- **中間**：連續的 3D 操作區，可以拖曳 EE 目標環。
- **右欄**：目標顏色、目前任務、夾爪、錄製狀態與時鐘。
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
- **錄製不受影響**：錄製器用自己的 render product，以相機原解析度讀原始 RGB。影像裡沒有任何 UI 或標註（見 [demo-capture.md](demo-capture.md)）。
- **感知開關與標註顯示分開**：
  - `--live-perception` 決定要不要跑感知；平常錄鍵鼠示範預設關閉。
  - `--annotations on|off` 決定相機格一開始顯示標註還是原圖；預設 on，執行中用 1／2 切換。
  - `--perception-stage capture` 只取圖、不估測，給效能對照用。
  - `--perception-delay 秒數` 讓每次估測故意晚一點完成，只用於測試，用來確認舊結果會被認出來。

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
  - 物理動作之後的等待，改用模擬時間：
    - `{settle: 秒數}`：模擬時間經過這麼久才進下一步；
    - `hold` 的 `sim_timeout`：以模擬秒數計算逾時。
  - UI 等待（`wait`、暫停期間）和服務逾時，仍用主機單調時鐘。
  - **原本的問題**：腳本只靠主機時間等待。開了感知後模擬只跑真實時間的約 16%，夾爪還沒合上就已經檢查，抓取因此失敗。
  - **改法**：`configs/capture/test_keyboard_yellow.yaml` 改成 `settle` 和 `sim_timeout`。
  - **測試**：`tests/test_capture.py` 模擬「0.16 倍速」的情況，`settle` 版本通過、`wait` 版本如預期失敗。
- **每秒紀錄**：`--metrics_log 路徑` 每秒寫一行 JSON（UI FPS、RTF、各相機的結果年齡與估測延遲），給效能對照用。

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
  - 不開感知時，注入腳本的抓放成功（234 幀）。
  - 開感知時抓取失敗。原因不是等待的時鐘（已改成模擬時間），而是鍵盤移動依主機時間計算：13 fps 時每幀約 6 mm，`until` 門檻被衝過頭，夾取時 y 偏了 4.6 mm（不開感知時 0.4 mm）。真人操作時，可以看著畫面修正。
  - 要不要讓鍵盤移動改用模擬時間計算，會改變操作手感，尚未決定。
- **偶發**：Isaac 第一次跑錄製共存時，在開頭遇到一次 GPU crash（Vulkan）。同一組合重跑兩次都正常。
