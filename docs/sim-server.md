# 實驗設定、選單與模擬伺服器

對應 issue [#3](https://github.com/jxuan083/agentic-robotic-simulator/issues/3)。

目前進度：

| 階段 | 內容 | 狀態 |
|---|---|---|
| P1 | 實驗設定檔、控制協定、選單 | 已完成 |
| P2 | Isaac 常駐伺服器 | 已完成 |
| P3 | MuJoCo backend | 已完成，見 [mujoco-backend.md](mujoco-backend.md) |
| P4 | 跨引擎比較 | 已完成，見 [cross-engine-comparison.md](cross-engine-comparison.md) |

選單的兩種執行方式：
- **伺服器在跑**：把請求送給伺服器，不必重開 Isaac，場景約 1 秒內重建完成。
- **伺服器沒在跑**（或加 `--direct`）：像以前一樣啟動一次性的程式。Isaac：手動操作用 `scripts/isaac_ee_teleop.py`，自動驗收用 `tools/run_acceptance.sh`；MuJoCo：`scripts/mujoco_ee_teleop.py` 與 `tools/run_acceptance_mujoco.sh`。
- MuJoCo 的常駐伺服器是主機上的背景程序（`tools/mujoco_server.sh up|down|status|logs`），協定與指令和 Isaac 相同，見 [mujoco-backend.md](mujoco-backend.md)。

## 常駐 Isaac 伺服器

```bash
export DISPLAY=:1                  # Isaac 視窗所在的 X display，依機器而定
docker/isaac/server.sh up          # 背景啟動，等到伺服器回應為止（冷啟動約 30 秒，含載入預設實驗）
uv run tools/sim.py                # 任何 terminal：選實驗與模式，送到伺服器
docker/isaac/server.sh status      # 容器狀態與伺服器狀態
docker/isaac/server.sh logs        # 追蹤伺服器 log
docker/isaac/server.sh down        # 停止並移除容器
```

- 伺服器跑在 `docker/isaac/compose.yaml` 定義的容器 `ars-isaac-server` 裡，映像、快取與掛載都和 `docker/isaac/run.sh` 相同。
- 啟動時載入的實驗由 `ISAAC_SERVER_EXPERIMENT` 指定，預設 `minimal_pick_place`。
- Isaac 視窗一直留在桌面上，手動操作照舊在視窗中進行。關掉選單的 terminal 不會關閉 Isaac。
- 選單在伺服器執行時也可以選「停止伺服器」；非互動用法是 `uv run tools/sim.py --server up|down|status`。

**透過伺服器跑自動驗收**：
- 伺服器先重建場景並載入驗收腳本，接著要求視窗管理員把 Isaac 視窗切到前景，然後開始注入事件。執行中請不要操作桌面。
- 選單會顯示進度，結束後印出結果。
- 結果寫在 `outputs/acceptance/<實驗>/`：`<情境>.json`、`.csv`，有錄影時另有 `.mp4`。檔案擁有者會改回主機使用者。
- 驗收結束後，伺服器重建一個一般的操作 session，選項和驗收前相同。
- 視窗失焦時輸入會被丟棄，結果裡會帶 `warning`，選單會印出 WARNING，該次結果不算數。

**實測**（2026-11-17）：

| 項目 | 結果 |
|---|---|
| 冷啟動到可以操作（含載入實驗） | 約 30 秒 |
| 新的 terminal 載入實驗（含拆掉舊場景） | 2.2–2.6 秒（`tools/server_check.py`，10 次） |
| 只重建場景（不含拆除） | 約 0.85 秒；拆除約 1.6 秒 |
| 經伺服器跑一個驗收情境 | 省掉每次約 25 秒的 Kit 啟動 |

## 主機端 Python 環境（uv）

選單、實驗設定、協定與純邏輯測試都在主機上執行，不需要 Isaac 容器。

```bash
uv sync                  # 依 uv.lock 建立 .venv（Python 3.11）
uv run pytest -q         # 純 Python 測試
uv run tools/sim.py      # 選單
```

- 依賴鎖定在 `pyproject.toml` 與 `uv.lock`。
- Isaac Sim 仍在自己的容器中執行，不使用這個環境；同一組測試也可以在容器內跑：`docker/isaac/run.sh /isaac-sim/python.sh -m pytest -q tests`。

## 實驗設定檔

一個實驗一個檔案：`configs/experiments/<name>.yaml`。檔案只引用其他設定，不重複寫數值，所以改場景只需要改一個地方。

```yaml
name: minimal_pick_place            # 必須等於檔名
description: "..."                  # 含 # 時要加引號，否則 YAML 會當成註解
backends: [isaac, mujoco]           # 支援的引擎：isaac、mujoco
scene: configs/scenes/lab_current_minimal.yaml
robots:
  - {id: arm0, model: so101}        # 安裝位置來自 scene，目前只支援一支手臂
teleop: configs/teleop.yaml
color_dr: configs/color_randomization.yaml
cameras: []                         # 目前必須是空的
acceptance:                         # configs/acceptance/ 中的情境
  - keyboard_pick_place               # 只用鍵盤：兩個引擎都能跑
  - {scenario: viewport_drag, backends: [isaac]}                       # 只在列出的引擎跑（滑鼠、面板步驟只有 Isaac）
  - {scenario: color_dr, options: {color_dr: true, color_dr_seed: 7}, backends: [isaac]}   # options：這個情境固定使用的選項
```

載入時會檢查以下項目，有問題時一次列出所有錯誤：
- 必要欄位都存在，沒有未知欄位（可以抓到拼字錯誤）。
- `name` 和檔名一致。
- backend 名稱有效。
- 引用的檔案都存在，驗收情境也都存在。
- 只有一支手臂，`cameras` 是空的。
- 情境選項只能是 `focus`、`color_dr`、`color_dr_seed`、`blocks`（積木數量 1–8）。
- 情境的 `backends` 必須是實驗 `backends` 的子集；沒寫就是實驗支援的所有引擎。選單、伺服器只列出該引擎能跑的情境。

新增實驗場景時（例如實驗室佈置），新增一個場景檔和一個實驗檔即可。相機與雙臂目前會被拒絕，等確認實驗室資訊後再開放。

## 選單

```bash
export DISPLAY=:1          # Isaac 視窗所在的 X display，依機器而定
uv run tools/sim.py        # 互動：模擬器 → 實驗 → 模式 → 選項
uv run tools/sim.py --list # 列出實驗、驗收情境與伺服器狀態
```

- 模式有兩種：
  - **手動操作**：可選專注模式（只有 Isaac）、顏色 DR 與 seed、積木數量。
  - **自動驗收**：選情境、選是否錄影（只錄模擬器視窗）。驗收執行中模擬器視窗要保持焦點。
- 選單會先印出實際執行的指令。
- 非互動用法給腳本與 agent 使用，加 `--dry-run` 只印指令、不執行：

```bash
uv run tools/sim.py --backend isaac --experiment minimal_pick_place --mode teleop --focus
uv run tools/sim.py --backend isaac --experiment minimal_pick_place --mode acceptance --scenario robustness --no-video
uv run tools/sim.py --backend mujoco --experiment minimal_pick_place --mode acceptance --scenario robustness_keyboard
uv run tools/sim.py --server up --backend mujoco
```

- 自動驗收的輸出放在 `outputs/acceptance/<實驗名稱>/`（Isaac）或 `outputs/acceptance/mujoco/<實驗名稱>/`（MuJoCo）。
- 情境若有固定選項（例如 `color_dr` 固定開啟 DR、seed 7），會覆蓋選單裡的選擇，並印出提示。

## 控制協定

模擬器與 client 之間的協定，和引擎無關，Isaac 與 MuJoCo 共用。

- 傳輸：TCP，**只監聽 127.0.0.1**，不提供跨機連線。埠號在 `configs/sim_server.yaml`：Isaac 47801、MuJoCo 47802。
- 格式：一行一個 JSON。

```text
→ {"id": 7, "cmd": "load_experiment", "args": {"name": "minimal_pick_place"}}
← {"id": 7, "ok": true, "result": {...}}
← {"id": 7, "ok": false, "error": {"code": "bad_args", "message": "..."}}
```

| 指令 | 參數 | 說明 |
|---|---|---|
| `hello` | `client`、`protocol_version`（選填） | 每個連線的第一個請求，回傳協定版本與 backend |
| `status` | | 目前實驗、選項（含面板改過的場景設定）、session、狀態 |
| `list_experiments` | | 伺服器可載入的實驗 |
| `load_experiment` | `name`、`options`（選填：`focus`、`color_dr`、`color_dr_seed`、`blocks`、`block_colors`） | 載入或切換實驗。`block_colors` 是每顆積木的顏色名稱，會決定積木數量 |
| `reset`、`pause`、`resume` | | 與面板按鈕相同（`reset` 也會套用面板 SCENE 中尚未套用的設定） |
| `get_state` | | EE、目標、關節、夾爪傾角、積木（`block` 是積木 0，`blocks` 是全部）、目前的場景設定 |
| `run_acceptance` | `scenario` | 立即回應，進度從 `status` 查詢 |
| `shutdown` | | 停止伺服器 |
| `task_order` | `counts`（例如 `{"red": 1, "blue": 2}`）、`mode`（`total`／`add`，預設 `total`） | 下需求或改單；第一次下單時任務執行器接手手臂。見 [task-executor.md](task-executor.md) |
| `task_answer` | `id`、`choice`，選填 `color`、`xy` | 回答執行器提出的問題 |
| `task_state` | | 帳本、手上物件、目前技能與階段、待回答的問題、停止延遲紀錄 |
| `task_stop` | | 停止任務，手臂交回鍵盤／滑鼠 |

錯誤碼：

| 錯誤碼 | 意思 |
|---|---|
| `bad_request` | 不是 JSON、不是物件，或單行超過 1 MB |
| `unknown_command` | 沒有這個指令 |
| `bad_args` | 參數缺少、多餘或型別錯誤 |
| `not_supported` | 指令有效，但這個 backend 還沒實作 |
| `failed` | 執行時發生例外；訊息裡有例外類型。伺服器會繼續服務 |

### Isaac 伺服器的實作方式

- `scripts/isaac_sim_server.py`：Kit 只啟動一次，主迴圈每幀依序做三件事：`poll()` 處理請求、跑一幀 session、檢查驗收是否結束。
- `src/agentic_sim/isaac/session.py` 的 `TeleopSession` 包含場景、`SimulationContext` 與操作 UI，可以拆掉後在同一個程序裡重建。單次執行的 `scripts/isaac_ee_teleop.py` 也用它，所以兩條路徑的每幀邏輯完全相同。
- 拆除順序：
  1. 停用 Isaac Lab 在 GUI 模式下的 STOP 處理。不停用的話，`stop()` 會一直 render 等使用者按 Play，永遠不會返回。
  2. 停止模擬。
  3. 放掉所有 asset 物件。
  4. `clear_all_callbacks()`、`clear_instance()`。
  5. 開新 stage。
- 專注模式的版面跨 session 保留，新的面板會在幾幀後停靠到原位。
- 伺服器收到 SIGTERM（`docker compose down`）會結束主迴圈，並寫完未完成的輸出。

### 協定的實作方式

- `SimServer`（`src/agentic_sim/server/core.py`）不開執行緒，由模擬器的主迴圈每幀呼叫一次 `poll()`。Kit 的 API 必須在主執行緒呼叫，這樣做處理請求時就在兩幀之間、主執行緒上。
- backend 只要提供 `cmd_<指令>(**args)` 方法即可，沒有實作的指令會自動回 `not_supported`。
- client 斷線不影響伺服器，新的 client 連上後看到相同的狀態。
- `SimClient`（`src/agentic_sim/server/client.py`）是阻塞式的 client，選單、腳本、agent 都用它。`probe(backend)` 回傳伺服器狀態，沒有伺服器時回傳 `None`。

## 驗證

**P1**（實驗設定檔、協定、選單）：
- 純 Python 測試在主機（uv）與 Isaac 容器內都通過，P2 完成後共 44 項。涵蓋：
  - 實驗設定的檢查與錯誤列表。
  - 啟動指令的組成。
  - 協定的編碼、解碼與錯誤處理，包括 numpy 型別的回傳值。
  - 實際在 localhost 上的 server／client 來回：handler 在主迴圈執行緒上執行、錯誤後繼續服務、client 斷線後狀態保留。
  - 選單端的伺服器流程：載入、驗收進度、固定選項、警告、錄影開始與停止。
- 經由選單執行真實的 `robustness` 驗收，17/17 通過。

**P2**（2026-11-17，常駐 Isaac 伺服器）：

| 項目 | 結果 |
|---|---|
| 同一個 Kit 程序內重建場景 20 次（`tools/probe_scene_rebuild.py`） | 每次都通過物理檢查（關節目標誤差 0.0008 rad、積木靜置在盤底）；重建 0.85 秒、拆除 1.6 秒；記憶體每次約 +65 MB |
| 執行中的伺服器（`tools/server_check.py`） | 10/10：連線、reset、pause／resume、錯誤後繼續服務、斷線重連後狀態相同；連續載入 10 次，每次 2.2–2.6 秒 |
| 重構後的單次執行（`scripts/isaac_ee_teleop.py`） | 第一輪 `viewport_drag` 11/14，其餘 6 個情境通過；`viewport_drag` 重跑 3 次都是 14/14（見已知問題 5） |
| 經由伺服器跑全部驗收 | 7 個情境全部通過，`viewport_drag` 共 3 次都是 14/14；沒有失焦警告 |

## 已知問題與限制

1. **每次重建場景，記憶體約增加 65 MB**：
   - 增長發生在 Kit／PhysX／renderer 內部，清空 undo 歷史、先關掉舊 stage 都沒有改善。GPU 記憶體不會增長，物理行為也每次都正確。
   - 只有載入實驗與跑自動驗收會重建；手動操作中的 reset 不會。
   - `status` 會回報 `rss_growth_mb`，超過 4000 MB 時帶 `warning`，選單會顯示，這時用 `server.sh down` 再 `up` 重啟。門檻可用 `--rss_warn_mb` 調整。
2. **Isaac 工具列的 Stop 按鈕**：按下後，Isaac Lab 會一直 render 等待 Play，這段時間伺服器不回應請求，按 Play 後恢復。單次執行的程式也有同樣的行為。
3. **視窗焦點**：伺服器透過 `carb.windowing` 要求切換焦點。視窗管理員可能拒絕搶焦點，這時結果會帶失焦警告，請手動點一下 Isaac 視窗再重跑。
4. **只監聽本機**：其他電腦無法連線，跨機操作不在範圍內。
5. **`viewport_drag` 偶發失敗**：
   - 重構後共跑 7 次（單次執行 4 次、伺服器 3 次），失敗 1 次，發生在單次執行：轉到側視角後，第一次按下沒有落在目標環上，接著縮小視窗那一步的點擊選到了 `/World`。
   - 同一份程式之後重跑都通過，重構前的 `main` 也通過，目前沒有重現，原因不明，先記錄觀察。

## 檔案

| 路徑 | 內容 |
|---|---|
| `configs/experiments/` | 實驗設定檔 |
| `configs/sim_server.yaml` | 各 backend 伺服器的位址與埠號 |
| `src/agentic_sim/experiments.py` | 實驗設定的載入與檢查 |
| `src/agentic_sim/launch.py` | 由 backend、實驗、模式與選項組成啟動指令 |
| `src/agentic_sim/server/` | 控制協定、伺服器核心、client |
| `tools/sim.py` | 選單 |
| `docker/isaac/compose.yaml`、`docker/isaac/server.sh` | 常駐 Isaac 伺服器的容器定義與管理腳本 |
| `scripts/isaac_sim_server.py` | Isaac 伺服器主程式 |
| `src/agentic_sim/isaac/session.py` | 可重建的操作 session（單次執行與伺服器共用） |
| `src/agentic_sim/server/runner.py` | 選單端的伺服器流程：手動操作、驗收進度、錄影 |
| `tools/server_check.py` | 檢查執行中的伺服器：指令、重新連線、連續重建的時間與記憶體 |
| `tools/probe_scene_rebuild.py` | 在同一個 Kit 程序內反覆重建場景的量測工具 |
| `scripts/mujoco_sim_server.py`、`tools/mujoco_server.sh` | MuJoCo 伺服器主程式與管理腳本（其餘見 [mujoco-backend.md](mujoco-backend.md)） |
| `pyproject.toml`、`uv.lock`、`.python-version` | 主機端 Python 環境 |
