# 實驗設定、選單與模擬伺服器

對應 issue [#3](https://github.com/jxuan083/agentic-robotic-simulator/issues/3)。

目前進度：

| 階段 | 內容 | 狀態 |
|---|---|---|
| P1 | 實驗設定檔、控制協定、選單 | 已完成 |
| P2 | Isaac 常駐伺服器 | 未開始 |
| P3 | MuJoCo backend | 未開始 |
| P4 | 跨引擎比較 | 未開始 |

在常駐伺服器完成前，選單直接啟動既有的程式：手動操作用 `scripts/isaac_ee_teleop.py`，自動驗收用 `tools/run_acceptance.sh`。

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
backends: [isaac]                   # 支援的引擎：isaac、mujoco
scene: configs/scenes/lab_current_minimal.yaml
robots:
  - {id: arm0, model: so101}        # 安裝位置來自 scene，目前只支援一支手臂
teleop: configs/teleop.yaml
color_dr: configs/color_randomization.yaml
cameras: []                         # 目前必須是空的
acceptance:                         # configs/acceptance/ 中的情境
  - robustness
  - {scenario: color_dr, options: {color_dr: true, color_dr_seed: 7}}   # 這個情境固定使用的選項
```

載入時會檢查以下項目，有問題時一次列出所有錯誤：
- 必要欄位都存在，沒有未知欄位（可以抓到拼字錯誤）。
- `name` 和檔名一致。
- backend 名稱有效。
- 引用的檔案都存在，驗收情境也都存在。
- 只有一支手臂，`cameras` 是空的。
- 情境選項只能是 `focus`、`color_dr`、`color_dr_seed`。

新增實驗場景時（例如實驗室佈置），新增一個場景檔和一個實驗檔即可。相機與雙臂目前會被拒絕，等確認實驗室資訊後再開放。

## 選單

```bash
export DISPLAY=:1          # Isaac 視窗所在的 X display，依機器而定
uv run tools/sim.py        # 互動：模擬器 → 實驗 → 模式 → 選項
uv run tools/sim.py --list # 列出實驗、驗收情境與伺服器狀態
```

- 模式有兩種：
  - **手動操作**：可選專注模式、顏色 DR 與 seed。
  - **自動驗收**：選情境、選是否錄影。驗收執行中 Isaac 視窗要保持焦點。
- 選單會先印出實際執行的指令。
- 非互動用法給腳本與 agent 使用，加 `--dry-run` 只印指令、不執行：

```bash
uv run tools/sim.py --backend isaac --experiment minimal_pick_place --mode teleop --focus
uv run tools/sim.py --backend isaac --experiment minimal_pick_place --mode acceptance --scenario robustness --no-video
```

- 自動驗收的輸出放在 `outputs/acceptance/<實驗名稱>/`。
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
| `status` | | 目前實驗、session、狀態 |
| `list_experiments` | | 伺服器可載入的實驗 |
| `load_experiment` | `name`、`options`（選填） | 載入或切換實驗 |
| `reset`、`pause`、`resume` | | 與面板按鈕相同 |
| `get_state` | | EE、目標、關節、積木 |
| `run_acceptance` | `scenario` | 立即回應，進度從 `status` 查詢 |
| `shutdown` | | 停止伺服器 |

錯誤碼：

| 錯誤碼 | 意思 |
|---|---|
| `bad_request` | 不是 JSON、不是物件，或單行超過 1 MB |
| `unknown_command` | 沒有這個指令 |
| `bad_args` | 參數缺少、多餘或型別錯誤 |
| `not_supported` | 指令有效，但這個 backend 還沒實作 |
| `failed` | 執行時發生例外；訊息裡有例外類型。伺服器會繼續服務 |

### 實作方式

- `SimServer`（`src/agentic_sim/server/core.py`）不開執行緒，由模擬器的主迴圈每幀呼叫一次 `poll()`。Kit 的 API 必須在主執行緒呼叫，這樣做處理請求時就在兩幀之間、主執行緒上。
- backend 只要提供 `cmd_<指令>(**args)` 方法即可，沒有實作的指令會自動回 `not_supported`。
- client 斷線不影響伺服器，新的 client 連上後看到相同的狀態。
- `SimClient`（`src/agentic_sim/server/client.py`）是阻塞式的 client，選單、腳本、agent 都用它。`probe(backend)` 回傳伺服器狀態，沒有伺服器時回傳 `None`。

## 驗證

- 純 Python 測試在主機（uv）與 Isaac 容器內都通過，共 38 項。新增的測試涵蓋：
  - 實驗設定的檢查與錯誤列表。
  - 啟動指令的組成。
  - 協定的編碼、解碼與各種錯誤。
  - 實際在 localhost 上的 server／client 來回，包括：handler 在主迴圈執行緒上執行、錯誤後繼續服務、client 斷線後狀態保留。
- 經由選單執行真實的 `robustness` 驗收，17/17 通過。

## 檔案

| 路徑 | 內容 |
|---|---|
| `configs/experiments/` | 實驗設定檔 |
| `configs/sim_server.yaml` | 各 backend 伺服器的位址與埠號 |
| `src/agentic_sim/experiments.py` | 實驗設定的載入與檢查 |
| `src/agentic_sim/launch.py` | 由 backend、實驗、模式與選項組成啟動指令 |
| `src/agentic_sim/server/` | 控制協定、伺服器核心、client |
| `tools/sim.py` | 選單 |
| `pyproject.toml`、`uv.lock`、`.python-version` | 主機端 Python 環境 |
