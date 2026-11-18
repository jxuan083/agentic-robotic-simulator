# 介面與設定規格

本文定義各模組之間交換的資料。欄位為設計內容，程式與 schema 檔尚未建立；格式（YAML／JSON）與命名在實作時確認。

## 設定規格

| 規格 | 內容 |
|---|---|
| `RobotProfile` | 關節名稱與順序、單位、限制、夾爪映射、設備與校正引用（ID／版本，不含原始值） |
| `SceneSpec` | 資產、桌面、托盤、基座、相機、光照、隨機種子；見 [scenes-and-tasks.md](scenes-and-tasks.md) |
| `TaskSpec` | 需求、干擾事件、成功條件、重試規則 |
| `InputProfile` | leader／keyboard／mouse／controller 的模式、映射與操作座標系 |
| `RuntimeConfig` | backend、政策、模型服務、記錄位置、執行角色 |
| `Policy manifest` | 技能範圍、接手條件、相機名稱、動作表示、頻率、正規化、資料版本、前處理 |

## 關節慣例

SO-101 單臂六個具名關節：`shoulder_pan`、`shoulder_lift`、`elbow_flex`、`wrist_flex`、`wrist_roll`、`gripper`。

- 介面一律使用**具名**關節，只有政策或資料集需要向量時才按上列順序排列。
- 既有實機程式以 LeRobot calibrated degrees 表示五個身體關節，夾爪為 0（關）–100（開）。這些值不是弧度，也不是模擬關節目標。
- 由來源狀態變成模擬或實機目標，必須在接收端依 `RobotProfile` 完成方向、零位、單位、限幅與夾爪映射。度轉弧度只是其中一步。

## 命令契約

所有控制來源（人工輸入、幾何技能、學習政策）輸出相同形式的命令，交由控制權層決定是否送往 backend。

| 欄位 | 說明 |
|---|---|
| `source` | `human_leader`／`human_keyboard`／`human_mouse`／`human_controller`／`policy`／`scripted` |
| `session_id`、`sequence` | 同一 session 內單調遞增；重複或亂序丟棄 |
| `target` | 關節目標或 EE 目標（二擇一，標明座標系與單位） |
| `gripper` | 夾爪目標（已映射到該 backend 的範圍） |

命令時效所需的時間欄位，在實作議題中與輸入 adapter 一起決定。

### 本機必要行為

- **輸入停止更新**：放開按鍵、視窗失焦、leader 讀數中斷，超過逾時即停止增量並保持，不無限重放最後一筆輸入。
- **控制權切換**：同一時間只有一個控制來源；接管、訂單變更、輸入中斷後，不執行舊命令。
- **清空舊動作**：切換控制權時清除政策動作佇列與內部狀態。
- **姿態對齊**：接管與 reset 後，先重新觀察並對齊姿態再開始動作。
- **Session**：新的 `session_id` 需要明確 re-arm／reset 才開始動作，不在 session 之間混合姿態。
- **不可達目標**：不可達的 EE 目標不執行。

## Leader 輸入 adapter

實體 SO-101 leader 與其操作的 Isaac Sim 在同一台電腦執行。leader 讀數經本機輸入 adapter 進入控制權層，與 keyboard／mouse 共用同一介面；其他電腦以 keyboard／mouse 操作各自的本機模擬。

adapter 可在同一程序內直接讀取 leader，或在必要時使用本機程序間通訊；選擇留待實作議題討論，目前不另訂協定。

既有 `sinica.so101.leader.v1` JSONL 狀態格式可沿用於**錄製與重播**（無硬體的開發與回歸測試），不代表需要建立網路服務：

| 要點 | 內容 |
|---|---|
| 格式 | JSONL，一行一個樣本 |
| 識別 | `schema_version`、`message_type: leader_state`、`session_id`、`sequence`、`control_mode: leader_read_only` |
| 時間 | `source_time.wall_time_ns` 與 `monotonic_time_ns`；用於樣本間時間差與重播節奏 |
| 每個關節 | `raw_encoder`、`calibrated`（值與單位）、`conversion`（身體關節附度轉弧度；夾爪標 `unmapped`） |
| 失效 | 讀取失敗不補造樣本 |
| 工具 | 既有離線工具支援 `validate`、`emit-example`、`replay`；實機讀取需明確開啟 |

待整合：現有格式以 `left`／`right` 雙臂為單位，單臂需加入 arm 選擇；接收端關節方向、零位與夾爪映射。

## 技能結果

技能（幾何或政策）執行後回報給任務執行器：

| 欄位 | 說明 |
|---|---|
| `skill`、`params` | 技能名稱與參數（例如目標顏色、位置） |
| `status` | `succeeded`／`failed`／`uncertain`／`cancelled` |
| `failure_reason` | 抓空、掉落、選錯色、不可達、逾時、被接管等 |
| `duration` | 各段耗時 |
| `evidence` | 用於核對的觀察（影像引用、感知結果）；模擬真值另欄標示 |

需求帳本只在外部核對確認後才增加完成數量，技能自報成功不直接計入。

模擬中的實作（#5）：`agentic_sim.task.skills.SkillResult`，另外記錄各階段耗時（`phases`）與結束時手上的積木（`holding`）；`evidence` 目前是模擬真值，標明 `"observation": "sim_truth"`。見 [task-executor.md](task-executor.md)。
