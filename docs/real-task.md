# 實機整單：聊天 → 執行器 → SO-101

分支 `feat/real-task-server`。內容是 `feat/real-teleop-record-replay`（實機）合併 `feat/chat-no-return`（Leo 的聊天四個分支）後，加上實機轉接層。

## 做法

聊天層和任務執行器的規則都不改：下單、改單、帳本、缺料提問、no_return（不搬回，只回報）都照原樣。

實機任務伺服器 `tools/real_task_server.py` 講和模擬伺服器相同的 `task_*` 協定，連接埠是 `configs/sim_server.yaml` 的 `real: 47803`。聊天 CLI 和網頁只要加 `--backend real` 就能用。

`agentic_sim/real_task/` 的內容如下。

| 模組 | 作用 |
|---|---|
| `executor.RealTaskExecutor` | 繼承 `TaskExecutor`，只換掉碰到實體世界的部分（見下表），決策邏輯沿用 |
| `skills` | `RealPick`／`RealPlace`／`RealHome`／`RealHold`／`RealObserve`。介面和模擬技能相同；硬體流程在背景執行緒跑，執行器照常逐幀 `step` |
| `tracker.Tracker` | 讓同一顆方塊在每次 D455 掃描後保持同一個編號（執行器用編號認方塊）。補料放在新位置的方塊，沿用同色、已不在盤內的編號 |
| `arm.FakeArm` | 模擬桌面，不接硬體也能跑整套流程（測試、週末開發、彩排對話） |
| `session` | 協定處理：`task_*`、`status`、`pause`／`resume`、`scan`、`task_continue`、`fake_scene`（只在模擬桌面） |

替換的部分：

| 執行器原本 | 實機 |
|---|---|
| 每幀用 IK 推末端的 Pick／Place／Home | 已上機驗證的流程：停放位 ⇄ 起姿固定路線、程式夾取（S1）、程式放置到米盤 P1–P4（11 cm 搬運） |
| 模擬真值判斷方塊在哪個盤 | 最近一次 D455 掃描（`Tracker.tray`） |
| 模擬真值判斷送達 | 放完回停放位，再用 D455 看米盤：同色方塊在格位 4 cm 內就算送達，否則記為「不確定」 |
| `ctrl.evaluate` 判斷可達 | 安全夾取區（橘色弧線）加 IK |
| `placement_slots` | `configs/real_slots.json` 的 P1–P4，經 D455 位姿換到基座座標 |

固定使用決賽設定：
- 只從最近一次觀測選方塊；觀測都在停放位進行，手臂不會擋到畫面。
- 缺料時只問「改色或等補料」。
- `no_return` 開啟。

`--standby-after N`：第 N 顆送達後手臂停在停放位，在操作者按 `c`（或送 `task_continue`）之前不會再觀測。決賽在這段時間取走缺料色。

## 使用

```bash
# 模擬桌面（不接硬體）
FAKE_TABLE=blue,blue,red,yellow STANDBY_AFTER=2 tools/lab_day.sh task-server
tools/lab_day.sh chat-web            # 需要模型服務，見下方「模型服務」

# 實機（終端機就是操作台：Esc 原地暫停；結束時回停放位、確認支撐後解扭力）
STANDBY_AFTER=2 tools/lab_day.sh task-server
tools/lab_day.sh chat-web
```

## 模型服務

| 位置 | 指令 | 備註 |
|---|---|---|
| 實驗室這台（2080 Ti） | `docker/llamacpp/serve.sh up`，然後 `Q4=1 tools/lab_day.sh chat-web` | Qwen3.5-9B Q4_K_M（llama.cpp），約 6 GB 顯存，延遲 P50 0.6 秒；舊語句表 141/144（BF16 144/144）。`lerobot` 要在 docker 群組 |
| 5090 | 那台 `docker/vllm/serve.sh up`，這台 `LLM_URL=http://<5090>:8000/v1 tools/lab_day.sh chat-web` | BF16，舊語句表 144/144；vLLM 只聽 127.0.0.1，要改對外或用 SSH 通道 |
| 比賽（Thor） | 練習場地驗證 | |

數字來自 `docs/chat-layer.md`（5090 上的比較，11/19）。

## 驗證狀態（2026-11-27）

- **模擬桌面**：`tests/test_real_task.py` 共 4 項通過，涵蓋：
  - 方塊編號；
  - 決賽流程：紅 → 藍 → 待命，取走藍色、提問、改黃、結單；
  - 夾空重試；
  - 等補料。
- **TCP 冒煙測試**：同一流程經協定跑通，每段動作模擬 1 秒，全程 20 秒。
- **既有測試**：Leo 的聊天、執行器與協定測試共 112 項通過。
- **尚未上機**：`RealArm` 的各段流程都已在其他工具上機驗證過，但串在一起還沒上機。
- **尚未接 LLM 測試**：實驗室這台（2080 Ti）帳號沒有 docker 權限，起不了 vLLM。

## 待辦

- 實機第一次整單：S1 抓取，`STANDBY_AFTER=2`。
- 聊天的「位置指定」（`pick_by_position`）在實機用 D455 座標，`pick_ref` 的上／左方向要依現場視角確認。
- S3（座標條件 ACT）取代 S1 的抓取：`RealArm.pick` 換成政策推論。
- 比賽現場在 Thor 上跑：RealSense、LeRobot、vLLM。

## 柴犬玩偶：方塊用程式夾、玩偶用 ACT 夾（11/28）

決賽 demo 的一幕：同一個聊天下單，執行器依物件種類選技能。規格與上機順序見 `notes/261128_玩偶規格與決賽用法.md`。

- **D455**：`agentic_sim.real_plush` 用高度和底面尺寸找出玩偶；桌上的玩偶算在「備料」裡。
- **聊天**：`configs/chat.yaml` 的 `plush`（柴犬、狗狗、玩偶、小熊…），下單方式和顏色一樣，回覆用「柴犬玩偶」。
- **執行器**：玩偶的 pick 參數是 `method: act`，方塊是 `program`。玩偶放在米盤 P0（`configs/real_plush.json`），不在程式夾取區內也照樣接單。
- **實機**：`PLUSH_POLICY=<…/pretrained_model> tools/lab_day.sh task-server`。沒有設定時，玩偶的訂單會以「沒有模型」失敗。ACT 迴圈在 `agentic_sim.real_policy`。
- **假桌面**：`FAKE_TABLE=red,red,blue,yellow,plush tools/lab_day.sh task-server`。

## 下一步：聊天裡的「收資料」技能（設計，11/27）

目標：在聊天頁說「開始自動收資料，五輪」或「我要手動示範」，由實機任務伺服器用它已開著的 follower 和 D455 執行。收資料和整單任務互斥，同時只能做一件。

**聊天層**（照 `pick_by_position`、`no_return` 的做法，用 `configs/chat.yaml` 的 `collection` 開關控制，預設關閉，模擬端不受影響）：

| 意圖 | 例句 | 送出的指令 |
|---|---|---|
| `collect_auto` | 「開始自動收資料，五輪」「讓手臂自己收十輪」 | `collect_start {mode: auto, rounds}`（1–20，沒說輪數就反問） |
| `collect_manual` | 「我要手動示範」「開始手動收資料」 | `collect_start {mode: manual}` |
| `collect_stop` | 「收到這裡就好」「停止收資料」 | `collect_stop`：做完這一筆、兩支手臂回停放位後結束 |
| `status` | 「收了幾筆？」 | 從 `collect_state` 回答 |

「停止收資料」和現有的 `stop`（緊急停止）是兩件事。緊急停止仍然只指向暫停按鈕和 Esc，聊天不直接控制手臂。

**伺服器**（`tools/real_task_server.py`）：
- 收資料在背景執行緒裡跑，使用伺服器已開著的裝置。
  - 自動：`auto_collect` 拆出「已開好裝置版」的迴圈。
  - 手動：`grasp_session`（起姿停留模式）同樣拆出。
- **手動收資料的前置檢查**（任一項不通過就拒絕，聊天回覆原因）：
  1. 設定檔裡的 leader 序列埠（by-id）存在；
  2. leader 開得起來，而且扭力是關的；
  3. leader 的校正和路線教示時的校正相同（`check_calibration(leader=True)`）；
  4. leader 在折疊位附近（離路線起點 25° 內）。不在的話，請操作者先擺回去。

  通過後才開 leader 的匯流排，收完就關。
- 手動示範時，操作者仍在伺服器終端按 `c`、`a`、`s`/`f`。手在 leader 上，不方便操作網頁。之後可以改用腳踏開關。
- 事件：`collect_started`、`collect_episode`（第幾筆、顏色、成功與否、自動或手動）、`collect_waiting`（需要操作者處理，例如補放方塊）、`collect_done`（各類筆數）。聊天頁要補這幾種事件的繁中模板（`zh.report_line`）。
- 安全：任何中止都照現行規則處理，兩支手臂先回停放位再解扭力，leader 只在有人扶住時才解扭力。

**工作量**：約 1 天，週末用假手臂、假 leader、腳本化終端機開發和測試。11/30 上機。

**風險**：新增意圖會改變 schema 和提示詞。開啟 `collection` 前要用 `tools/chat_eval.py` 重跑語句表，確認原有句子的結果沒變。
