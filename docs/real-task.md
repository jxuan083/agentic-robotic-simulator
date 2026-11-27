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
tools/lab_day.sh chat-web            # 需要模型服務；LLM_URL=http://<主機>:8000/v1 可指向別台

# 實機（終端機就是操作台：Esc 原地暫停；結束時回停放位、確認支撐後解扭力）
STANDBY_AFTER=2 tools/lab_day.sh task-server
tools/lab_day.sh chat-web
```

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
