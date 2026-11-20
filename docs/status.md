# 狀態、可沿用資產與待確認事項

更新：2026-11-20。

## 已完成

- 本 repo：README 與 docs 架構文件。
- 本 repo：Isaac Sim 最小桌面場景與 keyboard／mouse EE 操作（issue #1），以注入事件的驗收腳本完成抓放、不可達、失焦、暫停、reset 驗收。2026-11-17 加入主畫面滑鼠操作、專注模式與顏色 DR（預設關閉）；使用者實際操作確認手感可接受（2026-11-17）。之後加入面板改版（顏色 DR 與 seed、積木數量 1–8，Reset 套用）與傾斜抓取（到達範圍從約 0.28 m 擴大到約 0.40 m，傾斜上限 provisional）。見 [isaac-ee-teleop.md](isaac-ee-teleop.md)。
- 本 repo：實驗設定檔、選單、模擬伺服器控制協定與常駐 Isaac 伺服器（issue #3 的 P1、P2），見 [sim-server.md](sim-server.md)。MuJoCo backend（P3：同一份場景與控制器、和 Isaac 一致的鍵盤／滑鼠操作與操作面板、俯視圖、常駐伺服器）已完成，見 [mujoco-backend.md](mujoco-backend.md)；跨引擎比較（P4）已完成，見 [cross-engine-comparison.md](cross-engine-comparison.md)。
- 本 repo：需求帳本與可取消的技能執行器（issue #5）：改單、持物詢問、多交取回、缺料詢問在 Isaac 與 MuJoCo 的 GUI 驗收都通過。只用模擬真值，抓取為腳本化技能，尚未接 LLM、感知與 ACT。見 [task-executor.md](task-executor.md)。
- 本 repo：繁中命令列聊天層（issue #7）：自然語言下單、改單、回答問題與查詢進度，轉成任務執行器的結構化命令；模型 Qwen3.5-9B（本機 vLLM，可替換）。見 [chat-layer.md](chat-layer.md)。
- 本 repo：實驗室場景與相機視角（2026-11-20 配置，單臂）。依實驗室 RGB-D 錄影、關節讀值與照片重建托盤擺位與桌面，加入 D455 global 與腕部相機視角（Isaac、MuJoCo 共用場景檔），相機位姿由深度點與手臂網格的 ICP 推算。見 [lab-scene-cameras.md](lab-scene-cameras.md)。
- 既有實機工作區（尚未移入本 repo）：雙臂 SO-101 實機遙操作、相機串流、馬達校正、影像／狀態記錄、DAgger 接管與資料保存程式可運作。

## 可沿用與待補介面

| 既有資產 | 沿用方式 | 待補 |
|---|---|---|
| 馬達校正與 LeRobot 校正格式 | 實機端繼續使用，`RobotProfile` 以 ID 引用 | 單臂選擇；校正原始檔留在實機執行端 |
| 相機串流設定（top + 腕部，640×480／30 FPS，目前為 RGB） | 實機 backend 的相機設定來源；位置轉換採用 RealSense 深度 | 接入 RealSense 深度串流、RGB-depth 對齊、相機內外參校正（目前只有視覺姿態對齊工具） |
| 影像／狀態記錄（LeRobot dataset） | 作為 episode 格式基礎 | 追溯欄位（環境來源、版本、結果）；sim 資料命名空間 |
| DAgger 控制權切換、清空動作佇列、人工／政策來源標記 | 實機與模擬 DAgger 的行為依據 | 單臂化；接到命令契約；修正資料 → 檢查 → 再訓練 → 比較的串接 |
| DAgger 加權資料載入器 | 再訓練時沿用加權概念 | 對應到 ACT 訓練流程 |
| WebSocket 推論服務／執行端分離 | 推論架構參考 | 接到 policy adapter |
| Leader 讀取與 JSONL 狀態格式（`validate`／`emit-example`／`replay`） | 同機 leader 輸入的起點；JSONL 用於錄製與重播 | 單臂選擇、本機輸入 adapter、關節與夾爪映射 |
| ACT／Diffusion Policy 訓練設定 | ACT 主線起點 | 單臂積木資料；目前無 ACT checkpoint |

既有實機工作區的資料集與 π0.5 checkpoint 屬雙臂衣物任務，不直接用於單臂積木任務。ACT 需要針對本案單臂積木任務蒐集資料並訓練，但沿用既有 LeRobot ACT 訓練設定，不從零重寫。

## 預計實作

第 1 項是第一份 issue，已定案。第 2 項以後的拆分與先後，在後續 issue 討論時決定。

1. **Isaac Sim 最小桌面場景與 keyboard／mouse EE 操作**（第一份 issue，[#1](https://github.com/jxuan083/agentic-robotic-simulator/issues/1)）。單支 SO-101、桌面、儲存盤、放置盤與一顆積木。以 keyboard 或 mouse 控制夾爪中心，完成一次手動抓放。只建立這個流程需要的薄介面，不先建立完整的 `mock` 框架。
2. 模擬端 episode 記錄；共用型別與設定載入依需要擴充。
3. 資料檢查、分割與 manifest；`mock` backend 依測試需要建立。
4. ACT 訓練與 policy adapter；模擬端推論與評估。
5. Leader 接入同機模擬（單臂、本機輸入 adapter）。
6. 實機 backend：移植既有實機程式到命令契約與 episode 格式。
7. DAgger 完整閉環（sim 與 real）。
8. 感知、agent（任務執行器與需求帳本已在模擬中完成，#5）。
9. 任務隨機化、`competition_draft`、controller。

模擬端的相機視角與實驗室單臂配置已完成（[lab-scene-cameras.md](lab-scene-cameras.md)）。雙臂配置待比賽結束後再做。

## 待確認

| 項目 | 說明 |
|---|---|
| 軟體版本 | NVIDIA SO-101 Workshop 文件列 Isaac Sim 5.1.0／Isaac Lab 2.3.0／LeRobot 0.4.3；SimReady 文章採另一版本；既有實機工作區的不同環境分別使用 LeRobot 0.4.3 與 0.4.4。先維持各角色環境分開，選定共用資料格式與介面後再處理相容性。模擬端已實測鎖定 `nvcr.io/nvidia/isaac-lab:2.3.2`（Isaac Sim 5.1.0、`isaaclab` 0.54.2），見 [isaac-ee-teleop.md](isaac-ee-teleop.md#測試環境與版本) |
| 單臂選用 | 使用既有雙臂中的哪一支 follower／leader，或另配單臂 |
| RealSense 規格 | 已記錄：D455（韌體 5.12.6、USB 2.1），彩色 1280×720／15 FPS（全程錄影）或 640×480／30 FPS（RGB-D），深度 Z16、1 mm，內參與 depth-to-color 轉換由 SDK 讀出 |
| 相機配置 | Global：D455 位姿已由深度與關節讀值推算（`estimated`，殘差中位數 4.6 mm），未做標定板校正。腕部：Innomaker U20CAM-1080p-S1（640×480／30 FPS），內參與安裝位姿未量，模擬沿用官方相機座模型。見 [lab-scene-cameras.md](lab-scene-cameras.md#限制與待補) |
| EE 可行姿態與範圍 | 已定案：基座座標、兩指間抓取中心、向下抓取方向。模型驗證後的偏移、工作區與可達範圍已記錄於 [isaac-ee-teleop.md](isaac-ee-teleop.md#座標系ee-與限制)（provisional，待實機對照） |
| 比賽場地 | 桌面高度與擺法 |
| 托盤尺寸 | 外形與擺位已由 D455 投影核對（與商品標示相差 1–3%）；盤緣高度、壁厚、盤底只有約略值，內底與圓角待量 |
| Controller | 型號與映射 |
| Control Flow | 預留擴充，不是 ACT 基線的必要條件；接入時再取得對應程式與權重 |
| Agent 模型服務 | 聊天層的開發候選：Qwen3.5-9B，BF16（vLLM，`docker/vllm/serve.sh`）與 Q4_K_M（llama.cpp，`docker/llamacpp/serve.sh`）都可用，已在模擬端比較（見 [chat-layer.md](chat-layer.md)）。Q4 記憶體約 6 GiB；實機執行端與比賽平台待實測，語意比較還要補新語句 |
| Leader adapter 通訊方式 | 同程序直接讀取或本機程序間通訊，於實作議題決定 |
| 關節增益與重力下垂 | Isaac（Workshop 增益）靜止下垂 3–12 mm，MuJoCo（Menagerie 增益）< 0.5 mm，兩者都不是實機值。待實機量測各姿態的下垂與夾爪閉合角後再選定或調整，見 [cross-engine-comparison.md](cross-engine-comparison.md) |
| 傾斜切換處的命令平滑度 | 共用控制器在開始／結束傾斜時關節命令會抖動（硬增益下 EE 最多約 9 mm）；等實機數據回來後一起處理，見 [cross-engine-comparison.md](cross-engine-comparison.md#觀察) |
| 本機模擬效能 | 接 leader 的電腦執行選定 Isaac Sim 版本的相容性與效能待實測；有問題先回報討論，不改回跨機方案。目前模擬端實測約 40 fps（CPU PhysX），模擬約為真實時間的 0.67 倍，見 [isaac-ee-teleop.md](isaac-ee-teleop.md#效能) |

## 參考來源

| 來源 | 用途 | 查閱快照（2026-11-16） |
|---|---|---|
| [NVIDIA Sim-to-Real SO-101 Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop)／[教學](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html) | SO-101 資產、場景、LeRobot 接合；Workshop 使用 Isaac Lab 管理環境，不代表需訓練 RL | `ce807d9` |
| [NVIDIA SimReady 資產流程](https://developer.nvidia.com/blog/5-steps-to-create-simready-assets-for-robotics-with-frontier-ai-models/) | 資產尺度、外觀、物理與動態測試 | — |
| [Seeed SO-100M wiki](https://wiki.seeedstudio.com/lerobot_so100m_new/)、[LeRobot ACT](https://huggingface.co/docs/lerobot/act) | 單臂採集與 ACT 操作 | — |
| [LeHome solution](https://github.com/IliaLarchenko/lehome_solution) | 既有實機工作區的上游；沿用實機與 DAgger 經驗，不沿用其雙臂 checkpoint | `ca73828` |
| [EBiM benchmark](https://github.com/EBiM-Benchmark/benchmark) | 場景與任務組織 | — |
| [EBiM task2 submission](https://github.com/sevleete/ebim-task2-offline-submission) | 推論服務與執行端分離；不沿用其 FR3 動作參數 | `c7d3cd7` |

接手時以各 repo 當下狀態為準，快照僅供追溯。
