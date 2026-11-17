# Agentic Robotic Simulator

2026 台灣機器人挑戰賽（TRC）SO-101 單臂積木分揀專案。以同一個 repo 串起模擬與實機的完整競賽流程，讓資料採集、訓練、推論、DAgger 修正與 agent 任務執行共用同一套任務核心與介面。

> 目前為**架構與文件階段**，程式只有第一份 issue 的 Isaac Sim 最小場景與 keyboard／mouse EE 操作（[docs/isaac-ee-teleop.md](docs/isaac-ee-teleop.md)）。下列流程是預計整合的系統。可沿用的實機程式位於既有實機工作區，尚未移入本 repo。

## 完整流程

```mermaid
flowchart LR
    S[設備與場景設定] --> T[遙操作採集]
    T --> D[資料檢查與分割]
    D --> TR[訓練／再訓練 ACT]
    TR --> I[自主推論與評估]
    I --> DG[DAgger 人工修正]
    DG -->|修正資料| D
    I -->|新政策評估後| C[新舊政策比較]
    C --> A[Agent 任務執行]
```

每個階段可以獨立執行：採集、訓練、推論、評估都不需要啟動 agent。Agent 只在最上層呼叫已驗證的技能。

## 設計重點

- **共用任務核心，分開後端**：`mock`（無物理、測流程）、`isaac`（Isaac Sim）、`real`（SO-101 實機）。模擬先只做 Isaac Sim。
- **ACT 為主線**：政策經 adapter 接入，預留 Diffusion Policy、π0.5 等替代政策。
- **操作輸入共用一個介面**：模擬端先用 keyboard／mouse 操作末端（EE），再接實體 SO-101 leader，controller 最後。
- **場地與任務分開**：共用資產搭配 `lab_current`／`competition_draft` 場地配置，任務（一般分揀、干擾色、缺料）可載入任一場地。
- **DAgger 完整閉環**：人工接管 → 修正資料保存 → 檢查 → 明確納入再訓練 → 同條件比較新舊政策。
- **依賴環境分開**：同一 repo，模擬端、訓練端、實機執行端可使用各自的 Python 環境。

## 目前狀態

| 項目 | 狀態 |
|---|---|
| 實機遙操作、相機、校正、影像／狀態記錄 | 既有實機工作區可運作（雙臂配置），待移植與單臂化 |
| 實機 DAgger 接管（控制權切換、清空政策佇列、來源標記） | 既有程式可沿用；修正資料納入再訓練尚未串成流程 |
| SO-101 leader 讀取（JSONL 狀態格式、錄製與重播工具） | 既有雙臂版，待補單臂選擇與本機輸入 adapter |
| Isaac Sim 場景、keyboard／mouse EE 操作 | 第一份 issue（[#1](https://github.com/jxuan083/agentic-robotic-simulator/issues/1)）：已實作，含主畫面滑鼠操作、專注模式、顏色 DR（預設關閉，可在面板切換）、1–8 顆積木與傾斜抓取（擴大到達範圍）。注入事件的驗收腳本通過，使用者實際操作確認手感可接受。場景數值為 provisional（[docs/isaac-ee-teleop.md](docs/isaac-ee-teleop.md)） |
| 實驗選單、常駐模擬伺服器、MuJoCo backend | [#3](https://github.com/jxuan083/agentic-robotic-simulator/issues/3) 進行中：實驗設定檔、控制協定、選單與常駐 Isaac 伺服器已完成；MuJoCo 與跨引擎比較尚未開始（[docs/sim-server.md](docs/sim-server.md)） |
| 共用介面（RobotProfile、SceneSpec、TaskSpec 等） | 已定義文件，程式預計實作 |
| ACT 訓練與推論 adapter | 預計實作（既有工作區有 ACT 訓練設定，尚無 ACT checkpoint） |
| Agent、任務執行器、需求帳本 | 預計實作 |
| 評估報表 | 預計實作；驗收數字為目標，尚無成果 |

完整狀態、可沿用資產與待確認事項見 [docs/status.md](docs/status.md)。

## 執行環境角色

| 角色 | 用途 |
|---|---|
| 實機執行端 | 連接 SO-101 follower／leader 與相機，執行實機採集、推論與 DAgger 驗證 |
| 模擬端 | 執行 Isaac Sim 場景、keyboard／mouse 遙操作與模擬評估；以實體 leader 操作時，leader 接在執行模擬的同一台電腦 |
| 訓練端 | 執行 ACT 等政策訓練與離線評估 |

程式可在任何開發環境撰寫；涉及硬體的行為須在實機執行端驗證。機器專屬設定、原始校正與大型資料不進 repo，repo 只放範本、格式與版本說明（見 [docs/architecture.md](docs/architecture.md#設定與資料的存放)）。

## 文件

| 文件 | 內容 |
|---|---|
| [docs/architecture.md](docs/architecture.md) | 系統分層、後端、控制權、模組與環境劃分 |
| [docs/interfaces.md](docs/interfaces.md) | 設定規格、命令契約、leader 輸入 adapter、技能結果 |
| [docs/scenes-and-tasks.md](docs/scenes-and-tasks.md) | 資產、場地配置、任務定義與量測需求 |
| [docs/teleop-data-training.md](docs/teleop-data-training.md) | 遙操作、episode 資料、檢查與分割、ACT 訓練 |
| [docs/inference-dagger-evaluation.md](docs/inference-dagger-evaluation.md) | 自主推論、DAgger 閉環、再訓練比較與評估 |
| [docs/agent.md](docs/agent.md) | Agent、任務執行器、需求帳本與缺料流程 |
| [docs/isaac-ee-teleop.md](docs/isaac-ee-teleop.md) | Isaac Sim 最小場景與 keyboard／mouse EE 操作：環境、啟動、操作、驗收、已知問題 |
| [docs/sim-server.md](docs/sim-server.md) | 實驗設定檔、選單、主機端 uv 環境、模擬伺服器控制協定 |
| [docs/status.md](docs/status.md) | 已完成／預計實作／待確認、可沿用資產與參考來源 |
