# 實驗室環境一鍵入口、聊天視窗與相機畫面

一個指令就能開好整套實驗室環境：模型服務、模擬伺服器（含常駐相機畫面）和聊天視窗。用聊天視窗下單後，模擬手臂完成任務，可以從頭走一次競賽流程。場景與相機見 [lab-scene-cameras.md](lab-scene-cameras.md)，聊天規則見 [chat-layer.md](chat-layer.md)。

## 使用

```bash
uv run tools/lab.py                                       # MuJoCo、Q4 模型、lab_20261120、預設積木
uv run tools/lab.py --backend isaac --model bf16 --block-colors red,blue,blue,yellow
uv run tools/lab.py --wrist-lens pinhole                  # 腕部畫面用理想針孔
uv run tools/lab.py --no-chat                             # 只開模擬器
```

`tools/lab.py` 依序啟動下列元件，每一步都等前一步回應才繼續；任一步失敗就停下並說明原因：

1. **模型服務**：
   - Q4_K_M：llama.cpp，`docker/llamacpp/serve.sh`，port 8001，預設。
   - BF16：vLLM，`docker/vllm/serve.sh`，port 8000。
   - 已經在執行就直接沿用。
2. **模擬伺服器**：`tools/mujoco_server.sh` 或 `docker/isaac/server.sh`，再依參數載入 experiment（預設 `lab_20261120`）。已經在執行就沿用，並用這次的參數重新載入。
3. **聊天視窗**：`tools/chat_window.py`。

結束方式：關掉聊天視窗或按 Ctrl+C。只關這次啟動的元件，原本就在執行的服務不動。用 `--no-chat` 時，以 Ctrl+C 或關掉模擬器視窗結束。

`tools/sim.py`（選單、驗收）與 `tools/chat.py`（命令列、對話腳本）照舊可用。

## 聊天視窗

- 是獨立的 Tk 小視窗，和模擬器在不同程式中。可以單獨關掉、重開，不影響模擬器。
- 視窗內容：
  - 狀態列：模擬器、模型、暖機時間、連線狀態。
  - 待回答的問題列：黃色。
  - 對話紀錄。
  - 輸入框：按 Enter 送出。
- 規則和命令列聊天相同，沿用同一個 `ChatAgent`：
  - 回答綁定開始打字時畫面上的問題。打字途中換題，那句不會送出。
  - 追問綁定發問的題號。
  - 重開視窗時，先顯示目前待回答的問題。
- 單獨開：先開模擬器和模型服務，再執行 `uv run tools/chat_window.py --backend mujoco`。

## 相機畫面（常駐）

- 場景有感測相機時，D455 global 與腕部畫面固定在主畫面內，不接受操作：

| 引擎 | 位置 |
|---|---|
| MuJoCo | 3D 視圖右上角 |
| Isaac | 停靠在 3D 視窗下方，左邊 global、右邊腕部，沒有關閉鈕 |

- Isaac 的相機視窗只建立一次。Reset、換積木數量、伺服器重建 session 時，只把視窗改指到新 session 的相機；換到沒有相機的場景時先隱藏。

## 腕部畫面的鏡頭變形

腕部相機實機有桶形變形。模擬預設 `real` 模式，讓腕部畫面看起來像實機原始影像；`pinhole` 模式是理想針孔。

- **切換方式**：
  - `tools/lab.py --wrist-lens`
  - 兩個 teleop 腳本的 `--wrist_lens`
  - experiment 選項 `wrist_lens`
  - 預設值：場景檔的 `sensor_cameras.wrist.lens_mode`
- **鏡頭模型**：OpenCV 徑向模型，fx = fy = 400、k1 = −0.15（`estimated`）。
  - 依據：在 38 張 11/20 腕部實拍上，這組參數同時讓直邊變直，夾爪位置也和模擬一致。
  - 公開的 U20CAM 640×480 校正值（f 550、k1 −0.448）也能讓直邊變直，但模擬夾爪偏小偏下（重疊率 0.33），只留作參考。
  - 之後要用棋盤格校正我們自己的相機來取代。
- **MuJoCo 的做法**：先在畫面外算一張較寬的針孔畫面（960×720，垂直 85.7°），再逐像素重新取樣成 640×480。實作在 `src/agentic_sim/wrist_lens.py`。
- **Isaac 的做法**：腕部相機套用 RTX 的 OpenCV pinhole 鏡頭模型（`OmniLensDistortionOpenCvPinholeAPI`），係數相同。
- **兩個引擎一致**：用格線測試比對，相差不到 1 px。

## 實驗室場景的起始姿態

- 手臂從實驗室的收折休息姿態開始，即場景 `robot.start_pose_lerobot`（實機讀值），再以 0.8 rad/s 移到操作起點，約 1.3 秒。
- 夾取方向沿用切線方向（`robot.grasp_heading_deg: 90`），所以 home 時腕部旋轉約 93°。
- 實驗室操作者習慣把手腕放在 0° 左右，但模擬中這個方向在需要傾斜夾爪的遠處積木會滑落：兩指在傾斜平面內，一高一低。
- 積木角度不固定時，執行器會在每 90° 對稱的方向中挑最近的一個，手腕最多轉 ±45°。

## 驗收

- **競賽流程**（兩個引擎都用 `tools/lab.py` 一鍵啟動，用 Q4 模型）：
  1. 積木為一紅兩藍一黃，在聊天視窗打「我要兩顆紅色、一顆藍色」。
  2. 交付一紅、一藍後，跳出缺料問題。
  3. 在聊天視窗回答「換成黃色」。
  4. 交付黃色，帳本完成（一紅一藍一黃）。

  | 引擎 | 從下單到完成 |
  |---|---|
  | MuJoCo | 約 32 秒 |
  | Isaac | 約 56 秒 |

  - 這次的輸入由聊天視窗的 `--type` 依序打進輸入框，和手動打字走同一條路徑。
  - 關掉聊天視窗後，入口關掉它開的模擬伺服器和模型服務。
- **實驗室場景的任務情境**（MuJoCo，開發檢查）：缺料換色 8/8、持物交付 10/10、下降中改單 9/9。
- **Isaac 伺服器連續重建 3 次**（實驗室 → 最小場景 → 實驗室）：相機畫面保留在原位，並顯示新場景。
- **測試**：132 項全過。
- **最小場景的 GUI 回歸**：
  - MuJoCo：smoke 2/2、keyboard_pick_place 8/8。
  - Isaac：smoke 2/2、keyboard_pick_place 8/8、mouse_pick_place 10/10。

## 限制

- 鏡頭參數是由實拍推算的，還沒有用棋盤格校正。腕部相機的安裝位姿沿用官方相機座模型。
- 驗收用的是自動輸入。實際使用時，模型對新句型的判斷仍以 [chat-layer.md](chat-layer.md) 的語句表結果為準。
- 聊天視窗只送任務命令；要讓手臂停下，用模擬器的 Pause（P）。
