# B 方案：腕部定位與 ACT 接手

## 抓一顆積木時誰做什麼

| 階段 | 輸入與輸出 | 執行位置 |
|---|---|---|
| 選料 | Agent 讀訂單、交付進度和盤面，執行器指定來源盤與顏色。 | 任務層 |
| 感知 | Global 找候選方塊；腕部影像配合同幀相機姿態估計 XY、yaw，回報是否通過配對。 | 共用感知模組 `perception/localize.py`，由各後端提供影像和相機幾何 |
| 抓前定位 B | 移到目標上方，拍第一張腕部影像，依有效估計修正懸停姿態，再拍第二張。得到抓取姿態與兩次觀測紀錄。 | 線上技能策略 `task/wrist_alignment.py`；IK、關節控制與相機取幀由後端執行 |
| 近端抓取 | 本次 MuJoCo 以共用 IK 腳本下降、閉爪、抬穩。預計改由 ACT 讀腕部影像與關節狀態並輸出動作段。 | ACT 線上推論放在 pick 技能下的 policy adapter。控制權交給 ACT 後，由 ACT 完成局部微調至抬穩。 |
| 搬運、放置、核單 | 執行器搬到目標盤並放置，感知以新畫面核對，確認後更新帳本。 | 決賽整單流程，另行驗收 |

B 使用固定規則，可直接執行。實驗使用全域 RGB、假設已知支撐面及方塊邊長。比賽規劃的 RealSense RGB-D 粗定位接 B 須另外驗證。第二次腕部估計有效時優先使用，否則沿用第一次有效估計，再退回粗定位。這個退回規則服務於基準比較；實機若兩次都無法確認目標，建議回報不確定，請執行器重新觀測後再授權下降。

**ACT 的接手點。** 實機先記粗定位停穩、接手、閉爪及抬穩四個時間點。以第一筆紀錄確認 B 修正後的實際手臂和目標相對姿態，落在人工示範的接手範圍內，避免 B 和 ACT 重複做同一段修正。ACT 預計只訓練「接手→抬穩」，完整抓放保留供讀回與對照。紅、藍、黃各採粉→米、米→粉每方向 20 筆，共 120 筆，依完整 episode 分割訓練與驗證。資料整理、ACT 訓練及 DAgger 再訓練在離線訓練環境執行，輸出 checkpoint、前處理與正規化 manifest。部署端只載入凍結版本推論。實驗 C 的兩階段擬合 Q 也是離線訓練，目前和 B 同為 41/42。

## 單獨復現 B

在 repo 根目錄使用 MuJoCo 環境，並將 `MENAGERIE_DIR` 指向含 `robotstudio_so101` 的 Menagerie 目錄。依賴見 `requirements-mujoco.txt`。Linux 無視窗可用 EGL。

~~~bash
MUJOCO_GL=egl python tools/benchmark_finals_alignment.py --method B --out outputs/b_wrist_5090
~~~

讀 `summary.json` 的 `methods.B_wrist` 及 `trials.jsonl` 的每次腕部觀測、最終估計和首次提起時間。這個入口直接評估 42 場，無需訓練或權重。完整比較沿用 [四法實驗](finals-alignment-benchmark.md)。

**5060 對照。** [42 個場景及逐場結果](finals-alignment-5060.jsonl) 含四顆積木的世界座標（公尺）、各自 yaw（度）、指定目標、人工加入的粗定位偏差和 B 的結果。目標索引只供評分。指定方塊首次升高至少 35 mm 並連續保持 0.40 模擬秒算成功。5060 成功 41/42，中位 5.383 模擬秒；藍色 73031 的兩次腕部估計均未通過。這是模擬首次提起的成績。抓放整單、ACT 和真機皆須分開評估。

## 設計參考

- ASPIRE 用執行紀錄檢查技能，Agentic WAM 展示程式與學習動作的混合執行。本案採用的分工是讓 agent 選技能，讓定位和接觸動作各有可讀回的輸出。[ASPIRE](https://research.nvidia.com/labs/gear/aspire/) · [Agentic WAM](https://agentic-wam.github.io/)
- Gym-Anything 將場景、任務和評分明確化；ENPIRE 將策略執行、評估與改善串成循環。本案保留固定場景和獨立評分，用同一組條件比較策略。[Gym-Anything](https://gym-anything.github.io/robotics/) · [ENPIRE](https://research.nvidia.com/labs/gear/enpire/)
- ACT 原作從遙操作示範學習動作段；LeHome 提供 SO-ARM101 跨模擬與實機的參考。本案的單臂積木資料、接手幀與成功條件需獨立驗證。[ACT](https://tonyzhaozh.github.io/aloha/) · [LeHome](https://ilialarchenko.com/projects/lehome2026/)

