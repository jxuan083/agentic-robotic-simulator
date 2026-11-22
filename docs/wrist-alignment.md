# B 方案：腕部定位與 ACT 接手

本輪使用 `tools/benchmark_verified_overhead.py` 重播 234 場及 24 場相鄰紅塊。5060 分別為 234/234（首次 233）與 19/24（5 場停止），兩組抓錯鄰塊均為 0。[逐場對照資料](nearest-color-5060.jsonl) 含四顆的場景座標、顏色、yaw、注入偏差及首次／重抓穩定提起的物件 ID，供 Linux 與 Isaac Sim 復現。下方 42 場單獨入口保留最初的 B 基準。

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


## 擴增擺位與下降前的正上方驗證

在 MuJoCo 專用環境的 repo 根目錄執行。輸出使用新的目錄。

~~~bash
python tools/benchmark_verified_overhead.py --out outputs/verified_overhead_run
~~~

場景表在 `configs/benchmarks/finals_overhead_suite.yaml`。原粉盤 42 場作回歸；另設四組各 48 場，各組紅、藍、黃每色 16 場。四顆方塊各自隨機放置並旋轉。小間距組把設定的盤邊留白／鄰塊間距從 16／27 mm 降至 5／7 mm，這些數字是抽樣下限。大誤差組將初始全域姿態擾動上限由 14 mm／20° 提高至 22 mm／30°。米盤組改從米盤選料，仍採同一套估姿與抬升判定。

1. 全域粗定位後接近指定方塊。若全域沒有候選，以腕部查看來源盤的已知區域；本基準配置固定為兩紅一藍一黃，找紅時要看到兩顆再比到基座的距離。此搜尋條件只適用這個配置。
2. 腕部估姿失敗或第一次懸停不到位時，改變觀看位置、用新腕圖重新定位。換視角後仍找不到可靠姿態就結束本次嘗試。
3. 用確認後的估計命令手臂張爪移回目標正上方（EE 高度 80 mm，以場景座標為準）。讀實際關節值做正向運動學；控制點對影像估計的水平誤差、高度誤差各須 ≤3 mm，手臂各關節速率 ≤0.15 rad/s。通過後才下降。在下降階段沿用原腳本的固定指側 5 mm 抓點補償。量測點是手臂控制點；手腕允許傾斜。
4. 抬升後若爪位接近空爪閉合端（距閉合命令 ≤0.05 rad），再換視角確認目標仍在盤內。只有重新認得目標，才張爪回正上方，允許一次重抓。原始與重抓結果分開紀錄。

讀 `trials.jsonl` 的 `views` 看腕圖是否取得姿態，`retry` 看換視角原因，`hovers` 看實際到位情形，`first_grasp` 和 `empty_grasp_retry` 看接觸結果。`scored_xy_error_mm` 是事後用 MuJoCo 方塊真值核對的水平誤差，控制端只使用影像姿態與關節回讀。穩定提起用同一門檻：指定方塊中心升高至少 35 mm 並保持 0.40 模擬秒。

### 相鄰色塊與同色選取

先依訂單篩來源盤與顏色，再按候選中心到基座旋轉軸的桌面 XY 距離排序。初始全域影像只看出部分同色候選時，先用腕部複查。全域已通過擬合的候選會帶入複查，以桌面座標關聯同一顆（本場景積木邊長 29.5 mm，兩筆估計的桌面中心距離小於一個邊長時合併，腕部估計取代全域估計），每顆只計一次；每次新的盤面觀測都重新建立候選。本基準事先知道兩紅一藍一黃，這項數量檢查用於此次試驗。現場應以當下已核對的庫存為準。

選定後保存目標與同色鄰塊的估計位置。新的腕部估計須在配對距離內，且至少比其他同色候選更接近鎖定目標 3 mm。一般換視角仍無法辨識時，利用 10 cm 內的已知鄰塊估計投影，在可達姿態中最多再取兩個視角。排序依序是：目標與同色鄰塊的投影輪廓是否分開（間隙 ≥8 px，同色相接會被分割成同一個輪廓）、與任一鄰塊的重疊、同色鄰塊是否也完整在畫面內、目標離畫面邊緣的距離。確認後才回到目標正上方，辨不清就停止。

要重播相鄰紅塊專測，執行 `python tools/benchmark_verified_overhead.py --suite configs/benchmarks/adjacent_red_holdout.yaml --out outputs/adjacent_red_repro`。這 24 個場景另從新種子依序挑出，兩紅塊中心距離 ≤65 mm。以每次實際穩定離盤的物件 ID 核對選擇，若提起遠顆或連同鄰塊一起提起，另記錯抓。加入全域候選帶入與同色輪廓分離排序後，以原擺位重跑：相鄰紅塊由 19/24 提高到 21/24（恢復 90012、90015），主測仍為首次 233/234、含一次重抓 234/234，零退步、零錯抓。米盤 91006、91010、91026 的兩紅在全域已合併，掃描視角也未拍到分開的近顆，仍在下降前停止。

### Isaac Sim 重播：換視角後回到近顆正上方

`tools/isaac_nearest_hover.py` 在 Isaac Sim 內原樣執行同一個 `trial()`。四顆的位置、yaw、顏色和注入的粗定位偏差都依種子取自 [nearest-color-5060.jsonl](nearest-color-5060.jsonl)。控制只讀現有全域／腕部相機畫面（使用同一幀的相機矩陣）和實測關節；方塊真值只用於評分與疊圖。在 repo 根目錄用 Isaac 容器執行，有視窗時開 GUI，加 `--headless` 則不開。每次指定新的輸出目錄。

~~~bash
docker/isaac/run.sh /isaac-sim/python.sh tools/isaac_nearest_hover.py --seeds 85000 --lift_comp_mm 6 --out outputs/isaac_nearest_85000
docker/isaac/run.sh /isaac-sim/python.sh tools/isaac_nearest_hover.py --seeds 90010 --lift_comp_mm 6 --out outputs/isaac_nearest_90010
docker/isaac/run.sh /isaac-sim/python.sh tools/isaac_nearest_hover.py --seeds 91053 --lift_comp_mm 6 --out outputs/isaac_nearest_91053
~~~

三場（85000 粉盤近顆 ID 0、90010 粉盤 ID 1、91053 米盤 ID 0）都會換視角後重新確認近顆。原增益下，Isaac 懸停穩定在 73～74 mm，低於 80 mm 目標 6～7 mm（肩抬、肘關節的重力穩態誤差，保持 1 秒不變），下降前判定不放行。

`--lift_comp_mm 6` 是 Isaac 端的選用實驗，不改 `trial()`。只有在原判定未通過、已停穩、目標已確認且 XY ≤3 mm、只差高度偏低時，才補一次：補償量取「80 mm − 實測高度」與 6 mm 的較小值，從實測關節解 IK，無解就停。下降門檻仍以 80 mm 為準：實測高度 77～83 mm、EE 對影像估計 XY ≤3 mm、最大關節速率 ≤0.15 rad/s，三項須同一幀成立。

| 種子 | 補償前 | 補償量 | 補償後 | 真值 EE 對近顆 XY | 提起 ID |
|---|---:|---:|---:|---:|---|
| 85000 | 74.01 mm | 5.99 mm | 80.02 mm | 2.59 mm | 0（近顆） |
| 90010 | 73.04 mm | 6.00 mm | 79.06 mm | 2.54 mm | 1（近顆） |
| 91053 | 73.28 mm | 6.00 mm | 79.30 mm | 2.67 mm | 0（近顆） |

三場的腕部俯仰命令都在模型上限 95°，補償由肩抬與肘關節完成。這是模擬結果；是否把高度補償納入抓取流程，待同姿態的真機量測對照後再決定。

輸出目錄的 `summary.json` 逐場列出狀態、是否換視角、補償前後高度、是否下降及提起的 ID。`<seed>/trial.json` 是完整試驗紀錄；`hovers[-1].compensation` 含補償前後的關節命令與回讀，以及到關節限位的餘量。`<seed>/frames.json` 和各腕圖 PNG 記錄每張腕圖的模擬時間、實測關節、EE 位置及影像估計。`--hold_s 1.0`（不可與補償同用）會改為記錄懸停 1 秒內每一控制步的高度曲線。
