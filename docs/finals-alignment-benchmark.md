# 決賽配置的 MuJoCo 首次提起比較

在 Windows 5060 專用 checkout 根目錄執行。以下命令沿用既有 5060 Python 環境和 Menagerie 模型。

~~~powershell
$env:MENAGERIE_DIR = (Resolve-Path '..\5060-assets\mujoco_menagerie').Path
$env:MUJOCO_GL = 'glfw'
$python5060 = '..\agentic-robotic-simulator-5060\.venv-5060\Scripts\python.exe'

& $python5060 tools/benchmark_finals_alignment.py --out outputs/finals_alignment_repro
~~~

基準程式先訓練擬合 Q 策略，再用固定權重評估。--policy 可指定既有的權重檔。--smoke 使用另外兩組各三個場景的種子範圍。

在原生視窗選用同一種四顆方塊配置，可執行下列命令。Reset 會依種子序列抽出下一組姿態。

~~~powershell
& $python5060 scripts/mujoco_ee_teleop.py --scene configs/scenes/lab_20261120.yaml --random-blocks --random-config configs/random_blocks_finals.yaml --random-seed 73000
~~~

## 場景與流程

- 粉色盤內兩紅、一藍、一黃。在限定半徑及工作區內，四顆各自抽取連續 XY 與 yaw。場景抽樣依旋轉後的方塊輪廓檢查盤邊距與方塊間距，再檢查夾取和接近的 IK。米色盤是場景中的放置區。這個實驗以首次穩定提起為終點。
- 每個種子對四種方法提供相同的起始姿態與目標要求。紅色要求先抓離基座較近的一顆，評分用場景真值決定目標身分。控制方法由 RGB 偵測選出離基座最近的指定顏色。訓練與評估種子範圍分開，目標類別依序循環紅、藍、黃。
- 每個場景的原生 D455 RGB 估計加入同一組合成擾動。XY 在半徑 14 mm 的圓盤中抽樣，yaw 在 ±20° 中抽樣。D 先以深度修正 RGB 選中的目標姿態，再加入同樣的擾動向量與角度。擾動幅度是測試設定，實機的誤差分布要另行量測。
- A 用全域 RGB 姿態直接夾取。B 用兩次腕部 RGB 觀測和固定修正。C 共享 B 的初始觀測、再次拍攝時機及視覺欄位，第一步選擇腕部視角修正幅度，再由第二次觀測選擇夾取點局部偏移。策略訓練使用模擬夾取回饋與兩階段擬合 Q，評估時以影像估計選動作。
- D 用同一個 RGB 遮罩選出目標，從 MuJoCo 的 D455 視角渲染對齊的公尺深度，反投影估計方塊頂面和中心高度。Yaw 沿用 RGB 輪廓估計。深度是理想模擬量測，實機的深度缺洞、環境光及配準誤差需要另行驗證。
- 四種方法最後使用同一套用 IK 計算的對準、下降、閉爪、抬升、保持動作。指定方塊中心至少升高 35 mm，連續達 0.40 模擬秒，就記錄首次穩定提起時間。從第一次全域觀測開始計時。準備場景與機械臂歸位已先完成。模擬秒衡量動作流程，主機秒包含渲染、影像處理、IK、策略推論和物理推進。

## 結果檔

- summary.json 記錄各路徑的成功數、失敗原因、配對結果、成功案例時間中位數，以及把失敗計至共同時間上限的全樣本平均。
- trials.jsonl 記錄每個種子四顆的初始 XY/yaw、目標、注入偏差、各次感測估計、所選動作和時間。配對比較時使用同一列的四種結果。
- training.jsonl 保存訓練場景、腕部估計與各動作的實際 MuJoCo 回饋。fitted_q_policy.npz 保存供第二次執行載入的策略係數。

## 本輪結果

Windows 原生 MuJoCo 4.0.1 搭配 RTX 5060 Laptop GPU。訓練用種子 47000–47053，共 54 個場景、810 次動作路徑。四法評估共用 42 個保留場景（73000–73041，紅、藍、黃各 14 個）。

| 方法 | 首次穩定提起 | 成功案例中位數（模擬／主機秒） | 全樣本限制平均（模擬／主機秒） |
| --- | ---: | ---: | ---: |
| A 全域 RGB 直抓 | 31/42（73.8%） | 4.767／0.156 | 5.618／1.687 |
| B 腕部 RGB 固定修正 | 41/42（97.6%） | 5.383／0.476 | 5.456／0.634 |
| C 腕部 RGB 擬合 Q | 41/42（97.6%） | 5.383／0.472 | 5.457／0.600 |
| D 全域 D455 RGB-D 直抓 | 32/42（76.2%） | 4.767／0.167 | 5.546／1.557 |

全樣本限制平均把未成功抓取的場景計到 8 個模擬秒和 6 個主機秒。B、C 都救回 A 的 10 個失敗場景，兩者的唯一失敗同為藍色種子 73031，兩次腕部影像都未取得通過門檻的姿態。C 在 32 個場景選了額外抓點偏移，與 B 的成功場景完全相同。D 的深度在 42 個場景均可用，相較 A 多成功一個場景。加入擾動前的全域 XY 誤差中位數是 A 0.634 mm、D 0.478 mm。共用擾動後分別是 8.875 mm、9.084 mm。

四法的 [摘要](../outputs/finals_alignment_abcd/summary.json) 和 [逐場紀錄](../outputs/finals_alignment_abcd/trials.jsonl) 相對 checkout 路徑是 `outputs/finals_alignment_abcd/`。此次評估載入 [凍結策略](../outputs/finals_alignment/fitted_q_policy.npz)，其 [訓練紀錄](../outputs/finals_alignment/training.jsonl) 同在 `outputs/finals_alignment/`。

~~~powershell
& $python5060 tools/benchmark_finals_alignment.py --policy outputs/finals_alignment/fitted_q_policy.npz --out outputs/finals_alignment_abcd_repro
~~~
