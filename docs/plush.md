# 決賽玩偶：柴犬與江之電 305

決賽 demo 用的物件：方塊由程式夾取，玩偶由 ACT 夾取，執行器依物件種類選技能。本文件記錄玩偶規格，以及模擬端和實機端的用法。現場排程與照片原檔不放在 repo。

## 外形與尺寸

柴犬坐在江之電 305 電車裡的布面吊飾玩偶。下半部是有填充的方形車身，柴犬的頭和兩隻前腳從車頂露出來。原本附的珠鍊和紙吊牌已拆掉，示範與比賽都維持拆掉的狀態。

| 項目 | 數值 | 來源 |
|---|---|---|
| 長（電車長度） | 約 90 mm | 實測 |
| 寬（電車寬度） | 約 60 mm | 實測 |
| 車身高 | 約 50 mm | 實測 |
| 總高（含柴犬） | 約 100 mm | 實測 |
| 重量 | 28.6 g（不含珠鍊和吊牌） | 實測 |
| 柴犬頭寬 | 約 50–55 mm | 照片目測，待量 |

數值集中在 `agentic_sim/plush.py`，模擬和實機偵測共用。

方向性：
- 從上往下看，主軸是電車的長邊。
- 電車前後外觀相近，朝向以 180° 為週期（`yaw_fold` 2）。

## 系統裡的名字

- 物件代號 `plush`，繁中名稱「柴犬玩偶」。
- 聊天把它當成色票裡的一個「顏色」（`configs/chat.yaml`，`object: true`）。認得的說法有柴犬、狗狗、小狗、玩偶、布偶、電車、江之電、小熊。
- 回覆時不加「積木」。

## 夾法：從正上方夾柴犬頭的左右兩側

MuJoCo 研究在 `tools/sim_plush_grasp.py`。

| 夾法 | 結果 |
|---|---|
| 柴犬頭的左右兩側，從正上方下來 | 朝向 0–150° 全部成功：抬起約 52 mm，停 1 秒滑移不到 1 mm，夾爪收在約 30% |
| 車身寬邊（60 mm） | 失敗：手掌要在頭的上面，手指只碰得到車頂，斜的指面一夾就把車身往下推 |

- 夾爪要張到約 0.7 rad 以上（LeRobot 約 46%），指尖間距才有 68 mm。平常遙操作張開約 36%，間距 56 mm，比頭寬多不到 5 mm。
- SO-101 夾爪是單邊擺動，開得越大，可動的那一指尖端抬得越高。

## 決賽擺位：粉盤近右角

玩偶和方塊一起放在粉盤，位置在搆得到那一塊的近右角，也就是離基座最近、靠米盤的那一角。

- 模擬場景（`configs/scenes/lab_20261128_plush.yaml`）：
  - 位置 mount (215, 130) mm，r 0.25 m；
  - 車身中心到最遠的角是 54 mm，任何朝向都碰不到盤壁。
- 示範的變化範圍：x ±15 mm，y 0 到 +20 mm，朝向不限。
- 旁邊的方塊離車身 5 cm 以上。D455 會丟掉玩偶周圍 12 mm 內的方塊偵測，程式夾方塊時夾爪也需要空間。
- 場景的 `plush.block_xy` 在玩偶旁排了 4 格：(165, 215)、(205, 225)、(250, 205)、(170, 255) mm。
  - 4 顆都放得下，但方塊之間只剩約 1 cm，對程式夾取太擠。
  - 建議玩偶旁放 2–3 顆。
- 放入位置：米盤玩偶格位 P0（`configs/real_plush.json`），不和方塊格位 P1–P4 重疊。目前是由托盤尺寸推算的暫定值。

## 模擬

- 場景的 `plush:` 區段加入一個自由剛體：
  - `tray` 指定站在哪個托盤底板上；
  - `pos` 和 `yaw_deg` 是擺位；
  - `block_xy` 取代方塊的初始位置，避開玩偶。
- 沒有 `plush:` 的場景，模型和原本完全相同。
- 碰撞形狀：
  - 車身：圓角方塊；
  - 頭：橢球。
- 外觀：`assets/enoden305` 的網格和貼圖，只供顯示，不參與碰撞。
- 布面以高摩擦（1.0）近似，不做軟體模擬。

### 模擬示範與 ACT 筆數研究（2026-11-28）

用模擬量「成功率對示範筆數」，作為實機收幾筆的參考。模擬是剛體、光線一致、示範一致，成功率會比實機高，只能看趨勢。

**採集**（`tools/sim_plush_collect.py`，格式與擺位在 `agentic_sim.plush_sim_data`）：
- 每筆隨機擺位：x ±15 mm、y 0 到 +20 mm、朝向 0–180°，`block_xy` 隨機放 0–3 顆方塊。
- 起姿是場景 standby，每軸加 ±2°。
- 程式化夾頭：
  1. 夾爪開到 0.75–0.95 rad，經過頭上方 40–70 mm 的接近點；
  2. 下降、夾緊；
  3. 抬起 6 cm 停穩。

  停頓與速度隨機。抬 6 cm 而不是 5 cm，是為了讓 45 mm 判準有餘裕。
- 只收通過判準的示範：抬起 ≥ 45 mm，停留 1 秒內滑移 < 5 mm。
- 30 fps 直接寫成 LeRobot 資料集，features 與實機 ACT 資料集相同。
  - global 影像是 640×480，焦距取 1280×720 校正值的 0.6 倍（推算）。
  - 每次嘗試的 seed、擺位、結果都記在 `meta/sim_sources.jsonl`。
- 平行收的分片用 `tools/sim_plush_merge.py` 合併。

```bash
export MENAGERIE_DIR=<menagerie> MUJOCO_GL=egl
.venv-train/bin/python tools/sim_plush_collect.py --out outputs/plush/datasets/s1 --episodes 25 --seed 1000
.venv-train/bin/python tools/sim_plush_merge.py outputs/plush/datasets/s{1,2,3,4} --out outputs/plush/datasets/sim-plush-v1
.venv-train/bin/lerobot-train --config_path configs/real_training/act_right.yaml \
    --dataset.repo_id local/sim-plush --dataset.root outputs/plush/datasets/sim-plush-v1 \
    --dataset.episodes "[0,1,...,N-1]" --dataset.image_transforms.enable false \
    --output_dir outputs/plush/train/nN --steps 30000
.venv-train/bin/python tools/sim_plush_eval.py outputs/plush/train/nN/checkpoints/last/pretrained_model \
    --trials 50 --seed 900000 --json outputs/plush/eval/nN.json
```

**評估**（`tools/sim_plush_eval.py`）：
- 閉環執行 checkpoint，用實機的 `ActPolicy` 和 `real_policy.run` 的停止規則。
- 判定失敗類別：動作跳變、碰撞、抬不夠高、滑落、夾到車身、超時、沒夾到。
- 選項 `--cond-noise MM DEG` 在條件和目標圓上加偵測誤差。
- 選項 `--n-action-steps` 與 `--temporal-ensemble` 只在推論時改 ACT 的執行方式，不重新訓練。

**結果**：
- 設定：
  - 100 筆示範，前 N 筆訓練，每組 30000 步，batch 16。
  - 訓練時關掉影像增強：模擬影像一致，開了訓練慢三倍；實機訓練仍照 `act_right.yaml` 開著。
  - 每組評估 50 次，seed 900000–900049，和訓練用的 seed 不重疊。

| 訓練筆數 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|
| 成功（無偵測誤差） | 45 | 50 | 49 | 49 |
| 成功（偵測誤差 σ 5 mm／10°） | 42 | 48 | 47 | 48 |

- 25 筆以上已接近滿分，差 1–2 次不顯著。
  - 同一 seed 重跑結果不一定相同（GPU 推論有微小差異，閉環會放大）。
  - 50 次的成功率信賴區間約 ±5–10%。
- 失敗主要有三類：
  - **動作跳變**：
    - 新的一段動作（chunk）的第一步，和上一個命令差超過 6°。
    - 多發生在第 0 步：ACT 第一步的肘、肩轉預測和起姿有系統偏差，平均 3–4°，最大 5.6–6.5°。
    - 也會發生在第 100、200 步換段處。
  - **偵測誤差大**：xy 誤差約 11 mm 時，夾到頭偏側，抬起後滑落；每組都是同兩個擺位，和筆數無關。
  - **完成判定誤觸**：
    - 第一步的夾爪命令比起始值低 10% 以上，被當成已夾緊。
    - 手臂接著升到接近點並停住，在沒碰到玩偶時就判為完成。
- ACT 執行方式比較（只改推論）：
  - 原設定（每段 100 步全部執行）最好。
  - temporal ensembling 0.01：10 筆 39/50，100 筆 46/50。抬升被平均掉，多次抬不到 45 mm。
  - 每段只執行 25 步：10 筆 15/50，多半在起步的靜止段原地超時。
- 依這些結果，`agentic_sim.real_policy.Watch` 已修改，模擬評估和實機共用。改動後還沒重跑模擬評估。
  - 跳變 6°–15° 改為每步 6° 帶過去，超過 15° 才停。
  - 玩偶的夾爪命令要先張到 42%（`PLUSH_OPEN_PCT`）以上，之後的閉合才算夾緊。

## 實機

- **偵測**（`agentic_sim.real_plush`）：
  - D455 深度先轉成俯視高度圖，找出高 50–135 mm、底面長邊 65–130 mm、短邊 40–110 mm 的團塊；
  - 輸出中心、朝向和頭的位置；
  - 更高的團塊判為手臂或手；
  - 玩偶和手臂或手在俯視圖上要相隔約 3 cm，否則會連成一塊。
- **收示範**（`tools/lab_day.sh plush`）：
  1. D455 量好玩偶位置；
  2. 操作者從起姿夾起、抬穩後按 `c`；
  3. 程式依夾起時的高度把玩偶放到 P0。
- **轉檔**：
  - 條件是基座座標 mm 加上 sin/cos(2×朝向)；
  - 指令是 "Put the Shiba plush in the beige tray."（「把柴犬玩偶放進米色盤」）。
- **ACT 執行**（`agentic_sim.real_policy`）：
  - 夾起、抬穩後停止；
  - 動作一步跳 6°–15° 時，以每步 6° 帶過去；超過 15°、按 Esc 或超過 20 秒時，手臂原地停住；
  - 夾爪要先張到 42% 以上，之後的閉合才算夾緊。
- **任務伺服器**：用法見 `docs/real-task.md`。
  - `PLUSH_POLICY=<…/pretrained_model> tools/lab_day.sh task-server`；
  - 假桌面：`FAKE_TABLE=red,red,blue,yellow,plush`。
