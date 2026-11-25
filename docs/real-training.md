# 實機資料轉換與 ACT 訓練（訓練端）

把實機 episode（`control.jsonl`、各相機 `frames.jsonl` 與影像、`manifest.json`、`review.json`）轉成 LeRobot 資料集，再沿用 LeHome 的 ACT 設定訓練。目前實機正式示範是 0 筆，整條流程先用假裝置產生的 episode 打通。

## 環境

訓練端使用獨立的 venv，和實機執行端、模擬端的環境分開，安裝步驟見 [`requirements-train.txt`](../requirements-train.txt)。2026-11-25 在 RTX 5090 實測：

| 項目 | 版本 |
|---|---|
| Python | 3.10.14 |
| PyTorch | 2.7.1+cu128（CUDA 12.8，含 Blackwell sm_120） |
| torchvision | 0.22.1+cu128 |
| LeRobot | 0.4.3（和 LeHome 的 ACT 設定、`requirements-real.txt` 相同） |
| GPU／驅動 | RTX 5090／580.126.18 |

LeRobot 0.4.3 要求 PyTorch 低於 2.8。PyPI 上的 2.7.1 是 CUDA 12.6 版，不支援 5090，所以 PyTorch 要先從 cu128 套件來源安裝，再裝 LeRobot。

## 轉換

```bash
.venv-train/bin/python tools/real_to_lerobot.py outputs/261124-right --out outputs/datasets/right-act-v1
.venv-train/bin/python tools/check_real_dataset.py outputs/datasets/right-act-v1      # 逐幀和原始 episode 比對
```

**欄位對應**：每個 `control.jsonl` 列對應一個訓練幀，順序照 LeRobot 的「先觀測、再送動作」。

| LeRobot 欄位 | 來源 |
|---|---|
| `action`（6） | 該列的 `arms.<臂>.command` |
| `observation.state`（6） | **上一列**的 `arms.<臂>.readback`。實機每一列都是先送命令再讀回，所以上一列的讀回才是送出這個命令前看到的狀態 |
| `observation.images.<相機>` | 各相機影格中，離上一列讀回時間最近、而且在這列命令送出前拍到的那張。超過 `--max-offset-ms`（預設 25 ms，半個 20 Hz 週期）就排除整筆 |
| `timestamp`、`frame_index` | 依名目控制頻率排列；原始時間存在 `meta/trc_sources.jsonl` 的列號裡 |
| `task_is_policy`、`is_handoff` | DAgger 加權用的控制來源；交接幀標記 |
| `task` | 由任務卡產生的語言指令，預設英文，例如 "Put the red block in the beige tray."；`--language zh` 改成中文「把紅色方塊放進米色盤」。兩種都存在 `meta/trc_sources.jsonl` |

- 關節名稱與順序照 manifest 的 `joint_names`。
- 單位維持錄製時的值：五個身體關節是校正後的度數，夾爪是開啟 %，和 LeRobot SO-101 的慣例相同。
- **不經過暫定的關節映射**（`real_joints`），也不計算末端位姿。
- 只轉 RGB，深度本輪不轉。

**選段**（`--segment`）：

| 模式 | 範圍 |
|---|---|
| `handoff`（預設，ACT 用） | 從 `review.json` 的交接幀到 `training_range` 的終點 |
| `review` | 整個 `training_range` |
| `full` | 整筆，從第 1 列開始（第 0 列之前沒有讀回可當觀測） |

**起點抖動**（`--handoff-jitter k`）：另外從交接幀前、後各 1…k 幀起算，各存一段。原始 episode 保留了整段接近過程，所以同一筆示範可以涵蓋略有不同的起手姿態。

**排除規則**：每一筆都會印出收錄或排除，原因同時寫進 `meta/trc_conversion.json`。下列情況會排除：

- 讀回檢查沒通過；
- 操作者取消；
- 沒有 `review.json`（`full` 模式除外）；
- 控制間隔超過 1.5 個週期；
- 相機配對超出容許；
- 臂名不符；
- 錄製模式不在 `--modes` 裡（預設只收 `demonstration`；DAgger 回合要加 `--modes demonstration,dagger`）；
- `source: fake`（除非加 `--allow-fake`，只限流程測試）。

**追溯**：
- `meta/trc_sources.jsonl`：每個 LeRobot episode 一行，記錄來源 episode、選段、起點偏移、交接幀、`review.json` 各標記、任務卡、兩種語言的指令、程式版本、每幀對應的控制列與相機影格，以及相機配對時間差。
- `meta/trc_conversion.json`：轉換選項、版本、觀測規則，以及收錄和排除清單。
- 一次轉換的所有 episode 必須有相同的控制頻率與相機組；不同的要分開轉。

## 訓練與推論檢查

```bash
.venv-train/bin/lerobot-train --config_path configs/real_training/act_right.yaml \
    --dataset.root outputs/datasets/right-act-v1 --output_dir outputs/train/act-right-v1
.venv-train/bin/python tools/check_act_checkpoint.py outputs/train/act-right-v1/checkpoints/last/pretrained_model \
    outputs/datasets/right-act-v1
```

- `configs/real_training/act_right.yaml` 取自 LeHome 的 `train_act.yaml`（batch 16、30000 步），改成單臂。輸入輸出形狀由資料集讀取，任何相機解析度都能用。數值沒有針對本任務調過（`provisional`）。
- `act_right_smoke.yaml` 只用來檢查流程。
- `check_act_checkpoint.py` 會載入 checkpoint 和它存下的前後處理器，對資料集影格跑 `select_action`，確認推論路徑可用。不評估品質。

## 假資料驗證（2026-11-25）

```bash
python3 -S tools/real_teleop.py fake --arms right --output outputs/fake-training/fake-right
python3 -S tools/real_collection.py --fake --demo --session outputs/fake-training/session
.venv-train/bin/python tools/real_to_lerobot.py outputs/fake-training --out outputs/datasets/fake-act --allow-fake --handoff-jitter 1
.venv-train/bin/python tools/check_real_dataset.py outputs/datasets/fake-act
.venv-train/bin/lerobot-train --config_path configs/real_training/act_right_smoke.yaml
.venv-train/bin/python tools/check_act_checkpoint.py outputs/train/act_right_smoke/checkpoints/last/pretrained_model outputs/datasets/fake-act
```

| 步驟 | 結果 |
|---|---|
| 轉換 | 4 筆假 episode：收錄 2 筆示範（D001、D002），各 3 段起點抖動，共 6 段 36 幀；手動短錄與 DAgger 回合因錄製模式被排除，並寫出原因 |
| 讀回核對 | 6 段、36 幀全部一致：狀態、動作、語言指令、交接標記。影像經影片編碼後，每個像素通道的平均差最大 2.15（0–255） |
| 訓練 | ACT 52M 參數，20 步，loss 55 → 9，checkpoint 存檔，約 2 秒 |
| 推論 | 從 checkpoint 載入，在 `cuda:0` 上輸出 6 維動作，數值有限 |
| 單元測試 | `tests/test_real_dataset.py`：選段、觀測配對、起點抖動、排除原因、語言指令 |

假資料只有 6 幀的訓練段，影像也是合成的，只能證明流程能跑通，不代表訓練結果。

## 之後

- **VLA**（GR00T、π0.5）：需要的資料版本可能和 LeRobot 0.4 寫出的 v3.0 不同。原始 episode 是唯一真值，需要時另外轉，不必重錄；語言指令兩種都已保存。
- **ACT 的訓練段終點**（抬穩或含放置）還沒決定。轉換工具沿用 `review.json` 的 `training_range`，改成哪種只要重新標記或改選段。
