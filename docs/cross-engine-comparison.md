# 跨引擎比較（Isaac Sim vs MuJoCo）

對應 issue [#3](https://github.com/jxuan083/agentic-robotic-simulator/issues/3) 的 P4。同一條腳本化的 EE 目標序列分別送進 Isaac 與 MuJoCo，比較 EE 追蹤、關節軌跡、夾爪閉合角、積木軌跡與放置結果。**只記錄差異，不為了讓兩邊一致而調參數。**

場景數值都是 `provisional`，兩邊的關節增益也都不是實機量測值；這份比較說的是「兩個模擬器之間差多少、差在哪」，不是哪一個比較像實機。

## 方法

- **輸入**：`configs/trajectories/*.yaml`，關鍵幀之間用 smoothstep 內插（關鍵幀速度為 0）。軌跡直接送進共用的 EE 控制器（`TrajectoryDriver`），取代鍵盤與滑鼠輸入，用**模擬時間**取樣，所以和視窗焦點、畫面更新速度無關。
- **兩個引擎相同的部分**：場景 YAML、共用運動學（FK 差 < 0.01 mm）、IK、限速、60 Hz 控制、240 Hz 物理、底座貼桌。
- **不同的部分**：物理引擎、關節驅動增益（Isaac 用 Workshop 的設定，MuJoCo 用 Menagerie 的 kp 998）、接觸模型與參數。
- **兩條軌跡**：
  - `pick_place`：和抓放驗收相同的目標（接近、下降、夾取、抬起、搬運、放下、退回），16 秒。
  - `reach_sweep`：不碰積木，掃過垂直區、遠處傾斜（最大傾角約 42°）、高處 0.22 m、左右兩側，每個點停一下，20 秒。
- **重複性**：每條軌跡在每個引擎跑 3 次。
- **只比物理**：MuJoCo 逐幀重播 Isaac 第 1 次的關節與夾爪命令（`CommandReplay`，不經 IK），兩邊命令完全相同（差 0.0 rad）。

```bash
tools/run_trajectory.sh isaac  pick_place 1            # GUI，錄影只錄模擬器視窗
tools/run_trajectory.sh mujoco pick_place 1
uv run scripts/mujoco_ee_teleop.py --headless --trajectory outputs/compare/pick_place/isaac_1.csv \
    --log outputs/compare/pick_place/mujoco_replay.csv --result outputs/compare/pick_place/mujoco_replay.json --exit_after_script
DISPLAY= ISAAC_CONTAINER_NAME=ars-plot docker/isaac/run.sh /isaac-sim/python.sh tools/compare_engines.py
```

輸出在 `outputs/compare/<軌跡>/<引擎>_<次>.{csv,json,mp4}`，圖與摘要在 `outputs/figures/compare/`。

## 結果（2026-11-18）

| 項目 | Isaac | MuJoCo |
|---|---|---|
| `pick_place` 放進米盤 | 3/3 | 3/3 |
| 重複 3 次，EE 軌跡最大差 | 0.0 mm | 0.0 mm |
| 追蹤誤差，靜止時 平均（最大） | `pick_place` 4.3（6.2）mm；`reach_sweep` 7.0（12.5）mm | `pick_place` 0.1（0.3）mm；`reach_sweep` 0.2（3.7）mm |
| 追蹤誤差，移動中 平均（最大） | 4.6（6.9）／7.1（13.3）mm | 0.4（0.7）／1.1（9.1）mm |
| 靜止時 z 下垂（平均） | 3.6 mm／6.4 mm（遠處傾斜時約 11 mm） | 0.1 mm／0.2 mm |
| 夾住積木時的夾爪角 | 0.188 rad | 0.145 rad |
| 積木抬起最高 | 90.5 mm | 88.4 mm |
| 積木最後位置 | (0.188, −0.120) | (0.190, −0.113) |

兩個引擎之間：
- 同一條軌跡：EE 距離平均 4.4 mm（`pick_place`）、6.7 mm（`reach_sweep`），最大 13.6 mm；積木位置最大差 10 mm。
- 只比物理（相同命令）：EE 距離和上面幾乎一樣（平均 4.4／6.7 mm），所以差異幾乎全部來自物理與增益，不是 IK。

圖（交接資料夾 `figures/compare/`）：`<軌跡>_overview.png`（高度、追蹤誤差、夾爪角或下垂）、`<軌跡>_joints.png`（五個關節，實線量測、虛線命令）、`pick_place_block.png`（積木位置）、`<軌跡>_physics_only.png`（相同命令）。

## 觀察

1. **主要差異是關節增益**：Workshop 增益（Isaac）軟，重力下垂隨伸展增加，從 home 約 3 mm 到遠處約 12 mm；Menagerie 增益（MuJoCo）硬，靜止誤差 0.1–0.2 mm。兩者都不是實機量測，實機的下垂要在實驗室量。
2. **夾持接觸不同**：MuJoCo 的夾爪多壓進約 0.04 rad（0.145 vs 0.188 rad），是 Menagerie 夾爪的接觸設定（condim 6、priority）與 PhysX 的差異。兩邊都靠物理接觸夾住，搬運中積木都慢慢往下滑一點（Isaac 約 2 mm、MuJoCo 約 3 mm）。
3. **傾斜切換處的命令抖動（共用控制器）**：`reach_sweep` 開始和結束傾斜的地方（Wrist_Pitch 頂到極限附近），IK 的解在相鄰幀間跳動，命令的二階差分超過 0.01 rad 的幀兩個引擎都約 75 幀。軟的 Isaac 把它濾掉，硬的 MuJoCo 照著做，EE 出現最大約 9 mm 的短暫抖動。這是控制器的平滑度問題，不是引擎問題。
4. **IK 在手腕極限附近對參數很敏感**：Isaac 的運動學鏈從 USD 讀、MuJoCo 從匯出的 YAML 讀，數值只差浮點誤差，但在傾斜區兩邊的命令最多差 0.047 rad；把關節極限改 1e-7 rad 也會得到同樣大小的差。其他地方命令一致（`pick_place` 全程差 1e-5 rad）。
5. **兩個引擎都完全可重現**：同一條軌跡跑 3 次，EE 軌跡一模一樣（CPU PhysX 與 MuJoCo 都是確定性的）。

## 暫不處理（待實機數據）

以下項目等實驗室量測實機數據與環境後再處理，目前不為了讓兩個引擎一致而調參數：

- **關節增益**：量測實機在 home、遠處、傾斜等姿態的重力下垂與追蹤誤差，再決定兩個引擎的增益。
- **夾爪接觸**：量測實機夾住積木時的閉合角與是否下滑。
- **傾斜切換處的命令抖動**：可在傾斜區平滑命令，或在 IK 加入讓解保持連續的項；會改變操作手感，和增益一起調。
- **場景**：實驗室佈置（雙臂、墊板、容器等）確認後，用同樣的軌跡重跑這份比較。

## 檔案

| 路徑 | 內容 |
|---|---|
| `configs/trajectories/pick_place.yaml`、`reach_sweep.yaml` | 比較用的 EE 目標序列 |
| `src/agentic_sim/teleop/trajectory.py` | 軌跡載入與取樣、`TrajectoryDriver`、`CommandReplay`、結果格式 |
| `tools/run_trajectory.sh` | 在某個引擎跑一條軌跡（GUI、錄影只錄模擬器視窗） |
| `tools/compare_engines.py` | 對齊、計算與畫圖（Isaac 容器內執行，那裡有 matplotlib） |
| 兩個 session 的 `trajectory` 選項、`scripts/*_ee_teleop.py --trajectory` | 軌跡模式：取代所有輸入，跑完寫結果 JSON |
