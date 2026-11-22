# Isaac Sim 最小桌面場景與 keyboard／mouse EE 操作

對應 issue [#1](https://github.com/jxuan083/agentic-robotic-simulator/issues/1)。單支 SO-101、桌面、儲存盤（粉）、放置盤（米）與一顆積木；以 keyboard 或 mouse 控制夾爪抓取中心（EE），完成一次手動抓放。

> 場景尺寸與位置都是 **provisional**（見 [場景數值](#場景數值provisional)），不是實機驗證。模擬結果不代表實機可行。

不在本範圍：資料集錄製、ACT、DAgger、agent、leader、controller、跨機操作。MuJoCo 版見 [mujoco-backend.md](mujoco-backend.md)（同一份場景、按鍵與控制器）。

## 測試環境與版本

| 項目 | 版本／實測值 |
|---|---|
| OS | Ubuntu 22.04.3 |
| GPU／驅動 | NVIDIA RTX 5090 32 GB／580.126.18 |
| 容器 image | `nvcr.io/nvidia/isaac-lab:2.3.2` |
| Isaac Sim | 5.1.0-rc.19+release.26219（image 內建） |
| Isaac Lab | 0.54.2（image 內建，`isaaclab` 套件版本） |
| PyTorch／NumPy | 2.7.0+cu128／1.26.0（image 內建） |
| SO-101 資產 | [NVIDIA Sim-to-Real SO-101 Workshop](https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop) `ce807d9` |

版本以實測為準並鎖定，不跟著上游自動升級。這個 image 也是 Workshop 官方模擬容器所用的版本。在已有 Isaac Sim 5.1.0 image layer 的主機上，pull 這個 image 只會多佔約 3 GB。

## 資產來源與授權

- 機械臂：Workshop 的 `source/sim_to_real_so101/assets/usd/SO-ARM101-USD-NO-CAMERA.usd`，關節驅動參數沿用 Workshop `assets/so101.py` 的 `SO101_CFG`。
- 授權：Workshop repo 採 Apache-2.0（NVIDIA）。USD 沒有另附授權檔，所以依 repo 層級的授權處理。
- 本 repo **不複製**資產檔，由 `docker/isaac/fetch_assets.sh` 在固定 commit 取得，執行時以唯讀方式掛進容器。
- 桌面、托盤、積木都是程式產生的方塊幾何，不需要外部資產。

## 安裝與啟動

在模擬端主機的桌面 session 執行（需要 X display）：

```bash
# 1. 取得 image（一次）
docker pull nvcr.io/nvidia/isaac-lab:2.3.2

# 2. 取得 SO-101 資產（一次；預設放在 ~/ref/Sim-to-Real-SO-101-Workshop，可用 SO101_WORKSHOP_DIR 改）
docker/isaac/fetch_assets.sh

# 3. 啟動操作介面
export DISPLAY=:1                                  # 桌面的 display，依主機而定
docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py
```

- 第一次啟動要編譯 shader，需要數分鐘。快取存放在 `~/docker/isaac-lab-2.3.2/`，之後啟動到場景可操作約 20 秒（實測）。
- 關閉 Isaac Sim 視窗就會結束程式。
- `run.sh` 啟動的容器名稱是 `ars-isaac`，掛載方式：repo 掛在 `/workspace/ars`，資產唯讀掛在 `/workspace/so101_workshop`。結束時會把容器在 repo 裡建立的檔案改回目前使用者擁有。

常用參數：

| 參數 | 用途 |
|---|---|
| `--scene configs/scenes/lab_current_minimal.yaml` | 場景設定（預設值） |
| `--teleop configs/teleop.yaml` | EE、工作區、速度、IK 參數（預設值） |
| `--log outputs/run.csv` | 逐幀紀錄：目標、EE、積木、關節命令、輸入狀態 |
| `--profile` | 每 5 秒印出各階段耗時與 fps |
| `--device cuda:0` | 改用 GPU 物理（預設 `cpu`，見[效能](#效能)） |
| `--focus` | 以專注模式啟動 |
| `--color_dr` `--color_dr_seed N` | 開啟顏色 domain randomization（預設關閉），見[顏色 domain randomization](#顏色-domain-randomization) |
| `--color_dr_log path.jsonl` | 顏色紀錄位置（預設 `outputs/color_dr.jsonl`） |
| `--blocks N` | 積木數量 1–8（預設 1），見[多顆積木](#多顆積木) |

純邏輯測試（不需 GUI）：

```bash
docker/isaac/run.sh /isaac-sim/python.sh -m pytest -q tests
```

## 操作方式

畫面包含：
- 3D viewport：目標控制點（環）、實際 EE（藍色實心球）、座標軸。
- 右側的 **EE Teleop** 面板，由上而下：
  - 狀態卡：狀態標籤（綠／琥珀／紅）與原因、目標、實際 EE、Follow error、夾爪 yaw／傾角／jaw 命令、積木。
  - CONTROLS：Open／Close gripper、Yaw ±15、Pause（暫停時變成 Resume）、Reset、Focus view。
  - SCENE：積木數量（1–8 按鈕）、顏色 DR 開關、seed、目前配色。按 **Reset** 才套用，見[面板的場景設定](#面板的場景設定)。
  - 可收合的說明（滑鼠與鍵盤）與俯視圖。

Keyboard 與 mouse 寫入同一個 EE 目標，經同一個受限 IK 解算，可以交替使用。

### 標記與圖例（目標 ≠ 實際位置）

| 標記 | 代表 |
|---|---|
| 3D 畫面中的**中空環＋十字**（保持固定螢幕大小，下方有一條垂直線指到桌面） | **EE 目標**：操作者下的指令，用滑鼠拖曳的就是它 |
| 3D 畫面中的**藍色實心小球** | **實際 EE**：用實測關節角算出的夾爪抓取中心 |
| 環的顏色：綠 | 目標可達 |
| 環的顏色：琥珀 | 被工作區方框截住（例如抓取高度下限），不算錯誤 |
| 環的顏色：紅 | 被擋下，原因顯示在面板（見[畫面狀態](#畫面狀態)） |

- 面板會即時顯示「Follow error」，也就是目標與實際 EE 的距離。
- 俯視圖：中空環是目標，藍點是實際 EE，積木以它目前的顏色加白框顯示，灰點是基座。紅色不再代表積木，避免和「被擋下」混淆。

### Keyboard（焦點在 Isaac Sim 視窗時，操作不變）

| 按鍵 | 動作 |
|---|---|
| W／S | 目標往 +X／-X（前／後） |
| A／D | 目標往 +Y／-Y（左／右） |
| E／Q | 目標上升／下降 |
| Z／X | 夾爪繞垂直軸旋轉（yaw） |
| O／C | 按住時夾爪張開／閉合 |
| P | 暫停／繼續 |
| R | Reset |
| M | 切換專注模式＝操作工作區（全螢幕，只改版面） |
| 1／2 | global／wrist 相機格切換原圖或標註（只改顯示） |
| Shift（按住） | 滑鼠細調；鍵盤本身不受影響 |

- 移動量 = 速度 × 經過時間（0.08 m/s、yaw 1 rad/s），與 fps 無關；單幀最多計 0.1 s，卡頓後不會突然大跳。
- 只有新的按下事件會開始移動，系統的按鍵重複不會重新觸發。
- 按住的鍵（含 Shift）在放開、視窗失焦、暫停或 reset 時一律清除。之後即使鍵還按著，也要重新按下才會作用。

### Mouse（主要入口：3D 畫面中的目標環）

| 操作 | 動作 |
|---|---|
| 在**環上按下**左鍵後拖曳 | 目標在「目前高度的水平面」上跟著游標移動（相對移動） |
| 拖曳時按住 Shift | 細調：每次移動乘上 `fine_drag_scale`（預設 0.25） |
| 游標在環上（或拖曳中）滾輪 | 調高度，每格 `wheel_step`（5 mm） |
| Shift＋滾輪 | 每格 `fine_wheel_step`（1 mm） |
| 從其他地方拖曳、Alt＋拖曳、右鍵 | 照常是視角或 UI 操作，不會移動手臂 |
| 環以外的滾輪 | 照常縮放相機 |
| 面板 Open／Close gripper | 夾爪以 1.5 rad/s 轉到全開／閉合（不會瞬間夾緊） |
| 面板 Yaw -15／Yaw +15 | yaw 目標 ±15°，以 1 rad/s 逐步轉到 |
| 面板 Pause／Reset／Focus view | 暫停／reset（同時套用 SCENE 的變更）／專注模式 |
| 俯視圖（輔助，預設收合） | 左鍵拖曳一樣是相對移動；滾輪調高度；右鍵切換夾爪 |

- **按下不跳**：按下時目標不動。拖曳時每一幀只取游標在平面上的位移量，加到目前的有效目標上。
- **Shift 不跳**：倍率是在每個事件到達時套用，所以拖曳途中按下或放開 Shift，目標都不會跳動。
- **映射跟著視角**：每個事件都用 viewport 當下的投影，把游標射線和水平面求交點，所以視角、視窗尺寸、DPI 改變後依然正確。視線幾乎平行桌面時，該事件會被忽略。
- **滾輪會暫時鎖定在高度**：環會隨著高度往下移、離開游標，所以從環上開始滾之後，0.7 秒內的連續滾動都繼續調高度，相機不會被縮放。
- **碰到邊界**：
  - 一步走不到時，取這一步方向上最遠的可達點。水平拖曳時也會分別嘗試只動 x 或只動 y，讓目標能沿著邊界滑動。
  - 走不到的部分直接丟棄、不累積，所以往可達方向拖會立刻恢復。
  - 單次事件的移動量上限是 `max_drag_step`（3 cm），避免視線掠過平面時一次大跳。
- **舊輸入**：放開、失焦、暫停、reset 會結束拖曳並丟掉累積的滾輪量。
- **不會選到物件**：3D 畫面中的場景物件設為不可點選，框選也已關閉，拖曳不會選到物件或叫出變換 gizmo。Stage 面板仍可選取。

### 專注模式

- 按 M 或面板的 Focus view 切換。2026-11-22 起，專注模式就是全螢幕的操作工作區，見 [operator-workspace.md](operator-workspace.md)。
- 開啟時：
  - app 視窗全螢幕；
  - 隱藏 Stage、Layer、Render Settings、Property、Semantics Schema Editor、Content、Console、Main ToolBar、Status Bar，以及主選單列和各視窗的分頁標籤；
  - 面板停靠在 viewport 右側（約 28%），`Operator status` 停靠在面板上方；
  - viewport 佔滿其餘空間，場景有相機時並排在下方。
- 夾爪、yaw、暫停、reset、狀態顯示與鍵盤操作都照常可用。
- 再切一次會還原原本開著的視窗、原本在前面的分頁，以及浮動面板的位置。
- 切換不影響相機與 teleop 狀態。
- 加 `--focus` 參數可以直接以專注模式啟動。

### 畫面狀態

| 面板顯示 | 意義 |
|---|---|
| 右上 `input: ready／keyboard／mouse／focus lost` | 最後一個改變目標的輸入來源；失焦時顯示 `focus lost` |
| `PAUSED`／`fine x0.25` | 暫停中（所有輸入都丟棄）／細調中 |
| `OK`（綠） | 目標可達 |
| `WORKSPACE LIMIT`（琥珀）＋ `workspace z min` 等 | 被工作區方框截住，例如抓取高度下限。不算錯誤 |
| `JOINT LIMIT`（紅）＋關節名稱 | IK 停在某關節的極限，仍達不到位姿 |
| `OUT OF REACH`（紅）＋位置誤差與傾角 | 搆不到（即使夾爪傾斜到上限），且沒有關節頂到極限 |
| `JOINT JUMP`（紅）＋關節與變化量 | 有解，但單次更新某關節會變化超過 0.25 rad（例如接近手臂伸直的奇異點） |
| SCENE 下方的琥珀字＋琥珀色 Reset | 有尚未套用的場景設定，按 Reset 套用 |

- 被擋下時保留上一個有效目標；滑鼠拖曳時會沿著邊界取最遠的可達點。
- 每次拒絕都依原因分別計數，記錄在 log 中。

## 座標系、EE 與限制

- **基座座標（mount frame）**：+X 前、+Y 左、+Z 上，原點在基座正下方的桌面（z = 0 是桌面）。USD `base` link 繞 Z 轉 +90° 擺進來，原點位置是場景的 `robot.base_pos`：(0, 0, −0.0301) m。資產的 base link 原點在底座底面下方 30.1 mm（Workshop USD 與 Menagerie MJCF 一致），實驗室照片中底座直接夾在桌面上，所以原點在桌面下 30.1 mm、底座底面貼桌（2026-11-18 修正，之前底座懸空 30 mm）。x／y 尚未實測。場景的世界座標就等於 mount frame。
- viewport 中基座左後方有座標軸：紅 +X、綠 +Y、藍 +Z。
- **EE**：兩指間的抓取中心，相對 `gripper` link 的偏移是 **(0.0069, 0, -0.085) m**（provisional，量自 USD mesh）。
  - x：固定指內側面在 -0.0079，加上半個積木邊長，也就是積木貼住固定指時的中心。
  - z：指腹中段；指尖約在 -0.104，比 EE 低約 19 mm。
- **夾爪**：`Jaw` 角度為正時張開。張開設定 0.52 rad（指間約 53 mm），閉合命令是下限 -0.1745 rad；夾住 29.5 mm 積木時約停在 0.17–0.19 rad。
- **抓取方向**：夾爪盡量垂直朝下。yaw 預設讓指的開合方向沿著繞基座的切線，Z／X 或 Yaw 按鈕在這個方向上加偏移（限 ±90°）。
- **傾斜抓取**（2026-11-17 起）：垂直朝下搆不到的地方（遠處、高處），夾爪會往外傾斜，只傾斜到剛好搆得到，上限 `ik.max_tilt_deg`（預設 60°，provisional）。
  - 原因：垂直朝下時 Wrist_Pitch 很快頂到關節極限（1.66 rad），手臂其他關節其實還伸得出去。
  - 垂直搆得到的地方，解和以前相同（關節差 < 0.02 rad），手感不變。
  - 設成 4 就回到以前只能垂直的行為。
  - 60° 只依關節極限選定，沒有檢查實機的自身碰撞或桌面接觸；模擬中這些由物理處理。
- **工作區截斷**（`configs/teleop.yaml`）：x 0.06–0.45、y ±0.45、z 0.024–0.40 m。只是粗略的安全框，實際範圍由 IK 決定。z 下限讓指尖保持在盤底上方。
- **實際可達範圍**（從旋轉軸算的水平距離，IK 計算，見 `tools/reach_map.py`）：

  | 高度 z（桌面以上） | 只能垂直 | 傾斜 ≤ 60° |
  |---|---|---|
  | 0.024 m（抓取高度） | 0.28 m | 0.40 m |
  | 0.10 m | 0.23 m | 0.40 m |
  | 0.15 m | 搆不到 | 0.38 m |
  | 0.25 m | 搆不到 | 0.30 m |

  - 最高點：只能垂直約 0.11 m，傾斜約 0.28 m。
  - 以上是底座貼桌後（2026-11-18）的數字；修正前整個範圍高 30 mm（抓取高度的距離幾乎不變），前後對照見交接資料的 `reach_base_fix.png`。
  - home 是 (0.15, 0, 0.09) m：和修正前 (0.15, 0, 0.12) 是同一個手臂姿勢，只是整體低 30 mm；這裡垂直夾爪最高約 0.10 m。
  - 超出時顯示 OUT OF REACH 或 JOINT LIMIT，保留上一個有效目標。
- **IK**：numpy 實作，從上一個解開始迭代。
  - 位置與夾爪繞自身軸的角度是主要條件；剩下的一個自由度（傾角）在主要條件的零空間裡往垂直方向拉。頂到極限的關節在該步鎖住，其他關節繼續解。
  - 接受條件：位置誤差 ≤ 1.5 mm、夾爪繞軸角度誤差 ≤ 6°、傾角 ≤ `max_tilt_deg`。
  - 與上一個解相比，任一關節變化不超過 0.25 rad。
  - 計算量：鍵盤操作每幀約 2 ms；滑鼠頂著邊界拖曳時要做邊界搜尋，最壞約 20 ms（以前約 10 ms）。
- **關節命令**：送給機械臂的關節目標每關節限速 3 rad/s，即使目標大幅改變也不會跳動；夾爪命令限速 1.5 rad/s（按鈕與按鍵相同）。
- FK 與 PhysX 對照 20 組隨機姿態，最大誤差 0.000 mm／0.02°。

## 場景數值（provisional）

全部列在 `configs/scenes/lab_current_minimal.yaml`，每個欄位都標了 `source: provisional` 與說明。重點：

| 項目 | 數值 | 依據 |
|---|---|---|
| 桌面 | 0.80 × 1.00 m | 未量測 |
| 粉盤（儲存） | 外尺寸 30.5 × 23 cm，深約 1 cm，中心 (0.225, 0.150) | 尺寸取自商品標示；位置只為了讓部分盤面落在可達範圍內 |
| 米盤（放置） | 外尺寸 36.5 × 26.5 cm，深約 1 cm，中心 (0.255, -0.170) | 同上 |
| 積木 | 邊長 29.5 mm、16.69 g，初始位置 (0.170, 0.120) | 批次平均，不是單顆量測 |
| 多顆積木的位置 | 粉盤內 4 × 2 格，x 0.11／0.17／0.23／0.29，y 0.120／0.195 | 只為模擬選定，不是實驗室擺法 |
| 摩擦 | 積木 1.0／0.9，桌與盤 0.6／0.5 | 為模擬抓取所選，未量測 |
| 壁厚、盤底厚 | 3 mm／2 mm | 未量測 |

- 托盤由 5 片靜態方塊組成（盤底加四面牆），所以盤內是空的。
- 積木是剛體，抓取完全靠夾爪與積木的物理接觸，沒有吸附、瞬移或改父節點。

## 多顆積木

- 積木 1–8 顆：啟動參數 `--blocks N`、選單（`tools/sim.py` 的手動操作模式）或面板的 SCENE。
- 位置：積木 0 在原本的 (0.170, 0.120)，其餘依 `block.layout_xy` 排在粉盤內，間距 6 cm（x）與 7.5 cm（y），夾爪張開（約 5 cm）時放得進相鄰兩顆之間。
- 顏色：DR 關閉時積木 0 是原本的紅色，積木 1–7 用 palette 的第 1–7 色；DR 開啟時每顆各自抽色，見下節。
- 每顆都是獨立剛體，Reset 時全部回到初始位置。
- 面板顯示積木數量與放進米盤的數量；俯視圖畫出每顆積木。
- `get_state` 回傳 `blocks`（每顆位置）與 `blocks_in_placement`；`block` 仍是積木 0，驗收腳本相容。
- 驗收腳本以 1 顆積木撰寫；`multi_block_pick_place` 以 8 顆執行。

## 面板的場景設定

SCENE 區塊可以在操作中改積木數量、顏色 DR 開關與 seed，**按 Reset（按鈕或 R 鍵）才套用**：
- 有變更還沒套用時，下方出現琥珀色說明，Reset 按鈕變成琥珀色，積木數量按鈕以琥珀色標出新值。
- 顏色 DR、seed：在原場景直接套用。開啟或換 seed 時從該 seed 的第一組顏色開始，之後每次 Reset 抽下一組；關閉時還原基準配色。
- New seed：隨機產生一個 seed，並開啟 DR。
- 積木數量：要重建場景，約 2 秒。重建時保留 DR 設定、專注模式與相機視角。
- 常駐伺服器：`status` 回報的選項會反映面板的變更；從選單跑驗收時，若面板設定和驗收需要的不同，會先重新載入。
- 每次重建場景約多佔 65 MB 記憶體（見 [sim-server](sim-server.md) 的已知問題），積木數量不常改時影響很小。

## 顏色 domain randomization

設定在 `configs/color_randomization.yaml`，預設關閉。關閉時場景完全不受影響（積木紅、手臂原本的黃）。也可以在面板的 SCENE 開關，見[面板的場景設定](#面板的場景設定)。

```bash
docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py --color_dr --color_dr_seed 7
```

- **顏色來源**：積木八色 palette，見 `configs/palettes/block_colors.yaml`，版本 `block_colors_v1-provisional`。
  - 依序為 0 red、1 orange、2 yellow、3 light_green、4 dark_green、5 blue、6 cyan、7 purple。
  - 名稱來自實物照片與量測紀錄，RGB 是 provisional 的 sRGB 設計值，**不是實物校色值**。
  - `color_id` 固定不變，之後只調 RGB 時不改 ID，改升 `version`。
- **抽樣**：
  - 每個 session（啟動算 session 0，之後每次 reset 加一）抽一次。順序固定：先抽積木 0，再抽手臂外殼，然後依序抽其他積木；各自獨立，允許同色。
  - 增加積木不會改變積木 0 與外殼的抽樣序列。
  - 抽樣器是 `numpy.random.default_rng(seed)`，相同 seed 與相同 reset 次序可完全重現。
  - session 進行中顏色不變。
- **套用**：
  - palette 的 sRGB 值會轉成 linear 再寫入材質（renderer 的材質顏色是 linear）。
  - 積木：`/World/Block/geometry/material/Shader`（其他積木是 `/World/Block_<i>/...`）的 `inputs:diffuseColor`。
  - 手臂外殼：`material_a_3d_printed` 的 `inputs:diffuse_color_constant`，整個外殼同一色，**夾爪的兩個手指也跟著外殼變色**。這個 USD 的夾爪接觸面不是獨立的幾何或材質，接觸面不另外保留原色（2026-11-17 確認）。
  - 馬達用 `material_sts3215`，不受影響。
- **實作上的限制**：
  - 開啟時，手臂的 *visual* 群組會改成非 instance，否則 renderer 不會更新 instance 的材質。collision 群組維持 instance，物理不變。
- **不改的東西**：質量、摩擦、碰撞、幾何、光照、UI 與 EE 標記顏色。
- **紀錄**：每個 session 一行 JSON，內容包括：
  - `seed`、`palette_version`。
  - 積木與外殼各自的 `color_id`、`name`、palette `rgb`（sRGB）與 `linear`，以及 `written`（實際寫入 renderer 的值，Isaac 等於 `linear`；驗收檢查比對它，MuJoCo 版同一份腳本也能用）；其他積木在 `extra_blocks`。
  - `applied`：從 stage 讀回的實際材質值。

## 驗收

驗收腳本以真實的 carb keyboard／mouse 事件操作介面，走的路徑和人操作一樣：
- 鍵盤事件經 app window。
- 滑鼠事件經 3D viewport 的目標環（omni.ui.scene gesture）與面板按鈕。

腳本像看著畫面的操作者：把游標移到目標位置的螢幕投影點，再等目標跟上。它不直接改機械臂、目標或積木。

```bash
tools/run_acceptance.sh keyboard_pick_place   # 只用鍵盤抓放
tools/run_acceptance.sh mouse_pick_place      # 只用滑鼠抓放（3D 目標環＋滾輪＋按鈕）
ISAAC_ARGS=--focus tools/run_acceptance.sh mouse_pick_place   # 同上，專注模式
tools/run_acceptance.sh robustness            # 不可達（鍵盤、滑鼠）、失焦、暫停、reset
tools/run_acceptance.sh viewport_drag         # 按下不跳、Shift 細調、滾輪、邊界推回、側視角、縮小視窗
tools/run_acceptance.sh focus_mode            # 專注模式進出、按鈕與鍵盤可用、版面還原
tools/run_acceptance.sh color_dr_off          # DR 關閉時為基準配色
ISAAC_ARGS='--color_dr --color_dr_seed 7' tools/run_acceptance.sh color_dr   # DR 抽樣、套用、session 內不變
tools/run_acceptance.sh far_reach             # 傾斜抓取：放到米盤遠端、舉高
tools/run_acceptance.sh scene_settings        # 面板 SCENE：DR、seed、積木數量（重建），Reset 套用
ISAAC_ARGS='--blocks 8' tools/run_acceptance.sh multi_block_pick_place   # 8 顆積木時抓放
```

- 輸出在 `outputs/acceptance/<name>.{log,csv,json,mp4}`；`outputs/` 不進版控。
- 影片只錄 Isaac Sim 視窗（依 X window id 擷取，不會錄到桌面上其他視窗），工具是 `tools/record_isaac_window.sh`。

### 驗收結果（2026-11-17：面板改版、傾斜抓取、多顆積木）

在[測試環境](#測試環境與版本)執行。純邏輯測試 52/52 通過（主機 uv 環境）。

| 情境 | 單次執行（錄影） | 經常駐伺服器 |
|---|---|---|
| `keyboard_pick_place`、`mouse_pick_place`、`viewport_drag`、`color_dr`、`color_dr_off` | 8/8、10/10、14/14、6/6、4/4 | P2 已驗證，未重跑 |
| `robustness`（到達極限已移到 x ≈ 0.42） | 17/17 | 未重跑 |
| `focus_mode` | 10/10 | 未重跑 |
| `far_reach`（新）：抓起積木放到米盤遠端 (0.35, -0.17)，以前只能垂直時這個 y 只到 x 0.21；home 上方升到 0.25 m | 8/8 | 8/8 |
| `scene_settings`（新）：面板開 DR、換 seed、改成 4 顆（重建）、關 DR、改回 1 顆，全部按 Reset 套用 | 13/13 | 13/13（重建時腳本在新場景接續） |
| `multi_block_pick_place`（新）：8 顆積木時抓放積木 0，其他 7 顆不被碰動，Reset 全部歸位 | 10/10 | 10/10 |

- 有兩次執行因為 Isaac 視窗失去 OS 焦點（輸入依設計被丟棄）而判定無效，重跑後通過。
- `tools/server_check.py`：10/10，連續 10 次載入都在 2.6 秒內。
- 8 顆積木時 fps 37–40，和 1 顆時相同（物理約 13 ms／幀）。
- 到達極限處的下垂：手臂幾乎伸直時，EE 比目標低約 17 mm（x 0.42）、11 mm（x 0.36）；`far_reach` 停止操作後的跟隨誤差 p95 15 mm。`robustness` 的 z 容許值因此放寬到 25 mm（x 仍要求 1 cm 內），見[已知問題](#已知問題與限制)第 5 點。

### 驗收結果（2026-11-17：主畫面滑鼠操作、專注模式、顏色 DR）

在[測試環境](#測試環境與版本)執行，全部走新的操作路徑。純邏輯測試 19/19 通過。

| 情境 | 次數 | 結果 |
|---|---|---|
| `mouse_pick_place`（3D 目標環、滾輪、按鈕，專注模式錄影） | 3 | 30/30 |
| `viewport_drag`（按下不跳、拖曳準確、單幀不跳、滾輪不縮放相機、Shift 細調 5 mm／1 mm、邊界被擋並分類原因、推回立即恢復、側視角、縮小視窗、不會選到物件） | 3 | 42/42 |
| `robustness`（鍵盤與滑鼠不可達、失焦、暫停、reset） | 3 | 51/51 |
| `keyboard_pick_place`（鍵盤操作不變） | 1 | 8/8 |
| `focus_mode`（隱藏編輯區、拖曳、按鈕、鍵盤、reset、相機不變、版面與前景分頁還原） | 1 | 10/10 |
| `color_dr`（seed 7）：抽樣寫入材質、session 內不變、每次 reset 抽一次 | 2 | 12/12；兩次紀錄逐字相同，也和純 numpy 抽樣器的預期序列一致 |
| `color_dr_off`：基準配色、reset 不變色 | 1 | 4/4 |
| `keyboard_pick_place` 開啟 DR（seed 3）：換色不影響抓取 | 1 | 8/8 |

量測（`--log` 或驗收時寫入 result JSON 的 `metrics`）：

| 指標 | mouse_pick_place | viewport_drag | robustness |
|---|---|---|---|
| 滑鼠事件到目標更新的延遲 p50／p95 | 10–11／13 ms | 12–13／24–27 ms | 14–18／33–35 ms |
| 停止操作後的跟隨誤差 p50／p95 | 3.2／6.8 mm | 3.2／4.8–5.1 mm | 3.2／5.5 mm |
| 單幀關節命令最大變化 | 0.043 rad | 0.05 rad（= 3 rad/s 限速） | 0.05 rad |
| 被擋下的次數與原因 | 只有抓取高度下限（workspace） | 邊界推動時 55–60 次，原因為 joint limit | 119–120 次，原因為 joint limit |

- 延遲在邊界附近較高，因為那裡要做邊界搜尋，一次約 8 ms。
- 跟隨誤差主要是重力下垂；夾著積木時較大。

錄影（只錄應用程式視窗）：`mouse_pick_place.mp4`（專注模式）、`viewport_drag.mp4`（到縮小視窗那一步為止）、`focus_mode.mp4`、`dr_a/color_dr.mp4`，都在 `outputs/acceptance/` 下。

**真人操作**：2026-11-17 由使用者實際操作，手感可接受，參數維持預設（`fine_drag_scale` 0.25、滾輪 5 mm／1 mm）。上面的數據仍來自注入事件的自動驗收。

### 驗收結果（2026-11-16：第一版）

2026-11-16 在[測試環境](#測試環境與版本)執行。每個情境錄影 1 次，另外不錄影重複 3 次，**4 次全部通過**（共 132 項檢查）。

| Issue 驗收條件 | 情境與檢查 | 結果 |
|---|---|---|
| 依文件啟動；資產、比例、碰撞正確 | 本文件的啟動步驟。FK 與 PhysX 一致（0.000 mm）。托盤內部是空的，積木靜置在盤底 z = 0.01675 m（盤底 2 mm 加半邊長），手指無法穿過積木（見第 2 點） | 通過 |
| 只用 keyboard 完成接近、夾取、抬起、搬運、放下 | `keyboard_pick_place`：8 項 × 4 次 | 32/32 |
| 只用 mouse 完成接近、夾取、抬起、搬運、放下 | `mouse_pick_place`：9 項 × 4 次（包含「沒有任何按鍵事件」） | 36/36 |
| 以物理接觸夾取 | 兩個抓放情境都檢查：夾爪命令是全閉，但 jaw 停在約 0.17–0.19 rad（被積木擋住）；積木跟著抬起（最高 z 0.095／0.105 m）；鬆開後落進米盤 | 通過 |
| 不可達時不跳關節；放開輸入或失焦時增量停止 | `robustness`：keyboard 與 mouse 超出範圍時都顯示 UNREACHABLE，目標停在可達邊界，EE 停在最後有效目標；失焦後增量停止，回到焦點也不會恢復 | 通過 |
| Reset 不執行舊輸入 | `robustness`：按住 D 時按 R，目標回到 home，D 不再作用；暫停中按下的鍵在繼續後不會執行。兩個抓放情境結束時 reset，積木回到粉盤初始位置 | 通過 |
| 操作影片、測試環境、啟動方式、已知問題 | 影片在 `outputs/acceptance/*.mp4`；其餘內容見本文件 | 已提供 |

四次執行中的最大值：

| 情境 | 單幀關節命令最大變化 | 實測關節速度最大值 | 被拒絕的目標 | fps |
|---|---|---|---|---|
| keyboard_pick_place | 0.039 rad | 1.98 rad/s | 0 | 39–49 |
| mouse_pick_place | 0.039 rad | 2.88 rad/s | 0 | 39–48 |
| robustness | 0.050 rad（等於 3 rad/s 限速） | 2.13 rad/s | 73–81（刻意超出範圍） | 29–54 |

驗收範圍的限制：
- 本節數據來自注入事件的腳本；真人操作確認見上方第二版結果。
- 失焦是模擬的。
- 場景數值都是 provisional。

詳見[已知問題](#已知問題與限制)。

## 效能

在測試環境實測。每種設定各跑一次 keyboard 和一次 mouse 抓放，以下是 `--profile` 的平均每幀耗時：

| 設定 | 物理（4 substeps） | 渲染 | fps | 抓放驗收 |
|---|---|---|---|---|
| GPU PhysX、vsync 開（Isaac Lab 預設） | 29.4 ms | 17.4 ms | 20 | 17/17 |
| CPU PhysX、vsync 開 | 11.8 ms | 16.9 ms | 32 | 17/17 |
| GPU PhysX、vsync 關 | 30.1 ms | 10.6 ms | 22 | 17/17 |
| **CPU PhysX、vsync 與主迴圈 60 Hz 限速都關（預設）** | 12.5 ms | 9.1 ms | 41 | 17/17 |

**物理為什麼用 CPU**：渲染仍在 GPU 上，只有物理解算改用 CPU。
- GPU PhysX 是為上千個並行環境設計的（例如 RL 訓練）。每一步都有固定成本：kernel launch、GPU 與 CPU 同步、張量 API 搬資料。
- 本場景只有一支手臂和一顆積木，固定成本佔大部分：GPU 每個 substep 約 7 ms，CPU 只要約 3 ms。
- 測試主機的 CPU 是 `powersave` 模式，CPU 端的 launch 和同步更慢，又放大了這個差距。
- 四種設定的抓放驗收都通過，所以選 CPU 純粹是為了效能，不是 GPU 物理有問題。
- 之後若做大量並行模擬，再用 `--device cuda:0` 切回 GPU。

vsync 和 60 Hz 限速會讓每幀固定等到約 17 ms。關掉後渲染降到約 9 ms。

- 物理固定每幀推進 1/60 s（1/240 s × 4 substeps），所以 fps 低於 60 時，模擬會比真實時間慢；在 40 fps 時約 0.67 倍。
- 目標移動量依真實經過時間計算，因此快速移動時 EE 會落後目標，放開按鍵後會追上。
- 測試主機的 CPU governor 是 `powersave`（Isaac Sim 啟動時會警告）。這屬於主機設定，本專案沒有更改。

## 已知問題與限制

1. **場景數值都是 provisional**：托盤位置是配合可達範圍擺的，不是比賽或實驗室的實際擺法；量測後要更新 `configs/scenes/lab_current_minimal.yaml`。
2. **固定指一側要留間隙**：EE 定義成「積木貼住固定指時的中心」。目標若比積木中心更偏向活動指那側超過約 2 mm（固定指隨之移到積木上方），固定指會壓到積木頂面，無法下降。
   - 操作時讓積木稍微偏向活動指那側（約 5 mm），閉合時活動指會把積木推向固定指。
   - 驗收腳本就是這樣操作。
3. **失焦**：驗收腳本裡的失焦是模擬的。效能測試時真實失焦也發生過，所有輸入都被丟棄，符合設計。
   - 自動驗收必須讓 Isaac 視窗保持焦點，執行時不要操作桌面。
   - CSV 的 `os_focus` 欄記錄真實焦點，`run_acceptance.sh` 偵測到失焦會印 WARNING。
4. **手感參數**：2026-11-17 使用者實際操作後，手感可接受。主畫面拖曳的細調倍率與滾輪步距仍可在 `configs/teleop.yaml` 調整。
5. **重力下垂**：Workshop 的關節剛性偏低（Pitch stiffness 30），抬高時 EE 比目標低約 4–7 mm，停止操作後的跟隨誤差約 3–5 mm。抓放不受影響，實機對照時要注意。
   - 開放傾斜抓取後可以伸得更遠，下垂也跟著變大：x 0.36 約 11 mm、到達極限 x 0.42 約 17 mm。
   - 關節增益沿用 Workshop（與實機對應），沒有為此調高。需要時可以在控制器加重力補償，或在面板看實際 EE（藍球）操作。
6. **模擬慢於真實時間**：見[效能](#效能)。
7. **`SimulationApp.close()` 可能卡住**（Isaac Sim 5.1）：程式在輸出都寫完後直接以 `os._exit(0)` 結束。
8. **資產警告**：啟動時會出現 `getTypes called on non-existent path /World/Robot/wrist/visuals/...`，來自 Workshop USD 的引用，不影響碰撞與運動。
9. **俯視圖上的托盤以軸對齊矩形近似**：目前托盤 yaw 都是 0。
10. **使用了 Kit 內部介面**（Isaac Sim 5.1／Kit 107 實測可用，2026-11-17 確認接受；升版要重新確認）：
    - 滾輪在目標環上時，會略過 `omni.kit.manipulator.camera.viewport_camera_manipulator._zoom_operation`。
    - 用 `ViewportWindow` 的 layer 樹隱藏框選圖層。
    - 兩者失敗時只記 log，退回 Kit 原本的行為：滾輪也會縮放、框選矩形會出現。
11. **3D 畫面不能點選物件**：場景物件設為不可點選（`set_pickable`），要選取請用 Stage 面板。
12. **顏色 DR 換色後約 1 秒內有過渡**：reset 後 renderer 的時間累積會讓舊色與新色短暫混合，不影響物理或紀錄。
13. **改變視窗大小時錄影會中斷**：`record_isaac_window.sh` 依 window id 擷取，視窗尺寸一變 ffmpeg 就停止。`viewport_drag` 的影片因此在縮小視窗那一步結束，檢查結果不受影響。
14. **Kit 熱鍵**：W／E 在 Kit viewport 也是變換工具的熱鍵，但因為物件不可點選，不會造成變化。專注模式選用 M，是為了避開 F（Kit 的框選聚焦）。
15. **傾斜上限 60° 是 provisional**：只依關節極限選定；實機的自身碰撞、纜線與桌面接觸都沒有檢查。實機上線前要用實機確認，必要時調小 `ik.max_tilt_deg`。
16. **滑鼠頂著邊界拖曳時 IK 較重**：邊界搜尋最壞約 20 ms／幀（以前約 10 ms），fps 會短暫下降；一般操作每幀約 2 ms。
17. **改積木數量會重建場景**：約 2 秒，畫面會短暫空白；每次重建多佔約 65 MB 記憶體（Kit 內部，見 sim-server 文件）。單次執行時，`--log` 的 CSV 在重建後接著寫。
18. **基座懸空 30 mm（已修正，2026-11-18）**：Workshop USD 的基座底面在 base link 原點上方 30.1 mm，以前場景把原點放在桌面，底座懸空 30 mm、肩關節比實機高 30 mm。實驗室照片確認底座直接夾在桌面上，所以 `robot.base_pos` 改成 z −0.0301 m，共用運動學加上這個位置（目標座標仍以桌面為 0），home 從 z 0.12 改成 0.09（同一個手臂姿勢）。兩個引擎的驗收都重跑。
19. **傾斜開始處的瞬間拒絕（已修正）**：手腕剛頂到極限、夾爪開始傾斜的位置（例如 z 0.12 時 x 約 0.324），IK 偶爾會拒絕一幀並顯示 `joint_limit Wrist_Pitch`，接著又繼續前進。原因是零空間的「拉回垂直」步驟把手腕推過極限，被截斷後其他關節的補償不再一致。改成縮短零空間步驟、主要步驟越界時鎖住該關節後重解。`tools/ik_boundary_check.py`（4 個高度 × 5 個方向 × 30 次，隨機幀時間）：修正前 14/600 次，修正後 2/600 次；剩下的 2 次是單步關節跳動保護（Elbow 0.2503 rad，略超過 0.25）在垂直可達邊界拒絕一幀，屬於設計行為。

## 檔案

| 路徑 | 內容 |
|---|---|
| `scripts/isaac_ee_teleop.py` | 單次執行的主程式（每幀呼叫 session；積木數量改變時重建 session） |
| `src/agentic_sim/kinematics.py` | SO-101 FK／IK（numpy，不依賴 Isaac） |
| `src/agentic_sim/teleop/ee_target.py` | 共用 EE 目標：時間增量、相對移動與邊界滑動、拒絕原因分類、限速、reset |
| `src/agentic_sim/color_dr.py` | 顏色 palette 與 seed 抽樣（純 numpy） |
| `src/agentic_sim/isaac/viewport_handle.py` | 3D 畫面目標環：拖曳、滾輪、相機與框選處理 |
| `src/agentic_sim/isaac/focus_mode.py` | 專注模式版面切換 |
| `src/agentic_sim/isaac/color_randomizer.py` | 顏色 DR 套用與紀錄 |
| `src/agentic_sim/isaac/panel.py` | EE Teleop 面板（狀態、按鈕、SCENE 設定、說明、俯視圖） |
| `src/agentic_sim/isaac/session.py` | 一個操作 session：場景、輸入、IK、物理迴圈、面板、紀錄；重建時接續驗收腳本 |
| `src/agentic_sim/isaac/` 其他 | 場景建立（1–8 顆積木）、keyboard、資產載入、驗收輸入注入 |
| `configs/palettes/block_colors.yaml`／`configs/color_randomization.yaml` | 八色 palette（provisional）與 DR 設定 |
| `configs/so101_chain.yaml` | 從 USD 讀出的關節鏈（`tools/validate_kinematics.py` 產生並對照 PhysX） |
| `configs/teleop.yaml`／`configs/scenes/` | 操作參數與場景 |
| `configs/acceptance/` | 驗收腳本 |
| `docker/isaac/` | 容器啟動、資產取得 |
| `tools/` | 資產檢查、夾爪幾何量測、FK 驗證、可達範圍（`reach_map.py`／`plot_reach.py`）、驗收執行、視窗錄影 |
| `tests/` | 純邏輯測試 |

## 積木位置與八點預覽

啟動時加 `--live-perception`，可在鍵盤操作時顯示腕部相機推算的積木中心與八個頂點。使用含 sensor cameras 的場景，例如 `--scene configs/scenes/lab_20261120.yaml`。更新頻率預設上限 2 Hz；設定、座標定義與限制見 [即時感知預覽](perception.md#鍵盤操作時即時預覽isaac)。

## VS Code Remote／SSH 操作

沒有遠端桌面時執行 `docker/isaac/remote.sh run`，然後在 VS Code 轉發 **6080**，使用瀏覽器操作。詳見 [瀏覽器遠端桌面](remote-desktop.md)。
