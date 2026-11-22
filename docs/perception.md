# RGB 積木實例分割

模組位於 `agentic_sim.perception`，只接收 RGB `uint8` NumPy 影像，不依賴 Isaac Sim、MuJoCo 或模擬器真值。Ultralytics 將內部輸入當作 OpenCV BGR，因此推論前會明確轉換；CLI 用 OpenCV 讀圖後也會轉成 RGB 再呼叫核心。結果 mask 是原始影像大小的 bool array，留在記憶體中可交給後續 RGB-D 對應模組。

## 安裝與權重

建立獨立感知環境，避免把 Ultralytics、OpenCV 或 ByteTrack 依賴裝進模擬器／實機 Python：

```bash
python3 -m venv .venv-perception
.venv-perception/bin/python -m pip install -r requirements-perception.txt
```

`requirements-perception.txt` 明列 `lap` 追蹤依賴。推論時會設定 `YOLO_AUTOINSTALL=False`；若追蹤依賴不存在，程式會報錯並要求在這個獨立環境安裝，不會自動安裝套件。

設定範本是 [`../configs/perception.yaml`](../configs/perception.yaml)。請把自己訓練的單一 block 類別 segmentation checkpoint 放在 `weights/block-seg.pt`，或修改 `model_path`。模型、資料集及輸出檔不應提交至 repo。一般 YOLO COCO segmentation 權重不包含本專案的積木類別，不能當作可辨識積木的模型。repo 目前沒有適用 checkpoint，也尚未用真實積木相機影像驗證辨識準確率。

已有 Isaac Sim 時，可擷取 render-only 的 8 色 RGB 場景作為程式流程樣本（不是實拍，也不含偵測標註）：

```bash
docker/isaac/run.sh /isaac-sim/python.sh tools/capture_block_scene.py \
  --output outputs/block-scene-capture --frames 48 --fps 16 --width 1280 --height 720
```

此工具重新建構一個靜態 render-only 桌面／托盤／積木場景，只擷取 RGB，不含 SO-101 機械臂或任何運動，也不是原完整模擬場景的驗證。積木座標與顏色只用來建立渲染場景，不會輸出或偽造偵測結果。

## 傳統視覺基線

`--backend classical` 使用 HSV 飽和度／亮度閾值找彩色前景，再做形態閉合與去噪、連通元件、面積、最小邊長、長寬比及 solidity 篩選；可在 [`../configs/perception-classical.yaml`](../configs/perception-classical.yaml) 設 normalized ROI。ROI 在形態處理後會再套用，確保候選不長出 ROI。預設 `close_kernel: 3` 在合成近鄰測試中保留 4 像素間隔的兩個不同色物件；閉合仍可能合併貼在一起或只有極窄間隔的積木，需按影像調參。輪廓內部會填洞，並嘗試把同一連通前景的明暗面保留為一個候選。

此 backend 不載入 YOLO、PyTorch 或 checkpoint，只需要 NumPy、PyYAML、OpenCV。請用另外的環境安裝，避免改動模擬器／實機環境：

```bash
python3 -m venv .venv-perception-classical
.venv-perception-classical/bin/python -m pip install -r requirements-perception-classical.txt
PYTHONPATH=src .venv-perception-classical/bin/python -m agentic_sim.perception.cli \
  path/to/image.png --backend classical --config configs/perception-classical.yaml \
  --output outputs/classical-image
PYTHONPATH=src .venv-perception-classical/bin/python -m agentic_sim.perception.cli \
  path/to/video.mp4 --backend classical --track --jsonl --fps 30 \
  --config configs/perception-classical.yaml --output outputs/classical-video
```

`confidence` 是飽和度與形狀品質組合的啟發式排序分數，**不是模型機率**。傳統 tracker 以顏色相同及中心像素距離作一對一最近配對，允許設定最大距離和 missed-frame 期限；ID 在單一 detector 執行期遞增且不重用，`reset_tracker()` 清空活躍配對但不重設 ID 計數。相同顏色的積木交錯、互相遮擋、快速移動或分割合併時可能交換或失去 ID，不能保證遮擋前後是同一顆。

可選擇輸出人工審核用的 YOLO segmentation 候選標註：

```bash
PYTHONPATH=src .venv-perception-classical/bin/python -m agentic_sim.perception.cli \
  path/to/video.mp4 --backend classical --track --fps 30 \
  --config configs/perception-classical.yaml --output outputs/classical-video \
  --export-yolo outputs/block-label-candidates
```

輸出為 `images/` 與 `labels/` 同 stem；一張圖片一個 label txt，同一影格所有候選各佔一列。影片會保存相對應的原始 RGB frame PNG。`manifest.jsonl` 逐 detection 記錄 `label_line_index`、skip 原因及 `needs_review` 狀態；空候選影格也需人工判定，不能默認當作負樣本。含洞或多個不連通元件、無法安全表示為單一 polygon 的候選會跳過；該影格標為 `incomplete_needs_review`。每次執行只清除上一份 manifest 明列且位於該匯出目錄內的工具產物；若輸入本身是將被清除的已匯出影像，會拒絕執行。候選標註不是 ground truth，也不是可直接微調的模型；請人工修正 polygon、檢查漏檢與誤檢後再納入資料集，程式不會啟動訓練。

這是 RGB-only 的像素尺度前景基線，面積與形狀門檻不是 RGB-D 幾何驗證；同色物件貼合、背景帶有相近飽和色、陰影使飽和度下降或鏡面高光都可能造成合併、漏檢或誤檢。色彩規則依可設定 HSV 區間估計，亮暗綠與黃綠在不同照明下可能混淆；目前的場景測試也有一顆偏黃綠的積木被分到 yellow。不要將一次靜態 render 成功視為真實相機下的準確率。

## 推論

```bash
PYTHONPATH=src .venv-perception/bin/python -m agentic_sim.perception.cli path/to/image.jpg --config configs/perception.yaml --output outputs/perception
PYTHONPATH=src .venv-perception/bin/python -m agentic_sim.perception.cli path/to/video.mp4 --track --jsonl --output outputs/perception-video
```

預設不開視窗，可在有桌面的環境附加 `--preview`。每個影格輸出 JSON、每個 detection 的 PNG 二值 mask 與 overlay JPEG；影片加 `--jsonl` 時輸出逐影格 `detections.jsonl`。空偵測是 `"detections": []`。缺少權重、非 segmentation 模型或讀取不到輸入都會顯示明確錯誤。

## 輸出契約

`FrameDetections` 包含 `frame_id`、`timestamp_s`、`image_size`（height, width）及 `detections`。每筆 detection 包含 `detection_id`、`track_id`、`class_name`、`confidence`、`color`、`bbox_xyxy`、`centroid_uv` 和記憶體中的 `mask`。

- 像素座標以左上角為原點，u 向右、v 向下；質心取二值 mask 像素座標平均值。
- bbox 為模型輸出的連續 xyxy 邊界，順序 x1,y1,x2,y2，右／下邊界依一般半開區域解讀。
- `timestamp_s` 可為 null；`timestamp_source` 標記 `image`、`caller`、`frame_rate`、`media_position`，或在來源無有效時間時為 null。影片 FPS 有效時使用影格索引/FPS；否則只採用遞增的解碼器媒體位置，不以推論耗時偽裝影片時間。可用 CLI `--fps` 提供明確影格率。
- `detection_id` 每張影格從 0 起算，只在該 frame 有效。省略 `frame_id` 時 detector instance 會自增 frame ID。YOLO backend 的 `reset_tracker()` 清除關聯並將自增 frame ID 歸零；classical backend 清空活躍配對但保留不重用的 track ID 計數。YOLO 的 `track=False` 不執行追蹤、`track_id` 一律為 null；在追蹤和純預測間切換不會清除偵測。YOLO `track_id` 由 ByteTrack 產生，classical `track_id` 來自顏色／像素中心啟發式配對；沒有追蹤結果時為 null。遮擋太久、ID 重用或視角／影像中斷時，不保證仍是同一顆積木。
- JSON 不內嵌大型 mask；CLI 寫 mask PNG 並在 JSON 用 `mask_path` 指向檔案。

Python 呼叫範例：

```python
import cv2
from agentic_sim.perception import BlockPerception, PerceptionConfig

detector = BlockPerception(PerceptionConfig.from_yaml("configs/perception.yaml"))
rgb = cv2.cvtColor(cv2.imread("frame.jpg"), cv2.COLOR_BGR2RGB)
frame = detector.predict(rgb, timestamp_s=0.0, timestamp_source="caller")
for item in frame.detections:
    print(item.detection_id, item.track_id, item.color, item.centroid_uv)
    depth_for_this_mask = depth_image[item.mask]  # RGB-D geometry is a separate module
```

## 標註與微調

用 CVAT、Label Studio 或其他 polygon 標註工具逐顆描繪積木實例。資料應涵蓋預期相機高度、背景、照明、遮擋、積木旋轉與顏色；不要只用同一場景相鄰影格同時切到 train/val，避免資料洩漏。匯出 Ultralytics segmentation polygon 格式，每張圖一個同名 `.txt`，每列為 `class_id x1 y1 ... xn yn` 並將座標正規化至 0–1。單一 block 類別的 `data.yaml` 可寫成：

```yaml
path: /data/block-seg
train: images/train
val: images/val
names:
  0: block
```

在獨立感知環境安裝 `requirements-perception.txt` 後，以官方 Ultralytics CLI 微調並驗證：

```bash
yolo segment train model=yolo11n-seg.pt data=/data/block-seg/data.yaml imgsz=640 epochs=100
yolo segment val model=runs/segment/train/weights/best.pt data=/data/block-seg/data.yaml
```

選擇驗證集 mask mAP 與實際目標環境表現可接受的 `best.pt`，將其路徑設進設定檔。官方 `yolo11n-seg.pt` 是有效 segmentation 初始化權重，不代表它已辨識本專案積木。

## RGB-D 接點

深度模組可直接沿用 `FrameDetections` 的 frame/time/image size 與每個 detection 的同尺寸 bool `mask`。對齊後的 depth frame 在 `mask` 範圍抽取有效深度並計算 3D 幾何；相機內外參、深度尺度、RGB-D 對齊、抓取點與座標轉換由後續模組負責。本模組不假設相機型號或安裝位置。

## 八個幾何頂點

`tools/estimate_block_vertices.py` 對單張已校正相機 RGB 影像執行傳統分割，再將**已知邊長、直立、平放在已知支撐面**的立方體投影，擬合每個 mask 的完整外輪廓（頂面與可見側面）。最佳化未知的 x、y、yaw；高度由已知支撐面與邊長決定。沒有使用積木的初始位置、模擬真值或先前隨機抽樣記錄，也沒有假設影像 mask 質心就是頂面中心。

```bash
# 使用獨立 classical 環境；輸入必須是完整相機畫面，不能截取或縮放 GUI 小視窗
PYTHONPATH=src .venv-perception-classical/bin/python tools/estimate_block_vertices.py wrist.png \
  --camera wrist --scene configs/scenes/lab_20261120.yaml \
  --joints <pan_rad> <lift_rad> <elbow_rad> <wrist_pitch_rad> <wrist_roll_rad> \
  --output outputs/block-vertices
```

`--joints` 是與照片**同一曝光時刻**的五個實際關節角度，採本 repo shared 定義（弧度），不能使用目標關節命令或直接傳入 LeRobot 的度數。也可以用 `--joint-state snapshot.json`，內容含 `joint_positions_rad` 五個值及可選 `timestamp_s`。腕部相機會利用 FK 和場景內相機安裝位姿，計算 `mount_from_cv`；缺少關節讀值時拒絕估計。同步由呼叫端保證，此工具不能自行證明照片與讀值同步。

相機鏡頭模式預設沿用場景，使用理想針孔影像時需傳 `--wrist-lens pinhole`。原始變形影像會先依內參與畸變參數校正 mask，再擬合投影，最後把八點投影回原始影像。`--camera global` 可改用固定全域相機，此時不需要關節角度。輸入影像解析度必須與相機設定一致。

輸出 `vertices.png` 為八點與線框，`vertices.json` 包含各個 detection 的 `geometry`：

- `vertices[0..3]`：底面四點；`vertices[4..7]`：對應頂面四點。由上往下看，局部座標依 (-x,-y)、(+x,-y)、(+x,+y)、(-x,+y) 編號。
- `xyz_m`：mount 座標、公尺；`uv_px`：原圖左上角為原點的像素座標。
- 八點的 `source` 一律為 `geometry_inferred`，包括投影到可見面的點；它們不是直接觀測到的八個角點。`self_visible` 僅檢查立方體自遮擋，無法判斷機械臂或其他物體造成的遮擋。
- 底面點為橘色、頂面為青色；自遮擋點為空心，隱藏邊為虛線。標籤是 `detection_id:vertex_id`。
- `fit_iou` 為投影與 mask 的符合程度，不是三維定位準確率或抓取信心。低於 `--min-iou`（預設 .82）、畫面邊界截斷或支撐面在相機後方等情況，輸出 `rejected` 與空頂點清單。

**適用範圍與精度：**不支援未知高度、傾倒、堆疊或明顯外部遮擋；兩顆相貼時也可能無法分割。依賴正確邊長、支撐面高度、相機內外參與關節校正；現有腕部校正仍有 estimated/provisional 值，因此輸出是估計值。相同立方體有 90° 旋轉對稱性，yaw 以 π/2 取模，頂點編號不保證對應固定的實體材質角點或跨幀不變。

本工具提供 ACT 後續可使用的幾何資料與離線標記；Isaac 即時預覽見下一節，目前尚未將八點作為 ACT 模型輸入。若 ACT 要接收這些座標，需要在訓練與推論使用一致的座標系、編號與輸入格式。

Isaac 的腕部 viewport 相機現在會由 `WristCameraFollower` 每次物理更新後，以實際 gripper link 位姿設定相機 world pose，並使用 reset transform stack。這可避免 Fabric 已移動手臂、USD 相機仍沿用初始父節點變換的情況；一般固定相機不受影響。離線工具由關節 FK 計算外參；即時預覽使用同一渲染影格的相機 view matrix 配對影像，兩者都不讀取積木真值。


## 鍵盤操作時雙鏡即時預覽（Isaac 與 MuJoCo）

```bash
docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py \
  --scene configs/scenes/lab_20261120.yaml --random-blocks --random-seed 42 \
  --live-perception --perception-camera both --perception-hz 2 \
  --perception-json outputs/perception-live/isaac-latest.json
uv run scripts/mujoco_ee_teleop.py \
  --scene configs/scenes/lab_20261120.yaml --random-blocks --random-seed 42 \
  --live-perception --perception-camera both --perception-hz 2 \
  --perception-json outputs/perception-live/mujoco-latest.json
```

也可以沿用服務啟動入口：

```bash
python tools/lab.py --backend mujoco --no-chat --random-blocks --live-perception --perception-camera both
```

Isaac 的 global/wrist 預覽是兩個同時可見、標題不同的視窗；MuJoCo 在兩個相機 inset 顯示疊圖，並在旁列出各鏡頭的中心 XYZ、拒絕候選、age 與 stale 狀態。`--perception-camera global` 或 `wrist` 可單獨啟用一鏡。兩引擎都是獨立純影像估算，不讀取 block pose。可用 `--perception-json <path>` 定期覆寫 latest JSON；每鏡記錄含 frame、capture 時間、相機矩陣、候選中心、八角、fit/rejected 與 age。Reset 會清除前一場景顯示與 JSON 最新資料。

每次擷取完整 sensor 解析度的 RGB 及**同一渲染影格**的 view/projection matrix。腕部相機的 USD world pose 已由實際 gripper link 更新；view matrix 決定該影格外參，不會把較晚的關節角度套回舊影像。理想針孔內參取該次渲染 projection；畸變相機沿用場景的鏡頭內參與畸變。尚未使用深度影像。

一次最多一個背景推算工作，忙碌時不排隊累積舊影像；`--perception-hz` 是更新頻率上限，不保證每秒達到指定次數。疊圖和數字屬於同一張處理後的快照，`Snapshot age` 從渲染捕獲回報時開始計時，超過兩秒標示 `STALE`。不會把舊八點畫到正在移動的新相機畫面。Reset 會清除舊結果；畫面截斷、分割合併或擬合不佳的候選顯示 rejected。沒有候選時清空上一幀座標。

**支撐面高度**：Isaac 與 MuJoCo 的即時感知都用當次場景的支撐面（`agentic_sim.scene_layout.block_support_top`，和積木生成同一個值）：固定配置與 `--random-region storage` 是粉紅托盤底板 0.004 m，`--random-region table` 是桌面 0 m。估測中心高度為支撐面加半個邊長。

`get_state` 回傳的 `perception.cameras.global` 與 `.wrist` 各自包含 `latest`（相機矩陣、候選、中心、八點）、`age_s`、`stale`、`processing` 與 `error`，與原有模擬真值欄位分開。這裡的 detection ID 僅在該影格有效；移動中的腕部畫面不使用像素距離追蹤來假裝維持實體 ID。MuJoCo `--perception-json` 檔是相同的 `perception` state。八點仍受上一節的直立、已知邊長與支撐面假設限制，抬起、搬離托盤、傾倒或遮擋時不應供 ACT 當作可靠抓取座標。

若使用 VS Code Remote，改以 `docker/isaac/remote.sh run` 啟動，再轉發 6080；可於瀏覽器同時操作手臂及觀看八點預覽，見 [遠端桌面說明](remote-desktop.md)。

## 指定顏色定位：global 粗找、wrist 精修

給一個場景中存在的顏色，找出一顆該色積木，輸出它的位姿（mount 座標，公尺／弧度），之後交給 ACT；抓不到時由 wrist 確認後回報。核心在 `agentic_sim.perception.localize`，參數在 [`../configs/target_localization.yaml`](../configs/target_localization.yaml)，MuJoCo 執行與評估入口為 `tools/locate_target_color.py`。

1. **global 粗找**：手臂在 home，拍一張 global 影像，取擬合成功、顏色相符的積木當候選，**離手臂底座近的先試**（最可能抓得到）。global 會把深綠認成淺綠、淺綠認成黃（渲染評估中僅有這兩種混淆），所以要找深綠時也試 global 標為淺綠的積木，找淺綠時也試標為黃、深綠的（`global_lookalikes`），排在同色候選之後。最多 4 個候選。
2. **wrist 精修**：對每個候選，計算讓它落在 wrist 畫面中央的 EE 位置（`plan_wrist_view`：控制器 IK＋相機座 FK，求光軸與積木高度平面的交點並迭代修正）。EE 高度 0.10 m，夾爪保持垂直，相機約在積木上方 0.10 m、俯角約 45–57°；拍不到再試 0.12 m。經控制器（IK、限速、物理步進）移動到位後拍一張 wrist。
   - 候選 3 cm 內有該色的 wrist 擬合 → 確認，位姿取 wrist。
   - 有其他顏色的 wrist 擬合 → 這個候選是混色，換下一個。
   - 沒有擬合，但有一個至少一顆積木大小的輪廓蓋住候選，且候選頂面投影處的 wrist 像素是目標色（例如兩顆同色積木在畫面中相貼）→ 確認顏色，位姿取 global（`source.pose: global`）。黃、淺綠、深綠不走這條路（單看頂面分不出來，`color_only_excluded`），必須有完整擬合。
   - 都沒有 → 換高度或下一個候選。
3. **抓不抓得到**：確認後，用確認的位姿檢查抓取 IK（`random_blocks.grasp_ik_checker`：上方接近點與夾取點都要有解，試立方體四個 90° 朝向）。抓得到 → `found`；抓不到 → 記下並繼續試下一個候選（可能還有同色、抓得到的）。
4. **結果** `target.json`：
   - `found`：有一顆確認過、抓得到的積木。
   - `unreachable`：wrist 確認過該色積木，但都在抓取範圍外；附第一顆的位姿作為證據。只有 wrist 確認後才會這樣宣告。
   - `not_found`：沒有一顆被 wrist 確認，包括 global 看到但 wrist 看不到（遮擋、拍不到）的情況；`reason` 說明原因，交由上層（LLM）決定怎麼回應。
   - 位姿欄位：`center_xyz_m`、`yaw_rad_mod_pi_over_2`（立方體 90° 對稱）、八頂點 `vertices_xyz_m`、`fit_iou`、`graspable`，以及 `source`（位姿來自 wrist 或 global、影格號、相機外參、同一顆積木的 global 估計）。給 ACT 的格式尚未定案。

wrist 近拍時不同顏色的積木常在畫面中相貼，合成一個擬合失敗的輪廓。因此 wrist 的分割改為分層：先整塊擬合；失敗的輪廓再依色相區段切開（`ClassicalConfig.split_by_hue`）；仍失敗且是綠色時，再依亮度分淺綠／深綠（`split_green_by_value`）。只在上一層失敗時才往下切，避免把側面偏暗、色相落到相鄰區段的積木切壞；切開後仍擬合不了的部分保留為 rejected（顏色可作為證據）。在既有評估影像上，wrist 擬合成功數由 Isaac 12/80 提高到 44/80、MuJoCo 6/24 到 14/24，沒有多出誤檢或辨色錯誤；global 不分層（全切會讓 Isaac global 少 3 顆）。

**顏色相關的規則**（global 混色、哪些顏色不能只看頂面）都是依模擬調色盤的渲染結果訂的；實機積木色差明顯，換成實拍後應重新檢查這些設定。

```bash
# 評估：每場 8 顆隨機積木，目標色從當場存在的顏色隨機選；真值只在 target.json 寫出後用來評分
MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/locate_target_color.py --episodes 20 --seed 7
# 積木可放在托盤任何位置，包括抓不到的遠角（預設只放抓得到的位置）
MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/locate_target_color.py --episodes 20 --seed 3 --anywhere
# 瀏覽器桌面看一場（VS Code 轉發 6080，見 remote-desktop.md）
tools/mujoco_remote.sh --locate --seed 7
# 不開視窗、錄成影片：每場 episode_*/video.webm（左：場景與狀態列；右：最近一次 global／wrist 拍攝與擬合）
MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/locate_target_color.py --episodes 3 --seed 99 --video
```

### 抓取範圍

以抓取 IK 檢查粉紅托盤內每 1 cm 的積木中心（468 點，4 種朝向）：78% 任何朝向都抓得到，21% 完全抓不到，集中在遠離手臂的角落；分界約在離底座 0.44 m（托盤最遠內角 0.56 m）。這是運動學結果（傾斜上限 60° 暫定、不含自身碰撞與桌面），實機只會更小。預設的隨機積木只放在抓得到的位置，`--anywhere` 才會放到遠角。圖：`outputs/target-localization/storage_reach.png`。

### 結果（MuJoCo，2026-11-21）

5 組共 110 場，每場 8 顆積木。預期結果由真值決定：目標色有任何一顆抓得到 → 應回報 `found`，否則 `unreachable`。正確 = 回報與預期相同，且位姿與一顆目標色積木的中心、八頂點誤差都 ≤ 5 mm（`found` 時該積木還要抓得到）。

| 設定 | 場數 | 正確 | 預期 found → 回報 | 預期 unreachable → 回報 |
|---|---:|---:|---|---|
| 只放抓得到的位置（seed 20261121／7／99） | 50 | 48 | found 48、not_found 2 | — |
| `--anywhere`（seed 3／11） | 60 | 59 | found 54 | unreachable 6（其中 1 場位姿誤差 6.1 mm） |
| 合計 | 110 | **107** | | |

- **抓得到卻宣告抓不到：0 次**；拿錯顏色：0 次。
- 失敗的 3 場：2 場 `not_found`（目標與鄰近積木在 wrist 畫面中相貼、無法擬合，且是黃／綠不允許只看頂面）；1 場正確回報 `unreachable`，但遠角只能從近側斜看、目標被部分遮擋，位姿誤差 6.1 mm。
- 遠角的積木 wrist 只能從靠手臂這一側斜看，被前面的積木擋住時無法確認，會回報 `not_found` 而不是 `unreachable`。

| 量測（有位姿的 107 場） | 平均 | 中位數 | 最大 |
|---|---:|---:|---:|
| 中心誤差 | 0.51 mm | 0.31 mm | 6.11 mm |
| 八頂點 RMSE | 0.70 mm | 0.48 mm | 6.12 mm |
| yaw 誤差（mod 90°） | 1.1° | 0.7° | 7.7° |
| wrist 拍攝次數（110 場） | 1.4 | 1 | 7 |
| 模擬時間（110 場） | 1.9 s | 1.6 s | 6.2 s |

- 和先前「只用 wrist 盲掃 11 個點」的指定顏色搜尋（5/10）比，成功率大幅提高，多半只需一次 wrist 拍攝。
- 在模擬中 global 位姿本身就很準（中心誤差最大約 1.4 mm）；wrist 位姿平均略好，但偶有較大誤差。wrist 的主要貢獻是確認顏色與抓不到前的確認。實機的 global 相機外參仍是估計值（殘差中位數 4.6 mm），預期 wrist 近拍的價值更大，需實機驗證。
- 掃描過程中機械臂與積木沒有接觸（0 幀），積木最大位移 < 0.001 mm。
- 限制同前：RGB-only、已知邊長、直立平放在粉紅托盤；只在 MuJoCo 渲染影像上評估，Isaac 尚未接上這個入口，實體相機尚未驗證。

證據在 `outputs/target-localization/`（不進 repo）：`mujoco-seed{20261121,7,99}/`、`anywhere-seed{3,11}/` 各有 `evaluation.json`、`evaluation.csv`，每場目錄含 global 與每次 wrist 拍攝的原圖、疊圖、`target.json` 與事後評分；`video-seed99/` 是 3 場錄影。

## 深度高度檢查：找出疊放、傾斜、相貼的積木

定位之前，用 global D455 的深度（對齊到彩色影像）檢查每個色塊，找出需要先整理的積木：例如目標壓在別的積木下面、global 看不到。這是整理流程的第一步，只做判斷，還不動手臂。程式在 `agentic_sim.perception.height`，參數在 [`../configs/target_localization.yaml`](../configs/target_localization.yaml) 的 `height_check`。

做法：色塊的 mask 腐蝕 2 px（去掉邊緣的飛點）後反投影成點雲，用 RANSAC 找出最主要的平面（容許帶寬隨量到的雜訊放寬，重複擬合 3 次），這個平面就是看得到的頂面。

| 狀態 | 條件 | 意義 |
|---|---|---|
| `upright` | 頂面水平（≤ 5°）、高度 1 層、面積約一個面 | 正常 |
| `stacked` | 頂面水平、高度 2 層以上 | 下面還有積木（可能就是目標） |
| `tilted` | 頂面傾斜 > 5° | 傾斜，RGB 位姿不可信 |
| `merged` | 一層高，但頂面面積 > 1.6 個面 | 同色積木相貼 |
| `partial` | 頂面面積 < 0.5 個面 | 被部分遮住，或是疊放時露出的下層 |
| `off_layer` | 水平，但高度不在整數層 | 架在不平的東西上 |
| `side_face`、`no_depth` | 平面 > 70°（被色相切開的側面）、點太少 | 忽略，不算證據 |

另外，兩個頂面在 xy 上的間距（每邊補回 2 mm 腐蝕損失）小於 15 mm 時標為「太近」。15 mm 是暫定的夾爪需求，手指寬度還沒量。global 的分割改用依色相切開（相貼的不同色積木才會分開）；切出的側面由 `side_face` 排除。

```bash
# 每場 8 顆隨機積木，再隨機做 1–3 次：疊上去、斜靠在旁邊、推到相貼，物理落定後評估
MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/evaluate_height_check.py --episodes 40 --seed 5
# 只做某一種：--kinds stack | lean | touch；深度雜訊：--depth-noise-mm-at-1m（預設 5，隨 z² 放大）
```

### 結果（MuJoCo，2026-11-21）

4 組共 160 場（混合、只斜靠、只相貼、只疊放），深度雜訊 1 m 處 5 mm（D455 規格 4 m 處 ≤2% 換算，在托盤距離約 1.5 mm），邊緣加飛點。

| 項目 | 結果 |
|---|---|
| 被壓在下面的積木，上面那顆被標出來 | **77/78** |
| 有問題的色塊被標出（疊放、傾斜、被遮住、多顆相貼） | 135/136 |
| 正常平放的色塊誤報 | 11/1058（1.0%） |
| 傾斜角誤差（34 個傾斜色塊）中位數 | 0.5–1.0° |
| 「太近」（< 15 mm）被標出 | 370/404（92%）；多報 56 次，其中 51 次真實間距 15–18 mm |

- 雜訊：完美深度時誤報 3/259、傾斜誤差 0°；雜訊加到 8 mm 時，誤報 8/259（多是平放被判成傾斜），被壓住的積木 23/24。
- 「太近」的漏報大多是間距 12–15 mm 的邊界情況；另有 9 對直接相貼的積木沒被標出，原因尚未分析。
- 限制：
  - 只在 MuJoCo 測試，深度雜訊是依規格估的模型；D455 在 1280×720 的最近量測距離約 0.52 m，疊兩層的頂面離相機約 0.51 m，可能量不到，要用實機確認。
  - 斜靠的積木在物理落定時常常滑平（26 次只留下 7 次），所以傾斜的樣本較少。
  - 只檢查 global 看得到的部分；完全被蓋住的積木，要從上面那顆的 `stacked` 推斷。

證據在 `outputs/height-check/`（不進 repo）：`mix`、`lean`、`touch`、`stack`，以及雜訊比較 `mix-n0`、`mix-n8`。每場有 `global.png`（紅框是有問題的色塊，括號內是真值）和 `check.json`。

## 整理迴圈：先把擋住目標的積木移開

目標被壓住、斜靠、或周圍太擠時，先自動整理到定位與抓取可以處理的狀態，再交給 ACT。每次只為目前的目標移動，移完重新拍照判斷。規則在 `agentic_sim.task.declutter`（純函式），參數在 [`../configs/target_localization.yaml`](../configs/target_localization.yaml) 的 `declutter`，MuJoCo 執行與評估為 `tools/declutter_target.py`。

每一輪：
1. global RGB＋深度 → 依色相切開的色塊 → 深度高度檢查（上一節）→ 托盤內的物件清單。
2. 依規則決定下一步（由上往下，符合就做）：

| 情況 | 動作 |
|---|---|
| 目標平放，wrist 還沒確認顏色 | 先用 wrist 看（不花搬動次數；global 會混淆部分顏色） |
| 目標已確認、周圍空 | 定位完成，交給 ACT |
| 目標已確認但周圍太擠，或和同色積木黏在一起 | 夾爪有空間就把**目標本身**搬到空位（用 wrist 位姿，一次搞定）；沒有就搬開鄰居 |
| wrist 確認不了、而且周圍太擠 | 先搬開鄰居，再看一次 |
| 目標疊在別顆上／架在不平的東西上 | 把目標搬下來 |
| 目標傾斜 | 搬開它靠著的那顆（目標會倒平）；沒有就搬目標 |
| 目標只露出一部分，上面有疊放 | 搬開上面那顆 |
| 看不到目標，但有可疑處 | 搬開可疑的積木：先疊放、再傾斜／高度異常，近的先 |
| 都沒有 | 回報 `not_found` |

3. 「搬」是夾起來放到托盤空位：空位離其他積木至少 25 mm、離目標至少 6 cm，往外 1.5 cm 也還抓得到，從近手臂的位置開始選。夾爪方向預設轉動最小的那組面；手指那側空間不足 18 mm 時，才改用另一組面（並檢查 IK）。夾不起來就改用推（閉合夾爪，先確認路徑的 IK），推之前先回 home。同一顆失敗 2 次就不再碰。
4. 搬完回 home 再拍照；搬過之後，之前的定位結果作廢。如果搬的是**目標本身**（已確認、疊在別顆上、或傾斜），夾起來後先用 wrist 看夾爪裡積木的顏色。做法是把夾取時積木相對末端的位置投影到 wrist 影像，往中心縮一半避開手指（夾爪本身是黃色）。顏色對了就直接送到放置盤（`placement_beige`，離盤邊至少 2 cm），結果記為 `delivered`，不用放下來再定位、再抓一次。顏色不對（global 看錯的近似色），就照原本放回托盤的空位。
   - 有沒有夾起來，用機器人自己的感測判斷，不用模擬真值：
     - 夾爪實際角度 ≥ `grasp_check.jaw_hold_min_rad`（0.10 rad）：合上時會停在積木寬度，空夾會一路合到 −0.17 rad；
     - 末端高度（用量測關節角算 FK）確實升高了 3 cm。
   - MuJoCo 117 次抓取的夾爪角度：有夾住 0.127–0.234 rad，空夾或滑掉 ≤ 0.113 rad。
   - 有 3 次夾住了，但手臂卡在低處沒抬起來；只看夾爪會誤判，所以也要看末端高度。
   - 10 場驗證中，這個判斷和真值 12/12 一致。
5. 上限：搬 4 次、wrist 確認失敗 3 次。超過就回報 `gave_up`，交給 LLM。

目前的抓取、放置、推是腳本動作（`task/skills.py` 的 `PickAt`、`Place`、`Push`），代替 ACT。`PickAt` 用夾爪角度＋末端高度判斷有沒有夾起；`Place` 放下後的確認仍用模擬真值，和任務執行器相同。

直接送進放置盤的效果（seed 11、12 各 15 場，和原本同樣的場景）：
- 成功 26/30（原本 24/30）；
- 16 場直接送達，16 次夾在手上的顏色判斷都和真值一致；
- 平均整理時間 21.1 s（原本 24.9 s），而且原本的時間還沒算 ACT 之後再抓一次、搬去放置盤的時間。

這 30 場裡，夾起來的目標顏色剛好都對，所以「顏色不對就放回托盤」這條路線還沒在模擬中實際走過。

```bash
# 每場 8 顆隨機積木，再隨機疊放／斜靠／推到相貼；預設優先選被壓住看不到的顏色當目標
MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/declutter_target.py --episodes 15 --seed 11
# --target any：目標從所有顏色隨機選；--no-declutter：對照組，直接定位不整理；--video：錄影
```

### 結果（MuJoCo，2026-11-22）

80 場（seed 11–14 各 15 場優先選有問題的目標，seed 15 的 20 場隨機選目標），深度雜訊 1 m 處 5 mm。成功 = 回報 `found`，位姿與一顆目標色積木的中心、八頂點誤差都 ≤ 5 mm，而且那顆積木平放在底板、沒被壓、離其他積木 ≥ 15 mm、抓得到。

| 目標情況（依真值） | 整理迴圈 | 不整理直接定位 |
|---|---:|---:|
| 被壓在下面、global 看不到 | **15/16** | 0/16 |
| 疊放、傾斜、被壓或太擠 | **30/38** | 0/38（只看位姿：10/38） |
| 沒有問題 | 23/26 | 16/26（只看位姿：19/26） |
| 合計 | **68/80** | 16/80（只看位姿：29/80） |

- 平均搬 1.55 次（最多 4 次），整理花的模擬時間平均 22.5 s；沒有積木被弄出托盤。
- 腳本抓取失敗 33/121 次（大多是很擠或傾斜的積木），失敗後改推 30 次。
- 失敗的 12 場：
  - 約 5 場 wrist 確認不了黃色或綠色（模擬調色盤的顏色問題；實機要重新檢查）；
  - 約 3 場腳本抓取和推都失敗；
  - 2 場目標搬到空位後，定位判斷抓不到；
  - 其餘為邊界情況（間距 14.7 mm）、搬完用光次數。
- 限制：
  - 只在 MuJoCo；積木都在可達範圍內開始；
  - 抓取成功與否用模擬真值檢查；
  - 推的方向只考慮托盤邊界和 IK，沒有考慮推的路徑上有沒有其他積木。

證據在 `outputs/declutter/`（不進 repo）：`h1`–`h4`、`any`，以及對照組 `baseline-*`。每場的 `evaluation.json` 有每一輪的物件清單、決策和動作結果；`video-seed13/` 是 3 場錄影。
