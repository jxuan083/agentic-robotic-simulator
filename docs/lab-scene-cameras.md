# 實驗室場景與相機視角（2026-11-20 配置）

依 2026-11-20 實驗室採集的資料重建單臂場景，並在模擬中加入 global 與腕部兩個相機視角。Isaac 與 MuJoCo 共用同一份場景檔 `configs/scenes/lab_20261120.yaml`。

- 只建右臂（單臂）。雙臂配置要等比賽結束後再做，左臂不放進場景。
- 原本的 `lab_current_minimal` 不變：沿用無相機座的手臂，也沒有感測相機。既有驗收情境照舊使用它。

## 使用

```bash
# MuJoCo：3D 視圖右上角顯示 D455 與腕部相機的小畫面
uv run scripts/mujoco_ee_teleop.py --scene configs/scenes/lab_20261120.yaml
# Isaac：左上角兩個浮動視窗顯示 D455 與腕部相機
docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py --scene configs/scenes/lab_20261120.yaml
# 抓放軌跡（實驗室紅積木位置 → 米色盤），兩個引擎都可以用
uv run scripts/mujoco_ee_teleop.py --scene configs/scenes/lab_20261120.yaml --trajectory configs/trajectories/lab_pick_place.yaml
# 離線算出兩個相機的畫面；加上實機影格時，輸出「實機｜模擬｜50% 疊合」比較圖
uv run tools/render_sensor_cameras.py --lerobot <pan> <lift> <elbow> <wrist_flex> <wrist_roll> <gripper%> \
    --real-global <global.png> --real-wrist <wrist.png> --out outputs/lab_cameras/<name>
```

- 相機小畫面和視窗不接受操作，滑鼠事件會略過它們。
- RGB-D 錄影的 global 影格是 640 × 480，要比對時加 `--global-k 640 480 <fx> <fy> <cx> <cy>`，改用該串流的內參算圖，相機位姿不變。

## 座標

- 沿用 mount 座標：右臂基座的 base link 在桌面上的投影為原點，+X 朝前、+Y 朝左、+Z 朝上，單位為公尺。
- 基座底面貼在桌面上（z = 0）。D455 的深度平面擬合結果是桌面比基座底面低 1.6 mm。
- 場景檔的每個數值都標了來源：
  - `measured`：由裝置或 SDK 讀出，或用尺量得。
  - `estimated`：由採集資料推算得到，推算方法寫在 `note`。
  - `provisional`：商品標示、操作者的約略值，或為模擬選定的值。

## 相機

| 相機 | 型號 | 解析度 | 內參 | 位姿 |
|---|---|---|---|---|
| global | Intel RealSense D455 彩色串流 | 1280 × 720（`measured`） | librealsense 讀出的 fx 639.5、fy 639.0、cx 640.1、cy 369.3（`measured`） | 鏡頭在 (0.394, 0.181, 0.578) m，光軸偏離垂直 5.8°，畫面上方朝 +X（`estimated`） |
| 腕部 | Innomaker U20CAM-1080p-S1 | 640 × 480（`measured`） | 垂直視角 61°，方形像素，主點在中心（`provisional`） | 官方 SO-101 腕部相機座，位姿沿用 Menagerie `wrist_cam`（`provisional`） |

### global 相機位姿的推算

1. 從右臂的 RGB-D 錄影（`right_arm_sag_02`）取 8 幀，手臂姿態各不相同。
2. 用 D455 深度內參和 depth-to-color 轉換，把每幀深度圖換成彩色相機座標系中的點。
3. 用 RANSAC 擬合桌面，取高於桌面 40 mm 的點作為手臂點。
4. 用同一時刻的關節讀值擺出 SO-101 網格（Menagerie MJCF）。
5. 以剛體 ICP 同時對齊 8 幀，求出相機在 mount 座標系中的位姿。

結果：
- 殘差中位數 4.6 mm，p90 12.6 mm；85% 的點落在 10 mm 以內。
- 求出的 y = 0.181 m，與現場以捲尺量得的「相機支架距右臂約 18–19 cm」一致。
- 單臂抓放錄影（1280 × 720）和 RGB-D 錄影（640 × 480）中，桌上固定標籤的像素射線方向相同，可知兩次錄影之間相機沒有移動。

### 關節讀值的換算

實機回報 LeRobot 角度：五個手臂關節以度為單位，夾爪以開口百分比表示。換算方式沿用 NVIDIA Workshop 的線性對應，把 −100..100 對到 USD 各關節的範圍，實作在 `src/agentic_sim/real_joints.py`。

ICP 比較了兩種換算：

| 換算 | 殘差中位數 | 擬合出的桌面 |
|---|---|---|
| Workshop 線性對應 | 4.6 mm | 水平 |
| 角度直接相等 | 5.8 mm | 傾斜 5° |

因此採用 Workshop 對應。這個對應取決於各臂的校正檔，在逐關節核對前仍屬 `provisional`。

### 腕部相機

- 實驗室照片中，手臂裝的是官方腕部相機座，所以兩個引擎都換成帶相機座的手臂：
  - Isaac 使用 Workshop 的 `SO-ARM101-USD.usd`。這份 USD 和無相機版本的關節、連桿完全相同，只多了相機座外形與夾爪上的一個碰撞方塊。
  - MuJoCo 保留 Menagerie 的 `camera_mount` body。
- 視角取自兩份模型：Menagerie 的 `fovy` 是 61°；Workshop 的焦距 13.5 mm 換算成 4:3 時，垂直視角約 60°。廠商標示水平 103°，但有買家回報實際值小很多，所以不採用。
- 與實機腕部影格比對：兩指的位置與大小相近，指尖間距約 160 px 對 170 px；夾住積木時的開口也對得上。
- 實機鏡頭有明顯桶形變形，模擬是理想針孔，不模擬變形。要做像素級比對，需另外校正腕部相機的內參。

## 托盤、桌面與積木

| 項目 | 採用值 | 來源 |
|---|---|---|
| 粉色儲存盤中心、yaw | (0.294, 0.234) m、0.2° | `estimated`：在 D455 影格分割托盤，射線投影到盤緣高度的平面，取最小面積矩形。兩次錄影的結果相差 2 mm 以內 |
| 米色放置盤中心、yaw | (0.278, −0.158) m、−2.0° | `estimated`：同上 |
| 托盤外形 | 粉 305 × 230 mm、米 365 × 265 mm | `provisional`：商品標示。投影量得粉 308 × 230、米 355 × 253 |
| 托盤高度 | 粉約 17 mm、米約 23 mm（桌面到盤緣上緣）；壁厚 5 mm；盤底厚 4 mm | `provisional`：操作者約略值（內外尺寸差約 10 mm，盤底離桌面不超過 5 mm） |
| 桌面 | x 從 −0.05 m 到 0.64 m（後方隔板），寬 1.40 m，高 0.75 m | `provisional`：前緣依照片估計，後緣依 D455 影像，寬度與高度未量 |
| 積木 | 0 號與 1 號位置對應實驗室錄影中的紅、黃積木，其餘在粉盤可達區以 6 cm 間距排列 | `estimated`／`provisional` |

- 積木質量 16.69 g 是 64 顆的平均值（`measured`），邊長 29.5 mm 是近似值。
- 托盤擺位每輪可能不同（2026-11-20 的 RGB-D 錄影就是另一種擺法）。場景採用單臂抓放錄影 `single_right_stock_03` 的擺位。

## 驗收

- 測試 127 項全過：
  - 新增 `tests/test_lab_scene.py`，檢查來源標記、相機投影方向、托盤在 D455 視野內、積木在粉盤內，以及關節換算。
  - `tests/test_mujoco_backend.py` 新增一項，檢查兩個相機存在、global 位姿正確、腕部相機隨手臂移動，以及最小場景不受影響。
- GUI：兩個引擎都用 `lab_pick_place` 軌跡在可見視窗中抓起紅積木，放進米色盤（block in placement），過程中顯示兩個相機視角。
- 最小場景的回歸：
  - MuJoCo：smoke 2/2、keyboard_pick_place 8/8。
  - Isaac：smoke 2/2、keyboard_pick_place 8/8、mouse_pick_place 10/10。
- 實機對照：取 RGB-D 錄影 7 幀，用相同關節讀值算出模擬畫面。
  - global 視角中，模擬手臂與實機手臂重疊。
  - 腕部視角中，兩指的位置、大小與開口一致。

## 限制與待補

- **global 相機**：位姿由手臂點推算，沒有用標定板。D455 的鏡頭變形不模擬。
- **腕部相機**：內參未校正，位姿沿用模型值。要補：腕部支架側照、鏡頭相對夾爪中心的位置與朝向，以及內參校正。
- **桌面與托盤**：桌面寬、高與前緣位置都沒有量；托盤盤底高度與壁厚只有約略值。
- **關節換算**：未逐關節核對，下垂與增益仍等實機數據。
- **光照與材質**：沒有對照實機，模擬畫面比實機亮、顏色偏飽和，只能用來對照幾何，不能直接當訓練影像。
- **雙臂**：實驗室有雙臂配置（相機支架距左臂約 21–23 cm），比賽結束後再做。
