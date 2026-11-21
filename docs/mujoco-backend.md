# MuJoCo backend

## Live perception and browser desktop

MuJoCo can run the same image-only global and wrist perception preview as Isaac. Each stream renders RGB at its native sensor resolution and uses its synchronized camera pose; global calibration uses the centered intrinsics derived from the renderer's vertical FOV. Wrist distortion follows the configured rendered lens path. The UI shows per-camera age plus XYZ centers and rejected candidates; a periodic full state file is available with `--perception-json`.

For a host GUI inside the VS Code Remote browser desktop, use [the remote desktop guide](remote-desktop.md) and `tools/mujoco_remote.sh`. It expects a separate `.venv-mujoco` built from `requirements-mujoco.txt` and pinned Menagerie assets; it does not run MuJoCo in the Isaac container.

### Active wrist-view exploration

To run one interactive, controller-driven wrist sweep in the browser desktop:

```bash
tools/mujoco_remote.sh --full-scan --seed 20261121
```

The sweep visits six fixed observation points derived from configured pink-tray geometry, uses raised transit routes, and steps through the normal IK/controller path. It does not move blocks or use block poses to choose views or associate predictions. Press **R** to reset/cancel and clear the accumulated map, or close the viewer to cancel. Each reached view writes its RGB image, overlay, and JSON; `latest.json` separates estimates visible in the latest view from historical records.

For a reproducible paired evaluation (initial single wrist frame, accumulated multi-view estimates, and separate final single frame on the same 10 seeded episodes):

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/evaluate_perception_exploration.py \
  --episodes 10 --seed 20261121 --output outputs/perception-exploration/mujoco-final
```

Evaluation uses settled simulator poses only after predictions are saved. The sweep uses a 25 mm spatial-only association gate; close objects can merge and view-dependent duplicate clusters can remain. Reachability and observed contact/motion are logged, but routes are not certified collision-free. See the run report under `outputs/perception-exploration/mujoco-final/`.

### Random target-color search

The default wrist exploration command now chooses one unique color actually present in the generated block set and searches for it:

```bash
tools/mujoco_remote.sh --explore-wrist
```

For a reproducible 10-episode headless run:

```bash
MUJOCO_GL=egl PYTHONPATH=src .venv-mujoco/bin/python tools/search_target_color.py \
  --episodes 10 --seed 20261121 --output outputs/perception-target-search/mujoco-final
```

The target chooser sees only the generated color-name list; the image-only search receives only the selected color and rendered perception records. It captures an initial frame, visits up to six primary and four higher alternate wrist views, and stops at the first same-color candidate accepted in two physical camera poses at least 10 mm apart whose predicted centers differ by at most 5 mm and whose eight corners agree within 5 mm RMSE under cube quarter-turn symmetry. A bounded search ends as `confirmed`, `exhausted`, `unreachable`, or `cancelled`; color presence does not guarantee visibility. The GUI and `latest-target.json` show status, target, and the confirmed predicted XYZ/eight corners with source-frame/time/pose provenance. Press **R** to reset and clear the current target and map. Full tray-sweep evaluation remains available with `tools/mujoco_remote.sh --full-scan` or `tools/evaluate_perception_exploration.py`.

對應 issue [#3](https://github.com/jxuan083/agentic-robotic-simulator/issues/3) 的 P3。與 Isaac 版共用同一份場景設定、teleop 設定、按鍵、EE 控制器（IK、限制、拒絕原因、限速）與控制協定，在不能跑 Isaac Sim 的電腦上做物理驗證與流程驗證。

- 不需要 Docker 或 NVIDIA GPU，只要 Python 3.11 與 OpenGL（GLFW 視窗）。實測環境為 Linux；macOS、Windows 未測。
- 抓取只靠物理接觸。
- 場景數值與 Isaac 版相同，仍是 `provisional`。

## 環境建立

```bash
uv sync                              # 依 uv.lock 建立 .venv（含 mujoco 4.0.1）
tools/fetch_mujoco_assets.sh         # 取得 MuJoCo Menagerie 的 SO-101（固定 commit，只 checkout robotstudio_so101/）
uv run tools/mujoco_kinematics_check.py   # FK 對齊檢查（< 1 mm）
```

| 項目 | 版本 |
|---|---|
| MuJoCo | `mujoco==4.0.1`（`pyproject.toml`，`uv.lock` 鎖定）。不用 4.0.0：稀疏 Cholesky 的 bug 會讓求解變慢或發散 |
| SO-101 模型 | MuJoCo Menagerie `robotstudio_so101`，commit `3abca2e`（2026-09-28），Apache-2.0。來源是 TheRobotStudio `so101_new_calib.xml`。預設放在 `~/ref/mujoco_menagerie`，可用 `MENAGERIE_DIR` 改 |
| Python | 3.11（`.python-version`）。mujoco 4.x 需要 3.11 以上 |
| 視窗 | GLFW（`glfw` 套件隨 mujoco 安裝） |

## 啟動

```bash
export DISPLAY=:1                                             # 視窗所在的 X display，依機器而定
uv run scripts/mujoco_ee_teleop.py [--blocks 3] [--color_dr --color_dr_seed 7]   # 單次執行
tools/mujoco_server.sh up | down | status | logs              # 常駐伺服器（埠 47802）
uv run tools/sim.py                                           # 選單：選 mujoco
uv run tools/sim.py --backend mujoco --experiment minimal_pick_place --mode acceptance --scenario keyboard_pick_place
tools/run_acceptance_mujoco.sh keyboard_pick_place            # 不經伺服器跑一個驗收情境，錄影只錄 MuJoCo 視窗
```

- 伺服器是主機上的背景程序，啟動約 1 秒，載入實驗約 0.2 秒。關閉 MuJoCo 視窗也會結束伺服器（和 Isaac 相同）。
- `MUJOCO_SERVER_HEADLESS=1 tools/mujoco_server.sh up` 不開視窗，給腳本或 agent 使用。
- `scripts/mujoco_ee_teleop.py --hidden`：開發檢查用，視窗不顯示、焦點視為存在，可以讀回畫面截圖；**不算驗收**（驗收一律在看得到的視窗跑）。`--headless` 完全不開視窗，只能跑純鍵盤腳本。
- 驗收結果在 `outputs/acceptance/mujoco/<實驗>/`。

## 操作

視窗分成左邊的 3D 畫面與右邊的操作面板，功能和 Isaac 版一致（Isaac 的專注模式除外：MuJoCo 視窗沒有編輯器面板要藏）。

**鍵盤**（和 Isaac 相同）：W/S、A/D、E/Q 移動，Z/X 轉 yaw，O/C 開合夾爪，P 暫停，R reset。按住的鍵只在新的按下時開始；放開、視窗失焦、暫停、reset 都會丟掉輸入，不會補做。

**3D 畫面的滑鼠**（和 Isaac 的目標環規則相同）：
- 目標畫成面向相機、固定螢幕大小的圓環，往下有一條線指到桌面；實際 EE 是藍色小球。圓環顏色：綠＝OK、黃＝工作區邊界、紅＝被拒絕。
- 從圓環上開始的左鍵拖曳，在目標高度的水平面上相對移動目標（按下不會跳）；Shift 細調（0.25 倍）。
- 游標在圓環上（或拖曳中）時滾輪調高度，Shift 每格 1 mm；滾過一格後 0.7 秒內滾輪都留在高度，因為圓環會離開游標。
- 其他地方：左鍵拖曳旋轉視角、右鍵拖曳平移、滾輪縮放（Shift 切換水平方向）；在圓環上按 Alt + 拖曳也是移動視角。
- 搆不到的部分直接丟掉、不累積，往回拖立刻有反應。

**操作面板**（和 Isaac 面板相同的內容）：
- 狀態：彩色狀態標籤與拒絕原因、輸入來源（暫停、失焦時變色）、目標、實際 EE、跟隨誤差、夾爪 yaw／傾角／jaw、積木。
- CONTROLS：Open／Close gripper、Yaw −15／+15、Pause（暫停時變 Resume）、Reset、Reset view（相機回預設；取代 Isaac 的專注模式按鈕）。
- SCENE（按 Reset 才套用）：積木數量 1–8、顏色 DR 開關、seed（− ／ + 、點欄位直接打數字、New seed 隨機並開啟 DR）。有未套用的變更時 Reset 與數字按鈕變琥珀色，下方列出待套用的內容。改積木數量會重建場景（約 0.2 秒），相機視角保留。
- Help：可收合。
- Top view（預設展開）：正交相機從正上方看真實場景，+X 朝上、+Y 朝左（和 Isaac 的俯視圖相同），疊上托盤外框、目標環與 EE。左鍵拖曳相對移動目標、滾輪調高度、右鍵切換夾爪開合。

**為什麼自己開 GLFW 視窗**：MuJoCo 內建 viewer 的按鍵回呼只回報按下、不回報放開，做不出「按住才移動」與失焦處理，而且它自己的快捷鍵佔用了 W/S/A/D/E/Q/C。面板與俯視圖都用 MuJoCo 自己的 2D 繪圖（`mjr_label`、`mjr_text`、`mjr_rectangle`）與第二次 `mjr_render` 畫出，沒有新增套件。

## 模型與對齊

`configs/mujoco_so101.yaml` 記錄模型來源、對齊數值與物理設定。

**對齊方式**：共用的運動學鏈 `configs/so101_chain.yaml` 由 Workshop USD 量出。Menagerie 的 MJCF 和它同源（都來自 `so101_new_calib`），差異只有兩處：

| 差異 | 處理 |
|---|---|
| 基座原點不同 | MJCF 的 `base` 平移到 mount frame 的 (−0.01576, 0.02079, 0.03248) m，沒有旋轉。之後每個 link 的座標系都和共用鏈一致 |
| `wrist_roll` 的零點差 2.79° | MJCF 的關節範圍正好是 USD 範圍平移同一個量（校正偏移），所以在關節映射處理：`q_mujoco = q + 0.04868`。其他關節方向、零點都相同 |

`tools/mujoco_kinematics_check.py` 在關節極限內隨機取 2000 組角度，比較兩邊的 FK：

| 項目 | 結果 |
|---|---|
| EE（抓取中心）位置差 | 最大 0.004 mm |
| 夾爪座標系方向差 | 最大 0.0008° |
| 夾爪開合 | 0.18 rad 時兩指尖相距 30.5 mm（積木 29.5 mm），與 Isaac 版的接觸角度一致 |

**移除的部分**：Menagerie 模型帶腕部相機支架（含碰撞盒），Isaac 用的是無相機的 USD，所以刪掉 `camera_mount`，讓兩邊幾何相同。實驗室相機確認後再加回。

**基座高度**：底座底面貼在桌面上（MJCF 的 `base` 位置 = 場景的 `robot.base_pos`（z −0.0301）＋上表的平移），和 Isaac 場景相同。2026-11-18 以前兩邊都懸空 30 mm，見 [isaac-ee-teleop.md](isaac-ee-teleop.md) 已知問題 18。

## 物理

| 項目 | 設定 |
|---|---|
| 時間步長 | 物理 1/240 秒，控制 60 Hz，和 Isaac 場景相同（Menagerie 預設 0.005 秒） |
| 求解器 | 沿用 Menagerie：implicitfast、elliptic 摩擦錐、`impratio` 10 |
| 關節驅動 | 沿用 Menagerie 的 position actuator：kp 998.22、kv 2.731、力矩上限 ±2.94，關節阻尼 0.60、armature 0.028 |
| 夾爪接觸 | 沿用 Menagerie：condim 6、摩擦 1、priority 1（和積木接觸時用夾爪的參數） |
| 積木、托盤、桌面 | 尺寸、質量來自同一份場景 YAML。MuJoCo 每個幾何只有一個滑動摩擦係數，取場景的靜摩擦值 |
| speculative contacts | MuJoCo 4.0 的新功能，預設關閉（`mjENBL_SPECULATIVE` 是 opt-in），維持關閉 |

和 Isaac 的差異只記錄，不為了一致而調參數；跨引擎比較在 P4。

## 驗收

驗收腳本和 Isaac 共用（`configs/acceptance/`，步驟定義在 `src/agentic_sim/teleop/script_core.py`）。
- 按鍵事件送進 `KeyState.on_key`，滑鼠事件送進視窗的 `inject_move`／`inject_button`／`inject_scroll`，都是 GLFW 回呼呼叫的同一個函式，只省略 OS → GLFW 這一段。
- 游標位置由 3D 畫面的投影（圓環）、面板的按鈕位置與俯視圖的座標換算得出，和 Isaac 一樣模擬「看著畫面操作」的速度上限。
- 實驗設定檔用 `backends` 標記各情境能在哪個引擎跑：只有 `focus_mode` 是 Isaac 專用（編輯器面板），`top_view` 是 MuJoCo 專用（Isaac 的俯視圖預設收合）。

GUI 實跑（2026-11-18），錄影只錄 MuJoCo 視窗：

| 情境 | 單次執行 | 經常駐伺服器 |
|---|---|---|
| `keyboard_pick_place`：鍵盤抓放（物理接觸）、reset | 8/8 | 8/8 |
| `mouse_pick_place`：只用滑鼠（拖圓環、圓環上滾輪、面板按鈕） | 10/10 | 10/10 |
| `viewport_drag`：按下不跳、準確拖曳、圓環上滾輪不縮放相機、Shift 細調、頂邊界與回拖、換視角、縮小視窗 | 14/14 | 14/14 |
| `robustness`：不可達（鍵盤與滑鼠）、失焦、暫停、reset | 17/17 | 17/17 |
| `robustness_keyboard` | 14/14 | 14/14 |
| `far_reach`：傾斜抓取放到遠端 | 8/8 | 8/8 |
| `multi_block_pick_place`（8 顆） | 10/10 | 10/10 |
| `color_dr`（seed 7）／`color_dr_off` | 6/6／4/4 | 6/6／4/4 |
| `scene_settings`：面板開 DR、換 seed、改 4 顆（重建，腳本接續）、關 DR、改回 1 顆 | 13/13 | 13/13 |
| `top_view`（新）：俯視圖拖曳、滾輪調高度、推到邊界、往回拖、Reset 按鈕 | 7/7 | 7/7（共 2 次） |

- `top_view` 第一版的「往回拖」改過一次，見已知問題 4。
- `tools/server_check.py --backend mujoco`：10/10，載入 0.2 秒左右，記憶體穩定。
- 純邏輯測試：主機 65/65；Isaac 容器 59 passed、1 skipped（沒有 mujoco 的檔案）。面板與圓環的邏輯不需要 OpenGL，用假的投影測。
- FK 對齊：見上方「模型與對齊」。

## 已知問題與限制

1. **基座高度（已修正）**：以前兩個引擎的底座都懸空 30 mm；2026-11-18 依實驗室照片改成底座貼桌，home 跟著降到 z 0.09。所有驗收重跑。
2. **跟隨誤差比 Isaac 小很多**：Menagerie 的關節增益（kp 約 998）比 Workshop（Isaac）硬得多，靜止時跟隨誤差約 0.1 mm，Isaac 約 3 mm；伸直時也幾乎沒有下垂。兩者都不是實機量測值，P4 會並列比較。
3. **顏色 DR 的檢查比對「寫入 renderer 的值」**：MuJoCo 直接用 palette 的 sRGB 值，Isaac 寫 linear 值。紀錄裡兩邊都有 `written`（寫進 renderer 的值），`color_dr`、`scene_settings` 的檢查比對它，所以同一份腳本兩個引擎都能跑；基準色兩邊相同（積木 0.85/0.15/0.15、外殼 1.0/0.82/0.12）。
4. **從最遠的傾斜位置快速拖回**：一次大幅拖回時，每次更新的關節變化超過 `max_joint_step`（0.25 rad），部分步伐會被當成關節跳動拒絕；依設計拒絕的部分直接丟掉，所以目標會停在游標前面，要再拖一次。這是共用控制器的行為，Isaac 也一樣；`top_view` 情境改成「往內拖一小段要立刻有反應」。
5. **視窗焦點**：自動驗收時伺服器會要求視窗管理員把 MuJoCo 視窗切到前景；被拒絕時結果帶失焦警告，請手動點一下視窗再重跑。

## 檔案

| 路徑 | 內容 |
|---|---|
| `configs/mujoco_so101.yaml` | 模型來源、對齊數值、物理與視窗設定 |
| `tools/fetch_mujoco_assets.sh` | 取得固定版本的 Menagerie SO-101 |
| `src/agentic_sim/mujoco_backend/model.py` | 由場景 YAML 與 MJCF 組出 MuJoCo 模型（`MjSpec`），含俯視相機 |
| `src/agentic_sim/mujoco_backend/session.py` | 操作 session：控制器、物理步進、面板與圓環輸入、顏色 DR、重建、紀錄、狀態 |
| `src/agentic_sim/mujoco_backend/viewer.py` | GLFW 視窗：3D 畫面、投影與游標射線、滑鼠分派、俯視圖、腳本注入入口 |
| `src/agentic_sim/mujoco_backend/panel.py` | 操作面板（狀態、CONTROLS、SCENE、Help、Top view） |
| `src/agentic_sim/mujoco_backend/handle.py` | 3D 目標圓環的拖曳與滾輪規則 |
| `src/agentic_sim/mujoco_backend/draw2d.py` | 以 MuJoCo 2D 函式在視窗座標上畫圖 |
| `src/agentic_sim/mujoco_backend/script.py` | 驗收腳本的 MuJoCo 輸入（按鍵、滑鼠、面板） |
| `scripts/mujoco_ee_teleop.py`、`scripts/mujoco_sim_server.py` | 單次執行、常駐伺服器 |
| `tools/mujoco_server.sh`、`tools/run_acceptance_mujoco.sh` | 伺服器管理、單次驗收（錄影只錄 MuJoCo 視窗） |
| `tools/mujoco_kinematics_check.py` | FK 對齊檢查 |
| `src/agentic_sim/teleop/keys.py`、`src/agentic_sim/teleop/script_core.py`、`src/agentic_sim/scene_layout.py` | 和 Isaac 共用：按鍵對應、驗收腳本（鍵盤與滑鼠步驟）、場景幾何 |
| `configs/acceptance/top_view.yaml` | 俯視圖驗收（MuJoCo） |
