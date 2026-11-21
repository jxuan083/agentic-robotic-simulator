# 隨機積木配置

開啟隨機模式後，啟動環境、重新載入實驗，以及按 **R／Reset** 都會抽取新的積木配置。預設在粉紅儲存托盤內產生 1–8 顆，顏色從八色色盤獨立抽取，同色可以重複，米色放置盤保持空白。每輪的總數均勻抽取，各色數量由抽樣結果決定，不保證每種顏色都出現。

## 啟動

一鍵入口（含新實驗室場景與相機）：

```bash
uv run tools/lab.py --backend isaac --random-blocks --no-chat
# 若也需要聊天介面，移除 --no-chat
```

直接啟動個別引擎：

```bash
docker/isaac/run.sh /isaac-sim/python.sh scripts/isaac_ee_teleop.py \
  --scene configs/scenes/lab_20261120.yaml --random-blocks
uv run scripts/mujoco_ee_teleop.py \
  --scene configs/scenes/lab_20261120.yaml --random-blocks
```

兩個引擎共用 `agentic_sim.random_blocks`，攝影機、機械臂與托盤沿用原場景。未加 `--random-blocks` 時維持固定配置，既有驗收情境照舊執行。

## 合法位置

積木保持直立、平放且不堆疊；每顆獨立抽取設定範圍內的 yaw（預設 −180° 到 +180°），不抽 roll/pitch。產生器使用包住方塊水平投影的外接圓，保守檢查：

- 完整積木位於桌面與粉紅托盤內側，距離邊界至少 8 mm；托盤旋轉也會換算。
- 積木底面在托盤底板上方 0.5 mm 生成，再由物理引擎落穩。
- 積木外接圓之間至少留 6 mm 間隔，避免初始重疊；避開其他托盤及機械臂基座區域。
- 中心落在設定工作區，接近點和抓取點各自需通過 SO-101 的關節限制／傾斜 IK；方塊邊向對齊測試允許立方體的 90° 對稱，記錄在 `ik_orientation`。

這些是初始配置與端點檢查，不保證任意抓取順序或整段搬運路徑無碰撞，也沒有檢查每一段手臂掃掠體積。尚需由動作執行器做路徑與碰撞處理。

設定檔為 `configs/random_blocks.yaml`，可用 `--random-config <path>` 替換。可調整總數區間、候選顏色、間距、取樣工作區及重試次數。空間不足或無法取得足夠 IK 合法位置時，會在有界次數的重試後報錯，不會縮減要求數量、重疊放置或產生部分場景。`check_ik: false` 僅適合感知／幾何實驗，此時不保證抓取端點可達。

另有 `--random-region table` 可改為托盤外裸桌面取樣，此區不符合現有任務執行器「只從儲存盤取料」的邏輯；目前預設與分揀用途都使用 `storage`。

## 重現與 Reset

不指定 seed 時，每次啟動取得新的隨機 seed。設定 `--random-seed 42` 可重現同一串配置：首次為 episode 0，依序每次 Reset 為 1、2……。同樣 seed、設定檔、場景及 Reset 次數，在兩個引擎使用同一組配置；物理運動仍可能不同。程序重新啟動且帶同一 seed 會回到 episode 0。

每輪啟動會記錄 seed、episode、總數、各色數量、初始座標及每顆 `yaw_degs`；常駐伺服器 `get_state` 的 `random_spawn` 也可取得這份資料。Reset 會重建場景，以支援積木數量改變，因此比固定配置的位置重設更慢。相同 seed、episode、設定與引擎會重現角度；按 Reset 會前進 episode 並重新抽樣。

隨機模式由產生器決定顆數與顏色，`--blocks`／面板積木數量不會固定它。不能同時指定 `--block-colors`／`--block_colors`、固定 task scenario 或 color DR；隨機模式下 Reset 會關閉面板的 color DR，避免視覺顏色與任務的顏色身分不一致。色彩與位置不在操作途中改變。

此模組只負責產生模擬配置，感知模組仍需從相機影像辨識；`random_spawn` 是場景真值，不能當作視覺定位結果。
