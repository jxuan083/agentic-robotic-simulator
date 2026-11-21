# VS Code Remote：用瀏覽器操作 Isaac

Isaac 是桌面程式，VS Code Remote 終端機本身不會傳回視窗。此入口建立獨立 Xvfb/Openbox 桌面，以 [x11vnc](https://github.com/LibVNC/x11vnc) 與 [noVNC](https://github.com/novnc/noVNC) 傳送畫面及鍵盤／滑鼠輸入，Isaac 仍使用原有 NVIDIA GPU 容器。

## 啟動

在 VS Code 的遠端終端機，於 repo 根目錄執行：

```bash
docker/isaac/remote.sh run
```

預設啟動 `lab_20261120`、粉紅托盤隨機積木、global+wrist 即時頂點預覽。首次沒有 desktop image 時會建置包含 Ubuntu/Xvfb/Openbox/noVNC 的小型映像。需要 Docker 與現有 Isaac 資產。

1. 開啟 VS Code 的 **Ports／連接埠** 面板。
2. 選 **Forward a Port／轉發連接埠**，輸入 `6080`。
3. 使用該列的 **Open in Browser／在瀏覽器開啟**，網址後接 `/vnc.html?autoconnect=1&resize=scale`。
4. 若本機也是 `6080`，直接開 `http://localhost:6080/vnc.html?autoconnect=1&resize=scale`。
5. 點 Isaac 畫面讓它取得焦點，再使用面板列出的按鍵；W/S、A/D、E/Q 移動，O/C 開合，R 重設。建議使用外部瀏覽器，避免 VS Code 編輯器攔截按鍵。

如果 VS Code 改配另一個本機 port，以 Ports 面板顯示的 forwarded address 為準。可以同時看到 Isaac 的相機 viewport 和兩個各自標示 global/wrist 的 `Block perception` 視窗。

## 自訂啟動

```bash
docker/isaac/remote.sh run /isaac-sim/python.sh scripts/isaac_ee_teleop.py \
  --scene configs/scenes/lab_20261120.yaml --random-blocks --random-seed 8 \
  --live-perception --perception-camera global
```

`run` 會準備 DISPLAY、Xauthority 和共用 X11 socket，再呼叫原有 `docker/isaac/run.sh`。不改主機桌面，也不修改主機 Xserver 權限。若已有 `ars-isaac` 容器，先在原終端按 Ctrl+C 停止它，再執行此命令；已啟動且沒有掛載 X11 的程序無法直接搬到這個桌面。

只開啟桌面可用 `docker/isaac/remote.sh up`；查看狀態用 `status`，查看桌面服務日誌用 `logs`。模擬器日誌仍在啟動它的終端機。

## 停止

先在執行 `remote.sh run` 的終端機按 Ctrl+C 停止 Isaac，再執行：

```bash
docker/isaac/remote.sh down
```

`down` 只刪除桌面容器，不會停止 Isaac。關閉瀏覽器也不會停止模擬器。

## 連線範圍

瀏覽器端只發布到遠端主機的 `127.0.0.1:6080`，透過 VS Code SSH tunnel 連入；VNC 5900 不發布。此端點提供完整鍵盤滑鼠控制，維持 VS Code 轉發的私人存取範圍。無 VNC 密碼，與同主機 localhost 的其他服務一樣，主機本地使用者可連入。

可用 `ISAAC_DESKTOP_PORT` 改 host port；使用自訂 port 時也要在 VS Code 轉發相同值。`ISAAC_DESKTOP_DIR` 預設為 `$HOME/docker/ars-isaac-desktop`，保存專用 X11 socket 與授權檔。桌面 image 內的 NVIDIA_VISIBLE_DEVICES=void 不影響獨立 Isaac 容器的 `--gpus all`。

## 在同一瀏覽器桌面啟動 MuJoCo

MuJoCo 在 host 上執行，使用隔離 Python venv，不裝進 Isaac 容器或實機環境。首次建立環境：

```bash
python3 -m venv .venv-mujoco
.venv-mujoco/bin/pip install -r requirements-mujoco.txt
MENAGERIE_DIR="$HOME/ref/mujoco_menagerie" tools/fetch_mujoco_assets.sh
```

`tools/mujoco_remote.sh` 會沿用 `ars-isaac-desktop`，從 desktop container 複製 Xauthority 到 user-only 暫存檔，並安全連接 host 的 `:99` X socket。預設啟動隨機積木、兩鏡 live perception 與 `outputs/perception-live/latest.json`：

```bash
tools/mujoco_remote.sh
```

也可指定場景與相機模式，例如：

```bash
tools/mujoco_remote.sh --scene configs/scenes/lab_20261120.yaml --live-perception \
  --perception-camera wrist --perception-json outputs/perception-live/wrist.json
```

缺 SO-101 資產時，工具會列出 pinned Menagerie fetch 命令。這個 host launcher 不啟動、不重建也不停止 Isaac 容器。
