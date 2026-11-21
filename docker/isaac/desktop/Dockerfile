FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive NVIDIA_VISIBLE_DEVICES=void
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb xauth x11-utils x11vnc openbox novnc websockify python3 \
    && rm -rf /var/lib/apt/lists/*
COPY entrypoint.sh /usr/local/bin/ars-desktop
COPY configure_openbox.py /usr/local/bin/ars-configure-openbox.py
ENV DISPLAY=:99 XAUTHORITY=/tmp/.X11-unix/Xauthority
HEALTHCHECK --interval=2s --timeout=3s --start-period=5s --retries=15 \
  CMD xdpyinfo -display :99 >/dev/null && python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:6080/vnc.html', timeout=2)"
ENTRYPOINT ["bash", "/usr/local/bin/ars-desktop"]
