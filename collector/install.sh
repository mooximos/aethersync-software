#!/usr/bin/env bash
# Installs the collector on the Raspberry Pi. Safe to run again (idempotent).
# Run it as the normal user from the repository folder:   bash collector/install.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
USER_NAME="$(id -un)"
GROUP_NAME="$(id -gn)"
if [ "$USER_NAME" = "root" ]; then
  echo "Run this as your normal user (not with sudo); it asks for sudo itself."; exit 1
fi

echo "== 1/7 System packages"
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-opencv python3-scipy python3-matplotlib python3-yaml \
  python3-pil python3-requests python3-tqdm v4l-utils i2c-tools
sudo usermod -aG video,i2c "$USER_NAME"

echo "== 2/7 Python venv (with the system packages visible, for OpenCV, picamera2 and Hailo)"
if [ ! -x "$REPO/.venv/bin/python" ]; then
  python3 -m venv --system-site-packages "$REPO/.venv"
fi
"$REPO/.venv/bin/pip" install --quiet -r "$REPO/requirements-collector.txt"

echo "== 3/7 I2C (for the INA226 battery sensor)"
sudo raspi-config nonint do_i2c 0

echo "== 4/7 sudo rights, limited to shutdown, date and hwclock"
SUDOERS=/etc/sudoers.d/intersection-monitor
TMP="$(mktemp)"
{
  echo "$USER_NAME ALL=(root) NOPASSWD: $(command -v shutdown) -h now"
  echo "$USER_NAME ALL=(root) NOPASSWD: $(command -v date) -s *"
  if command -v hwclock >/dev/null; then echo "$USER_NAME ALL=(root) NOPASSWD: $(command -v hwclock) -w"; fi
} > "$TMP"
sudo visudo -cf "$TMP" >/dev/null
sudo install -m 0440 -o root -g root "$TMP" "$SUDOERS"
rm -f "$TMP"

echo "== 5/7 Let the collector set the RTC wake alarm (udev rule, no extra sudo rights)"
RULE=/etc/udev/rules.d/99-intersection-monitor-rtc.rules
echo "SUBSYSTEM==\"rtc\", KERNEL==\"rtc0\", RUN+=\"/bin/chgrp $GROUP_NAME /sys/class/rtc/rtc0/wakealarm\", RUN+=\"/bin/chmod g+w /sys/class/rtc/rtc0/wakealarm\"" \
  | sudo tee "$RULE" >/dev/null
sudo udevadm control --reload
if [ -e /sys/class/rtc/rtc0/wakealarm ]; then
  sudo chgrp "$GROUP_NAME" /sys/class/rtc/rtc0/wakealarm
  sudo chmod g+w /sys/class/rtc/rtc0/wakealarm
fi

echo "== 6/7 Settings file"
if [ ! -f "$REPO/collector/site.json" ]; then
  cp "$REPO/collector/site.example.json" "$REPO/collector/site.json"
  echo "   made collector/site.json from site.example.json - edit it (see README)"
fi
mkdir -p "$REPO/data"

echo "== 7/7 systemd service (starts at boot, restarts if it stops)"
sed -e "s|@USER@|$USER_NAME|g" -e "s|@REPO@|$REPO|g" "$REPO/collector/monitor.service" \
  | sudo tee /etc/systemd/system/intersection-monitor.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable intersection-monitor.service

cat <<NEXT

Done. Next steps:
  1. Edit collector/site.json (site name, latitude, longitude, each camera's usb_path).
  2. Check the hardware:   .venv/bin/python collector/check_hardware.py --all
     Fix every FAIL it prints (the fix is printed under it).
  3. Reboot once (for the new groups and I2C):   sudo reboot
  4. After the reboot, open http://10.42.0.1:8080 from a phone on the IntersectionMonitor Wi-Fi.
  Service log:   journalctl -u intersection-monitor -f
NEXT
