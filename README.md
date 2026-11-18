# Intersection Monitor

A science project that measures how long drivers wait at one traffic light, and
recommends better timing for it, backed by real data.

## What it does

1. **Measure** (Raspberry Pi on the signal pole, every weekday rush hour). Four cameras watch
   the four approaches. For every vehicle the Pi writes down when it arrived, whether and how
   long it waited, and when it crossed the stop line. From when queues start and stop it also
   works out when the light turns green and red. **No picture or video is ever saved**, only
   timestamped numbers.
2. **Describe** (laptop). A week or more of those numbers becomes the current situation:
   vehicles per hour, average and longest waits, queue lengths, and the light's current timing.
3. **Check the model.** A standard traffic-engineering delay model (HCM) is run with the
   measured traffic, and must predict the waits we actually measured.
4. **Improve.** The computer searches for the fixed timing with the least total waiting that
   still keeps every safety rule (yellow, all-red, pedestrian crossing time, minimum greens).
5. **Confirm.** The current, textbook (Webster) and recommended plans are tested in an
   independent traffic simulation (SUMO) and compared with statistics.
6. **Report.** A plain-English report for a science-fair judge or a city traffic engineer.

Three rules that are never broken:

- No camera image or video is ever written to disk. The live view on your phone is made in
  memory and disappears when you close it.
- The device only measures. It never controls the traffic light. The result is a
  recommendation for the city to review.
- No hand counting. Every number comes from the recorded data.

"Adaptive" in the project title means the timing adapts to the traffic that was actually
measured, by time of day and week by week. It does not change the light in real time.

---

## Try it now with the sample data (laptop, 3 commands)

You need Python 3.11 or newer and this folder (download it from GitHub with *Code -> Download
ZIP* and unzip it, or `git clone` it). Open a terminal in the folder. On Windows use
**PowerShell**; if `python` is not found, type `py` instead (and `py -m pip` for `pip`).

```bash
pip install -r requirements-analysis.txt
```

```bash
python -m tlopt all sample_data --intersection analysis/intersection.example.yaml
```

```bash
start results/REPORT.html
```

(On a Mac use `open` instead of `start`.) The second command takes about a minute. It reads
the sample data, measures the traffic, checks the model, finds the best timing, runs 60 traffic
simulations and writes the report. `sample_data/` is made-up traffic from
`tools/make_sample_data.py` (2 weekdays x 07:00-08:00 and 17:00-18:00), in exactly the same
format the Pi writes.

To check that everything works, run the tests (about 2 minutes). On a laptop the collector's
tests are skipped unless its packages are installed too (see *Test with a phone video*).

```bash
python -m pytest
```

---

## What is in this folder

| Folder | What it is |
|---|---|
| `collector/` | The program on the Raspberry Pi (`monitor`), its installer and `check_hardware.py` |
| `analysis/` | The program on the laptop (`tlopt`) and `intersection.example.yaml` |
| `common/` | The data format both programs share (it checks every file) |
| `tools/make_sample_data.py` | Makes realistic made-up data, for testing |
| `sample_data/` | Output of the sample generator |
| `docs/DATA_FORMAT.md` | Exactly what is in every data file, with examples |
| `docs/BUILD_NOTES.md` | Every design choice and default, and how accurate the measurements are |

---

## Part 1: Set up the Raspberry Pi (once, at home)

You need: the Pi 5 with the AI HAT+ (Hailo-8L), Active Cooler and RTC battery fitted; the 32 GB
microSD card; the 4 USB cameras; the INA226 sensor wired to I2C (SDA, SCL, 3.3 V, GND) with its
shunt in the + wire of the 12 V battery; and Wi-Fi internet at home for the installation.

1. **Card.** With Raspberry Pi Imager, write *Raspberry Pi OS Lite (64-bit)* to the card. In
   the Imager settings set a user name and password, your home Wi-Fi, and turn on SSH.
2. **AI chip.** Start the Pi, log in with SSH, then:
   ```bash
   sudo apt update && sudo apt full-upgrade -y && sudo apt install -y hailo-all git
   ```
   Reboot (`sudo reboot`). Check: `hailortcli fw-control identify` prints the chip's details.
3. **This project:**
   ```bash
   git clone https://github.com/mooximos/aethersync-software.git ~/intersection-monitor
   ```
   ```bash
   cd ~/intersection-monitor && bash collector/install.sh
   ```
   The installer is safe to run again. It creates a Python environment, turns on I2C, gives the
   program permission to shut down and to set the clock (nothing else), and starts the
   collector automatically at every boot.
4. **Wi-Fi hotspot** (so a phone can connect at the pole, where there is no internet). If the
   connection "Hotspot" does not exist yet, make it once (choose your own password of 8+
   characters):
   ```bash
   sudo nmcli device wifi hotspot ifname wlan0 con-name Hotspot ssid IntersectionMonitor password CHANGE-ME
   ```
   ```bash
   sudo nmcli connection modify Hotspot connection.autoconnect yes connection.autoconnect-priority 100
   ```
   From now on the Pi is at **10.42.0.1** on the *IntersectionMonitor* network.
5. **Settings.** Edit `collector/site.json` (`nano collector/site.json`). Replace every value
   that says REPLACE: the site name, your names, and each camera's `usb_path`. Also set
   `latitude` and `longitude` (right-click the intersection in an online map); in the Americas
   the longitude is negative, e.g. -121.87. The Pi will not start while they are still 0.0. The paths are listed by the next step. Camera A goes in the top USB 3
   port, B the bottom USB 3, C the top USB 2, D the bottom USB 2. Write down which camera
   looks at which approach (`approach` = the direction the traffic comes FROM).
   Optional: set `"web_password"` to protect the web page.
6. **Check the hardware:**
   ```bash
   .venv/bin/python collector/check_hardware.py --all
   ```
   Every line says PASS or FAIL. Under each FAIL it prints the fix. Do them all, then run it
   again. Two fixes that are usually needed the first time:
   - `EEPROM POWER_OFF_ON_HALT=1`: run `sudo -E rpi-eeprom-config --edit`, add the line
     `POWER_OFF_ON_HALT=1`, save, reboot. Without it the Pi cannot wake itself up.
   - `--hailo` saves the chip's raw output numbers to
     `collector/tests/fixtures/hailo_output_recorded.json`. Commit that file: the tests then check
     the program reads the real chip's output correctly.
7. **Reboot** (`sudo reboot`). The collector now runs by itself.

### Test with a phone video first (optional, on the laptop)

Before climbing the pole you can test the whole program on a short phone video of an
intersection, filmed from as high as you can safely get:

```bash
pip install -r requirements-collector.txt opencv-python-headless==5.0.0.93 ultralytics==8.4.215
```

```bash
python -m monitor --replay my_video.mp4 --camera A --detector ultralytics --site collector/site.json --out data_test
```

The rows get today's date and time unless you add `--start` with when you filmed it, written
`YYYY-MM-DDTHH:MM:SS` (local time).

First copy `collector/site.example.json` to `collector/site.json`. The replay writes real data
files to `data_test/`. Without a calibration for that video the vehicles are flagged
`uncalibrated`: you can still check detection, tracking and counting, but not distances,
speeds or waits. To get those, put a homography for the video in `site.json` (camera A's
`homography` and `calib_points`); the Calibrate page on the Pi does this for the real cameras.

### At the pole: mounting and calibration

- **Safety first.** Mounting at 5.8 m needs the city's permission and an adult with the right
  equipment. Never stand in the road to measure anything.
- Cover each camera's small light sensor (next to the lens) with black tape, so the infrared
  LEDs stay off.
- Point each camera along its approach so the line of waiting cars runs up the middle of the
  picture, and the stop line is near the bottom.
- Connect your phone to the *IntersectionMonitor* Wi-Fi and open **http://10.42.0.1:8080**.
  The web page has these pages:
  - **Status:** recording or idle, battery, temperatures, frames per second, vehicles counted
    in the last 15 minutes.
  - **Live view:** the cameras with lanes and stop line drawn on. It is made from memory only
    while the page is open; nothing is saved.
  - **Calibrate:** teaches each camera where the road is (next list).
  - **Download:** zip files of each day's data.
  - **Clock sync:** sets the Pi's clock from your phone.
- **Calibrate** each camera on the Calibrate page:
  1. Pick the camera and press *Get picture*. Recording must be on, so do this during a
     recording window or temporarily widen the schedule.
  2. Tap at least **6** points on the road whose position you know, spread over the picture:
     stop-line ends, lane-line ends, crosswalk corners. Do not use points on cars or poles.
  3. For each point type `x_m` (meters from the curb edge toward the center line) and `y_m`
     (meters before the stop line; 0 = on the stop line). Get the distances **without going
     into the road**: from the city's plan drawings, from an online map's measuring tool, or
     from standard markings. In the US, dashed lane lines are usually 10 ft (3.05 m) long with
     30 ft (9.14 m) gaps, and lanes about 12 ft (3.66 m) wide.
  4. Press *Check*. The page shows each point's error in meters. Fix or remove points with a
     big error. *Save* only works when the average error is below 0.5 m.
  5. Also fill in `lane_x_m` (where each lane starts and ends across the road) and
     `view_length_m` (how far up the road the camera can still see cars clearly, usually
     50-70 m from this height).
  6. Check the **Live view**: the drawn lane lines and stop line must sit on the real ones.
- When cars are waiting at a red light, run this over SSH:
  ```bash
  .venv/bin/python collector/check_hardware.py --jitter
  ```
  If it says FAIL, shorten `view_length_m`, or add a *tile* (an extra zoomed-in box) over the
  far part of the queue. Accuracy depends on this; see `docs/BUILD_NOTES.md`.
- Press **Clock sync** on the web page.

---

## Part 2: The weekly routine

The Pi records on weekdays 06:30-09:30 and 15:30-18:30 (change `schedule` in `site.json`).
Between windows it switches itself off and wakes up 10 minutes before the next one.
Nothing in the software is tied to particular dates: it records on the days and times in
`schedule` for as long as it has power, so you can start and stop a collection whenever you want.
To record on weekends too, add `"Sat", "Sun"` to the `days` of each window.

**Winter evenings.** From about November to January it is dark before 17:00. The cameras keep
recording at night (night mode), but the analysis leaves out twilight and night vehicles unless
you turn that off (`daylight_only` in `intersection.yaml`, see Part 3). Plan an evening study
period that is still light, or check the night accuracy first.

**The clock.** If the Pi's clock looks wrong when it starts (it is behind the files it already
wrote, or it disagrees with the clock chip because the clock battery is flat), it does **not**
record. The Status page then says *NOT RECORDING* in red; press **Clock sync** and it starts.

Once a week:

1. **Swap the battery** for a charged one (12 V LiFePO4). If the battery drops below 12.0 V
   for a minute, the Pi closes its files and switches off safely; it will not wake again until
   the battery is swapped.
2. Connect your phone to *IntersectionMonitor* and open http://10.42.0.1:8080.
   - If the Pi is asleep, it wakes up for the next window; or power-cycle it, and then press
     **Stay awake for 1 hour** so it does not go back to sleep while you work.
   - **Status:** check every camera shows about 10 frames per second, the temperatures are below
     80 C, and there is enough disk space.
   - **Clock sync:** press it every visit (the clock drifts a little).
   - **Download:** download every new day as a zip. Days you already downloaded are marked.
3. On the laptop, unzip each day into a `data/` folder (you get one `data/YYYY-MM-DD/` folder per day),
   then check the data:
   ```bash
   python -m tlopt check data
   ```
   Look at the quality table. Problems to look for: a camera with low frames per second, many
   `no_cross` or `low_conf` vehicles, outages, or lanes where the detector and the crossing
   counts differ by more than 3%.
4. Read `events.log` in each day folder for errors.

Tip: collect at least one full week (better two or three) of weekdays before drawing
conclusions. Write down any day that was not normal (a holiday, school break, a crash nearby,
road work) so you can leave it out in Part 3.

---

## Part 3: Run the analysis on your own data

1. Copy `analysis/intersection.example.yaml` to `analysis/intersection.yaml` and fill in what the
   cameras cannot see. Every key has a comment.
   - **Lanes:** the same lane numbers and movements as `site.json`.
   - **Geometry:** speed limit, grade, crosswalk lengths (feet), and the distance from the stop
     line to the far side of the intersection.
   - **Phases:** the signal's phases (NEMA phase numbers 1-8).
   - **Timing sheet:** the city's timing sheet, if you can get it. Ask the city's traffic
     department; without it the timing is worked out from the data.
   - **Periods:** the study periods, as clock times only (no dates), e.g. `am: "07:00-09:00"`.
     They must be inside the recording windows, and in daylight unless you turn off
     `daylight_only`. If a period has no usable data, the program says why and skips it.
   - **Which days:** `days` (default Monday to Friday) and `skip_dates` (unusual days, written
     `[YYYY-MM-DD, YYYY-MM-DD]`). Every day folder you give it is used except those; the report lists the
     days used and the ones left out. To study only some weeks, put only those day folders in
     the data folder, or give several folders: `python -m tlopt all data/week1 data/week2`.
2. Run everything:
   ```bash
   python -m tlopt all data
   ```
   Or one step at a time: `check data`, `measure data`, `optimize data`, `simulate`,
   `report`. Everything goes in `results/`:
   - `results/measures/`: the measured traffic, delays and quality tables (CSV)
   - `results/plans/`: the current, Webster and recommended plans (JSON) and the model check
   - `results/sumo/`: the simulation results
   - `results/REPORT.html`: the report, with charts (one file, easy to share)
   - `results/REPORT.md`: the same report as text
3. Read the report, especially section 5 (does the model match reality?) and section 9
   (limitations). Every number in the report can be traced back: `results/measures/vehicle_delays.csv`
   lists every vehicle with the file it came from.

---

## When something goes wrong

| Problem | What to do |
|---|---|
| The web page does not open | Is your phone on the *IntersectionMonitor* Wi-Fi? Is the Pi on (green LED)? Use http://, not https://. |
| "Cannot start: site.json is not valid" | The message names the wrong key, e.g. `cameras.B.approach`. Fix it in `collector/site.json`. |
| A camera shows 0 frames per second | Check its cable and port; the program retries every 5 seconds. Run `check_hardware.py --cameras`. |
| The Pi does not wake up | Run `check_hardware.py --rtc` and do the fixes it prints. |
| Collector log | `journalctl -u intersection-monitor -f` over SSH (Ctrl+C to stop watching). |
| Restart the collector | `sudo systemctl restart intersection-monitor` |
| `python -m tlopt` says no intersection.yaml | Make `analysis/intersection.yaml` (Part 3), or add `--intersection PATH`. |

## Privacy and safety

- Only numbers are stored; you can check: the data folders contain only `.csv`, `.csv.gz`,
  `.json` and `.log` files.
- Set a `web_password` in `site.json` if other people may join the hotspot.
- Work at the pole only with an adult, with the city's permission, and never in the road.
