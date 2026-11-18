# Data format

This is the contract between the two programs: the collector (Raspberry Pi) writes these
files and the analysis (laptop) reads them. `common/schema.py` and `common/site.py` check
every file against it. The text in section 2 below is copied word for word from the
project's build specification; the examples after it are real rows from `sample_data/`.

## 2. Data format (the contract between the two programs; implement exactly)

Folder per day: data/YYYY-MM-DD/. One set of files per local clock hour, suffix _HH (00-23). A row goes in the hour of its main time (t_left for vehicles, time for the others).
All times: ISO 8601 local time with UTC offset, 0.1 s precision, e.g. 2026-11-16T07:14:52.3-08:00. Use zoneinfo (America/Los_Angeles) so daylight-saving changes are correct.
CSV: UTF-8, comma separated, header row, no index column, empty string for missing values, "." decimal. Units: meters, seconds, m/s.

### 2.1 site.json — settings and calibration
Copied into each day folder at the first write of the day, and again as site_HHMMSS.json whenever settings change during the day.
Keys:
- site_name, latitude, longitude, time_zone, researchers (list), software_version, privacy ("no images or video are stored"), web_password (optional)
- device: {computer, ai_chip, mount_height_m}
- cameras: {A, B, C, D}, each with:
  - approach: N|S|E|W (the direction traffic comes FROM)
  - usb_path: its /dev/v4l/by-path/... name
  - lanes: {"1": {"movements": ["R","T"]}, "2": {"movements": ["T"]}, "3": {"movements": ["L"]}} (lane 1 = curbside)
  - lane_x_m: {"1": [x0, x1], ...} (lateral range in ground coordinates)
  - homography: 3x3 matrix, image pixels -> ground meters (null until calibrated)
  - calib_points: [{"px": [u, v], "m": [x, y]}, ...]
  - tiles: [[x, y, w, h], ...] (extra full-resolution crops in pixels; may be empty)
  - stop_line_zone_m: 6.0
  - view_length_m: meters from the stop line to the far edge of usable view
- coordinates: {"x_m": "meters from the curbside edge of the approach toward the center line", "y_m": "meters upstream of the stop line; 0 = stop line, negative = past it"}
- detection: {model: "yolov8s", hef_path, input_px: 640, frames_per_second: 10, min_confidence: 0.35, low_conf_flag_below: 0.65, classes: ["car","motorcycle","bus","truck"]}
- stopped_speed_mps: 0.5, moving_speed_mps: 1.0, reaction_time_s: 1.5, track_sample_rate_hz: 2
- schedule: {windows: [{days: ["Mon","Tue","Wed","Thu","Fri"], start: "06:30", end: "09:30"}, {days: [same], start: "15:30", end: "18:30"}], boot_grace_min: 15, wake_early_min: 10}
- power: {shutdown_below_v: 12.0, shutdown_after_s: 60}
- night: {switch_sun_elevation_deg: -2, night_exposure: 40, night_gain: null}
- event_codes: {"82": "detector_on", "81": "detector_off", "1": "green_start_inferred", "10": "red_inferred_by"}
- clock: {last_sync, offset_found_s}
Validate with a schema (pydantic or jsonschema). Refuse to start with a clear message naming the bad key.

### 2.2 vehicles_HH.csv — one row per vehicle, written when its track ends
Columns, in this order:
- vehicle_id: "<camera>-<counter>", counter at least 4 digits, resets daily (A-0412)
- camera: A|B|C|D
- approach: N|S|E|W
- lane: integer from lane_x_m, using the median x over the vehicle's last 3 s before the stop line; empty if outside all lanes
- turn: L|T|R, or U if unknown (rule in 4.6)
- type: car|motorcycle|bus|truck (confidence-weighted majority of detector classes over the track)
- conf: mean detection confidence over the track, 2 decimals
- t_entered: first confirmed track point inside the view zone (0 <= y <= view_length_m)
- t_stopped: start of the first stop episode before the stop line; empty if never stopped
- t_started: end of the last stop episode before crossing; empty if never stopped
- t_crossed: time y crossed 0, linearly interpolated between samples; empty if it never crossed
- t_left: last time the track was seen
- wait_s: total seconds in all stop episodes before crossing, 1 decimal; 0.0 if never stopped
- stops: number of stop episodes before crossing
- queue_pos: 1 + the number of stopped vehicles in the same lane with smaller y at t_stopped; empty if never stopped
- light: day|twilight|night from sun elevation at t_entered (above 0 deg day, -6 to 0 twilight, below -6 night)
- flags: semicolon-separated, empty if none:
  - low_conf: conf < low_conf_flag_below
  - started_in_view: the first point is closer than view_length_m - 15, or it was already present when the collector started
  - queue_past_view: stopped within 5 m of the far edge of the view
  - no_cross, short_track (< 1.0 s or < 5 points), night_mode, uncalibrated, cut_at_shutdown
  - id_switch_suspect: a jump of more than 4 m between consecutive samples
Stop episode: smoothed speed below stopped_speed_mps continuously for at least 1.0 s. It ends when speed stays above moving_speed_mps for at least 0.5 s (hysteresis).

### 2.3 tracks_HH.csv — one row per vehicle per 0.5 s
Columns: time, vehicle_id, x_m, y_m, speed_mps (1 decimal). Sample on each 0.5 s tick of the monotonic clock, using the nearest frame within 0.15 s; skip the tick if there is none. When the hour closes, gzip it to tracks_HH.csv.gz. All other files stay plain CSV.

### 2.4 signal_events_HH.csv — one row per event
Columns: time, event_code, event, approach, lane, detail
- 82 detector_on / 81 detector_off. Each lane has a virtual detector covering 0 <= y <= stop_line_zone_m across that lane. ON when a vehicle's ground point enters the empty zone; OFF when the zone becomes empty. detail = the vehicle_id that turned it on, or the last one to leave. Same codes as the Indiana high-resolution traffic signal event enumerations used by ATSPM.
- 1 green_start_inferred, per approach. When at least one vehicle is in a stop episode inside the stop-line zone and the first of them starts moving: time = that start minus reaction_time_s. detail = "first queued car moved at HH:MM:SS.s; minus 1.5 s reaction". At most one per approach per 20 s.
- 10 red_inferred_by, per approach. The first vehicle that begins a stop episode inside the stop-line zone after the approach has had at least one crossing since the last green_start_inferred. time = its stop start; detail = "first car stopped at the line". The light turned red at or before this time.
lane is empty for approach-level events.

### 2.5 system_HH.csv — one row per minute, on the minute
Columns: time, battery_v, current_a, power_w, cpu_temp_c, hailo_temp_c, fps_A, fps_B, fps_C, fps_D, dropped_frames, disk_free_gb
- fps = processed frames per second per camera over that minute
- dropped_frames = frames grabbed but not processed, all cameras
- An unavailable sensor gives an empty cell, never a made-up number.

### 2.6 Also in each day folder
- clock_sync.csv: time_before, time_after, offset_s, source (phone|laptop)
- events.log: plain-text log, one line per startup, shutdown, error or mode switch, each with an ISO time

## Examples (from sample_data/2026-05-12)

### site.json (one camera shown; the full file has all keys of 2.1)

```json
"A": {"approach": "N", "usb_path": "/dev/v4l/by-path/SAMPLE-DATA-camera-A-(no-real-camera)",
      "lanes": {"1": {"movements": ["R", "T"]}, "2": {"movements": ["T"]}, "3": {"movements": ["L"]}},
      "lane_x_m": {"1": [0.0, 3.5], "2": [3.5, 7.0], "3": [7.0, 10.5]},
      "homography": [[-0.0194709344, -0.0170240107, 23.9420971], [-0.0, 0.10231951, -48.9782587], [0.0, -0.0032426687, 1.0]],
      "calib_points": [{"px": [811.1, 478.7], "m": [0.0, 0.0]}, {"px": [1108.9, 478.7], "m": [10.5, 0.0]}],
      "tiles": [[640, 290, 640, 360]], "stop_line_zone_m": 6.0, "view_length_m": 120.0}
```

### vehicles_07.csv

```csv
vehicle_id,camera,approach,lane,turn,type,conf,t_entered,t_stopped,t_started,t_crossed,t_left,wait_s,stops,queue_pos,light,flags
A-0083,A,N,2,T,car,0.69,2026-05-12T07:03:45.1-07:00,2026-05-12T07:03:54.2-07:00,2026-05-12T07:04:08.4-07:00,2026-05-12T07:04:13.0-07:00,2026-05-12T07:04:14.4-07:00,14.2,1,5,day,
```

A vehicle that never stopped, and one with a flag:

```csv
C-0010,C,S,2,T,car,0.81,2026-05-12T07:00:01.9-07:00,,,2026-05-12T07:00:12.6-07:00,2026-05-12T07:00:13.8-07:00,0.0,0,,day,
A-0933,A,N,1,T,car,0.79,2026-05-12T17:00:00.0-07:00,2026-05-12T17:00:00.0-07:00,2026-05-12T17:00:02.2-07:00,2026-05-12T17:00:02.7-07:00,2026-05-12T17:00:05.6-07:00,2.2,1,1,day,started_in_view
```

### tracks_07.csv.gz (after unzipping)

```csv
time,vehicle_id,x_m,y_m,speed_mps
2026-05-12T07:01:41.5-07:00,A-0032,5.47,1.43,0.0
```

### signal_events_07.csv (one row of each event code)

```csv
time,event_code,event,approach,lane,detail
2026-05-12T07:00:00.0-07:00,82,detector_on,N,1,A-0001
2026-05-12T07:00:01.8-07:00,81,detector_off,S,3,C-0007
2026-05-12T07:00:00.1-07:00,1,green_start_inferred,S,,first queued car moved at 07:00:01.6; minus 1.5 s reaction
2026-05-12T07:00:54.5-07:00,10,red_inferred_by,N,,first car stopped at the line
```

### system_07.csv

```csv
time,battery_v,current_a,power_w,cpu_temp_c,hailo_temp_c,fps_A,fps_B,fps_C,fps_D,dropped_frames,disk_free_gb
2026-05-12T07:00:00.0-07:00,13.29,0.92,12.2,57.8,50.0,10.0,10.0,10.0,10.0,3,24.77
```

A minute with no battery sensor and no Hailo temperature (empty cells, never made-up numbers):

```csv
2026-11-17T00:00:00.0-08:00,,,,,,0.0,0.0,0.0,0.0,0,31.47
```

### clock_sync.csv

```csv
time_before,time_after,offset_s,source
2026-05-12T06:48:58.6-07:00,2026-05-12T06:48:59.8-07:00,1.2,phone
```

### events.log

```text
2026-05-12T06:48:59.8-07:00 clock synced from phone, offset +1.2 s
2026-05-12T07:00:00.0-07:00 startup: collector 1.0.0 (sample data generator)
```

## Details the specification left open

These are the choices made while building; `docs/BUILD_NOTES.md` explains each one.

- `x_m` and `y_m` in tracks have 2 decimals.
- `offset_s` in clock_sync.csv = `time_after` - `time_before`.
- `t_entered`, `wait_s` and `stops` may be empty for a vehicle flagged `uncalibrated`.
- A stop episode's start is the first slow sample (once the stop has lasted 1.0 s); its end is the
  first sample of the 0.5 s run above moving speed. A stop still open at the line ends at `t_crossed`.
- A vehicle row is written only for a track that was inside the view zone at some point.
- A detector occupied at startup turns ON at the start; one occupied at shutdown turns OFF then.
- `green_start_inferred` and `red_inferred_by` leave out lanes that allow a right turn (right turn
  on red looks like a green), unless the approach has no other lane.
- `id_switch_suspect` = a measurement more than 4 m from where the vehicle was expected and more
  than 4 standard deviations of the normal measurement noise there.
- `dropped_frames` counts decoded frames the main loop never took (it was busy), not the frames
  skipped on purpose to get down to `frames_per_second`.
- In the tracks file, `time` is the 0.5 s tick's time; the values come from the nearest frame.
- "rule in 4.6" in 2.2 refers to the turn rule, section 3.6 of the build specification: one movement in the lane -> that movement; else 8 m past the line, more than 3 m toward the curb = R, toward the center = L, otherwise T; else U.
