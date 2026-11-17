# Build notes (for picking the work back up)

The project is built in 4 stages from the original build brief. This file
records where the build stands and the decisions made so far.

## Status

| Stage | Contents | State |
|---|---|---|
| 1 | `common/`, `tools/make_sample_data.py`, `sample_data/` | **done** |
| 2 | `collector/` (with replay mode and `check_hardware.py`) | **done**, 161 tests pass in total (`python -m pytest`) |
| 3 | `analysis/` (with SUMO and the report) | not started |
| 4 | `README.md` (student guide), `docs/DATA_FORMAT.md` | not started (README is a placeholder) |

## Defaults chosen in stage 1

Data format details the brief left open:
- `x_m`, `y_m` are written with 2 decimals (speed has 1, as specified).
- A stop episode **starts** at the first sample below 0.5 m/s (once it has lasted 1.0 s) and
  **ends** at the first sample of the run above 1.0 m/s that lasts 0.5 s. `wait_s` is the sum of
  end - start. An episode still open at the stop line is cut at `t_crossed`.
- A vehicle row is only written for a track that was inside the view zone
  (0 <= y <= view_length_m) at some point, so `t_entered` is never empty for a calibrated camera.
  Cross traffic and departing traffic seen past the stop line get no row.
- For an `uncalibrated` camera, `t_entered`, `wait_s` and `stops` may be empty too (they depend on distance or speed).
- A detector that is occupied when the collector starts gets `detector_on` at the start time; one
  still occupied at shutdown gets `detector_off` at the shutdown time, so on/off always pair up.
- `detector_off` is stamped at the first sample where the zone is empty.
- `clock_sync.csv` `offset_s` = `time_after` - `time_before`.
- `site.json`: unknown keys are an error (catches typos); 1 to 4 cameras are allowed; a schedule
  window cannot cross midnight. The schema lives in `common/site.py` (pydantic), so pydantic is
  also needed on the laptop.
- `dropped_frames` (to be implemented in stage 2) should count frames that were due for
  processing at the target rate but were skipped because the Pi was busy - not the frames
  deliberately skipped to go from the camera's rate down to 10 per second.

Sample data generator:
- Dates default to 2026-05-12 and 2026-05-13 (Tue, Wed), because both 07:00 and 17:00 are in full
  daylight then. The generator does not compute the sun: hours given with `--night-hours` are
  labeled `night` + `night_mode`, everything else is `day`.
- Each block of neighbouring hours is one recording: it starts on the hour (vehicles already in
  view get `started_in_view`) and is shut down at HH:59:59.9 (`cut_at_shutdown`).
- Default splits leave 4 s of the 120 s cycle unused (after E/W green 56-110, yellow, all-red);
  it is simulated as extra all-red.
- Left turns run with their approach's green and do not yield; no right turn on red; no lane changes.
- Car following is Newell's model (a follower repeats its leader's path 1.5 s later, 6.5 m behind);
  measured through-lane saturation headway is 2.0 s. Right/left turners slow to 6/8 m/s, so
  turn lanes discharge slower.
- Default volumes (veh/h, left/through/right): AM N 90/700/90, S 70/500/80, E 60/340/70,
  W 80/420/60; PM N 80/540/90, S 100/720/100, E 90/500/80, W 70/460/90. Each quarter hour is
  scaled 0.85/1.0/1.15/1.0 and each day by a random factor (sd 4 %).
- Vehicle mix: 93 % car, 4 % truck, 2 % motorcycle, 1 % bus.
- Sample camera: pretend pinhole camera 30 m past the stop line, 5.8 m up, 15 degrees down, 96 degree lens.
- Cameras: A = N, B = E, C = S, D = W.

## Defaults chosen in stage 2 (collector)

- **supervision is pinned to 0.30.9**, the last release that still has `sv.ByteTrack` (0.31 removed it).
  ByteTrack keeps a lost track for `frame_rate / 30 * lost_track_buffer` frames, so
  `lost_track_buffer = 60` gives 2 s at any frame rate. ByteTrack itself only starts a new track
  for a box scoring at least 0.45 (its activation threshold 0.35 + 0.1).
- **Vehicle IDs** are given when a track first enters the view zone (0 <= y <= view_length_m), so a
  track that never does (cross traffic, cars seen only past the stop line) uses no number and gets
  no row. After a restart the counter continues from the highest ID already in today's files.
- **Ground point** = bottom-center of the box = the vehicle's front for a camera looking at oncoming
  traffic. If the cameras end up looking at the backs of cars, y_m is the rear bumper instead.
- **Smoothing**: constant-velocity Kalman filter; its position noise comes from the homography
  (1.5 px of box jitter turned into meters at that spot), and its acceleration noise is 2 m/s2.
  On fake boxes from the sample data, stops are detected about 0.9 s late and the end of a stop
  about 0.7 s late; the two mostly cancel in `wait_s`. `green_start_inferred` inherits the 0.7 s.
- **`id_switch_suspect`** = a measurement more than 4 m from where the filter expected the vehicle
  *and* more than 4 standard deviations of the normal noise there, after the first 5 frames.
  (Far away, one pixel is several meters, so a plain 4 m rule flagged almost every vehicle.)
- **Tracks file**: one row per 0.5 s tick of the monotonic clock; the time written is the tick's
  time, the values come from the nearest processed frame within 0.15 s.
- **Virtual detectors**: judged frame by frame; a vehicle the detector misses for under 1.0 s still
  counts as inside. A vehicle that disappears turns the zone off at the time it was last seen.
- **Right turn on red (decided)**: green and red inference ignore every lane that allows a right
  turn, unless the approach has no other lane. Crossings in any lane count.
- **Hour files close 30 s after the hour** (vehicle rows arrive up to ~2.5 s after `t_left`).
  A row for an hour that was already gzipped (late row, or a restart within the hour) unzips that
  file and appends; it is zipped again when the hour closes. At shutdown the current hour's tracks
  file is gzipped too.
- **Frames are decoded only when due**: every frame is grabbed (keeps the camera queue empty) but
  only about 10 per second are decoded. `dropped_frames` counts decoded frames the main loop never
  took (it was busy), not the frames skipped on purpose.
- **Rate control**: every 10 s, if the loop was busy more than 90% of the time, every camera drops
  1 fps (not below 4); after 60 s under 60% busy it goes back up 1 fps at a time.
- **Wake alarm permission**: the service runs as the normal user, so `install.sh` adds a udev rule
  that lets the user's group write `/sys/class/rtc/rtc0/wakealarm`. The sudoers lines stay limited
  to `shutdown -h now`, `date -s *` and `hwclock -w`.
- **No sleep** if the next window starts in under 10 minutes.
- **System rows** are written every minute while the service runs, recording or not (battery
  history); fps is 0.0 while idle.
- **Night mode** switches all cameras together, checked once a minute. Auto exposure is restored
  with menu value 3 (aperture priority), which is the UVC "auto" setting.
- **Web UI**: optional password is HTTP basic auth (any user name). The Calibrate page needs at
  least 4 points and warns below 6 (4 points always fit exactly). Pictures are made from the newest
  frame only while a page asks for them; a frame is held in RAM for 10 s after the last request.
- **Clock sync**: `source` is phone if the browser says Mobile/Android/iPhone/iPad, else laptop.
- **`collector/site.json` is not committed** (it may hold the web password); `site.example.json` is.

## Accuracy depends on box jitter (measure this on the Pi)

The end-to-end test runs the full pipeline on boxes made from the sample tracks (10 fps, 3% missed
detections). Result for 10 minutes, 4 cameras, all vehicles crossing inside the window:

| box jitter (1 sd) | waits within 1.0 s |
|---|---|
| 0.5 px (the test) | 100% |
| 0.75 px | 97% |
| 1.0 px | 94% |
| 1.5 px | 79% |

Counts per approach, lanes and turns stayed exact in all of these. The sample camera sees 120 m;
far away one pixel is several meters of road, so a stopped car there seems to move. Run
`check_hardware.py --jitter` while cars wait at a red light: if it is above 0.75 px, shorten
`view_length_m` or add a tile over the far part of the queue.

## Open questions for the project owners (asked at the end of stage 1)

1. **Right turn on red - DECIDED (yes).** A car that stops and then turns right on red looks like
   "the first queued car moved". Stage 2 rule: when inferring green, ignore every lane that allows
   a right turn (right-only or shared), unless the approach has no other lane.
2. **Darkness in the evening window.** With `daylight_only: true`, data after about 17:00 PST in
   November-January is excluded (sunset is before 17:00). The evening plan would then rest on
   15:30 to about 16:45 only.
3. **How far the cameras can really see.** At 5.8 m height with a 96 degree lens, one image pixel
   covers about 0.7 m of road at 60 m and about 3 m at 120 m, and a car at 120 m is about 13 pixels
   wide. Expect a usable `view_length_m` of roughly 50-70 m, not 120 m; queues longer than that
   will be reported as "at least". To be measured in the live view once the cameras are up.
4. **Seeing past the stop line.** The turn rule needs the track to continue 8 m past the stop
   line. A camera on the near-side pole looking back along the approach may not see that area;
   shared lanes would then report turn `U`.
