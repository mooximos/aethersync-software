"""Smaller parts: night mode controls, INA226, homography fitting, Kalman, cameras, sun, clock, IDs."""
import math

import numpy as np
import pytest

from common.timeutil import local_epoch
from helpers import ZONE, sample_site
from monitor import camera, nightmode
from monitor.clock import Clock
from monitor.ground import GroundMapper, Kalman, fit_homography
from monitor.power import INA226, LowBatteryWatch
from monitor.sun import light_label, sun_elevation

CTRLS_NEW = """
User Controls
                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
                           gain 0x00980913 (int)    : min=0 max=100 step=1 default=0 value=0
Camera Controls
                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=157 flags=inactive
"""
CTRLS_OLD = """
                  exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
              exposure_absolute 0x009a0902 (int)    : min=3 max=2047 step=1 default=250 value=250 flags=inactive
"""


@pytest.mark.parametrize("text,auto,exposure", [(CTRLS_NEW, "auto_exposure", "exposure_time_absolute"),
                                                (CTRLS_OLD, "exposure_auto", "exposure_absolute")])
def test_both_kinds_of_exposure_control_names(text, auto, exposure):
    controls = nightmode.ExposureControls.find(nightmode.parse_controls(text))
    assert (controls.auto, controls.exposure) == (auto, exposure)
    assert controls.settings(False, 40, None) == [(auto, 3)]
    night = controls.settings(True, 1, 50)
    assert night[0] == (auto, 1) and night[1] == (exposure, 1 if auto == "auto_exposure" else 3)   # clamped to min


def test_night_mode_runs_v4l2_ctl_and_describes_it():
    calls = []

    class Done:
        returncode = 0
    controls = nightmode.ExposureControls.find(nightmode.parse_controls(CTRLS_NEW))
    text = nightmode.apply("/dev/video0", controls, True, 40, 20, run=lambda cmd, **k: (calls.append(cmd), Done())[1])
    assert calls == [["v4l2-ctl", "-d", "/dev/video0", "-c", "auto_exposure=1", "-c", "exposure_time_absolute=40",
                      "-c", "gain=20"]]
    assert "exposure_time_absolute=40" in text and "FAILED" not in text


class FakeBus:
    def __init__(self, registers, fail=0):
        self.registers, self.fail, self.writes = registers, fail, []

    def write_i2c_block_data(self, address, register, data):
        self.writes.append((register, (data[0] << 8) | data[1]))

    def read_i2c_block_data(self, address, register, length):
        if self.fail:
            self.fail -= 1
            raise OSError("I2C error")
        value = self.registers[register]
        return [value >> 8, value & 0xFF]


def test_ina226_setup_and_reading():
    bus = FakeBus({0x02: round(13.2 / 0.00125), 0x04: 6554, 0xFE: 0x5449, 0xFF: 0x2260})
    sensor = INA226(smbus_factory=lambda n: bus)
    assert bus.writes == [(0x00, 0x4527), (0x05, 3355)]       # 16-sample average; 0.01 ohm, 5 A
    volts, amps = sensor.read()
    assert volts == pytest.approx(13.2, abs=0.002) and amps == pytest.approx(1.0, abs=0.001)
    assert sensor.ids() == (0x5449, 0x2260)


def test_ina226_retries_once_then_gives_empty_values():
    bus = FakeBus({0x02: 10560, 0x04: 0xFFFF}, fail=1)
    sensor = INA226(smbus_factory=lambda n: bus)
    assert sensor.read() == (13.2, pytest.approx(0.0, abs=0.001))   # 0xFFFF = -1 bit = -0.00015 A, rounded
    bus.fail = 2
    assert sensor.read() == (None, None)


def test_low_battery_needs_the_whole_hold_time():
    watch = LowBatteryWatch(12.0, 60)
    assert not watch.update(0, 11.9) and not watch.update(59, 11.9)
    assert not watch.update(60.5, 12.1)                       # recovered: start again
    assert not watch.update(61, 11.8) and watch.update(121, 11.8)
    assert not watch.update(200, None)                        # no sensor reading: never shuts down


def test_fit_homography_recovers_the_sample_camera():
    site = sample_site()
    truth = GroundMapper(site.cameras["A"].homography)
    points = []
    for x, y in [(0, 0), (10.5, 0), (0, 20), (10.5, 20), (0, 50), (10.5, 50), (5, 35)]:
        u, v = truth.to_image(x, y)
        points.append({"px": [u, v], "m": [x, y]})
    matrix, errors = fit_homography(points)
    assert max(errors) < 0.01
    assert GroundMapper(matrix).to_ground(*truth.to_image(3.0, 40.0)) == pytest.approx((3.0, 40.0), abs=0.01)
    points[6]["m"] = [5, 39]                                  # one point typed wrong by 4 m
    _, errors = fit_homography(points)
    assert errors[6] > 3.0 and max(errors[:6]) < 0.1          # RANSAC keeps the good points
    with pytest.raises(ValueError):
        fit_homography(points[:3])


def test_ground_noise_grows_with_distance():
    mapper = GroundMapper(sample_site().cameras["A"].homography)
    near = mapper.noise(*mapper.to_image(5.0, 2.0))
    far = mapper.noise(*mapper.to_image(5.0, 100.0))
    assert math.sqrt(far[1, 1]) > 5 * math.sqrt(near[1, 1])


def test_kalman_speed_settles_and_flags_jumps():
    noise = np.eye(2) * 0.05
    kf = Kalman(0.0, 0.0, 100.0, noise)
    for i in range(1, 40):
        x, y, speed = kf.update(i * 0.1, 0.0, 100.0 - 1.0 * i, noise)   # 10 m/s
        assert kf.jump_m == 0.0
    assert speed == pytest.approx(10.0, abs=0.3)
    kf.update(4.0, 0.0, 50.0, noise)                                   # 10 m off: a swapped track
    assert kf.jump_m > 4.0


def test_mjpeg_rate_parsing_and_choice():
    text = """ioctl: VIDIOC_ENUM_FMT
\t[0]: 'MJPG' (Motion-JPEG, compressed)
\t\tSize: Discrete 1920x1080
\t\t\tInterval: Discrete 0.033s (30.000 fps)
\t\t\tInterval: Discrete 0.067s (15.000 fps)
\t\tSize: Discrete 640x480
\t\t\tInterval: Discrete 0.008s (120.000 fps)
\t[1]: 'YUYV' (YUYV 4:2:2)
\t\tSize: Discrete 1920x1080
\t\t\tInterval: Discrete 0.200s (5.000 fps)
"""
    assert camera.parse_mjpeg_rates(text) == [15.0, 30.0]
    assert camera.choose_rate([15.0, 30.0], 10) == 15.0
    assert camera.choose_rate([5.0, 7.5], 10) == 7.5
    assert camera.choose_rate([], 10) == 15.0


def test_sun_and_light_labels():
    noon = local_epoch("2026-06-21", 13, 0, ZONE)
    midnight = local_epoch("2026-06-21", 0, 30, ZONE)
    assert sun_elevation(37.67, -121.87, noon) > 70 and sun_elevation(37.67, -121.87, midnight) < -20
    assert [light_label(e) for e in (5, 0, -3, -6, -7)] == ["day", "twilight", "twilight", "twilight", "night"]


def test_clock_offset_and_resync():
    wall, mono = [1000.0], [5.0]
    clock = Clock(time_fn=lambda: wall[0], mono_fn=lambda: mono[0])
    assert clock.offset == 995.0 and clock.wall(6.0) == 1001.0
    wall[0] = 2000.0                       # the system time was set
    assert clock.resync() == 1000.0 and clock.now() == 2000.0


def test_vehicle_ids_continue_after_a_restart(tmp_path):
    from helpers import PipelineRig, drive
    rig = PipelineRig(tmp_path)
    rig.run([drive(120.0, 13.4, [(10.0, 0.0)])])
    rig.close()
    again = PipelineRig(tmp_path)
    assert again.recorder.new_vehicle_id("A", 100.0) == "A-0002"
    assert again.recorder.new_vehicle_id("B", 100.0) == "B-0001"
