"""Image pixels -> ground meters, and a smoothed position and speed per track."""
from __future__ import annotations

import math

import cv2
import numpy as np

PIXEL_NOISE = 1.5        # how much a box's bottom edge jitters, pixels (1 standard deviation)
ACCEL_NOISE = 2.0        # how quickly vehicles can change speed, m/s2 (Kalman process noise).
                         # 4.0 halves the lag when a car starts moving but makes stop detection
                         # much worse when boxes jitter by 1 px or more (tested on sample data).
MAX_SPEED = 30.0


class GroundMapper:
    """Uses the camera's homography (site.json) to turn a pixel into (x_m, y_m)."""

    def __init__(self, homography: list[list[float]] | None) -> None:
        self.matrix = None if homography is None else np.array(homography, dtype=np.float64)
        self.inverse = None if self.matrix is None else np.linalg.inv(self.matrix)
        if self.inverse is not None:
            # A homography's overall sign is arbitrary; choose it so that points in
            # front of the camera (like the bottom-middle of the image) get w > 0.
            gx, gy, gw = self.matrix @ np.array([960.0, 1000.0, 1.0])
            if (self.inverse @ np.array([gx / gw, gy / gw, 1.0]))[2] < 0:
                self.inverse = -self.inverse

    @property
    def calibrated(self) -> bool:
        return self.matrix is not None

    def to_ground(self, u: float, v: float) -> tuple[float, float]:
        point = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float64), self.matrix)
        return float(point[0, 0, 0]), float(point[0, 0, 1])

    def to_image(self, x_m: float, y_m: float) -> tuple[float, float] | None:
        """Ground point -> pixel (for drawing on the live view). None if behind the camera."""
        x, y, w = self.inverse @ np.array([x_m, y_m, 1.0])
        return None if w <= 0 else (x / w, y / w)

    def noise(self, u: float, v: float) -> np.ndarray:
        """2x2 position uncertainty in m2 for a box bottom at (u, v): one pixel is
        a few cm close to the camera but meters far away."""
        here = np.array(self.to_ground(u, v))
        jac = np.column_stack([np.array(self.to_ground(u + 1, v)) - here,
                               np.array(self.to_ground(u, v + 1)) - here])
        return jac @ jac.T * PIXEL_NOISE ** 2 + np.eye(2) * 0.01


class Kalman:
    """Constant-velocity Kalman filter for one track. State: x, y, vx, vy."""

    def __init__(self, t: float, x: float, y: float, noise: np.ndarray) -> None:
        self.t = t
        self.state = np.array([x, y, 0.0, 0.0])
        self.cov = np.zeros((4, 4))
        self.cov[:2, :2] = noise
        self.cov[2:, 2:] = np.eye(2) * 15.0 ** 2   # speed unknown at first
        self.jump_m = 0.0
        self.updates = 0

    def update(self, t: float, x: float, y: float, noise: np.ndarray) -> tuple[float, float, float]:
        """Add one measurement. Returns smoothed (x, y, speed). self.jump_m = distance
        between the measurement and where the filter expected the vehicle."""
        dt = max(t - self.t, 1e-3)
        self.t = t
        move = np.eye(4)
        move[0, 2] = move[1, 3] = dt
        q_pos, q_cross, q_vel = dt ** 4 / 4, dt ** 3 / 2, dt ** 2
        process = np.kron(np.array([[q_pos, q_cross], [q_cross, q_vel]]), np.eye(2)) * ACCEL_NOISE ** 2
        self.state = move @ self.state
        self.cov = move @ self.cov @ move.T + process
        measured = np.array([x, y])
        residual = measured - self.state[:2]
        expected = self.cov[:2, :2] + noise
        # A "jump" is far from where the vehicle should be AND far beyond the normal
        # measurement noise there (4 standard deviations), once the filter has settled.
        surprise = float(residual @ np.linalg.solve(expected, residual))
        self.updates += 1
        self.jump_m = float(math.hypot(*residual)) if self.updates > 5 and surprise > 16.0 else 0.0
        gain = self.cov[:, :2] @ np.linalg.inv(expected)
        self.state = self.state + gain @ residual
        self.cov = (np.eye(4) - gain @ np.eye(2, 4)) @ self.cov
        speed = min(math.hypot(self.state[2], self.state[3]), MAX_SPEED)
        return float(self.state[0]), float(self.state[1]), speed


def fit_homography(points: list[dict]) -> tuple[list[list[float]], list[float]]:
    """Calibration: points [{"px": [u, v], "m": [x, y]}, ...] (at least 4).
    Returns (homography pixels -> meters, reprojection error of each point in meters)."""
    if len(points) < 4:
        raise ValueError("need at least 4 points")
    pixels = np.array([p["px"] for p in points], dtype=np.float64)
    meters = np.array([p["m"] for p in points], dtype=np.float64)
    matrix, _inliers = cv2.findHomography(pixels, meters, cv2.RANSAC, 0.5)
    if matrix is None:
        raise ValueError("these points do not give a homography (are 3 of them on one line?)")
    mapped = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    errors = [float(e) for e in np.hypot(*(mapped - meters).T)]
    return [[float(v) for v in row] for row in matrix / matrix[2, 2]], errors
