"""
kinematics.py — 6-DoF math · grasp classification · joint angles · dynamics
=============================================================================
All computation that turns raw MediaPipe landmarks into kinematic features,
now fused with IMU world-orientation data.

Key additions vs old pipeline:
  • world_hand_rpy()     — hand orientation in world frame (not camera-relative)
  • SmoothedDynamics     — Savitzky-Golay smoothed velocity + jerk
  • world_landmarks()    — landmarks transformed to world frame via IMU R
  • Everything else kept from original + improved
"""

from __future__ import annotations
import math
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter


# ── Finger / landmark topology (unchanged) ───────────────────────────────────

TIPS = [4, 8, 12, 16, 20]

LM_FINGER: dict[int, str] = {}
for _nm, _ids in [
    ("palm",   [0]),
    ("thumb",  [1, 2, 3, 4]),
    ("index",  [5, 6, 7, 8]),
    ("middle", [9, 10, 11, 12]),
    ("ring",   [13, 14, 15, 16]),
    ("pinky",  [17, 18, 19, 20]),
]:
    for _i in _ids:
        LM_FINGER[_i] = _nm

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]


# ── Palm frame + RPY ─────────────────────────────────────────────────────────

def palm_frame(lms_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build right-handed orthonormal palm frame from (21,3) world landmarks.
    Returns (origin, R3x3) where rows of R are [x_axis, y_axis, z_axis].
    Y is assumed flipped (Y-up) — apply hw() before calling.
    """
    wrist  = lms_w[0]
    middle = lms_w[9]
    index  = lms_w[5]
    pinky  = lms_w[17]

    y_ax = middle - wrist
    yn   = np.linalg.norm(y_ax)
    if yn < 1e-9:
        return wrist, np.eye(3)
    y_ax /= yn

    across = pinky - index
    x_ax   = across - np.dot(across, y_ax) * y_ax
    xn     = np.linalg.norm(x_ax)
    x_ax   = x_ax / xn if xn > 1e-9 else np.array([1., 0., 0.])

    z_ax = np.cross(x_ax, y_ax)
    return wrist, np.array([x_ax, y_ax, z_ax])


def rpy_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler decomposition → (roll, pitch, yaw) degrees."""
    Rm = R.T
    pitch = math.asin(max(-1., min(1., -Rm[2, 0])))
    cp    = math.cos(pitch)
    if abs(cp) < 1e-6:
        roll, yaw = math.atan2(-Rm[0, 1], Rm[1, 1]), 0.
    else:
        roll = math.atan2(Rm[2, 1], Rm[2, 2])
        yaw  = math.atan2(Rm[1, 0], Rm[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def world_hand_rpy(palm_R_cam: np.ndarray,
                   imu_quaternion_wxyz: np.ndarray
                   ) -> tuple[float, float, float]:
    """
    Compose camera-world orientation (from IMU) with hand-in-camera orientation
    (from MediaPipe palm frame) to get hand orientation in world frame.

    Parameters
    ----------
    palm_R_cam           : (3,3)  rotation matrix from palm_frame()
    imu_quaternion_wxyz  : (4,)   [w, x, y, z]  world quaternion from Madgwick

    Returns
    -------
    (roll, pitch, yaw) in degrees, world-frame ZYX Euler
    """
    w, x, y, z = imu_quaternion_wxyz
    R_cam_world  = Rotation.from_quat([x, y, z, w])   # scipy uses xyzw
    R_hand_cam   = Rotation.from_matrix(palm_R_cam.T)  # palm_R has rows=axes → transpose
    R_hand_world = R_cam_world * R_hand_cam
    yaw, pitch, roll = R_hand_world.as_euler('ZYX', degrees=True)
    return roll, pitch, yaw


def world_landmarks(lms_w: np.ndarray,
                    imu_quaternion_wxyz: np.ndarray) -> np.ndarray:
    """
    Rotate MediaPipe camera-frame landmarks into world frame using IMU quaternion.
    lms_w : (21,3) world landmarks from MediaPipe (Y-up flipped)
    Returns (21,3) in world frame.
    """
    w, x, y, z = imu_quaternion_wxyz
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    return (R @ lms_w.T).T


# ── Height normalisation ─────────────────────────────────────────────────────

def normalize_pose(xyz: np.ndarray,
                   op_height_cm: float,
                   robot_height_cm: float) -> np.ndarray:
    if op_height_cm <= 0 or robot_height_cm <= 0:
        return xyz.copy()
    return xyz * (robot_height_cm / op_height_cm)


# ── Joint angles ─────────────────────────────────────────────────────────────

def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b in degrees."""
    ba, bc = a - b, c - b
    na, nb = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos_t = np.dot(ba, bc) / (na * nb)
    return math.degrees(math.acos(float(np.clip(cos_t, -1., 1.))))


def finger_joint_angles(lms_w: np.ndarray) -> dict[str, float]:
    """14 finger-joint flexion angles from (21,3) world landmarks."""
    ja = joint_angle
    l  = lms_w
    return {
        "thumb_mcp":  ja(l[0],  l[1],  l[2]),
        "thumb_ip":   ja(l[1],  l[2],  l[3]),
        "idx_mcp":    ja(l[0],  l[5],  l[6]),
        "idx_pip":    ja(l[5],  l[6],  l[7]),
        "idx_dip":    ja(l[6],  l[7],  l[8]),
        "mid_mcp":    ja(l[0],  l[9],  l[10]),
        "mid_pip":    ja(l[9],  l[10], l[11]),
        "mid_dip":    ja(l[10], l[11], l[12]),
        "ring_mcp":   ja(l[0],  l[13], l[14]),
        "ring_pip":   ja(l[13], l[14], l[15]),
        "ring_dip":   ja(l[14], l[15], l[16]),
        "pinky_mcp":  ja(l[0],  l[17], l[18]),
        "pinky_pip":  ja(l[17], l[18], l[19]),
        "pinky_dip":  ja(l[18], l[19], l[20]),
    }


# ── Grasp classification ──────────────────────────────────────────────────────

def classify_grasp(lms_w: np.ndarray) -> tuple[str, float, str]:
    """
    Returns (grasp_type, aperture_m, contact_state).
    grasp_type : open | pinch | tripod | power | lateral | hook | unknown
    contact_state : open | partial | closed
    """
    wrist = lms_w[0]

    def _curl(tip_idx, mcp_idx):
        d_tip = np.linalg.norm(lms_w[tip_idx] - wrist)
        d_mcp = np.linalg.norm(lms_w[mcp_idx] - wrist)
        return (d_tip / d_mcp) if d_mcp > 1e-9 else 1.0

    c_t = _curl(4,  1)
    c_i = _curl(8,  5)
    c_m = _curl(12, 9)
    c_r = _curl(16, 13)
    c_p = _curl(20, 17)

    if   c_t > 0.8  and c_i > 0.8  and c_m > 0.8  and c_r > 0.8  and c_p > 0.8:
        grasp = "open"
    elif c_t < 0.7  and c_i < 0.7  and c_m > 0.75 and c_r > 0.75 and c_p > 0.75:
        grasp = "pinch"
    elif c_t < 0.7  and c_i < 0.7  and c_m < 0.7  and c_r > 0.75 and c_p > 0.75:
        grasp = "tripod"
    elif c_t < 0.65 and c_i < 0.65 and c_m < 0.65 and c_r < 0.65 and c_p < 0.65:
        grasp = "power"
    elif c_t > 0.8  and c_i < 0.65 and c_m < 0.65 and c_r < 0.65 and c_p < 0.65:
        grasp = "lateral" if np.linalg.norm(lms_w[4] - lms_w[5]) < 0.03 else "hook"
    else:
        grasp = "unknown"

    aperture = float(np.linalg.norm(lms_w[4] - lms_w[8]))
    contact  = "closed" if aperture < 0.02 else ("open" if aperture > 0.08 else "partial")
    return grasp, aperture, contact


# ── Smoothed dynamics (velocity + acceleration + jerk) ───────────────────────

class SmoothedDynamics:
    """
    Maintains a rolling window of end-effector positions and computes
    Savitzky-Golay smoothed velocity, acceleration, and jerk per update.

    Parameters
    ----------
    window : odd integer, history length for SG filter (must be ≥ polyorder+1)
    poly   : polynomial order for SG filter
    """

    def __init__(self, window: int = 9, poly: int = 2):
        assert window % 2 == 1 and window > poly, \
            "window must be odd and > poly"
        self.window  = window
        self.poly    = poly
        self._pos_q: deque[np.ndarray] = deque(maxlen=window)
        self._ts_q:  deque[float]      = deque(maxlen=window)

        self.velocity:     np.ndarray = np.zeros(3)
        self.speed:        float      = 0.0
        self.acceleration: np.ndarray = np.zeros(3)
        self.accel_mag:    float      = 0.0
        self.jerk:         np.ndarray = np.zeros(3)
        self.jerk_mag:     float      = 0.0

        self._prev_vel:   np.ndarray = np.zeros(3)
        self._prev_accel: np.ndarray = np.zeros(3)

    def update(self, pos: np.ndarray, ts: float):
        """Push new position + timestamp, recompute dynamics."""
        self._pos_q.append(pos.copy())
        self._ts_q.append(ts)

        n = len(self._pos_q)
        if n < 3:
            return   # not enough data yet

        pos_arr = np.array(self._pos_q)    # (n, 3)
        ts_arr  = np.array(self._ts_q)     # (n,)
        dt_mean = float(np.mean(np.diff(ts_arr)))
        if dt_mean < 1e-9:
            return

        wl = min(self.window, n)
        if wl % 2 == 0:
            wl -= 1
        if wl < self.poly + 1:
            return

        # Smooth positions, then finite-difference
        smoothed = savgol_filter(pos_arr, window_length=wl,
                                 polyorder=self.poly, axis=0)

        vel_arr   = np.diff(smoothed, axis=0) / dt_mean
        accel_arr = np.diff(vel_arr,  axis=0) / dt_mean
        jerk_arr  = np.diff(accel_arr, axis=0) / dt_mean

        self.velocity     = vel_arr[-1]
        self.speed        = float(np.linalg.norm(self.velocity))
        self.acceleration = accel_arr[-1] if len(accel_arr) else np.zeros(3)
        self.accel_mag    = float(np.linalg.norm(self.acceleration))
        self.jerk         = jerk_arr[-1] if len(jerk_arr) else np.zeros(3)
        self.jerk_mag     = float(np.linalg.norm(self.jerk))

    def as_dict(self) -> dict:
        return {
            "ee_vx": round(float(self.velocity[0]), 6),
            "ee_vy": round(float(self.velocity[1]), 6),
            "ee_vz": round(float(self.velocity[2]), 6),
            "ee_speed": round(self.speed, 6),
            "ee_ax": round(float(self.acceleration[0]), 6),
            "ee_ay": round(float(self.acceleration[1]), 6),
            "ee_az": round(float(self.acceleration[2]), 6),
            "ee_accel": round(self.accel_mag, 6),
            "ee_jx": round(float(self.jerk[0]), 6),
            "ee_jy": round(float(self.jerk[1]), 6),
            "ee_jz": round(float(self.jerk[2]), 6),
            "ee_jerk": round(self.jerk_mag, 6),
        }


# ── MediaPipe landmark helpers ────────────────────────────────────────────────

def hw(lm_list) -> np.ndarray:
    """World landmarks → (21,3) numpy, Y flipped to Y-up."""
    a = np.array([[l.x, l.y, l.z] for l in lm_list], dtype=np.float64)
    a[:, 1] *= -1
    return a


def hn(lm_list) -> list[tuple[float, float]]:
    """Normalised image landmarks → list of (x, y)."""
    return [(l.x, l.y) for l in lm_list]


def pw(lm_list) -> np.ndarray:
    """Pose world landmarks → (N,3) numpy, Y flipped."""
    a = np.array([[l.x, l.y, l.z] for l in lm_list], dtype=np.float64)
    a[:, 1] *= -1
    return a


# ── 3-D display scaling ──────────────────────────────────────────────────────

CUBE_H = 0.16   # half-side of the bounding cube used for 3-D hand display


def _cs(lms: np.ndarray, half: float = CUBE_H, fill: float = 0.60) -> np.ndarray:
    """Centre landmarks around wrist and scale to fit a display cube."""
    c = lms - lms[0]
    e = np.abs(c).max()
    return c * (half * fill / e) if e > 1e-9 else c
