"""
camera.py — Fisheye undistortion (Kannala-Brandt / OpenCV fisheye model)
=========================================================================
Calibration constants match the provided camera spec:
  fx/fy  : 1030.59 / 1032.82
  cx/cy  : 966.69  / 539.69
  k1..k4 : -0.1166 / -0.0236 / +0.0694 / -0.0463
  image  : 1920 x 1080

Usage:
    from camera import FisheyeUndistorter
    und = FisheyeUndistorter()
    undistorted_frame = und.undistort(raw_frame)
    new_K = und.new_K          # updated intrinsic matrix for downstream use
"""

import cv2
import numpy as np


# ── Default calibration (from camera spec sheet) ─────────────────────────────
DEFAULT_K = np.array([[1030.59,    0.0,  966.69],
                      [   0.0,  1032.82, 539.69],
                      [   0.0,    0.0,    1.0  ]], dtype=np.float64)

DEFAULT_D = np.array([-0.1166, -0.0236, 0.0694, -0.0463], dtype=np.float64)

DEFAULT_SIZE = (1920, 1080)


class FisheyeUndistorter:
    """
    Pre-computes the remap tables once; applies them per frame cheaply.

    Parameters
    ----------
    K        : (3,3) camera matrix  (None → use DEFAULT_K)
    D        : (4,)  distortion coefficients k1..k4  (None → use DEFAULT_D)
    img_size : (W, H) original image size
    balance  : 0.0 = crop to valid pixels only
               1.0 = keep all pixels (large black borders)
               0.5 = balanced (recommended default)
    """

    def __init__(self,
                 K:        np.ndarray | None = None,
                 D:        np.ndarray | None = None,
                 img_size: tuple[int,int]    = DEFAULT_SIZE,
                 balance:  float             = 0.5):

        self.K        = K if K is not None else DEFAULT_K.copy()
        self.D        = D if D is not None else DEFAULT_D.copy()
        self.img_size = img_size   # (W, H)

        # Estimate the new (undistorted) camera matrix
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K, self.D,
            img_size,
            np.eye(3),
            balance=balance,
        )

        # Build remap tables  — expensive, done once
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D,
            np.eye(3),
            self.new_K,
            img_size,
            cv2.CV_16SC2,
        )

        self.fx = float(self.new_K[0, 0])
        self.fy = float(self.new_K[1, 1])
        self.cx = float(self.new_K[0, 2])
        self.cy = float(self.new_K[1, 2])

    # ── Public API ────────────────────────────────────────────────────────────

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Apply the pre-computed remap.  Returns same size as input."""
        return cv2.remap(frame, self.map1, self.map2,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """
        Project 3-D points (N,3) into the undistorted image plane.
        Returns (N,2) pixel coordinates.
        """
        xyz = np.atleast_2d(xyz).astype(np.float64)
        u = xyz[:, 0] / xyz[:, 2] * self.fx + self.cx
        v = xyz[:, 1] / xyz[:, 2] * self.fy + self.cy
        return np.column_stack([u, v])

    def unproject_ray(self, uv: np.ndarray) -> np.ndarray:
        """
        Back-project pixel(s) (N,2) to unit direction vectors (N,3).
        Useful for ray-casting into world space.
        """
        uv = np.atleast_2d(uv).astype(np.float64)
        x  = (uv[:, 0] - self.cx) / self.fx
        y  = (uv[:, 1] - self.cy) / self.fy
        z  = np.ones(len(uv))
        dirs = np.column_stack([x, y, z])
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        return dirs / norms
