"""
segmentation.py — Hierarchical temporal step segmentation
==========================================================
Two modes:
  1. IMU-guided  — finds natural motion-valley boundaries from accelerometer
                   energy; falls back to equal split if insufficient valleys
  2. Equal split — uniform duration per step (old behaviour, kept as fallback)

Usage:
    from segmentation import StepTimeline
    tl = StepTimeline(
        steps=["Attach socket", "Loosen nuts", "Remove wheel"],
        ts_start=0.0, ts_end=45.0,
        imu_stream=imu,          # optional — enables smart detection
    )
    label, idx, total = tl.at(12.5)
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


@dataclass
class StepSegment:
    label:    str
    t_start:  float
    t_end:    float
    idx:      int       # 1-based
    total:    int


class StepTimeline:
    """
    Build and query a step timeline.

    Parameters
    ----------
    steps      : list of step label strings
    ts_start   : video start time (seconds)
    ts_end     : video end time (seconds)
    imu_stream : optional IMUStream (already fused) for boundary detection
    smooth_win : smoothing window for IMU energy (samples)
    min_dist   : minimum distance between valleys as fraction of total samples
    """

    def __init__(self,
                 steps:       list[str],
                 ts_start:    float = 0.0,
                 ts_end:      float = 0.0,
                 imu_stream   = None,
                 smooth_win:  int   = 50,
                 min_dist:    float = 0.10):

        self.steps    = steps if steps else [""]
        self.ts_start = ts_start
        self.ts_end   = ts_end if ts_end > ts_start else ts_start + 1.0
        self._method  = "equal_split"

        if len(self.steps) == 0:
            self.steps = [""]

        self._segments: list[StepSegment] = []

        if imu_stream is not None and len(imu_stream) > 10 and len(self.steps) > 1:
            self._build_imu(imu_stream, smooth_win, min_dist)
        else:
            self._build_equal()

    # ── Builders ─────────────────────────────────────────────────────────────

    def _build_equal(self):
        n        = len(self.steps)
        duration = self.ts_end - self.ts_start
        seg      = duration / n
        self._segments = [
            StepSegment(
                label   = self.steps[i],
                t_start = self.ts_start + i * seg,
                t_end   = self.ts_start + (i + 1) * seg,
                idx     = i + 1,
                total   = n,
            )
            for i in range(n)
        ]
        self._method = "equal_split"

    def _build_imu(self, imu_stream, smooth_win: int, min_dist: float):
        """
        Detect step boundaries at valleys of IMU acceleration energy
        within [ts_start, ts_end].
        """
        n_steps = len(self.steps)

        # Collect samples in the video time window
        samples = [s for s in imu_stream.samples
                   if self.ts_start <= s.ts <= self.ts_end]

        if len(samples) < n_steps * 4:
            self._build_equal()
            return

        times  = np.array([s.ts for s in samples])
        def _accel_mag(s):
            l2 = s.lin_ax * s.lin_ax + s.lin_ay * s.lin_ay + s.lin_az * s.lin_az
            if l2 > 1e-12:
                return float(np.sqrt(l2))
            return float(np.sqrt(s.ax * s.ax + s.ay * s.ay + s.az * s.az))

        energy = np.array([_accel_mag(s) for s in samples])

        # Smooth the energy envelope
        win = min(smooth_win, max(3, len(energy) // 10))
        if win % 2 == 0:
            win += 1
        smoothed = uniform_filter1d(energy, size=win)

        # Minimum distance between valleys
        min_d = max(1, int(len(times) * min_dist))

        # Find valleys (negative peaks = low motion moments)
        valleys, props = find_peaks(-smoothed,
                                    distance=min_d,
                                    prominence=0.1)

        n_boundaries = n_steps - 1
        boundaries: list[float] = []

        if len(valleys) >= n_boundaries:
            # Pick deepest valleys
            depths     = -smoothed[valleys]
            sorted_idx = np.argsort(depths)[-n_boundaries:]
            chosen     = sorted(valleys[sorted_idx])
            boundaries = [float(times[v]) for v in chosen]
        else:
            # Not enough valleys — fall back to equal split
            self._build_equal()
            return

        all_ts = [self.ts_start] + boundaries + [self.ts_end]
        self._segments = [
            StepSegment(
                label   = self.steps[i],
                t_start = all_ts[i],
                t_end   = all_ts[i + 1],
                idx     = i + 1,
                total   = n_steps,
            )
            for i in range(n_steps)
        ]
        self._method = "imu_detected"

    # ── Query ─────────────────────────────────────────────────────────────────

    def at(self, tc: float) -> tuple[str, int, int]:
        """Return (label, 1-based index, total) for timestamp tc."""
        for seg in self._segments:
            if seg.t_start <= tc < seg.t_end:
                return seg.label, seg.idx, seg.total
        if self._segments:
            s = self._segments[-1]
            return s.label, s.idx, s.total
        return "", 1, 1

    def full_at(self, tc: float) -> StepSegment | None:
        for seg in self._segments:
            if seg.t_start <= tc < seg.t_end:
                return seg
        return self._segments[-1] if self._segments else None

    @property
    def method(self) -> str:
        return self._method

    @property
    def total(self) -> int:
        return len(self._segments)

    def __repr__(self) -> str:
        return (f"StepTimeline(n={self.total}, "
                f"method='{self._method}', "
                f"[{self.ts_start:.1f}–{self.ts_end:.1f}s])")
