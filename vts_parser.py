"""
vts_parser.py — Auto-detect and parse .vts binary files
=========================================================
Tries four interpretations in order:

  1. VTK XML StructuredGrid  (<VTKFile type="StructuredGrid">)
  2. Packed float64 rows: [ts, x, y, z, ...]   (pose / point-cloud log)
  3. Packed float32 rows (same layout)
  4. Raw timestamp array: [ts0, ts1, ...]        (frame sync index)

Result is always a VTSData object with:
  .timestamps    np.ndarray  (N,)   seconds
  .poses         np.ndarray  (N,7)  [ts, x, y, z, qw, qx, qy, qz]  or None
  .format        str                detected format name

Usage:
    from vts_parser import VTSParser
    vts = VTSParser("recording.vts")
    print(vts)               # shows detected format + row count
    ts = vts.timestamps      # array of timestamps
    t_frame = vts.frame_ts(30, fps=30.0)   # timestamp of frame 30
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VTSData:
    format:     str
    timestamps: np.ndarray                          # (N,) seconds
    poses:      np.ndarray | None = None            # (N,7) if available
    raw:        np.ndarray | None = None            # full raw array
    meta:       dict = field(default_factory=dict)

    def frame_ts(self, frame_idx: int, fps: float) -> float:
        """
        Estimate the IMU/VTS timestamp corresponding to video frame `frame_idx`.
        If the .vts file contains a timestamp array, use direct lookup;
        otherwise fall back to frame_idx / fps.
        """
        if len(self.timestamps) > frame_idx:
            return float(self.timestamps[frame_idx])
        return frame_idx / fps

    def __repr__(self) -> str:
        p = f", poses={self.poses.shape}" if self.poses is not None else ""
        return (f"VTSData(format='{self.format}', "
                f"n={len(self.timestamps)}{p})")


class VTSParser:
    """
    Auto-detecting parser for binary .vts files.

    Parameters
    ----------
    path : path to the .vts file (can also accept None — returns empty VTSData)
    """

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.data: VTSData = self._parse()

    # ── Public ───────────────────────────────────────────────────────────────

    @property
    def timestamps(self) -> np.ndarray:
        return self.data.timestamps

    @property
    def poses(self) -> np.ndarray | None:
        return self.data.poses

    def frame_ts(self, frame_idx: int, fps: float) -> float:
        return self.data.frame_ts(frame_idx, fps)

    def __repr__(self) -> str:
        return repr(self.data)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self) -> VTSData:
        if self.path is None or not self.path.exists():
            return VTSData(format="none", timestamps=np.array([]))

        raw_bytes = self.path.read_bytes()

        # 1. TRI binary signature (e.g. TRIVTS01)
        if raw_bytes.startswith(b"TRIVTS"):
            return self._parse_tri_vts(raw_bytes)

        # 2. VTK XML StructuredGrid
        result = self._try_vtk_xml(raw_bytes)
        if result: return result

        # 3. Flat float-rows probe
        # Try Float64 rows
        result = self._try_float_rows(raw_bytes, np.float64)
        if result: return result

        # Try Float32 rows
        result = self._try_float_rows(raw_bytes, np.float32)
        if result: return result

        # 4. Unknown binary
        return VTSData(
            format="unknown_binary",
            timestamps=np.array([]),
            meta={"size_bytes": len(raw_bytes)},
        )

    def _parse_tri_vts(self, raw: bytes) -> VTSData:
        """Parse TRIVTS01: 32-byte header, then uint64 nanoseconds."""
        header_size = 32
        data = raw[header_size:]
        n = len(data) // 8
        if n == 0:
            return VTSData(format="tri_vts_empty", timestamps=np.array([]))
            
        ts_ns = np.frombuffer(data[:n*8], dtype=np.uint64).astype(np.float64)
        ts_sec = (ts_ns - ts_ns[0]) / 1e9
        
        return VTSData(
            format="tri_vts_u64",
            timestamps=ts_sec,
            meta={"n": n}
        )

    # ── Format probes ─────────────────────────────────────────────────────────

    def _try_vtk_xml(self, raw: bytes) -> VTSData | None:
        """Attempt parse as VTK XML StructuredGrid."""
        head = raw[:200]
        if b"<VTKFile" not in head and b"<?xml" not in head:
            return None

        try:
            # Try vtk library first
            import vtk
            reader = vtk.vtkXMLStructuredGridReader()
            reader.SetFileName(str(self.path))
            reader.Update()
            grid = reader.GetOutput()
            pts  = grid.GetPoints()
            n    = pts.GetNumberOfPoints()
            coords = np.array([pts.GetPoint(i) for i in range(n)])
            # Build synthetic timestamps if none present
            ts = np.arange(n, dtype=np.float64)
            poses = np.column_stack([ts, coords, np.tile([1,0,0,0], (n,1))])
            return VTSData(format="vtk_structured_grid",
                           timestamps=ts, poses=poses)
        except ImportError:
            pass

        # Fallback: parse XML manually for DataArray elements
        try:
            import xml.etree.ElementTree as ET
            text  = raw.decode("utf-8", errors="replace")
            root  = ET.fromstring(text)
            arrays = root.findall(".//{*}DataArray")
            values = []
            for arr in arrays:
                nums = [float(x) for x in arr.text.split() if x.strip()]
                values.extend(nums)
            if values:
                arr_np = np.array(values)
                return VTSData(format="vtk_xml_fallback",
                               timestamps=arr_np,
                               raw=arr_np)
        except Exception:
            pass

        return None

    def _try_float_rows(self, raw: bytes, dtype) -> VTSData | None:
        """
        Try interpreting the binary blob as packed rows of floats.
        Tests widths: 10, 9, 8, 7, 4, 1 — picks the one that
        (a) divides evenly and (b) has a plausible first column as timestamp.
        """
        item = np.dtype(dtype).itemsize
        total_items = len(raw) // item
        if total_items == 0:
            return None

        arr_flat = np.frombuffer(raw[:total_items * item], dtype=dtype).astype(np.float64)

        for width in [10, 9, 8, 7, 4, 1]:
            if total_items % width != 0:
                continue
            n_rows = total_items // width
            if n_rows < 2:
                continue
            arr = arr_flat.reshape(n_rows, width)

            # Heuristic: first column should be monotonically increasing timestamps
            col0 = arr[:, 0]
            is_monotone  = bool(np.all(np.diff(col0) >= 0))
            looks_like_ts = (col0[0] >= 0) and (col0[-1] < 1e7)

            if not (is_monotone and looks_like_ts):
                continue

            # Normalise timestamps to start at 0
            ts = col0 - col0[0]

            fmt = f"float{dtype().itemsize*8}_w{width}"

            if width >= 7:
                # Layout: [ts, x, y, z, qw, qx, qy, qz, ...]
                poses = arr[:, :8] if width >= 8 else None
                return VTSData(format=fmt, timestamps=ts,
                               poses=poses, raw=arr)
            else:
                return VTSData(format=fmt, timestamps=ts, raw=arr)

        return None
