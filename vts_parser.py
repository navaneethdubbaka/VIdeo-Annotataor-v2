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
    t_frame = vts.frame_ts(30, fps=30.0, time_origin_ns=imu_stream.time_origin_ns)
"""

from __future__ import annotations
import struct
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

    def timestamp_ns_for_frame(self, frame_idx: int) -> int | None:
        """Absolute nanosecond timestamp for a video frame index, if known."""
        m = self.meta.get("frame_ts_ns")
        if isinstance(m, dict):
            return m.get(int(frame_idx))
        return None

    def frame_ts(
        self,
        frame_idx: int,
        fps: float,
        time_origin_ns: int | None = None,
    ) -> float:
        """
        Timestamp in seconds aligned with IMUStream when `time_origin_ns` is the
        first IMU sample's `timestamp_ns` (TRIMU001 v2). Otherwise seconds since
        first VTS entry (legacy) or `frame_idx / fps`.
        """
        fi = int(frame_idx)
        ft = self.meta.get("frame_ts_ns")
        if isinstance(ft, dict):
            ns = ft.get(fi)
            if ns is not None:
                if time_origin_ns is not None:
                    return (int(ns) - int(time_origin_ns)) * 1e-9
                anchor = int(self.meta.get("vts_time_anchor_ns", 0))
                return (int(ns) - anchor) * 1e-9

        ts_pi = self.meta.get("ts_ns_per_index")
        if isinstance(ts_pi, np.ndarray) and fi < len(ts_pi):
            ns = int(ts_pi[fi])
            if time_origin_ns is not None:
                return (ns - int(time_origin_ns)) * 1e-9
            if len(self.timestamps) > fi:
                return float(self.timestamps[fi])
            return (ns - int(ts_pi[0])) * 1e-9

        if len(self.timestamps) > fi:
            return float(self.timestamps[fi])
        return fi / fps

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

    def frame_ts(
        self,
        frame_idx: int,
        fps: float,
        time_origin_ns: int | None = None,
    ) -> float:
        return self.data.frame_ts(frame_idx, fps, time_origin_ns=time_origin_ns)

    def timestamp_ns_for_frame(self, frame_idx: int) -> int | None:
        return self.data.timestamp_ns_for_frame(frame_idx)

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
        """TRIVTS: 32-byte header, then either (uint32 frame, uint64 ts_ns) × N or uint64 ts_ns × N."""
        header_size = 32
        data = raw[header_size:]
        if len(data) == 0:
            return VTSData(format="tri_vts_empty", timestamps=np.array([]))

        n12 = len(data) // 12
        if n12 > 0 and len(data) % 12 == 0:
            parsed = self._try_tri_vts_frame_map(data, n12)
            if parsed is not None:
                return parsed

        n = len(data) // 8
        if n == 0:
            return VTSData(format="tri_vts_empty", timestamps=np.array([]))

        ts_ns = np.frombuffer(data[: n * 8], dtype=np.uint64)
        ts_sec = (ts_ns.astype(np.float64) - float(ts_ns[0])) / 1e9

        return VTSData(
            format="tri_vts_u64",
            timestamps=ts_sec,
            meta={"n": n, "ts_ns_per_index": ts_ns.copy()},
        )

    def _try_tri_vts_frame_map(self, data: bytes, n12: int) -> VTSData | None:
        """12-byte records: frame (u32), timestamp_ns (u64). Returns None if implausible."""
        frames: list[int] = []
        ts_list: list[int] = []
        for i in range(n12):
            chunk = data[i * 12 : (i + 1) * 12]
            frame_u32, ts_ns = struct.unpack("<IQ", chunk)
            frames.append(int(frame_u32))
            ts_list.append(int(ts_ns))

        if not frames:
            return None

        ts_arr = np.array(ts_list, dtype=np.int64)
        # Distinguish from uint64-only streams when len is multiple of 24:
        # frame indices stay modest; timestamps do not look like two stacked u64s.
        if np.any(np.diff(ts_arr) < 0) or np.any(ts_arr < 0):
            return None
        if max(frames) > 50_000_000 or min(frames) < 0:
            return None
        # If "frames" are huge, this was probably raw u64 data reinterpreted.
        if max(frames) > n12 + 1_000_000:
            return None

        frame_ts_ns: dict[int, int] = {}
        for f, t in zip(frames, ts_list):
            frame_ts_ns[f] = t

        min_ts = int(np.min(ts_arr))
        max_f = max(frames)
        ts_by_index = np.zeros(max_f + 1, dtype=np.float64)
        for f, t in frame_ts_ns.items():
            if 0 <= f <= max_f:
                ts_by_index[f] = (t - min_ts) * 1e-9

        return VTSData(
            format="tri_vts_frame_u32_ts_u64",
            timestamps=ts_by_index,
            meta={
                "n": n12,
                "frame_ts_ns": frame_ts_ns,
                "vts_time_anchor_ns": min_ts,
            },
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
