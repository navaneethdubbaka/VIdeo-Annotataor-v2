"""
imu.py — IMU file parsing · Madgwick AHRS fusion · per-frame interpolation
===========================================================================
Handles:
  • Reading .imu files (CSV, binary float32/float64, or auto-detect)
  • 9-DoF sensor fusion via Madgwick filter (accel + gyro + mag)
  • Interpolating IMU state to any video frame timestamp

Expected .imu CSV columns (any order, header required):
    ts, ax, ay, az, gx, gy, gz, mx, my, mz
    ts  = seconds (float)
    a*  = m/s²   (or raw g — auto-scaled if |mean| > 5)
    g*  = rad/s  (or deg/s — auto-scaled if |max| > 50)
    m*  = µT     (or raw LSB — kept as-is for fusion)

Binary format: packed float64 rows of 10 values  [ts,ax,ay,az,gx,gy,gz,mx,my,mz]

Usage:
    from imu import IMUStream
    imu = IMUStream("recording.imu")
    imu.fuse()                            # run Madgwick over all samples
    state = imu.at(t_sec)                 # interpolated state at time t
    quat  = state.quaternion              # [w, x, y, z]  world orientation
    R     = state.rotation_matrix()       # (3,3)
"""

from __future__ import annotations
import struct, warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# ── Optional fast Madgwick via ahrs library ───────────────────────────────────
try:
    from ahrs.filters import Madgwick as _Madgwick
    _AHRS_OK = True
except ImportError:
    _AHRS_OK = False
    warnings.warn(
        "ahrs library not found — falling back to built-in Madgwick. "
        "Install with: pip install ahrs",
        ImportWarning, stacklevel=2,
    )


# ── Built-in Madgwick (no external dependency) ────────────────────────────────

class _MadgwickBuiltin:
    """Minimal Madgwick AHRS — MARG update (accel + gyro + mag)."""

    def __init__(self, frequency: float = 100.0, beta: float = 0.1):
        self.beta = beta
        self.dt   = 1.0 / frequency
        self.q    = np.array([1., 0., 0., 0.])   # [w, x, y, z]

    def update(self, gyr, acc, mag) -> np.ndarray:
        q  = self.q
        dt = self.dt

        # Normalise accelerometer
        an = np.linalg.norm(acc)
        if an < 1e-9:
            return q
        acc = acc / an

        # Normalise magnetometer
        mn = np.linalg.norm(mag)
        if mn < 1e-9:
            # Fall back to IMU-only update (no mag)
            return self._update_imu(gyr, acc)
        mag = mag / mn

        # Reference direction of Earth's magnetic field
        h  = self._quat_rotate(q, mag)
        b  = np.array([0., np.linalg.norm(h[0:2]), 0., h[2]])

        # Gradient descent step
        f  = np.array([
            2*(q[1]*q[3] - q[0]*q[2])         - acc[0],
            2*(q[0]*q[1] + q[2]*q[3])         - acc[1],
            2*(0.5 - q[1]**2 - q[2]**2)       - acc[2],
            2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - mag[0],
            2*b[1]*(q[1]*q[2] - q[0]*q[3])   + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - mag[1],
            2*b[1]*(q[0]*q[2] + q[1]*q[3])   + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - mag[2],
        ])

        J = np.array([
            [-2*q[2],  2*q[3], -2*q[0],  2*q[1]],
            [ 2*q[1],  2*q[0],  2*q[3],  2*q[2]],
            [      0, -4*q[1], -4*q[2],       0],
            [-2*b[3]*q[2],  2*b[3]*q[3], -4*b[1]*q[2]-2*b[3]*q[0],  -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1],  2*b[1]*q[2]+2*b[3]*q[0],  2*b[1]*q[1]+2*b[3]*q[3],  -2*b[1]*q[0]+2*b[3]*q[2]],
            [ 2*b[1]*q[2],  2*b[1]*q[3]-4*b[3]*q[1],  2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]],
        ])

        grad = J.T @ f
        gn   = np.linalg.norm(grad)
        if gn > 1e-9:
            grad /= gn

        # Integrate
        q_dot = 0.5 * self._quat_mul(q, np.array([0., gyr[0], gyr[1], gyr[2]])) - self.beta * grad
        q     = q + q_dot * dt
        q    /= np.linalg.norm(q)
        self.q = q
        return q

    def _update_imu(self, gyr, acc) -> np.ndarray:
        q  = self.q
        dt = self.dt
        f  = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - acc[0],
            2*(q[0]*q[1] + q[2]*q[3]) - acc[1],
            2*(0.5 - q[1]**2 - q[2]**2) - acc[2],
        ])
        J  = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [ 2*q[1], 2*q[0],  2*q[3], 2*q[2]],
            [      0,-4*q[1], -4*q[2],      0],
        ])
        grad = J.T @ f
        gn   = np.linalg.norm(grad)
        if gn > 1e-9: grad /= gn
        q_dot = 0.5 * self._quat_mul(q, np.array([0.,gyr[0],gyr[1],gyr[2]])) - self.beta * grad
        q = q + q_dot * dt
        q /= np.linalg.norm(q)
        self.q = q
        return q

    @staticmethod
    def _quat_mul(p, r):
        return np.array([
            p[0]*r[0] - p[1]*r[1] - p[2]*r[2] - p[3]*r[3],
            p[0]*r[1] + p[1]*r[0] + p[2]*r[3] - p[3]*r[2],
            p[0]*r[2] - p[1]*r[3] + p[2]*r[0] + p[3]*r[1],
            p[0]*r[3] + p[1]*r[2] - p[2]*r[1] + p[3]*r[0],
        ])

    @staticmethod
    def _quat_rotate(q, v):
        qv = np.array([0., v[0], v[1], v[2]])
        qc = np.array([q[0], -q[1], -q[2], -q[3]])
        return _MadgwickBuiltin._quat_mul(
            _MadgwickBuiltin._quat_mul(q, qv), qc)[1:]


# ── IMU sample dataclass ──────────────────────────────────────────────────────

@dataclass
class IMUSample:
    ts:  float
    ax:  float; ay: float; az: float   # m/s² (includes gravity unless from device lin only)
    gx:  float; gy: float; gz: float   # rad/s
    mx:  float; my: float; mz: float   # µT
    # TRIMU001 v2: gravity-removed body-frame linear acceleration (m/s²); else 0
    lin_ax: float = 0.0
    lin_ay: float = 0.0
    lin_az: float = 0.0
    # filled after fusion:
    quaternion:  np.ndarray = field(default_factory=lambda: np.array([1.,0.,0.,0.]))
    accel_world: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def rotation_matrix(self) -> np.ndarray:
        """(3,3) rotation matrix — camera-to-world."""
        w,x,y,z = self.quaternion
        return Rotation.from_quat([x, y, z, w]).as_matrix()

    def euler_deg(self) -> tuple[float,float,float]:
        """(roll, pitch, yaw) in degrees, ZYX convention."""
        w,x,y,z = self.quaternion
        r = Rotation.from_quat([x, y, z, w])
        yaw, pitch, roll = r.as_euler('ZYX', degrees=True)
        return roll, pitch, yaw


# ── Main IMU stream class ─────────────────────────────────────────────────────

GRAVITY = 9.80665   # m/s²

class IMUStream:
    """
    Load, parse, fuse, and query an .imu file.

    Parameters
    ----------
    path        : path to the .imu file
    beta        : Madgwick beta (higher = faster convergence, noisier)
    mag_enabled : use magnetometer for yaw correction (recommended True)
    """

    def __init__(self,
                 path:        str | Path,
                 beta:        float = 0.033,
                 mag_enabled: bool  = True):
        self.path        = Path(path)
        self.beta        = beta
        self.mag_enabled = mag_enabled
        self.samples:  list[IMUSample] = []
        self._fused    = False
        self._ts_arr:  np.ndarray | None = None
        self._q_arr:   np.ndarray | None = None
        # Nanosecond time of first sample (TRIMU001 v2); 0 if unknown — use with VTS sync
        self.time_origin_ns: int = 0
        # From v2 header only (optional)
        self.video_start_ns: int | None = None

        self._load()

    # ── Loading ──────────────────────────────────────────────────────────────

    def _load(self):
        raw = self.path.read_bytes()
        # 1. Check for TRI binary signatures
        if raw.startswith(b"TRIMU001"):
            self.samples = self._parse_trimu001(raw)
            return

        # 2. Try CSV (text)
        try:
            text = raw.decode("utf-8", errors="strict")
            self.samples = self._parse_csv(text)
            return
        except (UnicodeDecodeError, ValueError):
            pass
        # 3. Fallback: try raw binary float64/float32 rows of 10
        self.samples = self._parse_binary(raw)

    def _parse_trimu001(self, raw: bytes) -> list[IMUSample]:
        """Dispatch TRIMU001 v2 (64-byte header, 76-byte records) vs legacy (52 + 80)."""
        if len(raw) >= 64:
            version = struct.unpack_from("<I", raw, 8)[0]
            v2_ok = (len(raw) - 64) % 76 == 0 and len(raw) > 64
            legacy_ok = (len(raw) - 52) % 80 == 0 and len(raw) > 52
            if version == 2 and v2_ok:
                return self._parse_tri_imu_v2(raw)
            if v2_ok and not legacy_ok:
                return self._parse_tri_imu_v2(raw)
            if legacy_ok and version != 2:
                return self._parse_tri_imu_legacy(raw)
            if legacy_ok and not v2_ok:
                return self._parse_tri_imu_legacy(raw)
            if v2_ok:
                return self._parse_tri_imu_v2(raw)
        if len(raw) > 52 and (len(raw) - 52) % 80 == 0:
            return self._parse_tri_imu_legacy(raw)
        if len(raw) >= 64 and (len(raw) - 64) % 76 == 0:
            return self._parse_tri_imu_v2(raw)
        raise ValueError(
            f"TRIMU001 file size {len(raw)} does not match v2 (64+76n) or legacy (52+80n)"
        )

    def _parse_tri_imu_v2(self, raw: bytes) -> list[IMUSample]:
        """
        TRIMU001 format version 2: 64-byte header, 76-byte little-endian samples.
        Quaternion on disk is XYZW (SciPy); IMUSample stores WXYZ.
        """
        header = raw[:64]
        (
            magic,
            version,
            sample_rate_hz,
            _accel_fs,
            _gyro_fs,
            start_time_ns,
            video_start_ns,
        ) = struct.unpack("<8s I I 2B 6x Q Q 24x", header)
        if not magic.startswith(b"TRIMU001"):
            raise ValueError("Invalid TRIMU001 v2 magic")
        self.video_start_ns = int(video_start_ns)
        _ = version, sample_rate_hz, start_time_ns  # available for diagnostics

        record_size = 76
        data = raw[64:]
        n = len(data) // record_size
        if n == 0:
            self.time_origin_ns = 0
            return []

        samples: list[IMUSample] = []
        anchor_ns: int | None = None
        for i in range(n):
            chunk = data[i * record_size : (i + 1) * record_size]
            unpacked = struct.unpack("<Q 3f 3f 3f f 4f 3f", chunk)
            ts_ns = int(unpacked[0])
            if anchor_ns is None:
                anchor_ns = ts_ns
            accel = unpacked[1:4]
            gyro = unpacked[4:7]
            mag = unpacked[7:10]
            qx, qy, qz, qw = unpacked[11:15]
            lin = unpacked[15:18]
            q_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
            nq = float(np.linalg.norm(q_xyzw))
            if 0.9 < nq < 1.1:
                q_xyzw = q_xyzw / nq
                qw_n, qx_n, qy_n, qz_n = float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])
                quat_wxyz = np.array([qw_n, qx_n, qy_n, qz_n])
            else:
                quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

            samples.append(
                IMUSample(
                    ts=(ts_ns - anchor_ns) * 1e-9,
                    ax=float(accel[0]),
                    ay=float(accel[1]),
                    az=float(accel[2]),
                    gx=float(gyro[0]),
                    gy=float(gyro[1]),
                    gz=float(gyro[2]),
                    mx=float(mag[0]),
                    my=float(mag[1]),
                    mz=float(mag[2]),
                    lin_ax=float(lin[0]),
                    lin_ay=float(lin[1]),
                    lin_az=float(lin[2]),
                    quaternion=quat_wxyz,
                )
            )
        self.time_origin_ns = int(anchor_ns) if anchor_ns is not None else 0
        return samples

    def _parse_tri_imu_legacy(self, raw: bytes) -> list[IMUSample]:
        """Parse TRIMU001 legacy: 52-byte header, 80-byte (20-float) records."""
        header_size = 52
        record_size = 80
        data = raw[header_size:]
        n = len(data) // record_size
        if n == 0: return []
        
        # We read as both uint32 (for clock) and float32 (for data)
        arr_f = np.frombuffer(data[:n*record_size], dtype=np.float32).reshape(n, 20)
        arr_u = np.frombuffer(data[:n*record_size], dtype=np.uint32).reshape(n, 20)
        
        # Col 0: Clock (e.g. 10 MHz uint32)
        # Col 5,6,7: acc; 8,9,10: gyro; 11,12,13: mag; 14,15,16,17: qx,qy,qz,qw
        samples = []
        t0 = None
        for i in range(n):
            # Convert clock to seconds (assume 10MHz if not sure)
            clock = float(arr_u[i, 0])
            if t0 is None and clock > 0: t0 = clock
            ts = (clock - t0) / 1e7 if t0 is not None else 0.0
            
            s = IMUSample(
                ts=ts,
                ax=float(arr_f[i, 5]), ay=float(arr_f[i, 6]), az=float(arr_f[i, 7]),
                gx=float(arr_f[i, 8]), gy=float(arr_f[i, 9]), gz=float(arr_f[i, 10]),
                mx=float(arr_f[i, 11]), my=float(arr_f[i, 12]), mz=float(arr_f[i, 13]),
            )
            
            # Pre-fused orientation (indices 14-17 are qx, qy, qz, qw)
            q_raw = arr_f[i, 14:18]
            norm_q = np.linalg.norm(q_raw)
            if 0.9 < norm_q < 1.1:
                qx, qy, qz, qw = q_raw / norm_q
                s.quaternion = np.array([float(qw), float(qx), float(qy), float(qz)])
                
            samples.append(s)
            
        return samples

    def _parse_csv(self, text: str) -> list[IMUSample]:
        import csv, io
        reader = csv.DictReader(io.StringIO(text))
        # Accept flexible column names
        col_map = {}
        for col in reader.fieldnames or []:
            lc = col.strip().lower()
            for key in ["ts","ax","ay","az","gx","gy","gz","mx","my","mz"]:
                if lc == key or lc.startswith(key):
                    col_map[key] = col
                    break

        required = {"ts","ax","ay","az","gx","gy","gz"}
        if not required.issubset(col_map):
            raise ValueError(f"CSV missing columns. Found: {reader.fieldnames}")

        rows = []
        for r in reader:
            def g(k, default=0.0):
                return float(r[col_map[k]]) if k in col_map else default
            rows.append([g("ts"),g("ax"),g("ay"),g("az"),
                         g("gx"),g("gy"),g("gz"),
                         g("mx"),g("my"),g("mz")])

        return self._scale_and_build(rows)

    def _parse_binary(self, raw: bytes) -> list[IMUSample]:
        """Try packed float64 rows of 10, then float32, with monotonicity check."""
        # Heuristic to detect best format
        best_arr = None
        
        for dtype in [np.float64, np.float32]:
            item_size = 10 * np.dtype(dtype).itemsize
            n = len(raw) // item_size
            if n < 2: continue
            
            arr = np.frombuffer(raw[:n * item_size], dtype=dtype).reshape(n, 10).astype(np.float64)
            # Check if timestamps (col 0) are monotone and plausible
            ts = arr[:, 0]
            if np.all(np.diff(ts[ts > 0]) >= 0) and np.any(ts > 0):
                best_arr = arr
                break
        
        if best_arr is None:
            # Last resort: just take float32 if length permits
            if len(raw) >= 40:
                n = len(raw) // 40
                best_arr = np.frombuffer(raw[:n * 40], dtype=np.float32).reshape(n, 10).astype(np.float64)
            else:
                raise ValueError("Cannot parse binary .imu: unexpected size or no valid data found")
                
        return self._scale_and_build(best_arr.tolist())

    def _scale_and_build(self, rows: list) -> list[IMUSample]:
        arr  = np.array(rows, dtype=np.float64)  # (N,10)
        
        # Remove zero rows (padding/init records)
        if len(arr) > 0:
            # Keep rows where at least one sensor value (index 1..9) is non-zero
            # or the timestamp is notably non-zero.
            mask = np.any(arr[:, 1:] != 0, axis=1) | (arr[:, 0] > 0)
            arr = arr[mask]

        if len(arr) == 0:
            return []

        ts   = arr[:, 0]
        accs = arr[:, 1:4]
        gyrs = arr[:, 4:7]
        mags = arr[:, 7:10]

        # Auto-detect unit: if median |accel| >> 1 g → likely in raw LSB or g
        median_a = float(np.median(np.linalg.norm(accs, axis=1)))
        if median_a > 20.0:     # raw LSB (4096 LSB/g) → m/s²
            accs = accs / 4096.0 * GRAVITY
        elif median_a > 2.0:    # in g → m/s²
            accs = accs * GRAVITY

        # Auto-detect gyro: if max > 50 → likely deg/s
        max_g = float(np.max(np.abs(gyrs)))
        if max_g > 50.0:
            gyrs = np.deg2rad(gyrs)

        # Ensure timestamps start near 0 (some firmwares use epoch time)
        if ts[0] > 1e9:
            ts = ts - ts[0]

        samples = []
        for i in range(len(ts)):
            s = IMUSample(
                ts=float(ts[i]),
                ax=accs[i,0], ay=accs[i,1], az=accs[i,2],
                gx=gyrs[i,0], gy=gyrs[i,1], gz=gyrs[i,2],
                mx=mags[i,0], my=mags[i,1], mz=mags[i,2],
            )
            samples.append(s)
        return samples

    # ── Fusion ───────────────────────────────────────────────────────────────

    def fuse(self):
        """
        Run Madgwick AHRS over all samples, storing quaternion + world accel
        on each IMUSample in-place.
        """
        if not self.samples:
            raise RuntimeError("No IMU samples loaded.")

        # Estimate frequency from timestamps
        ts = np.array([s.ts for s in self.samples])
        dts = np.diff(ts)
        median_dt = float(np.median(dts[dts > 0]))
        freq = 1.0 / median_dt if median_dt > 0 else 100.0

        if _AHRS_OK:
            filt = _Madgwick(frequency=freq, beta=self.beta)
            q = np.array([1., 0., 0., 0.])
        else:
            filt = _MadgwickBuiltin(frequency=freq, beta=self.beta)

        # Gravity vector in world frame (pointing down = -Z)
        g_world = np.array([0., 0., -GRAVITY])

        for s in self.samples:
            acc = np.array([s.ax, s.ay, s.az])
            gyr = np.array([s.gx, s.gy, s.gz])
            mag = np.array([s.mx, s.my, s.mz])

            # If we already have a pre-fused orientation from the stream, use it.
            if s.quaternion is not None and np.linalg.norm(s.quaternion) > 0.9:
                # Synchronize Madgwick filter state to pre-fused orientation
                q = s.quaternion.copy()
            else:
                if _AHRS_OK:
                    if self.mag_enabled and np.linalg.norm(mag) > 1e-3:
                        q = filt.updateMARG(q, gyr=gyr, acc=acc, mag=mag)
                    else:
                        q = filt.updateIMU(q, gyr=gyr, acc=acc)
                    s.quaternion = q.copy()
                else:
                    if self.mag_enabled and np.linalg.norm(mag) > 1e-3:
                        s.quaternion = filt.update(gyr, acc, mag).copy()
                    else:
                        s.quaternion = filt._update_imu(gyr, acc).copy()

            # World-frame acceleration: device linear (gravity already removed) vs derived
            R = s.rotation_matrix()
            lin = np.array([s.lin_ax, s.lin_ay, s.lin_az])
            if float(np.linalg.norm(lin)) > 1e-9:
                s.accel_world = R @ lin
            else:
                acc_world = R @ acc
                s.accel_world = acc_world - g_world

        self._fused = True

        # Build interpolation arrays
        self._ts_arr = np.array([s.ts for s in self.samples])
        self._q_arr  = np.array([s.quaternion for s in self.samples])

    # ── Query ────────────────────────────────────────────────────────────────

    def at(self, t_sec: float) -> IMUSample:
        """
        Return an IMUSample interpolated to timestamp t_sec.
        Quaternion is slerp'd; scalars are linearly interpolated.
        """
        if not self._fused:
            raise RuntimeError("Call .fuse() before querying.")

        ts = self._ts_arr
        t  = float(t_sec)

        if t <= ts[0]:
            return self.samples[0]
        if t >= ts[-1]:
            return self.samples[-1]

        idx = int(np.searchsorted(ts, t)) - 1
        idx = max(0, min(idx, len(self.samples) - 2))

        s0, s1 = self.samples[idx], self.samples[idx + 1]
        dt_seg = ts[idx + 1] - ts[idx]
        alpha  = (t - ts[idx]) / dt_seg if dt_seg > 1e-12 else 0.0

        def lerp(a, b): return a + (b - a) * alpha

        # Slerp quaternion
        r0 = Rotation.from_quat([s0.quaternion[1], s0.quaternion[2],
                                  s0.quaternion[3], s0.quaternion[0]])
        r1 = Rotation.from_quat([s1.quaternion[1], s1.quaternion[2],
                                  s1.quaternion[3], s1.quaternion[0]])
        slerp_fn = Slerp([0, 1], Rotation.concatenate([r0, r1]))
        r_interp = slerp_fn(alpha)
        xyzw     = r_interp.as_quat()
        q_interp = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])  # wxyz

        out = IMUSample(
            ts = t,
            ax = lerp(s0.ax, s1.ax), ay = lerp(s0.ay, s1.ay), az = lerp(s0.az, s1.az),
            gx = lerp(s0.gx, s1.gx), gy = lerp(s0.gy, s1.gy), gz = lerp(s0.gz, s1.gz),
            mx = lerp(s0.mx, s1.mx), my = lerp(s0.my, s1.my), mz = lerp(s0.mz, s1.mz),
            lin_ax = lerp(s0.lin_ax, s1.lin_ax),
            lin_ay = lerp(s0.lin_ay, s1.lin_ay),
            lin_az = lerp(s0.lin_az, s1.lin_az),
            quaternion  = q_interp,
            accel_world = lerp(s0.accel_world, s1.accel_world),
        )
        return out

    # ── Diagnostics ──────────────────────────────────────────────────────────

    @property
    def duration(self) -> float:
        if not self.samples: return 0.0
        return self.samples[-1].ts - self.samples[0].ts

    @property
    def hz(self) -> float:
        if len(self.samples) < 2: return 0.0
        ts = np.array([s.ts for s in self.samples])
        return float(1.0 / np.median(np.diff(ts)))

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        status = "fused" if self._fused else "raw"
        return (f"IMUStream({self.path.name}, "
                f"n={len(self)}, hz={self.hz:.1f}, "
                f"dur={self.duration:.1f}s, {status})")
