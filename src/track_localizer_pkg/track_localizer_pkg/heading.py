"""Track-frame vehicle tracking and heading estimation.

The BEV detector returns a centroid only — it has no orientation. Heading has to
be inferred from motion, and the model choice decides whether it meets the 3 deg
gate or not.

Why CTRV and not constant-velocity
----------------------------------
A constant-velocity filter estimates a velocity *vector* and heading falls out
as ``atan2(vy, vx)``. Measured on synthetic tracks with 6 cm centroid noise that
gives 2.8 deg RMS on a straight — already at the gate — and it comes apart in a
corner, because the velocity direction lags the true heading by roughly the
filter's time constant times the turn rate.

CTRV puts heading and turn rate *in the state*::

    x = [px, py, v, psi, psi_dot]

so a corner is a state the filter tracks rather than a disturbance it fights.
The motion model is nonlinear, so this is an unscented filter; at dt = 0.1 s the
sigma-point spread is small and the update is cheap.

Measured accuracy, and the limit callers must respect
-----------------------------------------------------
Against synthetic tracks at 10 Hz with 6 cm centroid noise (see the version note
in docs/ver for the sweep):

===========================  ==================
condition                    heading RMS
===========================  ==================
straight 1.5 m/s             2.8 deg
straight 0.8 m/s             3.8 deg
steady corner, r = 4.3 m     3.2 deg
corner *entry* transient     17 deg peak, ~1.3 s to recover
===========================  ==================

So the 3 deg target holds on a straight and is missed while cornering, and no
amount of retuning fixes it: tightening the process noise improves the straight
and makes the transient strictly worse (at accel 0.05 the peak grows to 26 deg
and takes 2.9 s to settle). Heading inferred from position alone simply cannot
track a step in turn rate. Closing that gap needs an actual orientation
*measurement* — rectangle/L-shape fitting fused as a second update — which is
deliberately not in this version.

Practical consequence: use this pose for **position-based** decisions (zone
arrival, geofence, stop lines), which is what lane-following navigation needs.
Do not close a steering loop on the heading through corners.

Heading is observable only through motion. Standing still, position measurements
carry no orientation information at all, and no filter recovers it. Below
``min_speed_mps`` the estimate is held and ``heading_valid`` goes False. That is
exactly the commanded-stop case, so a consumer that ignores the flag will steer
on a stale heading every time the car stops.

Known bias, not corrected here: the centroid of a single-sided LiDAR return sits
on the face turned toward the sensor, so it walks by up to half the vehicle width
as the car rotates. Measure it with validate_track_pose.py before claiming 5 cm.
"""

import math

import numpy as np

EPS_YAW_RATE = 1e-4


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def _ctrv_step(state, dt):
    """CTRV motion model. state = [px, py, v, psi, psi_dot]."""
    px, py, v, psi, omega = state
    if abs(omega) > EPS_YAW_RATE:
        px = px + (v / omega) * (math.sin(psi + omega * dt) - math.sin(psi))
        py = py + (v / omega) * (-math.cos(psi + omega * dt) + math.cos(psi))
    else:
        px = px + v * math.cos(psi) * dt
        py = py + v * math.sin(psi) * dt
    return np.array([px, py, v, wrap_angle(psi + omega * dt), omega])


class CTRVTracker:
    """Unscented Kalman filter over [px, py, v, psi, psi_dot].

    Measurements are position only. ``accel_std`` and ``yaw_accel_std`` are the
    process noise: how hard the car can change speed and turn rate between
    frames. Raise them if the filter lags in corners, lower them if the pose is
    jittery on a straight.
    """

    N = 5

    def __init__(
        self,
        accel_std=0.3,
        yaw_accel_std=0.5,
        measurement_std=0.06,
        min_speed_mps=0.35,
        max_coast_seconds=0.5,
        init_travel_m=0.30,
        beta=2.0,
    ):
        self.accel_std = float(accel_std)
        self.yaw_accel_std = float(yaw_accel_std)
        self.r_std = float(measurement_std)
        self.min_speed = float(min_speed_mps)
        self.max_coast = float(max_coast_seconds)
        self.init_travel = float(init_travel_m)

        # Unscented transform weights, alpha = 1 / kappa = 0 so lambda = 0 and
        # every weight is non-negative. The scaled form with a small alpha gives
        # w_c[0] a large negative value, which drives the covariance non-PSD and
        # the Cholesky then fails mid-run.
        n = self.N
        self.lam = 0.0
        self.wm = np.full(2 * n + 1, 1.0 / (2.0 * n))
        self.wc = self.wm.copy()
        self.wm[0] = 0.0
        self.wc[0] = beta

        self.x = None
        self.P = None
        self.last_stamp = None
        self.heading = None
        self.heading_valid = False
        self._init_buf = []

    # ------------------------------------------------------------------

    def reset(self):
        self.x = None
        self.P = None
        self.last_stamp = None
        self.heading = None
        self.heading_valid = False
        self._init_buf = []

    def _seed(self, stamp, forward, lateral):
        """Buffer measurements until the car has moved far enough to fix heading.

        Initialising with a huge heading variance does not work: sigma points
        would be spread several radians in psi, wrap around, and the covariance
        stops meaning anything. Instead wait for ``init_travel_m`` of travel and
        seed psi from the displacement direction, which is a real observation.
        """
        self._init_buf.append((stamp, float(forward), float(lateral)))
        if len(self._init_buf) < 2:
            return False
        # Drop anything older than the coast window so a car that sat still for a
        # while does not seed heading from a stale, far-away first sample.
        self._init_buf = [b for b in self._init_buf if stamp - b[0] <= max(self.max_coast * 4.0, 1.0)]
        if len(self._init_buf) < 2:
            return False

        t0, x0, y0 = self._init_buf[0]
        t1, x1, y1 = self._init_buf[-1]
        dx, dy = x1 - x0, y1 - y0
        travel = math.hypot(dx, dy)
        dt = t1 - t0
        if travel < self.init_travel or dt <= 0.0:
            return False

        psi = math.atan2(dy, dx)
        v = travel / dt
        # Heading uncertainty from the displacement: measurement noise over the
        # baseline length, floored so a short baseline is not over-trusted.
        psi_std = max(math.atan2(self.r_std * math.sqrt(2.0), travel), math.radians(5.0))
        self.x = np.array([x1, y1, v, psi, 0.0])
        self.P = np.diag([
            self.r_std ** 2,
            self.r_std ** 2,
            max(v * 0.5, 0.3) ** 2,
            psi_std ** 2,
            1.0,
        ])
        self.last_stamp = stamp
        self.heading = psi
        self.heading_valid = v >= self.min_speed
        self._init_buf = []
        return True

    def _sigma_points(self):
        n = self.N
        P = self.P.copy()
        # Keep the covariance symmetric positive-definite; a stale asymmetry from
        # accumulated round-off makes the Cholesky fail mid-run.
        P = 0.5 * (P + P.T) + np.eye(n) * 1e-9
        try:
            L = np.linalg.cholesky((n + self.lam) * P)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky((n + self.lam) * (P + np.eye(n) * 1e-6))
        pts = np.zeros((2 * n + 1, n))
        pts[0] = self.x
        for i in range(n):
            pts[i + 1] = self.x + L[:, i]
            pts[n + i + 1] = self.x - L[:, i]
        for p in pts:
            p[3] = wrap_angle(p[3])
        return pts

    def _mean(self, pts):
        m = self.wm @ pts
        # Angles cannot be averaged linearly — a set straddling +-pi would mean
        # near zero. Average the unit vectors instead.
        s = self.wm @ np.sin(pts[:, 3])
        c = self.wm @ np.cos(pts[:, 3])
        m[3] = math.atan2(s, c)
        return m

    def _process_noise(self, dt, psi):
        G = np.array([
            [0.5 * dt * dt * math.cos(psi), 0.0],
            [0.5 * dt * dt * math.sin(psi), 0.0],
            [dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [0.0, dt],
        ])
        return G @ np.diag([self.accel_std ** 2, self.yaw_accel_std ** 2]) @ G.T

    def _predict(self, dt):
        pts = self._sigma_points()
        prop = np.array([_ctrv_step(p, dt) for p in pts])
        x = self._mean(prop)
        P = np.zeros((self.N, self.N))
        for i, p in enumerate(prop):
            d = p - x
            d[3] = wrap_angle(d[3])
            P += self.wc[i] * np.outer(d, d)
        self.x = x
        self.P = P + self._process_noise(dt, x[3])
        return prop

    # ------------------------------------------------------------------

    def predict_only(self, stamp):
        """Advance without a measurement. False once the gap is too long."""
        if self.x is None or self.last_stamp is None:
            return False
        dt = stamp - self.last_stamp
        if dt <= 0.0 or dt > self.max_coast:
            return False
        self._predict(dt)
        self.last_stamp = stamp
        self.heading_valid = False
        return True

    def update(self, stamp, forward, lateral):
        z = np.array([float(forward), float(lateral)])

        if self.x is None or self.last_stamp is None:
            self._seed(stamp, z[0], z[1])
            return self.estimate(fallback=z)

        dt = stamp - self.last_stamp
        if dt > self.max_coast:
            # Long gap: a velocity and turn rate fitted across the seam would be
            # fiction. Start over from the seeding path.
            self.reset()
            self._seed(stamp, z[0], z[1])
            return self.estimate(fallback=z)
        if dt > 0.0:
            prop = self._predict(dt)
        else:
            prop = self._sigma_points()

        zpts = prop[:, :2]
        zhat = self.wm @ zpts
        S = np.eye(2) * (self.r_std ** 2)
        C = np.zeros((self.N, 2))
        for i in range(len(prop)):
            dz = zpts[i] - zhat
            dx = prop[i] - self.x
            dx[3] = wrap_angle(dx[3])
            S += self.wc[i] * np.outer(dz, dz)
            C += self.wc[i] * np.outer(dx, dz)

        K = C @ np.linalg.inv(S)
        self.x = self.x + K @ (z - zhat)
        self.x[3] = wrap_angle(self.x[3])
        P = self.P - K @ S @ K.T
        # The subtractive form is not guaranteed PSD in floating point; a single
        # negative eigenvalue here fails the next Cholesky and kills the node.
        self.P = 0.5 * (P + P.T)

        self.last_stamp = stamp
        self._update_heading()
        return self.estimate()

    def _update_heading(self):
        # CTRV's psi is the direction the body points. Speed can come out
        # negative if the filter fits the motion as reversing; fold that into the
        # heading so psi always means "which way the car faces".
        if self.x[2] < 0.0:
            self.x[2] = -self.x[2]
            self.x[3] = wrap_angle(self.x[3] + math.pi)
        if abs(self.x[2]) >= self.min_speed:
            self.heading = float(self.x[3])
            self.heading_valid = True
        else:
            # Hold the last heading. At rest, position measurements carry no
            # orientation information, so recomputing would track pure noise.
            self.heading_valid = False

    # ------------------------------------------------------------------

    def speed(self):
        return 0.0 if self.x is None else float(abs(self.x[2]))

    def estimate(self, fallback=None):
        if self.x is None:
            # Still seeding. Publish the raw position so the consumer has
            # something, but never a heading — there is no observation yet.
            if fallback is None:
                return None
            return {
                "forward": float(fallback[0]),
                "lateral": float(fallback[1]),
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
                "yaw_rate": 0.0,
                "heading": None,
                "heading_valid": False,
                "pos_std": self.r_std,
                "heading_std": float(math.pi),
            }
        psi = float(self.x[3])
        v = float(self.x[2])
        return {
            "forward": float(self.x[0]),
            "lateral": float(self.x[1]),
            "vx": v * math.cos(psi),
            "vy": v * math.sin(psi),
            "speed": abs(v),
            "yaw_rate": float(self.x[4]),
            "heading": self.heading,
            "heading_valid": bool(self.heading_valid),
            "pos_std": float(math.sqrt(max(self.P[0, 0], 0.0) + max(self.P[1, 1], 0.0))),
            "heading_std": float(math.sqrt(max(self.P[3, 3], 0.0))),
        }


def yaw_to_quaternion(yaw):
    """Yaw about +z -> (x, y, z, w)."""
    half = 0.5 * float(yaw)
    return 0.0, 0.0, math.sin(half), math.cos(half)
