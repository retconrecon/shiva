"""SHIVA motion model — the piece SAM3 does not have.

WHY
---
SHIVA carries no motion model of any kind. There is no Kalman filter, no
velocity, no predicted position anywhere in sam3/. Its BoT-SORT port fuses mask
IoU with an appearance embedding and nothing else, which is BoT-SORT with the
part that actually survives an occlusion removed: in the original, identity is
carried through overlap by the Kalman prediction, not by IoU. Mask IoU between
t and t-1 is a degenerate motion model that degrades exactly when two animals
touch, which is the only moment identity is at risk.

The sibling repo `shiv` (SAM2.1 + SAMURAI) does have this filter. ZEUS runs it
as a separate later pass and then reconciles. This module lifts the idea
upstream so the prediction is available *during* the frame it is needed, where
it can prevent a swap instead of flagging one afterwards.

WHAT IT IS FOR
--------------
The consumer is the contested-pixel decision in
`_suppress_object_pw_area_shrinkage`, which is the partition written into the
memory bank. Today that partition is a pure per-pixel argmax over object
logits, with no notion of whose body a pixel belonged to a frame ago, and its
result feeds memory, so a stolen pixel raises the thief's logit there next
frame. That feedback is why takeovers are gradual and one-directional. See
`identity_swap_fix_design.md`, which specified this fix in 2026-06 and was
never implemented.

THE ONE RULE THAT MAKES OR BREAKS IT
------------------------------------
Feed the decision the *predicted* centroid from filtered history, never the
current frame's mask centroid. By the time a takeover is visible the
instantaneous centroid is already drifting toward the thief, so a naive
"where was it last frame" prior would reinforce the bug rather than resist it.
A constant-velocity filter has inertia that a slow per-frame creep cannot move,
which is precisely the property needed.

COORDINATES
-----------
Everything is in normalized image coordinates [0, 1], so the filter is
independent of the resolution juggling between the 912 frames, the 1008 model
input, and whatever the mask downsampler emits.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ConstantVelocityKalman", "ShivaMotionModel"]


class ConstantVelocityKalman:
    """2D constant-velocity Kalman filter over normalized coordinates.

    State is [x, y, vx, vy]; measurement is [x, y]. dt is one frame, so the
    velocity unit is normalized-image-widths per frame.

    Defaults are set for the ZEUS fish setup: 912 px at 50 fps, animals moving
    at roughly 5-20 px/frame, i.e. 0.005-0.022 normalized. process_std is the
    per-frame acceleration the filter treats as unsurprising; measurement_std
    reflects how much a mask centroid jitters on a stationary animal (body bend
    and fin flicker move it a few px even when the fish does not translate).
    """

    def __init__(
        self,
        xy: Tuple[float, float],
        process_std: float = 0.004,
        measurement_std: float = 0.004,
        initial_velocity_std: float = 0.02,
    ):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], dtype=np.float64)
        self.P = np.diag(
            [
                measurement_std**2,
                measurement_std**2,
                initial_velocity_std**2,
                initial_velocity_std**2,
            ]
        ).astype(np.float64)
        self._q = float(process_std) ** 2
        self._r = float(measurement_std) ** 2
        self.age = 0
        self.misses = 0

    # F for dt=1: position += velocity.
    @staticmethod
    def _predict_state(x: np.ndarray) -> np.ndarray:
        return np.array([x[0] + x[2], x[1] + x[3], x[2], x[3]], dtype=np.float64)

    @staticmethod
    def _predict_cov(P: np.ndarray, q: float) -> np.ndarray:
        # P' = F P F^T + Q, written out for the constant-velocity F.
        F = np.array(
            [[1.0, 0.0, 1.0, 0.0],
             [0.0, 1.0, 0.0, 1.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        # Discrete white-noise acceleration: position and velocity noise are
        # correlated, which matters here because a coasting track must widen its
        # position uncertainty at the right rate or the prior stays overconfident
        # through a long occlusion.
        Q = q * np.array(
            [[0.25, 0.0, 0.5, 0.0],
             [0.0, 0.25, 0.0, 0.5],
             [0.5, 0.0, 1.0, 0.0],
             [0.0, 0.5, 0.0, 1.0]],
            dtype=np.float64,
        )
        return F @ P @ F.T + Q

    def predict(self) -> Tuple[float, float, float]:
        """Advance one frame. Returns (x, y, position_std).

        This mutates state: call exactly once per frame, before observe/miss.
        """
        self.x = self._predict_state(self.x)
        self.P = self._predict_cov(self.P, self._q)
        self.age += 1
        return float(self.x[0]), float(self.x[1]), self.position_std

    @property
    def position_std(self) -> float:
        """1-sigma positional uncertainty, averaged over x and y."""
        return float(np.sqrt(max(0.5 * (self.P[0, 0] + self.P[1, 1]), 1e-12)))

    def observe(self, xy: Tuple[float, float]) -> None:
        """Correct with a measured centroid."""
        z = np.array([xy[0], xy[1]], dtype=np.float64)
        # S = H P H^T + R, a 2x2; inverted in closed form.
        S = self.P[:2, :2] + np.eye(2) * self._r
        det = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
        if abs(det) < 1e-18:
            return
        S_inv = np.array([[S[1, 1], -S[0, 1]], [-S[1, 0], S[0, 0]]]) / det
        K = self.P[:, :2] @ S_inv            # 4x2
        self.x = self.x + K @ (z - self.x[:2])
        self.P = self.P - K @ self.P[:2, :]
        self.P = 0.5 * (self.P + self.P.T)   # keep symmetric against drift
        self.misses = 0

    def miss(self) -> None:
        """No usable measurement this frame: coast on the motion model.

        Updating from an in-occlusion mask means fitting the corruption, so the
        caller is expected to route occluded frames here rather than to
        observe(). predict() has already widened P, which is what makes the
        prior back off on its own the longer the animal stays hidden.
        """
        self.misses += 1


class ShivaMotionModel:
    """One constant-velocity filter per tracked object, keyed by SAM3 obj_id."""

    def __init__(
        self,
        process_std: float = 0.004,
        measurement_std: float = 0.004,
        max_jump: float = 0.12,
    ):
        self.filters: Dict[int, ConstantVelocityKalman] = {}
        self.process_std = process_std
        self.measurement_std = measurement_std
        # A measurement farther than this from the prediction is treated as a
        # miss rather than an observation. At 912 px, 0.12 is ~110 px in one
        # frame at 50 fps, which no fish does; a jump that large is the mask
        # landing on a neighbour, and feeding it in would teach the filter the
        # very swap it exists to resist. ZEUS applies the same idea downstream
        # as its pass-1.7 Kalman jump rejector, on a 150 px threshold.
        self.max_jump = max_jump
        self._predictions: Dict[int, Tuple[float, float, float]] = {}

    def predict(self) -> Dict[int, Tuple[float, float, float]]:
        """Advance every filter one frame. Returns {oid: (x, y, std)}."""
        self._predictions = {
            oid: f.predict() for oid, f in self.filters.items()
        }
        return self._predictions

    @property
    def predictions(self) -> Dict[int, Tuple[float, float, float]]:
        return self._predictions

    def observe(self, oid: int, xy: Optional[Tuple[float, float]]) -> bool:
        """Feed a measured centroid. Returns True if it was accepted.

        None (no mask, or an unusable one) counts as a miss and coasts.
        """
        if xy is None or not np.isfinite(xy).all():
            if oid in self.filters:
                self.filters[oid].miss()
            return False
        if oid not in self.filters:
            self.filters[oid] = ConstantVelocityKalman(
                xy,
                process_std=self.process_std,
                measurement_std=self.measurement_std,
            )
            return True
        f = self.filters[oid]
        pred = self._predictions.get(oid)
        if pred is not None:
            d = float(np.hypot(xy[0] - pred[0], xy[1] - pred[1]))
            if d > self.max_jump:
                f.miss()
                return False
        f.observe(xy)
        return True

    def swap(self, oid_a: int, oid_b: int) -> None:
        """Exchange two objects' filters when the verifier corrects an identity.

        Without this the motion model keeps asserting the pre-swap trajectory
        for both animals and fights the correction.
        """
        fa = self.filters.pop(oid_a, None)
        fb = self.filters.pop(oid_b, None)
        if fb is not None:
            self.filters[oid_a] = fb
        if fa is not None:
            self.filters[oid_b] = fa
        pa = self._predictions.pop(oid_a, None)
        pb = self._predictions.pop(oid_b, None)
        if pb is not None:
            self._predictions[oid_a] = pb
        if pa is not None:
            self._predictions[oid_b] = pa

    def remove(self, oid: int) -> None:
        self.filters.pop(oid, None)
        self._predictions.pop(oid, None)
