"""SHIVA closed-world constraint check — the measurement SHIVA does not have.

THE CONSTRAINT NOBODY ELSE CAN USE
----------------------------------
Every strong method in the multi-animal tracking literature imposes hard
constraints it had to derive under the assumption of an UNKNOWN target count:
idtracker.ai rejects duplicate identity assignments, DeepLabCut pushes exactly
n_tracks units of flow, TNT zeroes overlap edges. This setup has a strictly
stronger fact available: there are exactly N animals, they are all present in
every one of the 60,000 frames, and none is ever born or dies.

That means the per-frame assignment of identities to animals must be a
PERMUTATION. Not "at most one identity per animal", which is what mutual
exclusion buys. Exactly one, always.

SHIVA enforces this nowhere except implicitly at the pixel level, and measures
it nowhere at all.

WHAT THIS COMPUTES
------------------
An adaptation of TRex's `uniqueness` (Walter & Couzin, eLife 2021), whose form
is

    uniqueness = (|distinct assigned identities| / |objects in frame|)
                 * mean assignment probability

and whose first factor is exactly 1 if and only if the frame's assignment is a
permutation. TRex uses it as a label-free training signal; here it is a
label-free per-frame swap detector.

The mask analogue of "two objects were assigned the same identity" is "two
identities are sitting on the same animal", which shows up as mask overlap, and
"one identity is carrying two animals", which shows up as an area roughly equal
to the sum of two body medians. Both are in the project's own eyeballed failure
taxonomy: absorption, detachment-drift, partial collapse, label stacking.

WHY THIS BEFORE ANY TREATMENT
-----------------------------
The identity metrics available today are all blind to the dominant failure:
`coverage_full` measures presence, not identity, so a mask parked on the wrong
animal still counts; the tracker's own swap counter reads 0 on a video with
three eyeball-confirmed swaps. Nothing in the pipeline emits a per-frame signal
that says "this frame violates the closed-world constraint". Without one, an
A/B compares coverage numbers; with one, it counts constraint violations
directly.

This module only READS. It cannot change tracking, so it is safe to leave on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from sam3.model.shiva_instrumentation import bump as _shiva_bump

logger = logging.getLogger(__name__)

__all__ = ["ClosedWorldReport", "evaluate_frame", "DEFAULTS"]


DEFAULTS = {
    # Two identities whose masks overlap by more than this fraction of the
    # smaller mask are, in practice, on the same animal. Intersection-over-
    # minimum rather than IoU: a small drifting mask sitting entirely on top of
    # a large one has IoM 1.0 but a low IoU, and that is exactly the
    # detachment-drift / label-stacking case worth catching.
    "overlap_thresh": 0.30,
    # Area at or above (own median + this * other median) means one identity is
    # carrying two bodies. Matches the absorption rule already validated in the
    # project's offline audit (corpus_audit_v4).
    "absorption_frac": 0.60,
    # Below this fraction of its own running median, a mask has cratered.
    "collapse_frac": 0.30,
    # Ignore masks smaller than this many pixels when testing overlap; they are
    # noise, not an identity claim.
    "min_area": 50,
}


@dataclass
class ClosedWorldReport:
    """Per-frame verdict on whether the assignment is a valid permutation."""

    frame_idx: int
    n_expected: int
    n_present: int
    uniqueness: float                                   # 1.0 == constraint satisfied
    max_overlap: float = 0.0
    overlap_pair: Optional[Tuple[int, int]] = None
    absorbed: List[int] = field(default_factory=list)   # oids carrying two bodies
    collapsed: List[int] = field(default_factory=list)  # oids whose mask cratered
    missing: List[int] = field(default_factory=list)    # oids with no mask
    violations: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def __str__(self) -> str:
        if self.ok:
            return f"frame {self.frame_idx}: closed-world OK (uniqueness=1.00)"
        return (
            f"frame {self.frame_idx}: uniqueness={self.uniqueness:.2f} "
            + "; ".join(self.violations)
        )


def _iom(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over the smaller area. See DEFAULTS['overlap_thresh']."""
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    smaller = min(int(a.sum()), int(b.sum()))
    return inter / max(smaller, 1)


def evaluate_frame(
    frame_idx: int,
    masks: Dict[int, np.ndarray],
    n_expected: int,
    median_areas: Optional[Dict[int, float]] = None,
    cfg: Optional[dict] = None,
) -> ClosedWorldReport:
    """Score one frame against the closed-world permutation constraint.

    Args:
        frame_idx: for reporting only.
        masks: {obj_id: (H, W) bool}. May be missing entries for lost objects.
        n_expected: N. The number of animals that are ALWAYS present.
        median_areas: {obj_id: running median area}. Absorption and collapse
            checks are skipped for objects without one, which is correct early
            in a video before the medians have stabilised.
        cfg: overrides for DEFAULTS.

    Returns:
        ClosedWorldReport. `uniqueness == 1.0` means the frame is consistent
        with exactly N animals each claimed by exactly one identity. It is a
        necessary condition, not a sufficient one: a frame where all four masks
        are cleanly on the wrong four animals scores 1.0. It catches the
        transition, not the steady state, which is why it pairs with a
        forward/backward permutation check rather than replacing one.
    """
    c = {**DEFAULTS, **(cfg or {})}
    areas = {oid: int(m.sum()) for oid, m in masks.items()}
    live = {oid: m for oid, m in masks.items() if areas[oid] >= c["min_area"]}

    report = ClosedWorldReport(
        frame_idx=frame_idx,
        n_expected=n_expected,
        n_present=len(live),
        uniqueness=0.0,
    )

    # --- cardinality: are all N accounted for? ---
    for oid, a in areas.items():
        if a < c["min_area"]:
            report.missing.append(oid)
    n_missing = n_expected - len(live)
    if n_missing > 0:
        report.violations.append(f"{n_missing} identity(s) absent")

    # --- mutual exclusion: are two identities on the same animal? ---
    oids = sorted(live)
    for i in range(len(oids)):
        for j in range(i + 1, len(oids)):
            ov = _iom(live[oids[i]], live[oids[j]])
            if ov > report.max_overlap:
                report.max_overlap = ov
                report.overlap_pair = (oids[i], oids[j])
    if report.max_overlap > c["overlap_thresh"]:
        a, b = report.overlap_pair
        report.violations.append(
            f"identities {a} and {b} overlap by {report.max_overlap:.0%} of the smaller mask"
        )

    # --- absorption and collapse, against each object's own running median ---
    if median_areas:
        for oid in oids:
            med = median_areas.get(oid)
            if not med:
                continue
            others = [median_areas[o] for o in oids if o != oid and median_areas.get(o)]
            if others and areas[oid] >= med + c["absorption_frac"] * min(others):
                report.absorbed.append(oid)
            if areas[oid] < c["collapse_frac"] * med:
                report.collapsed.append(oid)
        if report.absorbed:
            report.violations.append(
                f"identity(s) {report.absorbed} large enough to be carrying two bodies"
            )
        if report.collapsed:
            report.violations.append(f"identity(s) {report.collapsed} collapsed")

    # --- the TRex-form scalar ---
    # First factor: fraction of expected identities that are present AND not
    # sharing an animal with another identity. Second factor: how cleanly
    # separated they are. Both are 1.0 exactly when the frame is a clean
    # permutation.
    n_conflicted = 0
    if report.overlap_pair and report.max_overlap > c["overlap_thresh"]:
        n_conflicted = 1  # the pair collapses to one distinct animal
    distinct = max(len(live) - n_conflicted, 0)
    occupancy = distinct / max(n_expected, 1)
    separation = 1.0 - min(report.max_overlap, 1.0)
    report.uniqueness = float(min(occupancy, 1.0) * separation)

    if report.violations:
        _shiva_bump("closed_world_violation")
        if report.max_overlap > c["overlap_thresh"]:
            _shiva_bump("closed_world_overlap")
        if report.absorbed:
            _shiva_bump("closed_world_absorption")
        if report.collapsed:
            _shiva_bump("closed_world_collapse")
        if n_missing > 0:
            _shiva_bump("closed_world_missing")
    else:
        _shiva_bump("closed_world_ok")

    return report
