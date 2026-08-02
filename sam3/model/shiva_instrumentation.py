"""SHIVA hook instrumentation — proof that a feature actually executed.

WHY THIS EXISTS
---------------
Every SHIVA feature is a plain attribute planted on a SAM3 model object and read
back via ``getattr(..., default)`` deep inside SAM3's forward pass. That pattern
fails silently in both directions: a typo, a wrapper/inner MRO mismatch, or an
enclosing guard that is False makes the feature a no-op, and nothing anywhere
says so. The run completes, the metrics look plausible, and the flag reads True
in the config that gets written to results.json.

This has already happened at least five times on this codebase:

  1. ``occlusion_memory_freeze``  — gated on ``non_overlap_masks_for_mem_enc``,
     forced False at sam3_multiplex_base.py:110. Dead.
  2. ``identity_verification``    — runs, but never writes a mask.
  3. ``pixel_paint_enabled``      — grown mask never yielded; the memory
     injection raises and is swallowed at debug level.
  4. the crossing embedding freeze — ``crossing_active`` was never passed by the
     one caller, so it was always False. Fixed 2026-08.
  5. the ``_immediate_overlap`` reconditioning guard — tested
     ``hasattr(adt_result, 'iou_matrix')`` on a class with no such field, so it
     was always False. Fixed 2026-08.

Items 1-3 cost real GPU runs before anyone noticed; a downstream grant document
still credits item 1 for an identity result it cannot have produced.

THE RULE
--------
A config flag is a REQUEST. A counter is EVIDENCE. Do not A/B a SHIVA feature,
and do not report a number that depends on one, until a run shows its counter
nonzero. ``assert_fired()`` turns that from a discipline into a check.

USAGE
-----
    from sam3.model.shiva_instrumentation import bump

    bump("botsort_association")          # at the hook's point of no return

    # end of run
    from sam3.model.shiva_instrumentation import format_report, assert_fired
    print(format_report())
    assert_fired(["botsort_association"])   # raises if the feature never ran

Counters are process-global and cheap (a dict increment). ``reset()`` is called
by ``ShivaTracker.__init__`` so counts are per-session.

COMPILE SAFETY
--------------
``_suppress_object_pw_area_shrinkage`` is wrapped by ``compile_wrapper(...,
fullgraph=True)`` at sam3_multiplex_tracking.py:1365. A Python-level counter
bump inside a compiled region is a graph break, which is an error under
fullgraph. Call sites inside compiled functions must guard with
``if not torch.compiler.is_compiling():`` — dynamo folds that branch away at
trace time, so the eager path counts and the compiled path stays whole.
"""

from __future__ import annotations

import threading
from typing import Dict, Iterable, List

__all__ = [
    "bump",
    "get",
    "snapshot",
    "reset",
    "format_report",
    "assert_fired",
    "KNOWN_HOOKS",
]

# Every hook we expect to be able to prove. Listing them here means the report
# can show a hook at 0 rather than omitting it, which is the whole point: a
# missing line is invisible, an explicit zero is a finding.
KNOWN_HOOKS: Dict[str, str] = {
    # --- association / identity (inside SAM3's forward pass) ---
    "botsort_association": "BoT-SORT Hungarian association dispatched",
    "botsort_hungarian_match": "a (det, trk) pair accepted by Hungarian",
    "appearance_update": "appearance embedding EMA-updated",
    "appearance_frozen_by_crossing": "embedding update BLOCKED mid-crossing (2026-08 fix #4)",
    "appearance_rejected_far": "embedding update rejected, distance > max_update_distance",
    "recondition_suppressed": "reconditioning gated off (identity ambiguous)",
    "immediate_overlap_suppressed": "gated off by first-frame overlap (2026-08 fix #5)",
    "sentinel_yellow": "SENTINEL degraded to YELLOW",
    "sentinel_red": "SENTINEL degraded to RED",
    # --- mask partition / memory (inside SAM3's forward pass) ---
    "keep_contested_rescue": "object handed its won partition instead of deleted (keep-contested)",
    "memory_freeze_applied": "object's memory frozen during occlusion",
    "temporal_boundary_prior": "temporal boundary prior applied to the partition",
    "tiab_forward": "TIAB module forward pass executed",
    # --- MAP contested-pixel partition + motion model (2026-08) ---
    "map_partition_frames_active": "a frame where the prior overturned the argmax",
    "map_partition_pixels_flipped": "pixels reassigned by the prior vs bare argmax",
    "map_partition_skipped_batch_mismatch": "prior dropped, object count changed",
    "motion_observe": "Kalman accepted a measured centroid",
    "motion_miss": "no usable centroid, filter coasted",
    "motion_jump_rejected": "centroid rejected as an implausible jump",
    "motion_prior_dropped_stale": "prior withheld, track coasting past horizon",
    # --- closed-world constraint check (read-only measurement) ---
    "closed_world_ok": "frame satisfied the exactly-N permutation constraint",
    "closed_world_violation": "frame VIOLATED the closed-world constraint",
    "closed_world_overlap": "two identities sitting on the same animal",
    "closed_world_absorption": "one identity large enough to carry two bodies",
    "closed_world_collapse": "a mask cratered below its running median",
    "closed_world_missing": "fewer than N identities present",
    # --- post-yield hooks (inside ShivaTracker) ---
    "pixel_paint_recovery": "pixel-paint recovered a missing mask",
    "bfs_mask_completion": "BFS completion filled unclaimed foreground",
    "confidence_injection": "low-confidence object re-injected into memory",
    "identity_swap_detected": "identity verifier emitted a SwapEvent",
    "identity_swap_applied": "apply_swap() actually swapped two identities",
    "memory_pruned": "memory pruning evicted at least one frame",
}

_lock = threading.Lock()
_counts: Dict[str, int] = {}


def bump(name: str, n: int = 1) -> None:
    """Increment a hook counter. Cheap, thread-safe, never raises."""
    if n <= 0:
        return
    with _lock:
        _counts[name] = _counts.get(name, 0) + int(n)


def get(name: str) -> int:
    """Current count for one hook (0 if it never fired)."""
    with _lock:
        return _counts.get(name, 0)


def snapshot() -> Dict[str, int]:
    """Copy of all counters, including known hooks that never fired."""
    with _lock:
        out = {k: 0 for k in KNOWN_HOOKS}
        out.update(_counts)
        return out


def reset() -> None:
    """Clear all counters. Called per tracking session."""
    with _lock:
        _counts.clear()


def format_report(only_expected: Iterable[str] | None = None) -> str:
    """Human-readable truth table of which hooks ran.

    ``only_expected`` marks the hooks the caller believes it enabled, so the
    report can call out a requested-but-silent feature explicitly.
    """
    counts = snapshot()
    expected = set(only_expected or ())
    width = max((len(k) for k in counts), default=10)

    fired = sorted(
        ((k, v) for k, v in counts.items() if v > 0),
        key=lambda kv: -kv[1],
    )
    silent = sorted(k for k, v in counts.items() if v == 0)

    lines: List[str] = ["", "=" * (width + 30), "SHIVA hook instrumentation", "=" * (width + 30)]

    if fired:
        lines.append("FIRED:")
        for k, v in fired:
            lines.append(f"  {k:<{width}}  {v:>10,}   {KNOWN_HOOKS.get(k, '')}")
    else:
        lines.append("FIRED: (nothing — no SHIVA hook executed this run)")

    if silent:
        lines.append("")
        lines.append("SILENT (count == 0):")
        for k in silent:
            flag = "  <-- REQUESTED BUT NEVER RAN" if k in expected else ""
            lines.append(f"  {k:<{width}}  {0:>10}   {KNOWN_HOOKS.get(k, '')}{flag}")

    broken = sorted(expected - {k for k, v in counts.items() if v > 0})
    lines.append("")
    if broken:
        lines.append(f"VERDICT: {len(broken)} requested feature(s) NEVER RAN: {', '.join(broken)}")
        lines.append("         Any result attributed to them is unsupported.")
    elif expected:
        lines.append(f"VERDICT: all {len(expected)} requested feature(s) executed.")
    lines.append("=" * (width + 30))
    return "\n".join(lines)


def assert_fired(names: Iterable[str]) -> None:
    """Raise if any named hook never fired. Use in CI and smoke runs."""
    names = list(names)
    dead = [n for n in names if get(n) == 0]
    if dead:
        raise AssertionError(
            "SHIVA feature(s) enabled but never executed: "
            + ", ".join(dead)
            + "\n"
            + format_report(only_expected=names)
        )
