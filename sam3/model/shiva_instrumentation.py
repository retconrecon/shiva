"""SHIVA hook instrumentation — proof that a feature actually executed.

WHY THIS EXISTS
---------------
Every SHIVA feature is a plain attribute planted on a SAM3 model object and read
back via ``getattr(..., default)`` deep inside SAM3's forward pass. That pattern
fails silently in both directions: a typo, a wrapper/inner MRO mismatch, or an
enclosing guard that is False makes the feature a no-op, and nothing anywhere
says so. The run completes, the metrics look plausible, and the flag reads True
in the config that gets written to results.json.

This happened at least five times on this codebase, and the 2026-08 cleanup
deleted four of the five features outright rather than repair them:

  1. ``occlusion_memory_freeze``  — gated on ``non_overlap_masks_for_mem_enc``,
     which upstream hardcodes false at sam3_multiplex_base.py:110. ZEUS passed
     it True in production for months. REMOVED.
  2. ``identity_verification``    — ran, but never wrote a mask. REMOVED.
  3. ``pixel_paint_enabled``      — the recovered mask is returned but its only
     effect on tracking is a memory injection whose failure was swallowed at
     debug level. KEPT, and now counted: see ``pixel_paint_inject_failed``.
  4. the crossing embedding freeze — ``crossing_active`` was never passed by the
     one caller, so it was always False. REMOVED with BoT-SORT.
  5. the ``_immediate_overlap`` reconditioning guard — tested
     ``hasattr(adt_result, 'iou_matrix')`` on a class with no such field, so it
     was always False. REMOVED with BoT-SORT.

Items 1-3 cost real GPU runs before anyone noticed; a downstream grant document
credits item 1 for an identity result it cannot have produced. The removed code
is recoverable from the tag ``archive/tiab-pre-cleanup-2026-08-01``.

THE RULE
--------
A config flag is a REQUEST. A counter is EVIDENCE. Do not A/B a SHIVA feature,
and do not report a number that depends on one, until a run shows its counter
nonzero. ``assert_fired()`` turns that from a discipline into a check.

USAGE
-----
    from sam3.model.shiva_instrumentation import bump

    bump("pixel_paint_recovery")         # at the hook's point of no return

    # end of run
    from sam3.model.shiva_instrumentation import format_report, assert_fired
    print(format_report())
    assert_fired(["pixel_paint_recovery"])  # raises if the feature never ran

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
    # --- memory management ---
    "memory_pruned": "memory pruning evicted at least one frame",
    # --- pixel-paint (inside ShivaTracker, post-yield) ---
    "pixel_paint_recovery": "pixel-paint recovered a missing mask",
    "bfs_mask_completion": "BFS completion filled unclaimed foreground",
    "pixel_paint_injected": "recovered mask written into SAM3's memory",
    "pixel_paint_inject_failed": "memory injection RAISED — recovery is a no-op",
    "pixel_paint_inject_unavailable": "no add_new_masks on the inner model",
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
