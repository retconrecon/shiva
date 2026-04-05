"""SHIVA Identity Verifier — crossing-triggered swap detection.

Monitors pairwise mask IoU to detect crossing events. When two fish
separate after overlapping, compares pre- vs post-crossing visual
embeddings to detect identity swaps.

The key insight: we don't ask "which fish is this?" (impossible for
visually identical targets). We ask "did the identity assignment change
across this specific crossing?" — comparing pre- vs post-crossing
embeddings provides relative context.

Architecture:
    4-phase crossing state machine per object pair:
    CLEAR → APPROACHING → OVERLAPPING → SEPARATING → CLEAR

    At APPROACHING: capture pre-crossing reference embeddings
    At SEPARATING → CLEAR: compare post-crossing vs pre-crossing
    If sim_swapped > sim_correct + margin → swap detected

Usage:
    from sam3.model.shiva_identity_verifier import ShivaIdentityVerifier

    verifier = ShivaIdentityVerifier(n_objects=4)
    # ... in tracking loop ...
    swap_events = verifier.update(frame_idx, bool_masks, pairwise_ious, frame_bgr)
    for swap in swap_events:
        print(f"SWAP: {swap.oid_a} <-> {swap.oid_b} (margin={swap.margin:.3f})")

Ported from shiv/sam2/sam2/identity_verifier.py (461 lines) with adaptations
for SAM3.1's multiplex pipeline.
"""

import enum
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crossing state machine
# ---------------------------------------------------------------------------

class CrossingPhase(enum.Enum):
    CLEAR = "clear"
    APPROACHING = "approaching"
    OVERLAPPING = "overlapping"
    SEPARATING = "separating"


@dataclass
class PairCrossingState:
    """Mutable state for one object pair's crossing lifecycle."""
    phase: CrossingPhase = CrossingPhase.CLEAR
    ref_embeddings: Dict[int, np.ndarray] = field(default_factory=dict)
    ref_centroids: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    approach_frame: Optional[int] = None
    overlap_frames: int = 0
    separating_retry_count: int = 0  # retries when frame_bgr unavailable


@dataclass
class SwapEvent:
    """A detected identity swap between two objects."""
    oid_a: int
    oid_b: int
    margin: float
    frame_idx: int
    sim_keep: float
    sim_swap: float
    approach_frame: Optional[int] = None
    action: str = "swap"  # "swap", "no_swap", "deferred", "timeout"
    overlap_frames: int = 0  # how many frames the pair spent in OVERLAPPING


# ---------------------------------------------------------------------------
# ShivaIdentityVerifier
# ---------------------------------------------------------------------------

class ShivaIdentityVerifier:
    """Crossing-triggered identity verification for multi-object tracking.

    Parameters
    ----------
    n_objects : int
        Expected number of tracked objects.
    proximity_threshold : float
        IoU level marking entering/exiting the crossing zone.
    overlap_threshold : float
        IoU level marking full overlap.
    swap_margin : float
        How much sim_swapped must exceed sim_correct to declare a swap.
    min_crop_size : int
        Minimum crop width/height in pixels; skip degenerate crops.
    max_overlap_frames : int
        Abandon crossing event if OVERLAPPING persists longer than this.
    n_bins : int
        Histogram bins for embedding extraction.
    """

    def __init__(
        self,
        n_objects: int,
        proximity_threshold: float = 0.1,
        overlap_threshold: float = 0.8,
        swap_margin: float = 0.05,
        min_crop_size: int = 16,
        max_overlap_frames: int = 300,
        n_bins: int = None,  # defaults to SHIVA_HISTOGRAM_BINS
        appearance_store=None,  # optional: use store's extract_embedding for swap comparison
    ):
        self.n_objects = n_objects
        self.proximity_threshold = proximity_threshold
        self.overlap_threshold = overlap_threshold
        self.swap_margin = swap_margin
        self.min_crop_size = min_crop_size
        self.max_overlap_frames = max_overlap_frames
        from sam3.model.shiva_appearance import SHIVA_HISTOGRAM_BINS
        self.n_bins = n_bins if n_bins is not None else SHIVA_HISTOGRAM_BINS
        self._appearance_store = appearance_store

        # list() copies on _pair_states iteration are load-bearing —
        # notify_swap and update both mutate the dict during processing
        self._pair_states: Dict[Tuple[int, int], PairCrossingState] = {}
        self.swap_log: List[SwapEvent] = []
        self._max_log_entries = 10000

    def reset_object(self, obj_id: int) -> None:
        """Reset all pair states involving obj_id.

        Call when an object dies, is re-initialized, or has its memory
        purged. Prevents stale pair states from corrupting swap decisions.
        """
        for pair_key in list(self._pair_states.keys()):
            if obj_id in pair_key:
                state = self._pair_states[pair_key]
                if state.phase != CrossingPhase.CLEAR:
                    logger.debug(
                        "Resetting pair %s (object %d reset, was %s)",
                        pair_key, obj_id, state.phase.value,
                    )
                del self._pair_states[pair_key]

    def notify_swap(self, oid_a: int, oid_b: int,
                    bool_masks: Optional[Dict[int, np.ndarray]] = None,
                    frame_bgr: Optional[np.ndarray] = None) -> None:
        """Invalidate sibling pair states after a swap on (oid_a, oid_b).

        Instead of dropping sibling pairs entirely (which loses secondary
        swaps in three-body crossings), marks their ref_embeddings as stale
        and re-snapshots with corrected identities if frame data is available.
        """
        swapped_pair = (min(oid_a, oid_b), max(oid_a, oid_b))
        for pair_key in list(self._pair_states.keys()):
            if pair_key == swapped_pair:
                continue
            if oid_a in pair_key or oid_b in pair_key:
                state = self._pair_states[pair_key]
                if state.phase == CrossingPhase.CLEAR:
                    continue
                # Re-snapshot with corrected (post-swap) identities if possible
                if bool_masks is not None and frame_bgr is not None:
                    ok = self._snapshot_embeddings(pair_key, bool_masks, frame_bgr)
                    if ok:
                        logger.info(
                            "Re-snapshotted sibling pair %s after swap(%d, %d) (was %s)",
                            pair_key, oid_a, oid_b, state.phase.value,
                        )
                        continue
                # Fallback: reset if can't re-snapshot
                logger.info(
                    "Resetting sibling pair %s after swap(%d, %d) — "
                    "could not re-snapshot (was %s)",
                    pair_key, oid_a, oid_b, state.phase.value,
                )
                del self._pair_states[pair_key]

    def _get_pair_state(self, oid_a: int, oid_b: int) -> PairCrossingState:
        """Get or create state for an ordered pair."""
        key = (min(oid_a, oid_b), max(oid_a, oid_b))
        if key not in self._pair_states:
            self._pair_states[key] = PairCrossingState()
        return self._pair_states[key]

    def _extract_embedding(self, mask: np.ndarray, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding from masked crop.

        Uses the active appearance store if provided (supports OSNet, histogram,
        or any future backend). Falls back to shared extract_masked_histogram
        with strict validation (min crop size + 10% coverage).
        """
        if self._appearance_store is not None:
            return self._appearance_store.extract_embedding(mask, frame_bgr)
        from sam3.model.shiva_appearance import extract_masked_histogram
        return extract_masked_histogram(
            mask, frame_bgr, n_bins=self.n_bins,
            min_crop_size=self.min_crop_size, min_coverage=0.1,
        )

    def _snapshot_embeddings(
        self, pair: Tuple[int, int], bool_masks: Dict[int, np.ndarray],
        frame_bgr: np.ndarray,
    ) -> bool:
        """Capture and store reference embeddings for a pair.

        Returns True if both embeddings were successfully extracted.
        """
        state = self._pair_states[pair]
        state.ref_embeddings = {}
        state.ref_centroids = {}

        for oid in pair:
            mask = bool_masks.get(oid)
            if mask is None:
                return False
            emb = self._extract_embedding(mask, frame_bgr)
            if emb is None:
                logger.debug("Cannot snapshot embedding for oid %d: bad crop", oid)
                return False
            state.ref_embeddings[oid] = emb
            # Capture reference centroid for trajectory continuity check
            # Use largest-component centroid to match tracking pipeline
            from scipy.ndimage import label as _ndlabel
            labeled, n_comp = _ndlabel(mask)
            if n_comp >= 1:
                if n_comp == 1:
                    ys, xs = np.where(mask)
                else:
                    comp_sizes = [(labeled == c).sum() for c in range(1, n_comp + 1)]
                    largest = max(range(1, n_comp + 1), key=lambda c: comp_sizes[c - 1])
                    ys, xs = np.where(labeled == largest)
                state.ref_centroids[oid] = (float(xs.mean()), float(ys.mean()))

        return True

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity using the same metric as the active appearance store.

        Histogram intersection for histogram backend, cosine for OSNet.
        """
        metric = getattr(self._appearance_store, 'distance_metric', 'histogram_intersection')
        if metric == "cosine":
            # Cosine similarity for L2-normalized embeddings
            return float(np.dot(a, b))
        else:
            # Histogram intersection similarity
            return float(np.minimum(a, b).sum())

    def _check_swap(
        self, pair: Tuple[int, int], bool_masks: Dict[int, np.ndarray],
        frame_bgr: np.ndarray, frame_idx: int,
    ) -> Optional[SwapEvent]:
        """Compare pre- vs post-crossing embeddings for a pair."""
        state = self._pair_states[pair]
        oid_a, oid_b = pair

        ref_a = state.ref_embeddings.get(oid_a)
        ref_b = state.ref_embeddings.get(oid_b)
        if ref_a is None or ref_b is None:
            return None

        # Extract post-crossing embeddings
        mask_a = bool_masks.get(oid_a)
        mask_b = bool_masks.get(oid_b)
        if mask_a is None or mask_b is None:
            return None

        cur_a = self._extract_embedding(mask_a, frame_bgr)
        cur_b = self._extract_embedding(mask_b, frame_bgr)
        if cur_a is None or cur_b is None:
            return None

        # Compare: keep assignment vs swapped assignment
        # Uses histogram intersection similarity (same metric as appearance store)
        sim_keep = (
            self._similarity(cur_a, ref_a)
            + self._similarity(cur_b, ref_b)
        )
        sim_swap = (
            self._similarity(cur_a, ref_b)
            + self._similarity(cur_b, ref_a)
        )
        margin = sim_swap - sim_keep

        # Trajectory continuity tiebreaker: when appearance is ambiguous
        # (same-species animals with similar histograms), check whether
        # post-crossing centroids are closer to the keep or swap assignment.
        # This provides an identity signal independent of visual similarity.
        traj_margin = 0.0
        ref_c_a = state.ref_centroids.get(oid_a)
        ref_c_b = state.ref_centroids.get(oid_b)
        if ref_c_a is not None and ref_c_b is not None:
            # Use largest-component centroid for post-crossing comparison
            from scipy.ndimage import label as _ndlabel
            def _lc_centroid(mask):
                labeled, n_comp = _ndlabel(mask)
                if n_comp == 0:
                    return None
                if n_comp == 1:
                    ys, xs = np.where(mask)
                else:
                    sizes = [(labeled == c).sum() for c in range(1, n_comp + 1)]
                    largest = max(range(1, n_comp + 1), key=lambda c: sizes[c - 1])
                    ys, xs = np.where(labeled == largest)
                return (float(xs.mean()), float(ys.mean()))

            cur_c_a = _lc_centroid(mask_a)
            cur_c_b = _lc_centroid(mask_b)
            if cur_c_a is not None and cur_c_b is not None:
                # Distance under keep assignment
                d_keep = (
                    np.sqrt((cur_c_a[0] - ref_c_a[0])**2 + (cur_c_a[1] - ref_c_a[1])**2)
                    + np.sqrt((cur_c_b[0] - ref_c_b[0])**2 + (cur_c_b[1] - ref_c_b[1])**2)
                )
                # Distance under swap assignment
                d_swap = (
                    np.sqrt((cur_c_a[0] - ref_c_b[0])**2 + (cur_c_a[1] - ref_c_b[1])**2)
                    + np.sqrt((cur_c_b[0] - ref_c_a[0])**2 + (cur_c_b[1] - ref_c_a[1])**2)
                )
                # Normalize to a similarity-like scale (positive = swap is better)
                if d_keep + d_swap > 0:
                    traj_margin = (d_keep - d_swap) / (d_keep + d_swap)

        # Combined decision: appearance margin + trajectory tiebreaker
        # traj_margin > 0 means swap is closer to trajectory continuity
        combined_margin = margin + 0.3 * traj_margin

        action = "swap" if combined_margin > self.swap_margin else "no_swap"

        return SwapEvent(
            oid_a=oid_a,
            oid_b=oid_b,
            margin=combined_margin,
            frame_idx=frame_idx,
            sim_keep=sim_keep,
            sim_swap=sim_swap,
            approach_frame=state.approach_frame,
            action=action,
            overlap_frames=state.overlap_frames,
        )

    def update(
        self,
        frame_idx: int,
        bool_masks: Dict[int, np.ndarray],
        pairwise_ious: Dict[Tuple[int, int], float],
        frame_bgr: Optional[np.ndarray] = None,
    ) -> List[SwapEvent]:
        """Drive the crossing state machine for all pairs.

        Returns list of swap events emitted this frame (may be empty).
        """
        events = []

        # Prune pair states for objects no longer active
        active_objs = set()
        for a, b in pairwise_ious.keys():
            active_objs.add(a)
            active_objs.add(b)
        for pair_key in list(self._pair_states.keys()):
            if pair_key[0] not in active_objs or pair_key[1] not in active_objs:
                del self._pair_states[pair_key]

        # Hysteresis deadband: enter at proximity_threshold,
        # exit at half that to prevent CLEAR↔APPROACHING flapping
        exit_threshold = self.proximity_threshold * 0.5

        for (oid_a, oid_b), iou in pairwise_ious.items():
            pair = (min(oid_a, oid_b), max(oid_a, oid_b))
            state = self._get_pair_state(*pair)

            if state.phase == CrossingPhase.CLEAR:
                if iou > self.proximity_threshold:
                    state.phase = CrossingPhase.APPROACHING
                    state.approach_frame = frame_idx
                    if frame_bgr is not None:
                        ok = self._snapshot_embeddings(pair, bool_masks, frame_bgr)
                        if not ok:
                            state.phase = CrossingPhase.CLEAR
                            state.approach_frame = None
                    else:
                        state.phase = CrossingPhase.CLEAR
                        state.approach_frame = None

            elif state.phase == CrossingPhase.APPROACHING:
                if iou > self.overlap_threshold:
                    state.phase = CrossingPhase.OVERLAPPING
                    state.overlap_frames = 0
                elif iou <= exit_threshold:
                    # False alarm — never reached overlap
                    state.phase = CrossingPhase.CLEAR
                    state.ref_embeddings = {}
                    state.ref_centroids = {}
                    state.approach_frame = None

            elif state.phase == CrossingPhase.OVERLAPPING:
                state.overlap_frames += 1
                if (self.max_overlap_frames > 0
                        and state.overlap_frames > self.max_overlap_frames):
                    logger.info(
                        "Overlap timeout for pair %s after %d frames",
                        pair, state.overlap_frames,
                    )
                    # Emit timeout event so consumers can flag low-confidence window
                    events.append(SwapEvent(
                        oid_a=pair[0], oid_b=pair[1], margin=0.0,
                        frame_idx=frame_idx, sim_keep=0.0, sim_swap=0.0,
                        approach_frame=state.approach_frame, action="timeout",
                        overlap_frames=state.overlap_frames,
                    ))
                    state.phase = CrossingPhase.CLEAR
                    state.ref_embeddings = {}
                    state.ref_centroids = {}
                    state.approach_frame = None
                    state.overlap_frames = 0
                elif iou < self.overlap_threshold:
                    state.phase = CrossingPhase.SEPARATING

            elif state.phase == CrossingPhase.SEPARATING:
                if iou > self.overlap_threshold:
                    state.phase = CrossingPhase.OVERLAPPING
                    state.separating_retry_count = 0
                elif iou < exit_threshold:
                    # Fully separated — run swap check
                    if frame_bgr is not None:
                        event = self._check_swap(pair, bool_masks, frame_bgr, frame_idx)
                        if event is not None:
                            events.append(event)
                        # Reset state after successful check
                        state.phase = CrossingPhase.CLEAR
                        state.ref_embeddings = {}
                        state.approach_frame = None
                        state.overlap_frames = 0
                        state.separating_retry_count = 0
                    else:
                        # Retry on next frame — don't discard the crossing
                        state.separating_retry_count += 1
                        if state.separating_retry_count > 10:
                            logger.warning(
                                "Crossing for pair %s: frame_bgr unavailable "
                                "for %d frames — abandoning swap check",
                                pair, state.separating_retry_count,
                            )
                            state.phase = CrossingPhase.CLEAR
                            state.ref_embeddings = {}
                            state.approach_frame = None
                            state.overlap_frames = 0
                            state.separating_retry_count = 0

        # Resolve conflicting swaps: highest margin wins
        actual_swaps = [e for e in events if e.action == "swap"]
        if len(actual_swaps) > 1:
            actual_swaps.sort(key=lambda e: e.margin, reverse=True)
            claimed_oids = set()
            for event in actual_swaps:
                if event.oid_a in claimed_oids or event.oid_b in claimed_oids:
                    logger.warning(
                        "Deferring conflicting swap %s at frame %d (oid already claimed)",
                        (event.oid_a, event.oid_b), frame_idx,
                    )
                    event.action = "deferred"
                else:
                    claimed_oids.add(event.oid_a)
                    claimed_oids.add(event.oid_b)

        # Three-body safety: reset sibling pairs after each applied swap
        for event in events:
            if event.action == "swap":
                self.notify_swap(event.oid_a, event.oid_b,
                                 bool_masks=bool_masks, frame_bgr=frame_bgr)

        self.swap_log.extend(events)
        if len(self.swap_log) > self._max_log_entries:
            self.swap_log = self.swap_log[-self._max_log_entries:]
        return events

    def get_crossing_states(self) -> Dict[Tuple[int, int], str]:
        """Return current state of all crossing pairs for diagnostics."""
        return {
            pair: state.phase.value
            for pair, state in self._pair_states.items()
        }
