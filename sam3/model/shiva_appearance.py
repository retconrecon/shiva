"""SHIVA Appearance Store — per-track EMA histogram embeddings.

Zero-training, zero-parameter appearance model using grayscale histograms
of masked fish regions. Used by BoT-SORT association to resolve identity
ambiguity during occlusions.

Usage:
    from sam3.model.shiva_appearance import ShivaAppearanceStore

    store = ShivaAppearanceStore(n_bins=32)
    hist = store.extract_histogram(mask_bool, frame_pixels)
    store.update(obj_id, hist)
    trk_embs = store.get([0, 1, 2, 3])
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


SHIVA_HISTOGRAM_BINS = 32


def extract_masked_histogram(mask, frame_pixels, n_bins=SHIVA_HISTOGRAM_BINS,
                              min_crop_size=0, min_coverage=0.0):
    """Extract grayscale histogram from masked region of frame.

    Shared function used by both the appearance store (BoT-SORT) and
    the identity verifier (crossing swap detection).

    Args:
        mask: (H, W) bool — the object's mask
        frame_pixels: (H, W, 3) BGR or (H, W) grayscale
        n_bins: number of histogram bins
        min_crop_size: minimum bounding box width/height (0 to disable)
        min_coverage: minimum mask-to-crop area ratio (0.0 to disable)

    Returns:
        (n_bins,) normalized float64 histogram, or None if validation fails.
    """
    if frame_pixels.ndim == 3:
        gray = cv2.cvtColor(frame_pixels, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_pixels

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    # Crop validation
    if min_crop_size > 0:
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
            return None
        if min_coverage > 0:
            crop_mask = mask[y1:y2, x1:x2]
            if crop_mask.size == 0 or crop_mask.sum() / crop_mask.size < min_coverage:
                return None

    vals = gray[mask].astype(np.int32)
    if len(vals) == 0:
        return None

    hist, _ = np.histogram(vals, bins=n_bins, range=(0, 256))
    total = hist.sum()
    if total > 0:
        hist = hist.astype(np.float64) / total
    else:
        return None
    return hist


def histogram_intersection_distance(h1, h2):
    """Histogram intersection distance (0 = identical, 1 = disjoint)."""
    return 1.0 - np.minimum(h1, h2).sum()


class ShivaAppearanceStore:
    """Per-track EMA appearance embeddings for identity-aware association."""

    distance_metric = "histogram_intersection"  # explicit metric declaration

    def __init__(self, n_bins=32, ema_alpha=0.9, max_consecutive_rejects=50):
        self.n_bins = n_bins
        self.ema_alpha = ema_alpha
        self.embeddings = {}  # {obj_id: np.ndarray histogram}
        self._consecutive_rejects = {}  # {obj_id: int}
        self._max_consecutive_rejects = max_consecutive_rejects

    def update(self, obj_id, histogram, max_update_distance=0.4,
               crossing_active=False):
        """Update track's appearance model with exponential moving average.

        Skips the update if the new histogram diverges too much from the
        current embedding — a large appearance change likely indicates an
        identity swap, and updating would lock the wrong identity in.

        If updates are rejected for too many consecutive frames (e.g., after
        an undetected swap or lighting change), force-resets the embedding
        from the current observation to unstick it — but ONLY when no
        crossing is active. During crossings, the current embedding is a
        blend of overlapping animals and force-reset would lock in the
        wrong identity.
        """
        if obj_id not in self.embeddings:
            self.embeddings[obj_id] = histogram.copy()
            self._consecutive_rejects[obj_id] = 0
        else:
            dist = histogram_intersection_distance(
                self.embeddings[obj_id], histogram
            )
            if dist < max_update_distance:
                self.embeddings[obj_id] = (
                    self.ema_alpha * self.embeddings[obj_id]
                    + (1 - self.ema_alpha) * histogram
                )
                self._consecutive_rejects[obj_id] = 0
            else:
                self._consecutive_rejects[obj_id] = (
                    self._consecutive_rejects.get(obj_id, 0) + 1
                )
                # Force-reset after too many consecutive rejects to unstick
                # embeddings frozen by undetected swaps or lighting changes.
                # Suppress during crossings — current observation is likely
                # contaminated by overlapping animals.
                if (self._consecutive_rejects[obj_id] >= self._max_consecutive_rejects
                        and not crossing_active):
                    self.embeddings[obj_id] = histogram.copy()
                    self._consecutive_rejects[obj_id] = 0
                    logger.info(
                        "Appearance embedding reset for obj %s after %d "
                        "consecutive rejected updates",
                        obj_id, self._max_consecutive_rejects,
                    )

    def get(self, obj_ids):
        """Get appearance embeddings for a list of track IDs.

        Returns uniform histogram fallback for tracks that don't have
        an embedding yet, so one missing track doesn't disable appearance
        matching for all tracks.

        Returns:
            (M, n_bins) numpy array. Never returns None.
        """
        uniform = np.ones(self.n_bins, dtype=np.float64) / self.n_bins
        embs = []
        for oid in obj_ids:
            if oid in self.embeddings:
                embs.append(self.embeddings[oid])
            else:
                embs.append(uniform)
        if not embs:
            return np.zeros((0, self.n_bins), dtype=np.float64)
        return np.stack(embs)

    def remove(self, obj_id):
        """Remove a track's embedding (e.g., track died)."""
        self.embeddings.pop(obj_id, None)

    def extract_histogram(self, mask, frame_pixels):
        """Extract grayscale histogram from masked region of frame.

        Returns uniform fallback if extraction fails (empty mask, etc.).
        """
        uniform = np.ones(self.n_bins, dtype=np.float64) / self.n_bins
        hist = extract_masked_histogram(mask, frame_pixels, n_bins=self.n_bins)
        return hist if hist is not None else uniform

    def extract_embedding(self, mask, frame_pixels):
        """Alias for extract_histogram with strict validation.

        Provides interface compatibility with ShivaOSNetAppearanceStore.
        Returns None on failure (matching OSNet contract for the identity
        verifier's needs).
        """
        return extract_masked_histogram(
            mask, frame_pixels, n_bins=self.n_bins,
            min_crop_size=16, min_coverage=0.1,
        )
