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

import cv2
import numpy as np


class ShivaAppearanceStore:
    """Per-track EMA appearance embeddings for identity-aware association."""

    def __init__(self, n_bins=32, ema_alpha=0.9):
        self.n_bins = n_bins
        self.ema_alpha = ema_alpha
        self.embeddings = {}  # {obj_id: np.ndarray histogram}

    def update(self, obj_id, histogram, max_update_distance=0.4):
        """Update track's appearance model with exponential moving average.

        Skips the update if the new histogram diverges too much from the
        current embedding — a large appearance change likely indicates an
        identity swap, and updating would lock the wrong identity in.
        """
        if obj_id not in self.embeddings:
            self.embeddings[obj_id] = histogram.copy()
        else:
            dist = self.histogram_intersection_distance(
                self.embeddings[obj_id], histogram
            )
            if dist < max_update_distance:
                self.embeddings[obj_id] = (
                    self.ema_alpha * self.embeddings[obj_id]
                    + (1 - self.ema_alpha) * histogram
                )

    def get(self, obj_ids):
        """Get appearance embeddings for a list of track IDs.

        Returns:
            (M, n_bins) numpy array, or None if any track has no embedding yet.
        """
        embs = []
        for oid in obj_ids:
            if oid not in self.embeddings:
                return None
            embs.append(self.embeddings[oid])
        if not embs:
            return None
        return np.stack(embs)

    def remove(self, obj_id):
        """Remove a track's embedding (e.g., track died)."""
        self.embeddings.pop(obj_id, None)

    def extract_histogram(self, mask, frame_pixels):
        """Extract grayscale histogram from masked region of frame.

        Args:
            mask: (H, W) bool — the object's mask
            frame_pixels: (H, W, 3) BGR or (H, W) grayscale — raw frame pixels

        Returns:
            (n_bins,) normalized float64 histogram
        """
        if frame_pixels.ndim == 3:
            gray = cv2.cvtColor(frame_pixels, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_pixels
        vals = gray[mask].astype(np.int32)
        if len(vals) == 0:
            return np.ones(self.n_bins, dtype=np.float64) / self.n_bins
        hist, _ = np.histogram(vals, bins=self.n_bins, range=(0, 256))
        total = hist.sum()
        if total > 0:
            hist = hist.astype(np.float64) / total
        else:
            hist = np.ones(self.n_bins, dtype=np.float64) / self.n_bins
        return hist

    @staticmethod
    def histogram_intersection_distance(h1, h2):
        """Histogram intersection distance (0 = identical, 1 = disjoint)."""
        return 1.0 - np.minimum(h1, h2).sum()
