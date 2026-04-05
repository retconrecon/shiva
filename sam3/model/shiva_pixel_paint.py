"""SHIVA Pixel-Paint Recovery — area-gated mask recovery via background subtraction.

When a fish's mask disappears or shrinks to an artifact, detects via area
fingerprint and recovers via pixel-paint blob detection. Unclaimed blobs
(not covered by healthy masks) are matched to missing fish via Hungarian
matching on an area-distance cost matrix.

Usage:
    from sam3.model.shiva_pixel_paint import ShivaPixelPaintRecovery

    pp = ShivaPixelPaintRecovery(frame_dir, n_animals=4)
    pp.build_background_model()
    # ... during tracking loop ...
    pp.update_median_areas(frame_bool_masks)
    recovery = pp.check_and_recover(frame_idx, frame_bool_masks, frame_bgr=frame)

IMPORTANT: Always pass frame_bgr from the caller rather than relying on the
filename-based fallback. The fallback assumes contiguous 5-digit zero-padded
JPEG filenames ({:05d}.jpg) and will silently return {} on other naming schemes.
"""

import logging
import os
from collections import defaultdict, deque

import cv2
import numpy as np
from scipy.ndimage import label as ndlabel
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


class ShivaPixelPaintRecovery:
    """Detects lost masks and recovers them via background subtraction."""

    def __init__(self, frame_dir, n_animals, bg_sample_frames=300,
                 area_lower=0.4, min_fish_area=200, max_fish_area=20000,
                 n_frames=None, last_known_centroids=None,
                 bg_std_multiplier=2.5, bg_additive_offset=8.0):
        self.frame_dir = frame_dir
        self.n_animals = n_animals
        self.bg_sample_frames = bg_sample_frames
        self.area_lower = area_lower
        self.min_fish_area = min_fish_area
        self.max_fish_area = max_fish_area
        self.n_frames = n_frames
        self.bg_std_multiplier = bg_std_multiplier
        self.bg_additive_offset = bg_additive_offset
        self.median_areas = {}  # {oid: float}
        self.bg_median = None
        self.bg_std = None
        # defaultdict so arbitrary object IDs (not just 0..N-1) work without KeyError
        self._area_history = defaultdict(lambda: deque(maxlen=500))
        self._clean_count = 0
        self.recovery_log = []  # [{"frame": int, "oid": int, "blob_area": int, "cost": float}]
        self._max_log_entries = 10000
        # Last known centroid per object — used for spatial matching in recovery
        self.last_known_centroids = last_known_centroids or {}

    def build_background_model(self):
        """Build median + std background from sampled frames."""
        # Sort numerically — extract digits from filename, fall back to lexicographic
        import re
        def _numeric_key(f):
            digits = re.findall(r'\d+', os.path.splitext(f)[0])
            return int(digits[-1]) if digits else 0

        frame_files = sorted(
            (f for f in os.listdir(self.frame_dir)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))),
            key=_numeric_key,
        )
        n_total = self.n_frames if self.n_frames else len(frame_files)
        indices = np.linspace(
            0, min(n_total, len(frame_files)) - 1,
            min(self.bg_sample_frames, len(frame_files)),
            dtype=int,
        )
        samples = []
        for idx in indices:
            frame = cv2.imread(
                os.path.join(self.frame_dir, frame_files[idx]),
                cv2.IMREAD_GRAYSCALE,
            )
            if frame is not None:
                samples.append(frame.astype(np.float32))
        if not samples:
            raise RuntimeError(
                f"Failed to load any frames from {self.frame_dir} for background model"
            )
        stack = np.stack(samples)
        self.bg_median = np.median(stack, axis=0)
        self.bg_std = np.maximum(np.std(stack, axis=0), 1.0)

    def set_initial_areas(self, oid_to_area):
        """Seed area estimates from frame-0 detections (e.g., bounding box areas).

        Gives the outlier filter a reference point from the very first frame,
        preventing contamination during the bootstrap window before enough
        samples have been collected for a reliable median.
        """
        for oid, area in oid_to_area.items():
            self.median_areas[oid] = float(area)

    def update_median_areas(self, frame_masks):
        """Feed frame masks to build per-fish running median area.

        Computes per-fish independently (no all-present gate),
        recomputes periodically (not frozen), filters outliers.

        Args:
            frame_masks: {oid: bool_mask} for this frame
        """
        for oid, mask in frame_masks.items():
            area = int(mask.sum())
            if area < self.min_fish_area:
                continue
            history = self._area_history[oid]

            # Outlier filter: reject if area changes too much vs running median.
            # Tighter bounds during bootstrap (first 100 samples) to prevent
            # early identity swaps from contaminating the median.
            if history and self.median_areas.get(oid):
                ratio = area / self.median_areas[oid]
                if len(history) < 100:
                    # Strict during bootstrap
                    if ratio > 1.5 or ratio < 0.5:
                        continue
                else:
                    # Relaxed after median stabilizes
                    if ratio > 2.5 or ratio < 0.25:
                        continue

            history.append(area)

            # Recompute median once we have enough samples, then periodically
            if len(history) >= 30 and (len(history) <= 100 or len(history) % 50 == 0):
                self.median_areas[oid] = float(np.median(history[-500:]))
                # deque(maxlen=500) handles trimming automatically

    def update_last_centroid(self, oid, cx, cy):
        """Update last known centroid for an object (for spatial matching)."""
        self.last_known_centroids[oid] = (cx, cy)

    def _get_not_water(self, gray):
        """Background subtraction -> binary not-water mask."""
        diff = np.abs(gray.astype(np.float32) - self.bg_median)
        not_water = diff > (self.bg_std * self.bg_std_multiplier + self.bg_additive_offset)
        mask_u8 = not_water.astype(np.uint8)
        closed = cv2.morphologyEx(
            mask_u8, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        opened = cv2.morphologyEx(
            closed, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        return opened.astype(bool)

    @staticmethod
    def complete_masks_with_foreground(frame_masks, not_water):
        """Fill unclaimed foreground pixels into the nearest mask via BFS.

        SAM3.1's masks often don't cover thin appendages like fins. These
        unclaimed foreground pixels cause centroid displacement when the
        nearest-centroid paint rule assigns them to the wrong fish.

        Algorithm: multi-source BFS from each mask's boundary pixels into
        unclaimed foreground. All masks expand simultaneously — each
        unclaimed pixel goes to whichever mask boundary is closest by
        geodesic distance along the foreground. This gives exact nearest-
        boundary assignment (more precise than iterative dilation).

        Args:
            frame_masks: {oid: bool_mask} — SAM3.1 output masks
            not_water: (H, W) bool — foreground from background subtraction

        Returns:
            {oid: bool_mask} — completed masks with fins filled in
        """
        from collections import deque

        if not frame_masks or not_water is None:
            return frame_masks

        oids = sorted(frame_masks.keys())
        if len(oids) < 1:
            return frame_masks

        H, W = not_water.shape
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # Build claimed mask (union of all fish masks)
        claimed = np.zeros((H, W), dtype=bool)
        for mask in frame_masks.values():
            claimed |= mask

        # Unclaimed foreground = fish body parts not covered by any mask
        unclaimed_fg = not_water & ~claimed

        if not unclaimed_fg.any():
            return frame_masks

        # BFS assignment map: which oid owns each unclaimed pixel (-1 = unassigned)
        assignment = np.full((H, W), -1, dtype=np.int32)
        bfs_queue = deque()

        # Seed BFS from each mask's boundary — the edge pixels adjacent to
        # unclaimed foreground
        oid_to_idx = {oid: i for i, oid in enumerate(oids)}
        for oid in oids:
            mask_u8 = frame_masks[oid].astype(np.uint8)
            dilated = cv2.dilate(mask_u8, kernel)
            # Boundary = pixels in the dilation that are unclaimed foreground
            boundary = (dilated > 0) & unclaimed_fg
            by, bx = np.where(boundary)
            idx = oid_to_idx[oid]
            for pi in range(len(bx)):
                px, py = int(bx[pi]), int(by[pi])
                if assignment[py, px] == -1:
                    assignment[py, px] = idx
                    bfs_queue.append((px, py, idx))

        # BFS flood — all sources expand simultaneously (FIFO = breadth-first)
        while bfs_queue:
            x, y, idx = bfs_queue.popleft()
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < W and 0 <= ny < H
                        and unclaimed_fg[ny, nx]
                        and assignment[ny, nx] == -1):
                    assignment[ny, nx] = idx
                    bfs_queue.append((nx, ny, idx))

        # Build completed masks
        completed = {}
        for oid in oids:
            idx = oid_to_idx[oid]
            completed[oid] = frame_masks[oid] | (assignment == idx)

        return completed

    @staticmethod
    def _largest_component_centroid(mask):
        """Centroid of the largest connected component in a bool mask.

        When non-overlap constraints carve a mask into disconnected
        fragments, the center-of-mass of all fragments may fall on water.
        Use the largest fragment's centroid instead.
        """
        labeled, n_comp = ndlabel(mask)
        if n_comp <= 1:
            ys, xs = np.where(mask)
            if len(xs) == 0:
                return None, None
            return float(xs.mean()), float(ys.mean())

        best_area = 0
        best_cx, best_cy = None, None
        for i in range(1, n_comp + 1):
            comp = labeled == i
            area = int(comp.sum())
            if area > best_area:
                best_area = area
                ys, xs = np.where(comp)
                best_cx, best_cy = float(xs.mean()), float(ys.mean())
        return best_cx, best_cy

    def check_and_recover(self, frame_idx, frame_masks, frame_bgr=None):
        """Check for lost masks and return recovery masks.

        Uses Hungarian matching (T3 fix) on a cost matrix combining area
        distance and centroid distance to last-known position.

        Args:
            frame_idx: current frame index
            frame_masks: {oid: bool_mask} from tracker
            frame_bgr: BGR frame for pixel-paint. ALWAYS pass this from the
                caller to avoid synchronous disk I/O (T9) and filename format
                assumptions. The None fallback is for backwards compat only.

        Returns:
            {oid: bool_mask} for fish needing recovery. Empty if all healthy.
        """
        if not self.median_areas:
            return {}
        if self.bg_median is None:
            return {}

        # Find missing/artifact masks — only consider OIDs that appear in
        # both median_areas AND the current frame's expected set. Objects
        # that were removed from tracking should not trigger recovery.
        known_oids = sorted(
            oid for oid in self.median_areas.keys()
            if oid < self.n_animals  # only expected animals
        )
        missing_oids = []
        healthy_masks = {}
        for oid in known_oids:
            if oid not in frame_masks:
                missing_oids.append(oid)
                continue
            area = int(frame_masks[oid].sum())
            if area < self.median_areas[oid] * self.area_lower:
                missing_oids.append(oid)
            else:
                healthy_masks[oid] = frame_masks[oid]

        if not missing_oids:
            return {}

        # Load frame — prefer caller-provided to avoid sync disk I/O
        if frame_bgr is None:
            logger.warning(
                "frame_bgr=None at frame %d — falling back to disk read "
                "(assumes {:05d}.jpg naming). Pass frame_bgr from caller.",
                frame_idx,
            )
            frame_bgr = cv2.imread(
                os.path.join(self.frame_dir, f"{frame_idx:05d}.jpg")
            )
        if frame_bgr is None:
            return {}

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Verify resolution matches background model
        if gray.shape != self.bg_median.shape:
            logger.warning(
                "Frame resolution %s != background model %s — resizing frame",
                gray.shape, self.bg_median.shape,
            )
            gray = cv2.resize(gray, (self.bg_median.shape[1], self.bg_median.shape[0]))

        # Background subtraction
        not_water = self._get_not_water(gray)

        # Guard against stale background model — if most of the frame is
        # classified as foreground, the model has drifted and blob detection
        # would produce false recovery masks
        fg_fraction = not_water.sum() / not_water.size
        if fg_fraction > 0.5:
            logger.debug(
                "Frame %d: %.0f%% foreground — background model may be stale, "
                "skipping recovery", frame_idx, fg_fraction * 100,
            )
            return {}

        # Subtract healthy masks (with dilation to avoid edge noise)
        unclaimed = not_water.copy()
        for oid, mask in healthy_masks.items():
            unclaimed[mask] = False
            dilated = cv2.dilate(
                mask.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            unclaimed[dilated > 0] = False

        # Find unclaimed blobs — use per-fish area as lower bound
        labeled, n_comp = ndlabel(unclaimed)
        min_blob_area = self.min_fish_area // 2
        if self.median_areas:
            min_blob_area = max(
                min_blob_area,
                int(min(self.median_areas.values()) * self.area_lower * 0.5),
            )

        blobs = []
        for i in range(1, n_comp + 1):
            bm = labeled == i
            area = int(bm.sum())
            if min_blob_area <= area <= self.max_fish_area:
                ys, xs = np.where(bm)
                blobs.append({
                    "mask": bm,
                    "area": area,
                    "cx": float(xs.mean()),
                    "cy": float(ys.mean()),
                })

        if not blobs:
            return {}

        # --- T3 fix: Hungarian matching instead of greedy ---
        n_missing = len(missing_oids)
        n_blobs = len(blobs)
        cost = np.full((n_missing, n_blobs), 1e6, dtype=np.float64)

        for mi, oid in enumerate(missing_oids):
            median = self.median_areas.get(oid, 1000)
            last_cent = self.last_known_centroids.get(oid)
            for bi, blob in enumerate(blobs):
                # Area distance (normalized by median)
                area_dist = abs(blob["area"] - median) / max(median, 1)
                # Spatial distance (if last centroid known)
                if last_cent is not None:
                    spatial_dist = np.sqrt(
                        (blob["cx"] - last_cent[0]) ** 2
                        + (blob["cy"] - last_cent[1]) ** 2
                    )
                    # Normalize by frame diagonal so matching is resolution-invariant
                    if self.bg_median is not None:
                        frame_diag = np.sqrt(self.bg_median.shape[0]**2 + self.bg_median.shape[1]**2)
                    else:
                        frame_diag = 1000.0
                    spatial_dist /= (frame_diag / 2.0)
                else:
                    spatial_dist = 0.0
                cost[mi, bi] = area_dist + spatial_dist

        # Guard against NaN/Inf from degenerate area/spatial values
        if not np.isfinite(cost).all():
            cost = np.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=0.0)

        row_ind, col_ind = linear_sum_assignment(cost)

        # Reject matches above a maximum cost threshold
        max_cost = 3.0  # area ratio > 3x or centroid > 1500px away
        recoveries = {}
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < max_cost:
                oid = missing_oids[r]
                blob = blobs[c]
                recoveries[oid] = blob["mask"]
                # Log with cost for provenance (caller sets was_applied)
                self.recovery_log.append({
                    "frame": frame_idx,
                    "oid": oid,
                    "blob_area": blob["area"],
                    "cost": float(cost[r, c]),
                    "was_applied": None,  # caller should set True/False
                })

        if len(self.recovery_log) > self._max_log_entries:
            self.recovery_log = self.recovery_log[-self._max_log_entries:]

        return recoveries
