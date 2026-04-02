"""SHIVA Tracker — canonical integration of memory pruning + pixel-paint recovery.

Wraps SAM3.1's handle_stream_request with SHIVA's memory management and
mask recovery. This is the single integration point for all SHIVA modules
(prevents copy-paste of integration logic across experiment scripts).

Usage:
    from sam3.model_builder import build_sam3_predictor
    from sam3.model.shiva_tracker import ShivaTracker

    predictor = build_sam3_predictor(version="sam3.1", use_fa3=False)

    # Start session, add prompts...
    session_id = predictor.handle_request({
        "type": "start_session", "resource_path": frame_dir
    })["session_id"]
    predictor.handle_request({
        "type": "add_prompt", "session_id": session_id,
        "frame_index": 0, "bounding_boxes": boxes, "bounding_box_labels": labels,
    })

    # Track with SHIVA
    shiva = ShivaTracker(predictor, session_id, frame_dir, n_animals=4)
    for frame_idx, outputs, recovery_masks in shiva.track(n_frames=5000):
        # outputs: SAM3.1 output dict (out_obj_ids, out_binary_masks, etc.)
        # recovery_masks: {oid: bool_mask} from pixel-paint (may be empty)
        pass
"""

import logging

import numpy as np
import torch

from sam3.model.shiva_memory_pruning import prune_output_dict
from sam3.model.shiva_pixel_paint import ShivaPixelPaintRecovery

logger = logging.getLogger(__name__)


class ShivaTracker:
    """Wraps SAM3.1 propagation with SHIVA memory pruning and mask recovery."""

    def __init__(self, predictor, session_id, frame_dir, n_animals,
                 max_recent_frames=500, max_landmark_frames=50,
                 pixel_paint_enabled=True, botsort_enabled=False,
                 n_frames=None):
        self.predictor = predictor
        self.session_id = session_id
        self.n_animals = n_animals
        self.max_recent_frames = max_recent_frames
        self.max_landmark_frames = max_landmark_frames
        self.pixel_paint_enabled = pixel_paint_enabled
        self.botsort_enabled = botsort_enabled

        # Access inference_state through predictor internals
        session = predictor._all_inference_states.get(session_id)
        if session is None:
            raise RuntimeError(f"Session {session_id} not found in predictor")
        self._inference_state = session["state"]

        # Initialize pixel-paint recovery
        self.pixel_paint = None
        if pixel_paint_enabled:
            self.pixel_paint = ShivaPixelPaintRecovery(
                frame_dir, n_animals, n_frames=n_frames,
            )
            self.pixel_paint.build_background_model()

        # Initialize BoT-SORT association
        self._model = predictor.model if hasattr(predictor, 'model') else None
        if botsort_enabled:
            from sam3.model.shiva_appearance import ShivaAppearanceStore
            self._appearance_store = ShivaAppearanceStore(n_bins=32)
            if self._model is not None:
                self._model.use_botsort_association = True
                self._model._shiva_appearance_store = self._appearance_store
                logger.info("BoT-SORT association enabled on model")
            else:
                logger.warning("Could not access predictor.model for BoT-SORT")

        self.prune_stats = []
        self._min_healthy_area = 100  # minimum mask area to count as "healthy"

    def track(self, n_frames=None, propagation_direction="forward"):
        """Run SAM3.1 propagation with SHIVA hooks.

        Yields:
            (frame_idx, outputs, recovery_masks) tuples.
            - outputs: SAM3.1 output dict
            - recovery_masks: {oid: bool_mask} from pixel-paint, empty if healthy
        """
        request = {
            "type": "propagate_in_video",
            "session_id": self.session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": 0,
        }
        if n_frames is not None:
            request["max_frame_num_to_track"] = n_frames

        # Wrap generator so exceptions trigger cleanup — an aborted run
        # would otherwise leave inference_state dirty.
        _seeded_areas = False
        gen = self.predictor.handle_stream_request(request)
        try:
            for result in gen:
                frame_idx = result["frame_index"]
                outputs = result.get("outputs", {})

                # Extract bool masks for pixel-paint
                obj_ids = outputs.get("out_obj_ids", [])
                raw_masks = outputs.get("out_binary_masks", [])
                frame_bool = {}
                for i, oid in enumerate(obj_ids):
                    oid = int(oid)
                    m = raw_masks[i]
                    if hasattr(m, 'cpu'):
                        m = m.cpu().numpy()
                    m = m.squeeze().astype(bool)
                    frame_bool[oid] = m

                # Seed area fingerprints from first frame's masks so the
                # outlier filter has a reference point during bootstrap
                if not _seeded_areas and self.pixel_paint is not None and frame_bool:
                    initial = {
                        oid: int(m.sum()) for oid, m in frame_bool.items()
                        if int(m.sum()) > self.pixel_paint.min_fish_area
                    }
                    if initial:
                        self.pixel_paint.set_initial_areas(initial)
                        _seeded_areas = True

                # Update centroids only for healthy masks,
                # using largest connected component (not naive mean)
                if self.pixel_paint is not None:
                    for oid, m in frame_bool.items():
                        if int(m.sum()) <= 0:
                            continue
                        area = int(m.sum())
                        median = self.pixel_paint.median_areas.get(oid)
                        # Skip centroid update for artifact masks
                        if median is None or area >= median * self.pixel_paint.area_lower:
                            # Largest component centroid handles disconnected masks
                            cx, cy = ShivaPixelPaintRecovery._largest_component_centroid(m)
                            if cx is not None:
                                self.pixel_paint.update_last_centroid(oid, cx, cy)

                # Auto-update SENTINEL status for BoT-SORT adaptive gating
                if self.botsort_enabled and self._model is not None:
                    n_healthy = sum(
                        1 for m in frame_bool.values()
                        if int(m.sum()) > self._min_healthy_area
                    )
                    if n_healthy >= self.n_animals:
                        self._model._shiva_sentinel_status = "GREEN"
                    elif n_healthy >= self.n_animals - 1:
                        self._model._shiva_sentinel_status = "YELLOW"
                    else:
                        self._model._shiva_sentinel_status = "RED"

                # Memory pruning
                stats = prune_output_dict(
                    self._inference_state, frame_idx,
                    self.max_recent_frames, self.max_landmark_frames,
                )
                if stats is not None:
                    self.prune_stats.append((frame_idx, stats))

                # Load frame for pixel-paint recovery (avoid filename fallback)
                frame_bgr = None
                if self.pixel_paint is not None:
                    fc = self._inference_state.get("feature_cache")
                    if fc is not None and frame_idx in fc:
                        ft = fc[frame_idx][0]
                        if ft is not None:
                            ft = ft.cpu().float()
                            if ft.ndim == 3 and ft.shape[0] in (1, 3):
                                ft = ft.permute(1, 2, 0)
                            # Denormalize model tensor to uint8
                            ft = (ft * 0.5 + 0.5) * 255.0
                            frame_bgr = ft.clamp(0, 255).to(torch.uint8).numpy()

                # Pixel-paint recovery
                recovery_masks = {}
                if self.pixel_paint is not None:
                    self.pixel_paint.update_median_areas(frame_bool)
                    recovery_masks = self.pixel_paint.check_and_recover(
                        frame_idx, frame_bool, frame_bgr=frame_bgr,
                    )

                yield frame_idx, outputs, recovery_masks
        except Exception:
            # Close the generator to trigger its finally blocks, then
            # reset session so re-propagation starts from clean state.
            gen.close()
            try:
                self.predictor.handle_request({
                    "type": "reset_session",
                    "session_id": self.session_id,
                })
            except Exception:
                pass
            raise
        finally:
            gen.close()  # Idempotent — ensures deterministic cleanup on normal exit too
