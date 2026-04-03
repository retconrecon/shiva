"""SHIVA Confidence-Triggered Mid-Tracking Mask Injection.

Monitors per-object confidence (object_score_logits) during tracking.
When confidence drops below threshold, runs pixel-paint to find the
real fish body and injects the corrected mask via add_new_masks().

The injected mask is stored in non_cond_frame_outputs but protected from
pruning via _shiva_protected_frames. Memory features from the original
entry are preserved so the next frame's attention doesn't crash.

Usage:
    from sam3.model.shiva_confidence_injection import ShivaConfidenceInjector

    injector = ShivaConfidenceInjector(predictor, session_id, pixel_paint, ...)
    # ... inside tracking loop, after each yield ...
    injector.check_and_inject(frame_idx, frame_bool, id_mapping)
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ShivaConfidenceInjector:
    """Monitors per-object confidence and injects pixel-paint masks when low.

    Parameters
    ----------
    predictor : Sam3BasePredictor
        The SAM3.1 predictor with handle_request/handle_stream_request API.
    session_id : str
        Active tracking session ID.
    pixel_paint : ShivaPixelPaintRecovery
        Pixel-paint recovery instance (for bg model and blob detection).
    confidence_threshold : float
        Sigmoid confidence below which injection triggers (default 0.3).
    cooldown_frames : int
        Minimum frames between injections for the same object.
    """

    def __init__(self, predictor, session_id, pixel_paint,
                 confidence_threshold=0.3, cooldown_frames=16):
        self.predictor = predictor
        self.session_id = session_id
        self.pixel_paint = pixel_paint
        self.confidence_threshold = confidence_threshold
        self.cooldown_frames = cooldown_frames

        session = predictor._all_inference_states.get(session_id)
        if session is None:
            raise RuntimeError(f"Session {session_id} not found")
        self._inference_state = session["state"]

        # P0-3: verify add_new_masks exists on the model
        if not hasattr(predictor.model, 'add_new_masks'):
            raise NotImplementedError(
                "Confidence injection requires VideoTrackingMultiplexDemo. "
                "Model class %s does not support add_new_masks()."
                % type(predictor.model).__name__
            )

        self._last_injection_frame = {}  # {oid: frame_idx}
        self.injection_log = []  # [(frame_idx, oid, confidence, success)]

    def check_and_inject(self, frame_idx, frame_bool, id_mapping=None,
                         frame_bgr=None, already_claimed=None):
        """Check per-object confidence and inject pixel-paint masks if low.

        When multiple fish have low confidence simultaneously, all are collected
        first, then pixel-paint's check_and_recover (with Hungarian matching)
        finds recovery masks for all of them in one pass — no greedy ordering
        bias, no duplicated blob detection pipeline.

        If pixel-paint already recovered a fish (via already_claimed), that
        mask is reused for injection instead of re-running blob detection.

        Args:
            frame_idx: current frame index
            frame_bool: {oid: bool_mask} from tracker output
            id_mapping: {sam3_obj_id: animal_oid} mapping (or None for identity)
            frame_bgr: BGR frame for pixel-paint (optional)
            already_claimed: {oid: bool_mask} from pixel-paint recovery

        Returns:
            list of (oid, confidence) tuples for injected objects
        """
        if self.pixel_paint is None or self.pixel_paint.bg_median is None:
            return []

        # --- Phase 1: Identify all low-confidence objects ---
        low_conf_objects = []  # [(obj_idx, sam3_obj_id, oid, conf)]

        # Read live mapping from inference_state (refreshed by SAM3.1 on
        # object add/remove, so always current)
        idx_to_id = self._inference_state.get("obj_idx_to_id", {})

        for obj_idx in self._inference_state.get("output_dict_per_obj", {}):
            # Validate mapping is still valid for this obj_idx
            if obj_idx not in idx_to_id:
                continue
            non_cond = self._inference_state["output_dict_per_obj"][obj_idx].get(
                "non_cond_frame_outputs", {}
            )
            if frame_idx not in non_cond:
                continue

            entry = non_cond[frame_idx]
            logits = entry.get("object_score_logits")
            if logits is None:
                continue

            # Extract scalar confidence
            if hasattr(logits, 'sigmoid'):
                conf = float(logits.sigmoid().mean().item())
            elif hasattr(logits, 'item'):
                conf = 1.0 / (1.0 + np.exp(-float(logits.item())))
            else:
                conf = 1.0 / (1.0 + np.exp(-float(logits)))

            if conf >= self.confidence_threshold:
                continue

            # Map SAM3.1 obj_idx → user-facing obj_id → animal oid
            sam3_obj_id = idx_to_id.get(obj_idx, obj_idx)
            if id_mapping is not None:
                oid = id_mapping.get(sam3_obj_id, sam3_obj_id)
            else:
                oid = sam3_obj_id

            # Cooldown check
            last = self._last_injection_frame.get(oid, -999)
            if frame_idx - last < self.cooldown_frames:
                continue

            low_conf_objects.append((obj_idx, sam3_obj_id, oid, conf))

        if not low_conf_objects:
            return []

        # --- Phase 2: Get recovery masks ---
        # First, check if pixel-paint already recovered any of these fish
        already = already_claimed or {}
        need_recovery = []
        have_recovery = {}

        for obj_idx, sam3_obj_id, oid, conf in low_conf_objects:
            if oid in already:
                have_recovery[oid] = already[oid]
                logger.info(
                    "Confidence drop: fish %d frame %d conf=%.3f — "
                    "reusing pixel-paint recovery mask",
                    oid, frame_idx, conf,
                )
            else:
                need_recovery.append((obj_idx, sam3_obj_id, oid, conf))

        # For fish not already recovered, run pixel-paint with Hungarian matching
        if need_recovery and frame_bgr is not None:
            # Build a synthetic frame_masks dict where the low-conf fish are
            # marked as missing (empty mask) so check_and_recover finds them
            synthetic_masks = {}
            for oid, mask in frame_bool.items():
                needs_it = any(oid == nr[2] for nr in need_recovery)
                if needs_it:
                    # Mark as missing by giving it a zero-area mask
                    synthetic_masks[oid] = np.zeros_like(mask)
                else:
                    synthetic_masks[oid] = mask

            pp_recovery = self.pixel_paint.check_and_recover(
                frame_idx, synthetic_masks, frame_bgr=frame_bgr,
            )
            have_recovery.update(pp_recovery)

        # --- Phase 3: Inject recovered masks ---
        injected = []
        for obj_idx, sam3_obj_id, oid, conf in low_conf_objects:
            recovery_mask = have_recovery.get(oid)
            if recovery_mask is None:
                self.injection_log.append((frame_idx, oid, conf, False))
                continue

            success = self._inject_mask(frame_idx, sam3_obj_id, recovery_mask)
            self._last_injection_frame[oid] = frame_idx
            self.injection_log.append((frame_idx, oid, conf, success))

            if success:
                injected.append((oid, conf))
                logger.info(
                    "Injected recovery mask for fish %d at frame %d "
                    "(conf=%.3f, area=%d)",
                    oid, frame_idx, conf, int(recovery_mask.sum()),
                )

        return injected

    def _inject_mask(self, frame_idx, sam3_obj_id, mask_bool):
        """Inject a corrected mask into SAM3.1 via add_new_masks().

        add_new_masks() sets run_mem_encoder=False, so the new entry has
        maskmem_features=None. We preserve the original entry's memory
        features to prevent a crash on the next frame when
        _prepare_memory_conditioned_features tries to use them.
        """
        try:
            # Snapshot existing maskmem_features before injection overwrites them
            obj_idx_to_id = self._inference_state.get("obj_idx_to_id", {})
            id_to_idx = {v: k for k, v in obj_idx_to_id.items()}
            obj_idx = id_to_idx.get(sam3_obj_id)

            saved_features = {}
            if obj_idx is not None:
                per_obj = self._inference_state.get("output_dict_per_obj", {})
                if obj_idx in per_obj:
                    entry = per_obj[obj_idx].get("non_cond_frame_outputs", {}).get(frame_idx)
                    if entry is not None:
                        for key in ("maskmem_features", "maskmem_pos_enc"):
                            if key in entry and entry[key] is not None:
                                saved_features[key] = entry[key]

            mask_tensor = torch.from_numpy(
                mask_bool.astype(np.float32)
            ).unsqueeze(0)  # (1, H, W)

            self.predictor.model.add_new_masks(
                inference_state=self._inference_state,
                frame_idx=frame_idx,
                obj_ids=[sam3_obj_id],
                masks=mask_tensor,
            )

            # Restore maskmem_features in BOTH per-obj and consolidated dicts
            # so the next frame's _prepare_memory_conditioned_features doesn't
            # crash trying to access None
            if obj_idx is not None and saved_features:
                # Per-obj dict
                per_obj = self._inference_state.get("output_dict_per_obj", {})
                if obj_idx in per_obj:
                    entry = per_obj[obj_idx].get("non_cond_frame_outputs", {}).get(frame_idx)
                    if entry is not None:
                        for key, val in saved_features.items():
                            if key not in entry or entry[key] is None:
                                entry[key] = val

                # Consolidated dict (memory attention reads from here)
                main_dict = self._inference_state.get("output_dict", {})
                main_entry = main_dict.get("non_cond_frame_outputs", {}).get(frame_idx)
                if main_entry is not None:
                    for key, val in saved_features.items():
                        if key not in main_entry or main_entry[key] is None:
                            main_entry[key] = val

            # Protect this frame from pruning, with a cap to prevent
            # unbounded VRAM growth
            protected = self._inference_state.setdefault("_shiva_protected_frames", set())
            protected.add(frame_idx)
            _MAX_PROTECTED = 100
            if len(protected) > _MAX_PROTECTED:
                # Evict oldest protected frame
                oldest = min(protected)
                protected.discard(oldest)

            return True
        except Exception as e:
            logger.error("Mask injection failed at frame %d: %s", frame_idx, e)
            return False
