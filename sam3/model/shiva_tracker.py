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
    shiva = ShivaTracker(predictor, session_id, frame_dir, n_animals=4,
                         identity_verification=True)
    for frame_idx, outputs, recovery_masks, swap_events in shiva.track(n_frames=5000):
        # outputs: SAM3.1 output dict
        # recovery_masks: {oid: bool_mask} from pixel-paint (may be empty)
        # swap_events: list of SwapEvent from identity verifier (may be empty)
        pass
"""

import logging

import cv2
import numpy as np
import torch

from sam3.model.shiva_memory_pruning import prune_output_dict
from sam3.model.shiva_pixel_paint import ShivaPixelPaintRecovery

# SAM3.1 feature cache normalization: (pixel / 255 - 0.5) / 0.5
# Reverse: (tensor * 0.5 + 0.5) * 255
# Defined here: sam3/model/sam3_multiplex_base.py run_backbone_and_detection
_DENORM_MEAN = 0.5
_DENORM_STD = 0.5


def denormalize_feature_cache_to_bgr(tensor):
    """Convert SAM3.1 model-input tensor to uint8 BGR numpy array.

    The model normalizes with mean=0.5, std=0.5:
        tensor = (pixel / 255.0 - mean) / std
    This reverses it:
        pixel = (tensor * std + mean) * 255.0
    """
    ft = tensor.cpu().float()
    if ft.ndim == 3 and ft.shape[0] in (1, 3):
        ft = ft.permute(1, 2, 0)  # C,H,W -> H,W,C
    ft = (ft * _DENORM_STD + _DENORM_MEAN) * 255.0
    return ft.clamp(0, 255).to(torch.uint8).numpy()

logger = logging.getLogger(__name__)


class ShivaTracker:
    """Wraps SAM3.1 propagation with SHIVA memory pruning and mask recovery.

    Supports context manager for automatic session cleanup:
        with ShivaTracker(predictor, session_id, ...) as shiva:
            for frame_idx, outputs, recovery, swaps in shiva.track():
                ...
    """

    def __enter__(self):
        return self

    # Attributes set on the model that must be cleaned up between sessions
    _SHIVA_MODEL_ATTRS = [
        '_shiva_sentinel_status', '_shiva_crossing_active',
        'use_botsort_association', '_shiva_appearance_store',
        '_shiva_frame_pixels', '_prev_non_overlap_assignment',
        '_prev_non_overlap_batch_size',
        '_shiva_diag_logged', '_shiva_nonoverlap_diag_logged',
        'occlusion_memory_freeze', 'occlusion_freeze_threshold',
        'temporal_boundary_prior', '_tiab_module',
        '_tiab_appearance_embs', '_tiab_centroid_history',
        '_tiab_expected_B', '_tiab_prior_warned',
        # Extraction callback + legacy hook attrs
        '_tiab_extract_callback',
        '_tiab_extractor', '_tiab_extract_buffer', '_tiab_original_encode',
    ]

    def __exit__(self, *exc):
        # Clean up model attributes to prevent bleed into next session
        if self._model is not None:
            for attr in self._SHIVA_MODEL_ATTRS:
                if hasattr(self._model, attr):
                    delattr(self._model, attr)
                # Also clean inner model
                _inner = getattr(self._model, 'tracker', None)
                if _inner is not None:
                    _inner = getattr(_inner, 'model', _inner)
                    if hasattr(_inner, attr):
                        delattr(_inner, attr)
        try:
            self.predictor.handle_request({
                "type": "close_session",
                "session_id": self.session_id,
            })
        except Exception as e:
            logger.debug("Session close failed: %s", e)

    def __init__(self, predictor, session_id, frame_dir, n_animals,
                 max_recent_frames=500, max_landmark_frames=50,
                 pixel_paint_enabled=True, botsort_enabled=False,
                 identity_verification=False, appearance_backend="histogram",
                 confidence_injection=False, confidence_threshold=0.3,
                 occlusion_memory_freeze=False, occlusion_freeze_threshold=0.5,
                 temporal_boundary_prior=0.0,
                 tiab_enabled=False, tiab_checkpoint=None,
                 tiab_appearance_dim=512, tiab_trajectory_len=16,
                 n_frames=None):
        self.predictor = predictor
        self.session_id = session_id
        self.n_animals = n_animals
        self.frame_dir = frame_dir
        self.max_recent_frames = max_recent_frames
        self.max_landmark_frames = max_landmark_frames
        self.pixel_paint_enabled = pixel_paint_enabled
        self.botsort_enabled = botsort_enabled
        self.occlusion_memory_freeze = occlusion_memory_freeze
        self.occlusion_freeze_threshold = occlusion_freeze_threshold
        self.temporal_boundary_prior = temporal_boundary_prior
        self.identity_verification = identity_verification

        # Access inference_state through predictor internals
        session = predictor._all_inference_states.get(session_id)
        if session is None:
            raise RuntimeError(f"Session {session_id} not found in predictor")
        self._inference_state = session["state"]

        # Reset SENTINEL status from any previous session and set memory freeze flags
        self._model = predictor.model if hasattr(predictor, 'model') else None
        if self._model is not None:
            # Clean stale SHIVA attrs from BOTH outer and inner model
            # (mirrors __exit__ cleanup for non-context-manager usage)
            _inner = self._model
            if hasattr(self._model, 'tracker') and hasattr(self._model.tracker, 'model'):
                _inner = self._model.tracker.model
            for attr in self._SHIVA_MODEL_ATTRS:
                if hasattr(self._model, attr):
                    delattr(self._model, attr)
                if _inner is not self._model and hasattr(_inner, attr):
                    delattr(_inner, attr)
            # Set fresh state
            self._model._shiva_sentinel_status = "GREEN"
            self._model._shiva_crossing_active = False
            _inner.occlusion_memory_freeze = occlusion_memory_freeze
            _inner.occlusion_freeze_threshold = occlusion_freeze_threshold
            _inner.temporal_boundary_prior = temporal_boundary_prior
            if occlusion_memory_freeze:
                logger.info("Occlusion memory freeze enabled on %s (threshold=%.2f)",
                            type(_inner).__name__, occlusion_freeze_threshold)
            if temporal_boundary_prior > 0:
                logger.info("Temporal boundary prior enabled on %s (alpha=%.1f)",
                            type(_inner).__name__, temporal_boundary_prior)

        # Initialize pixel-paint recovery
        self.pixel_paint = None
        if pixel_paint_enabled:
            self.pixel_paint = ShivaPixelPaintRecovery(
                frame_dir, n_animals, n_frames=n_frames,
            )
            self.pixel_paint.build_background_model()

        # Initialize BoT-SORT association
        self.appearance_backend = appearance_backend
        if botsort_enabled:
            if appearance_backend == "osnet":
                from sam3.model.shiva_appearance_osnet import ShivaOSNetAppearanceStore
                self._appearance_store = ShivaOSNetAppearanceStore(ema_alpha=0.9)
                logger.info("Appearance backend: OSNet-AIN (512-dim)")
            else:
                from sam3.model.shiva_appearance import ShivaAppearanceStore, SHIVA_HISTOGRAM_BINS
                self._appearance_store = ShivaAppearanceStore(n_bins=SHIVA_HISTOGRAM_BINS)
                logger.info("Appearance backend: histogram (%d-bin)", SHIVA_HISTOGRAM_BINS)
            if self._model is not None:
                self._model.use_botsort_association = True
                self._model._shiva_appearance_store = self._appearance_store
                logger.info("BoT-SORT association enabled on model")
            else:
                logger.warning("Could not access predictor.model for BoT-SORT")

        # Initialize identity verifier — pass appearance store so it uses
        # the same backend (OSNet or histogram) as BoT-SORT
        self.verifier = None
        if identity_verification:
            from sam3.model.shiva_identity_verifier import ShivaIdentityVerifier
            _store = getattr(self, '_appearance_store', None)
            self.verifier = ShivaIdentityVerifier(
                n_objects=n_animals, appearance_store=_store,
            )
            logger.info("Identity verification enabled (backend: %s)",
                        getattr(_store, 'distance_metric', 'histogram'))

        # Initialize confidence-triggered mask injection
        self.injector = None
        if confidence_injection and self.pixel_paint is not None:
            from sam3.model.shiva_confidence_injection import ShivaConfidenceInjector
            self.injector = ShivaConfidenceInjector(
                predictor, session_id, self.pixel_paint,
                confidence_threshold=confidence_threshold,
            )
            logger.info("Confidence injection enabled (threshold=%.2f)", confidence_threshold)

        # Initialize TIAB boundary refinement module
        self.tiab_enabled = tiab_enabled
        self._tiab_trajectory_len = tiab_trajectory_len
        self._tiab_appearance_dim = tiab_appearance_dim
        self._centroid_history = {}  # {oid: deque of (cx_norm, cy_norm)}
        if tiab_enabled:
            from sam3.model.tiab import TemporalIdentityBoundaryModule
            tiab_module = TemporalIdentityBoundaryModule(
                appearance_dim=tiab_appearance_dim,
                trajectory_len=tiab_trajectory_len,
            )
            if tiab_checkpoint is not None:
                ckpt = torch.load(tiab_checkpoint, map_location="cpu", weights_only=True)
                state_dict = ckpt.get("model_state_dict", ckpt)
                tiab_module.load_state_dict(state_dict)
                logger.info("TIAB weights loaded from %s", tiab_checkpoint)
            tiab_module = tiab_module.cuda().eval()
            # Attach to inner model where _encode_new_memory reads it
            _inner = self._model
            if hasattr(self._model, 'tracker') and hasattr(self._model.tracker, 'model'):
                _inner = self._model.tracker.model
            _inner._tiab_module = tiab_module
            logger.info("TIAB boundary refinement enabled (%d params)",
                        sum(p.numel() for p in tiab_module.parameters()))

        self.prune_stats = []
        self._applied_swaps = set()  # deduplication for apply_swap

        # Derive healthy area threshold from pixel-paint so SENTINEL and
        # pixel-paint agree on what constitutes a healthy mask
        if self.pixel_paint is not None:
            self._min_healthy_area = self.pixel_paint.min_fish_area
        else:
            self._min_healthy_area = 200

        # Validate denormalization formula against disk-loaded frame 0
        self._validate_denormalization(frame_dir)

    def _validate_denormalization(self, frame_dir):
        """One-time check that feature_cache denormalization produces valid pixels."""
        fc = self._inference_state.get("feature_cache")
        if fc is None or 0 not in fc:
            return
        ft = fc[0][0]
        if ft is None:
            return
        bgr = denormalize_feature_cache_to_bgr(ft)
        vmin, vmax = float(bgr.min()), float(bgr.max())
        if vmin < 0 or vmax > 255:
            logger.error(
                "Feature cache denormalization out of range [%.1f, %.1f] — "
                "pixel-paint and identity verification will be unreliable. "
                "The model may use different normalization than mean=%.1f, std=%.1f.",
                vmin, vmax, _DENORM_MEAN, _DENORM_STD,
            )

    def track(self, n_frames=None, propagation_direction="forward"):
        """Run SAM3.1 propagation with SHIVA hooks.

        Yields:
            (frame_idx, outputs, recovery_masks, swap_events) tuples.
            - outputs: SAM3.1 output dict
            - recovery_masks: {oid: bool_mask} from pixel-paint, empty if healthy
            - swap_events: list of SwapEvent from identity verifier, empty if none
        """
        # Sync memory freeze flags to the INNER tracker model
        if self._model is not None:
            _inner = self._model
            if hasattr(self._model, 'tracker') and hasattr(self._model.tracker, 'model'):
                _inner = self._model.tracker.model
            for attr in ('occlusion_memory_freeze', 'occlusion_freeze_threshold',
                        'temporal_boundary_prior'):
                if hasattr(self, attr):
                    setattr(_inner, attr, getattr(self, attr))

        request = {
            "type": "propagate_in_video",
            "session_id": self.session_id,
            "propagation_direction": propagation_direction,
            "start_frame_index": 0,
        }
        if n_frames is not None:
            request["max_frame_num_to_track"] = n_frames

        # Initialize TIAB attributes with zeros for frame 0.
        # Frame 0's _encode_new_memory runs before we see any output,
        # so we need placeholder attributes. Real values are set after
        # each frame via _update_tiab_attributes.
        if self.tiab_enabled and self._model is not None:
            _inner = self._model
            if hasattr(self._model, 'tracker') and hasattr(self._model.tracker, 'model'):
                _inner = self._model.tracker.model
            device = next(_inner._tiab_module.parameters()).device
            _inner._tiab_appearance_embs = torch.zeros(
                self.n_animals, self._tiab_appearance_dim, device=device,
            )
            _inner._tiab_centroid_history = torch.full(
                (self.n_animals, self._tiab_trajectory_len, 2), 0.5, device=device,
            )

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

                # Get denormalized frame early — needed for mask completion
                frame_bgr = None
                if self._model is not None and hasattr(self._model, '_shiva_frame_pixels'):
                    frame_bgr = self._model._shiva_frame_pixels
                if frame_bgr is None:
                    fc = self._inference_state.get("feature_cache")
                    if fc is not None and frame_idx in fc:
                        ft = fc[frame_idx][0]
                        if ft is not None:
                            frame_bgr = denormalize_feature_cache_to_bgr(ft)

                # Seed area fingerprints from RAW masks (before BFS completion)
                # so the baseline isn't inflated by BFS-assigned appendages.
                # Delay seeding if any pair overlaps (crossing at frame 0 would
                # poison the seed with argmax-carved mask areas).
                if not _seeded_areas and self.pixel_paint is not None and frame_bool:
                    # Check that no pair has significant IoU (crossing)
                    _has_overlap = False
                    if len(frame_bool) >= 2:
                        _oids = sorted(frame_bool.keys())
                        for _i in range(len(_oids)):
                            for _j in range(_i + 1, len(_oids)):
                                _mi, _mj = frame_bool[_oids[_i]], frame_bool[_oids[_j]]
                                _inter = int((_mi & _mj).sum())
                                _union = int((_mi | _mj).sum())
                                if _union > 0 and _inter / _union > 0.05:
                                    _has_overlap = True
                                    break
                            if _has_overlap:
                                break
                    # Force-seed after 100 frames even with overlap — dense
                    # scenes may never have a clean frame, and no seeding
                    # disables recovery entirely.
                    if not _has_overlap or frame_idx >= 100:
                        initial = {
                            oid: int(m.sum()) for oid, m in frame_bool.items()
                            if int(m.sum()) > self.pixel_paint.min_fish_area
                        }
                        if initial:
                            self.pixel_paint.set_initial_areas(initial)
                            _seeded_areas = True
                            if _has_overlap and frame_idx >= 100:
                                logger.warning(
                                    "Force-seeded area fingerprints at frame %d "
                                    "despite overlap (no clean frame in first 100)",
                                    frame_idx,
                                )

                # Update median areas from raw masks (before BFS completion)
                if self.pixel_paint is not None and frame_bool:
                    self.pixel_paint.update_median_areas(frame_bool)

                # Complete masks by filling unclaimed foreground (fins, appendages)
                # into the nearest mask. This prevents centroid displacement from
                # unclaimed body parts and eliminates fin-color flashing.
                if self.pixel_paint is not None and frame_bgr is not None and frame_bool:
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    not_water = self.pixel_paint._get_not_water(gray)
                    # Resize not_water to match mask resolution if needed
                    # (bg model is built at disk resolution, masks are at SAM3.1 resolution)
                    _sample_mask = next(iter(frame_bool.values()))
                    if not_water.shape != _sample_mask.shape:
                        not_water = cv2.resize(
                            not_water.astype(np.uint8),
                            (_sample_mask.shape[1], _sample_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    frame_bool = ShivaPixelPaintRecovery.complete_masks_with_foreground(
                        frame_bool, not_water,
                    )

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
                # Uses per-object adaptive threshold from pixel-paint so SENTINEL
                # agrees with pixel-paint on what constitutes a healthy mask
                if self.botsort_enabled and self._model is not None:
                    n_healthy = 0
                    for oid, m in frame_bool.items():
                        area = int(m.sum())
                        if self.pixel_paint is not None:
                            median = self.pixel_paint.median_areas.get(oid)
                            thresh = max(
                                self._min_healthy_area,
                                int(median * self.pixel_paint.area_lower) if median else 0,
                            )
                        else:
                            thresh = self._min_healthy_area
                        if area > thresh:
                            n_healthy += 1
                    # Identity signal: check if any mask's appearance is
                    # closer to a different object's store than its own.
                    # This catches silent identity swaps that area checks miss.
                    identity_mismatch = False
                    if getattr(self, '_appearance_store', None) is not None and len(frame_bool) >= 2 and frame_bgr is not None:
                        from sam3.model.shiva_appearance import histogram_intersection_distance
                        _dist_fn = histogram_intersection_distance
                        if getattr(self._appearance_store, 'distance_metric', '') == 'cosine':
                            _dist_fn = lambda a, b: 1.0 - float(np.dot(a, b))
                        oids_list = sorted(frame_bool.keys())
                        for oid in oids_list:
                            own_emb = self._appearance_store.embeddings.get(oid)
                            if own_emb is None:
                                continue
                            cur_emb = self._appearance_store.extract_embedding(
                                frame_bool[oid], frame_bgr,
                            )
                            if cur_emb is None:
                                continue
                            own_dist = _dist_fn(cur_emb, own_emb)
                            for other_oid in oids_list:
                                if other_oid == oid:
                                    continue
                                other_emb = self._appearance_store.embeddings.get(other_oid)
                                if other_emb is None:
                                    continue
                                other_dist = _dist_fn(cur_emb, other_emb)
                                # Relative check: mismatch if closer to any
                                # other object than to own. No absolute margin
                                # — works for same-species where all distances
                                # are close but relative ordering still matters.
                                if other_dist < own_dist:
                                    identity_mismatch = True
                                    break
                            if identity_mismatch:
                                break

                    if n_healthy >= self.n_animals and not identity_mismatch:
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

                # frame_bgr already computed above (before mask completion)

                # Pixel-paint recovery
                recovery_masks = {}
                if self.pixel_paint is not None:
                    # update_median_areas already called above on raw masks
                    recovery_masks = self.pixel_paint.check_and_recover(
                        frame_idx, frame_bool, frame_bgr=frame_bgr,
                    )
                    # Mark yielded recoveries as applied in the log
                    if recovery_masks and self.pixel_paint.recovery_log:
                        for entry in self.pixel_paint.recovery_log[-len(recovery_masks):]:
                            if entry.get("frame") == frame_idx:
                                entry["was_applied"] = True

                    # Update centroids from recovery blobs so spatial matching
                    # stays current during multi-frame loss events
                    for oid, rmask in recovery_masks.items():
                        cx, cy = ShivaPixelPaintRecovery._largest_component_centroid(rmask)
                        if cx is not None:
                            self.pixel_paint.update_last_centroid(oid, cx, cy)

                    # Inject recovery masks into SAM3.1's memory so future frames
                    # attend to corrected masks, not the bad ones. Batched into a
                    # single add_new_masks + consolidate call to avoid O(N) encoder
                    # passes (4 simultaneous recoveries = 1 pass instead of 4).
                    if recovery_masks and self.injector is None:
                        _inner = self._model
                        if hasattr(self._model, 'tracker') and hasattr(self._model.tracker, 'model'):
                            _inner = self._model.tracker.model
                        if hasattr(_inner, 'add_new_masks'):
                            try:
                                all_oids = sorted(recovery_masks.keys())
                                all_masks = torch.stack([
                                    torch.from_numpy(recovery_masks[oid].astype(np.float32)).unsqueeze(0)
                                    for oid in all_oids
                                ]).to('cuda')
                                _inner.add_new_masks(
                                    inference_state=self._inference_state,
                                    frame_idx=frame_idx,
                                    obj_ids=all_oids,
                                    masks=all_masks,
                                )
                                _inner._consolidate_temp_output_across_obj(
                                    inference_state=self._inference_state,
                                    frame_idx=frame_idx,
                                    is_cond=False,
                                    run_mem_encoder=True,
                                )
                            except Exception as e:
                                logger.debug("Batched recovery injection failed: %s", e)

                # Confidence-triggered mid-tracking mask injection
                # Pass recovery_masks so injector doesn't claim blobs already
                # assigned by pixel-paint
                if self.injector is not None:
                    self.injector.check_and_inject(
                        frame_idx, frame_bool, frame_bgr=frame_bgr,
                        already_claimed=recovery_masks,
                    )

                # Identity verification — detect swaps after crossings
                # Filter out artifact masks to prevent false crossing events
                swap_events = []
                if self.verifier is not None and frame_bool:
                    healthy_for_verifier = {}
                    for oid, m in frame_bool.items():
                        area = int(m.sum())
                        if self.pixel_paint is not None:
                            median = self.pixel_paint.median_areas.get(oid)
                            thresh = median * self.pixel_paint.area_lower if median else self._min_healthy_area
                        else:
                            thresh = self._min_healthy_area
                        if area >= thresh:
                            healthy_for_verifier[oid] = m
                    if healthy_for_verifier:
                        pairwise_ious = self._compute_pairwise_ious(healthy_for_verifier)
                        swap_events = self.verifier.update(
                            frame_idx, healthy_for_verifier, pairwise_ious,
                            frame_bgr=frame_bgr,
                        )
                    # Signal to reconditioning gate whether any crossing is active
                    if self._model is not None:
                        states = self.verifier.get_crossing_states()
                        self._model._shiva_crossing_active = any(
                            s != "clear" for s in states.values()
                        )

                # Update TIAB per-frame attributes for the NEXT frame's
                # _encode_new_memory. Must happen after centroid extraction
                # (so history is current) and before yield (so attributes
                # are set when the generator advances to the next frame).
                if self.tiab_enabled and self._model is not None:
                    self._update_tiab_attributes(frame_bool, frame_bgr)

                yield frame_idx, outputs, recovery_masks, swap_events
        except Exception:
            # Close the generator to trigger its finally blocks, then
            # reset session so re-propagation starts from clean state.
            gen.close()
            try:
                self.predictor.handle_request({
                    "type": "reset_session",
                    "session_id": self.session_id,
                })
            except Exception as e:
                logger.debug("Session reset failed during error handling: %s", e)
            # Free zombie GPU tensors from generator locals (hotstart_buffer, etc.)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            raise
        finally:
            gen.close()  # Idempotent — ensures deterministic cleanup on normal exit too

    def apply_swap(self, oid_a, oid_b, frame_idx=None):
        """Atomically swap all internal state for two object IDs.

        Call this when the identity verifier detects a swap. Swaps:
        - pixel_paint.last_known_centroids
        - pixel_paint.median_areas
        - pixel_paint._area_history
        - appearance_store.embeddings (if BoT-SORT enabled)

        The consumer is still responsible for swapping its own output
        data structures (stored masks, CSV labels, etc.).

        Idempotent per (pair, frame): prevents accidental double-swap
        within the same crossing event, but allows the same pair to
        swap again in a later crossing.
        """
        key = (min(oid_a, oid_b), max(oid_a, oid_b), frame_idx)
        if key in self._applied_swaps:
            logger.warning("Ignoring duplicate apply_swap(%d, %d) at frame %s",
                           oid_a, oid_b, frame_idx)
            return
        self._applied_swaps.add(key)

        if self.pixel_paint is not None:
            pp = self.pixel_paint
            # Swap centroids
            c_a = pp.last_known_centroids.get(oid_a)
            c_b = pp.last_known_centroids.get(oid_b)
            if c_a is not None:
                pp.last_known_centroids[oid_b] = c_a
            elif oid_b in pp.last_known_centroids:
                del pp.last_known_centroids[oid_b]
            if c_b is not None:
                pp.last_known_centroids[oid_a] = c_b
            elif oid_a in pp.last_known_centroids:
                del pp.last_known_centroids[oid_a]

            # Swap median areas (mirror centroid's None-aware deletion pattern)
            m_a = pp.median_areas.get(oid_a)
            m_b = pp.median_areas.get(oid_b)
            if m_a is not None:
                pp.median_areas[oid_b] = m_a
            elif oid_b in pp.median_areas:
                del pp.median_areas[oid_b]
            if m_b is not None:
                pp.median_areas[oid_a] = m_b
            elif oid_a in pp.median_areas:
                del pp.median_areas[oid_a]

            # Swap area history
            h_a = pp._area_history.get(oid_a, [])
            h_b = pp._area_history.get(oid_b, [])
            pp._area_history[oid_a] = h_b
            pp._area_history[oid_b] = h_a

        # Swap appearance embeddings + reset reject counters
        if self.botsort_enabled and hasattr(self, '_appearance_store'):
            store = self._appearance_store
            e_a = store.embeddings.get(oid_a)
            e_b = store.embeddings.get(oid_b)
            if e_a is not None:
                store.embeddings[oid_b] = e_a
            elif oid_b in store.embeddings:
                del store.embeddings[oid_b]
            if e_b is not None:
                store.embeddings[oid_a] = e_b
            elif oid_a in store.embeddings:
                del store.embeddings[oid_a]
            # Reset consecutive-reject counters so the now-correct
            # embeddings aren't force-reset by stale rejection history
            if hasattr(store, '_consecutive_rejects'):
                store._consecutive_rejects[oid_a] = 0
                store._consecutive_rejects[oid_b] = 0

        # Swap obj_ptrs in the inner tracker's recent memory bank
        # so memory attention retrieves the correct identity context
        try:
            session = self.predictor._all_inference_states.get(
                self.session_id, {}
            ).get("state", {})
            for inner_state in session.get("sam2_inference_states", []):
                non_cond = inner_state.get("output_dict", {}).get(
                    "non_cond_frame_outputs", {}
                )
                obj_id_to_idx = inner_state.get("obj_id_to_idx", {})
                idx_a = obj_id_to_idx.get(oid_a)
                idx_b = obj_id_to_idx.get(oid_b)
                if idx_a is None or idx_b is None:
                    continue
                # Swap obj_ptr in recent frames (num_maskmem window)
                recent = sorted(non_cond.keys(), reverse=True)[:7]
                for f in recent:
                    entry = non_cond[f]
                    ptr = entry.get("obj_ptr")
                    if ptr is not None and idx_a < ptr.shape[0] and idx_b < ptr.shape[0]:
                        ptr[idx_a], ptr[idx_b] = ptr[idx_b].clone(), ptr[idx_a].clone()
        except Exception as e:
            logger.warning(
                "obj_ptr swap failed for oid %d <-> %d: %s. "
                "Memory attention may retrieve wrong identity for ~7 frames.",
                oid_a, oid_b, e,
            )

        # Swap TIAB centroid history so trajectory conditioning
        # reflects the corrected identity assignment
        if self.tiab_enabled:
            h_a = self._centroid_history.get(oid_a)
            h_b = self._centroid_history.get(oid_b)
            if h_a is not None:
                self._centroid_history[oid_b] = h_a
            elif oid_b in self._centroid_history:
                del self._centroid_history[oid_b]
            if h_b is not None:
                self._centroid_history[oid_a] = h_b
            elif oid_a in self._centroid_history:
                del self._centroid_history[oid_a]

        logger.info("Applied swap: oid %d <-> %d", oid_a, oid_b)

    def _update_tiab_attributes(self, frame_bool, frame_bgr):
        """Set per-frame TIAB attributes on the inner model.

        Updates _tiab_appearance_embs and _tiab_centroid_history so the
        next frame's _encode_new_memory can run TIAB boundary refinement.
        """
        from collections import deque

        _inner = self._model
        if hasattr(self._model, 'tracker') and hasattr(self._model.tracker, 'model'):
            _inner = self._model.tracker.model

        if not hasattr(_inner, '_tiab_module') or _inner._tiab_module is None:
            return

        # Use SAM3.1's obj_id_to_idx ordering if available to match
        # the batch dimension in pred_masks_high_res. Fall back to sorted
        # oids if the inner state doesn't expose the mapping.
        _obj_id_to_idx = None
        for s in self._inference_state.get("sam2_inference_states", []):
            if "obj_id_to_idx" in s:
                _obj_id_to_idx = s["obj_id_to_idx"]
                break
        if _obj_id_to_idx is not None:
            # Order oids by their SAM3.1 batch index
            oids = sorted(
                [oid for oid in frame_bool if oid in _obj_id_to_idx],
                key=lambda o: _obj_id_to_idx[o],
            )
        else:
            oids = sorted(frame_bool.keys())
        B = len(oids)
        if B == 0:
            _inner._tiab_appearance_embs = None
            _inner._tiab_centroid_history = None
            return

        K = self._tiab_trajectory_len
        device = next(_inner._tiab_module.parameters()).device

        # Get image dimensions for normalizing centroids
        h = self._inference_state.get("orig_height", 1008)
        w = self._inference_state.get("orig_width", 1008)

        # Update centroid history per object using largest-component centroid
        # (matches eval's _largest_component_centroid, not naive mean)
        for oid in oids:
            mask = frame_bool[oid]
            cx, cy = ShivaPixelPaintRecovery._largest_component_centroid(mask)
            if cx is not None:
                cx_norm = cx / w
                cy_norm = cy / h
            else:
                cx_norm, cy_norm = 0.5, 0.5

            if oid not in self._centroid_history:
                self._centroid_history[oid] = deque(maxlen=K)
            self._centroid_history[oid].append((cx_norm, cy_norm))

        # Build centroid history tensor [B, K, 2]
        history = torch.zeros(B, K, 2, device=device)
        for i, oid in enumerate(oids):
            h_list = list(self._centroid_history.get(oid, [(0.5, 0.5)]))
            # Pad with earliest position if not enough history
            while len(h_list) < K:
                h_list.insert(0, h_list[0])
            h_list = h_list[-K:]
            history[i] = torch.tensor(h_list, device=device)

        _inner._tiab_centroid_history = history

        # Build appearance embeddings [B, D]
        # Use the appearance store if available, otherwise zeros
        appear = torch.zeros(B, self._tiab_appearance_dim, device=device)
        store = getattr(self, '_appearance_store', None)
        if store is not None and frame_bgr is not None:
            for i, oid in enumerate(oids):
                emb = store.extract_embedding(frame_bool[oid], frame_bgr)
                if emb is not None:
                    # Pad or truncate to appearance_dim
                    emb_t = torch.from_numpy(np.asarray(emb, dtype=np.float32)).to(device)
                    dim = min(emb_t.shape[0], self._tiab_appearance_dim)
                    appear[i, :dim] = emb_t[:dim]

        _inner._tiab_appearance_embs = appear
        _inner._tiab_expected_B = B

    @staticmethod
    def _compute_pairwise_ious(bool_masks):
        """Compute IoU for all (N choose 2) pairs."""
        oids = sorted(bool_masks.keys())
        ious = {}
        for i in range(len(oids)):
            for j in range(i + 1, len(oids)):
                oi, oj = oids[i], oids[j]
                intersection = int((bool_masks[oi] & bool_masks[oj]).sum())
                union = int((bool_masks[oi] | bool_masks[oj]).sum())
                ious[(oi, oj)] = float(intersection / max(union, 1))
        return ious
