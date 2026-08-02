"""SHIVA Tracker — the single integration point for SHIVA on top of SAM3.1.

Wraps SAM3.1's `handle_stream_request` with memory pruning (so a 60k-frame
video fits in one session) and pixel-paint mask recovery. That is the whole
surface. Every other SHIVA feature that once lived here — BoT-SORT
association, the appearance stores, the identity verifier, confidence
injection, TIAB, the occlusion memory freeze — was unreachable from the only
caller and has been removed. They are recoverable from the tag
`archive/tiab-pre-cleanup-2026-08-01`.

THE RULE THIS MODULE EXISTS TO ENFORCE
--------------------------------------
A config flag is a REQUEST. A counter is EVIDENCE. Five SHIVA features were
shipped, reported on, and in one case credited in a grant document while
provably never executing. Every feature here registers an expected hook at
construction and the teardown report prints any that stayed at zero. Do not
report a number that depends on a feature until its counter is nonzero.

Usage:
    from sam3.model_builder import build_sam3_predictor
    from sam3.model.shiva_tracker import ShivaTracker

    predictor = build_sam3_predictor(version="sam3.1", use_fa3=False)
    session_id = predictor.handle_request({
        "type": "start_session", "resource_path": frame_dir,
        "lazy_loading_frames": True,
    })["session_id"]
    predictor.handle_request({
        "type": "add_prompt", "session_id": session_id, "frame_index": 0,
        "bounding_boxes": boxes, "bounding_box_labels": labels,
    })

    with ShivaTracker(predictor, session_id, frame_dir, n_animals=4) as shiva:
        for frame_idx, outputs, recovery_masks in shiva.track(n_frames=5000):
            # outputs: SAM3.1 output dict (out_obj_ids, out_binary_masks, ...)
            # recovery_masks: {oid: bool_mask} from pixel-paint, empty if healthy
            ...

NOTE ON recovery_masks: these are NOT folded into `outputs`. A caller that
wants pixel-paint to affect its stored masks must merge them itself. ZEUS
currently does not, and counts them as events only.
"""

import logging

import cv2
import numpy as np
import torch

from sam3.model import shiva_instrumentation
from sam3.model.shiva_closed_world import evaluate_frame as _closed_world_eval
from sam3.model.shiva_instrumentation import bump as _shiva_bump
from sam3.model.shiva_memory_pruning import prune_output_dict
from sam3.model.shiva_pixel_paint import ShivaPixelPaintRecovery

# SAM3.1 feature cache normalization: (pixel / 255 - 0.5) / 0.5
# Reverse: (tensor * 0.5 + 0.5) * 255
# Defined here: sam3/model/sam3_multiplex_base.py run_backbone_and_detection
_DENORM_MEAN = 0.5
_DENORM_STD = 0.5

logger = logging.getLogger(__name__)


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


class ShivaTracker:
    """SAM3.1 propagation with SHIVA memory pruning and pixel-paint recovery.

    Supports context manager for automatic session cleanup:
        with ShivaTracker(predictor, session_id, ...) as shiva:
            for frame_idx, outputs, recovery in shiva.track():
                ...
    """

    # Cap on the rolling closed-world violation log, so a 60k-frame run cannot
    # grow it without bound. The instrumentation counters carry the totals.
    _MAX_CW_LOG = 5000

    def __init__(self, predictor, session_id, frame_dir, n_animals,
                 max_recent_frames=500, max_landmark_frames=50,
                 pixel_paint_enabled=True, diagnostics_path=None,
                 n_frames=None):
        self.predictor = predictor
        self.session_id = session_id
        self.n_animals = n_animals
        self.frame_dir = frame_dir
        self.max_recent_frames = max_recent_frames
        self.max_landmark_frames = max_landmark_frames
        self.pixel_paint_enabled = pixel_paint_enabled
        # Optional JSON dump of the end-of-run diagnostics. The truth table is
        # printed regardless; this makes it machine-readable for an A/B.
        self.diagnostics_path = diagnostics_path

        # Counters are per-session, and every feature the caller turned on
        # becomes a claim checked at teardown.
        shiva_instrumentation.reset()
        self._expected_hooks = []
        if pixel_paint_enabled:
            self._expected_hooks.append('pixel_paint_recovery')

        # Access inference_state through predictor internals
        session = predictor._all_inference_states.get(session_id)
        if session is None:
            raise RuntimeError(f"Session {session_id} not found in predictor")
        self._inference_state = session["state"]
        self._model = predictor.model if hasattr(predictor, 'model') else None

        # Initialize pixel-paint recovery
        self.pixel_paint = None
        if pixel_paint_enabled:
            self.pixel_paint = ShivaPixelPaintRecovery(
                frame_dir, n_animals, n_frames=n_frames,
            )
            self.pixel_paint.build_background_model()

        self.prune_stats = []
        # Rolling log of frames that failed the closed-world check.
        self.closed_world_violations = []

        # Derive healthy area threshold from pixel-paint so any health test
        # agrees with pixel-paint on what constitutes a healthy mask.
        if self.pixel_paint is not None:
            self._min_healthy_area = self.pixel_paint.min_fish_area
        else:
            self._min_healthy_area = 200

        # Validate denormalization formula against disk-loaded frame 0
        self._validate_denormalization(frame_dir)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Surface the diagnostics before tearing anything down.
        #
        # This PRINTS rather than only logging, deliberately. ZEUS's track.py
        # configures no logging at all, so the root logger sits at WARNING and a
        # logger.info() report would produce nothing -- which is precisely the
        # silent-failure pattern this instrumentation exists to catch. A
        # diagnostic that depends on the caller configuring logging, or on the
        # caller remembering to read an attribute, is a diagnostic that will not
        # fire when it matters. Self-surfacing is the whole point.
        try:
            report = shiva_instrumentation.format_report(
                only_expected=self._expected_hooks
            )
            cw = self.closed_world_violations
            if cw:
                report += (
                    f"\nCLOSED-WORLD: {len(cw)} frame(s) violated the exactly-N "
                    f"permutation constraint"
                    + (" (log capped)" if len(cw) >= self._MAX_CW_LOG else "")
                    + "\n  first: " + str(cw[0])
                    + "\n  last:  " + str(cw[-1])
                )
            else:
                report += "\nCLOSED-WORLD: no violations detected."
            print(report, flush=True)
            logger.info("%s", report)
            if self.diagnostics_path:
                import json
                with open(self.diagnostics_path, "w") as f:
                    json.dump(self.diagnostics(), f, indent=2)
                print(f"[shiva] diagnostics written to {self.diagnostics_path}",
                      flush=True)
        except Exception as e:  # never let reporting break teardown
            logger.debug("Instrumentation report failed: %s", e)
        try:
            self.predictor.handle_request({
                "type": "close_session",
                "session_id": self.session_id,
            })
        except Exception as e:
            logger.debug("Session close failed: %s", e)
        # Break reference cycles so GC can free session memory.
        self._inference_state = None
        self._model = None
        self.pixel_paint = None

    def diagnostics(self):
        """Machine-readable summary of what actually happened this session."""
        counters = shiva_instrumentation.snapshot()
        by_type = {}
        for r in self.closed_world_violations:
            for v in r.violations:
                key = v.split(" ")[0] if v[0].isdigit() else " ".join(v.split(" ")[:2])
                by_type[key] = by_type.get(key, 0) + 1
        return {
            "counters": counters,
            "requested_hooks": list(self._expected_hooks),
            "silent_requested_hooks": [
                h for h in self._expected_hooks if counters.get(h, 0) == 0
            ],
            "closed_world": {
                "violation_frames_logged": len(self.closed_world_violations),
                "log_capped_at": self._MAX_CW_LOG,
                "by_type": by_type,
                "first_violations": [
                    str(r) for r in self.closed_world_violations[:20]
                ],
            },
        }

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
                "pixel-paint will be unreliable. The model may use different "
                "normalization than mean=%.1f, std=%.1f.",
                vmin, vmax, _DENORM_MEAN, _DENORM_STD,
            )

    def track(self, n_frames=None, propagation_direction="forward"):
        """Run SAM3.1 propagation with SHIVA hooks.

        Yields:
            (frame_idx, outputs, recovery_masks) tuples.
            - outputs: SAM3.1 output dict, unmodified
            - recovery_masks: {oid: bool_mask} from pixel-paint, empty if healthy.
              NOT merged into `outputs` — see the module docstring.
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

                # Get denormalized frame early — needed for mask completion
                frame_bgr = None
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
                    # Resize not_water to match mask resolution if needed (bg
                    # model is built at disk resolution, masks at SAM3.1's).
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
                    _shiva_bump('bfs_mask_completion')

                # Update centroids only for healthy masks, using largest
                # connected component (not naive mean)
                if self.pixel_paint is not None:
                    for oid, m in frame_bool.items():
                        if int(m.sum()) <= 0:
                            continue
                        area = int(m.sum())
                        median = self.pixel_paint.median_areas.get(oid)
                        # Skip centroid update for artifact masks
                        if median is None or area >= median * self.pixel_paint.area_lower:
                            cx, cy = ShivaPixelPaintRecovery._largest_component_centroid(m)
                            if cx is not None:
                                self.pixel_paint.update_last_centroid(oid, cx, cy)

                # Closed-world constraint check. Read-only: it cannot change
                # tracking, so it stays on. This is the per-frame identity
                # signal the pipeline has never had -- coverage measures
                # presence and is blind to a mask parked on the wrong animal,
                # and the tracker's own swap counter reads 0 on a video with
                # three confirmed swaps. Without this an A/B compares coverage;
                # with it, an A/B counts constraint violations.
                cw_report = _closed_world_eval(
                    frame_idx, frame_bool, self.n_animals,
                    median_areas=(
                        self.pixel_paint.median_areas if self.pixel_paint else None
                    ),
                )
                if not cw_report.ok:
                    self.closed_world_violations.append(cw_report)
                    if len(self.closed_world_violations) > self._MAX_CW_LOG:
                        self.closed_world_violations.pop(0)

                # Memory pruning — the reason a 60k-frame video fits in one
                # session. Unconditional.
                stats = prune_output_dict(
                    self._inference_state, frame_idx,
                    self.max_recent_frames, self.max_landmark_frames,
                )
                if stats is not None:
                    self.prune_stats.append((frame_idx, stats))
                    _shiva_bump('memory_pruned')

                # Pixel-paint recovery
                recovery_masks = {}
                if self.pixel_paint is not None:
                    recovery_masks = self.pixel_paint.check_and_recover(
                        frame_idx, frame_bool, frame_bgr=frame_bgr,
                    )
                    if recovery_masks:
                        _shiva_bump('pixel_paint_recovery', len(recovery_masks))
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
                    #
                    # This is pixel-paint's ONLY effect on tracking: the returned
                    # masks are not merged into `outputs`. A failure here used to
                    # be swallowed at debug level, which made a total no-op
                    # indistinguishable from a working feature. It is counted now.
                    if recovery_masks:
                        _inner = self._model
                        if (hasattr(self._model, 'tracker')
                                and hasattr(self._model.tracker, 'model')):
                            _inner = self._model.tracker.model
                        if hasattr(_inner, 'add_new_masks'):
                            try:
                                all_oids = sorted(recovery_masks.keys())
                                all_masks = torch.stack([
                                    torch.from_numpy(
                                        recovery_masks[oid].astype(np.float32)
                                    ).unsqueeze(0)
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
                                _shiva_bump('pixel_paint_injected', len(all_oids))
                            except Exception as e:
                                _shiva_bump('pixel_paint_inject_failed')
                                if not getattr(self, '_inject_warned', False):
                                    self._inject_warned = True
                                    logger.warning(
                                        "Pixel-paint memory injection failed "
                                        "(first occurrence, frame %d): %s. "
                                        "Recovery is now a no-op on tracking.",
                                        frame_idx, e, exc_info=True,
                                    )
                        else:
                            _shiva_bump('pixel_paint_inject_unavailable')

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
            except Exception as e:
                logger.debug("Session reset failed during error handling: %s", e)
            # Free zombie GPU tensors from generator locals (hotstart_buffer, etc.)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            raise
        finally:
            gen.close()  # Idempotent — deterministic cleanup on normal exit too
