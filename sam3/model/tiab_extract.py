"""TIAB feature extraction — hooks into SAM3.1 tracking to capture
per-frame tensors needed for TIAB training.

Sets a callback on the inner tracker model that fires inside
_encode_new_memory, capturing pred_masks (before non-overlap constraint)
and pix_feat (backbone features at stride 16).

Usage in experiment scripts:
    from sam3.model.tiab_extract import setup_extraction, teardown_extraction

    # After start_session, before propagation:
    extractor = TIABExtractor(save_dir, gt_data, n_animals)
    setup_extraction(predictor, session_id, extractor)

    # Normal tracking loop
    for result in predictor.handle_stream_request({...}):
        frame_idx = result["frame_index"]
        obj_ids = result["outputs"]["out_obj_ids"]
        masks = result["outputs"]["out_binary_masks"]
        # Convert masks to bool dict for the extractor
        output_masks = {}
        for i, oid in enumerate(obj_ids):
            m = masks[i].cpu().numpy().squeeze().astype(bool) if hasattr(masks[i], 'cpu') else masks[i].squeeze().astype(bool)
            output_masks[int(oid)] = m
        extractor.capture_output(frame_idx, output_masks, obj_ids)

    teardown_extraction(predictor, session_id)
    extractor.finalize()
"""

import torch
from sam3.model.tiab_dataset import TIABExtractor


def setup_extraction(predictor, session_id, extractor):
    """Install extraction hook on the inner tracker model.

    Sets a callback that fires inside _encode_new_memory to capture
    pred_masks_high_res and pix_feat before the non-overlap constraint.
    """
    session = predictor._all_inference_states.get(session_id, {})
    state = session.get("state", {})

    # Find the inner model (where _encode_new_memory runs)
    model = predictor.model
    inner = model
    if hasattr(model, 'tracker') and hasattr(model.tracker, 'model'):
        inner = model.tracker.model

    # Store the extractor and a buffer for pre-constraint tensors
    inner._tiab_extractor = extractor
    inner._tiab_extract_buffer = {}

    # Monkey-patch _encode_new_memory to capture tensors
    original_encode = inner._encode_new_memory.__func__

    def patched_encode(self, image, current_vision_feats, feat_sizes,
                       pred_masks_high_res, object_score_logits,
                       is_mask_from_pts, **kwargs):
        # Capture pre-constraint tensors
        ext = getattr(self, '_tiab_extractor', None)
        buf = getattr(self, '_tiab_extract_buffer', None)
        if ext is not None and buf is not None:
            B = current_vision_feats[-1].size(1)
            C = self.hidden_dim
            H, W = feat_sizes[-1]
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            buf["pred_masks"] = pred_masks_high_res.detach()
            buf["pix_feat"] = pix_feat.detach()
            buf["object_scores"] = object_score_logits.detach()

        return original_encode(
            self, image, current_vision_feats, feat_sizes,
            pred_masks_high_res, object_score_logits,
            is_mask_from_pts, **kwargs,
        )

    import types
    inner._encode_new_memory = types.MethodType(patched_encode, inner)
    inner._tiab_original_encode = original_encode

    print("TIAB extraction hook installed")


def teardown_extraction(predictor, session_id):
    """Remove extraction hook and restore original _encode_new_memory."""
    model = predictor.model
    inner = model
    if hasattr(model, 'tracker') and hasattr(model.tracker, 'model'):
        inner = model.tracker.model

    original = getattr(inner, '_tiab_original_encode', None)
    if original is not None:
        import types
        inner._encode_new_memory = types.MethodType(original, inner)

    for attr in ('_tiab_extractor', '_tiab_extract_buffer', '_tiab_original_encode'):
        if hasattr(inner, attr):
            delattr(inner, attr)

    print("TIAB extraction hook removed")


class ExtractionCapture:
    """Helper that bridges the tracking loop output with the extractor.

    The extraction hook captures pre-constraint tensors inside
    _encode_new_memory. This class captures the post-tracking output
    (masks, obj_ids) and calls the extractor with both.
    """

    def __init__(self, extractor, inner_model):
        self.extractor = extractor
        self.inner = inner_model

    def on_frame(self, frame_idx, output_masks, obj_ids):
        """Called per frame in the tracking loop after yield.

        Args:
            frame_idx: current frame
            output_masks: dict {oid: bool_mask}
            obj_ids: list of object IDs from SAM3.1 output
        """
        buf = getattr(self.inner, '_tiab_extract_buffer', {})
        if not buf:
            return

        self.extractor.on_frame(
            frame_idx=frame_idx,
            pred_masks_pre_constraint=buf.get("pred_masks"),
            pix_feat=buf.get("pix_feat"),
            object_score_logits=buf.get("object_scores"),
            output_masks=output_masks,
            obj_ids=obj_ids,
        )

        # Clear buffer
        self.inner._tiab_extract_buffer = {}
