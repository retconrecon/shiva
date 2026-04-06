"""TIAB feature extraction — captures per-frame tensors from inside
SAM3.1's _encode_new_memory for TIAB training.

Uses a callback attribute (_tiab_extract_callback) on the inner tracker
model, called directly from _encode_new_memory in video_tracking_multiplex.py.
No monkey-patching — the callback is a first-class integration point.

Usage in experiment scripts:

    from sam3.model.tiab_extract import TIABExtractionSession
    from sam3.model.tiab_dataset import TIABExtractor

    extractor = TIABExtractor(save_dir, gt_data, n_animals)

    with TIABExtractionSession(predictor, session_id, extractor) as extract:
        for result in predictor.handle_stream_request({...}):
            frame_idx = result["frame_index"]
            obj_ids = result["outputs"]["out_obj_ids"]
            masks = result["outputs"]["out_binary_masks"]
            output_masks = {int(oid): masks[i].cpu().numpy().squeeze().astype(bool)
                           for i, oid in enumerate(obj_ids)}
            extract.on_frame(frame_idx, output_masks, obj_ids)

    # extractor.finalize() is called automatically by __exit__
"""


class TIABExtractionSession:
    """Manages the extraction callback lifecycle on the inner tracker model.

    Sets _tiab_extract_callback on the inner model, which is called from
    _encode_new_memory with (pred_masks, pix_feat, object_scores) before
    the non-overlap constraint. The callback buffers these tensors; on_frame()
    pairs them with the tracking loop's output masks and saves to disk.
    """

    def __init__(self, predictor, session_id, extractor):
        self.predictor = predictor
        self.session_id = session_id
        self.extractor = extractor
        self._buffer = {}

        # Find the inner model where _encode_new_memory runs
        model = predictor.model
        self._inner = model
        if hasattr(model, 'tracker') and hasattr(model.tracker, 'model'):
            self._inner = model.tracker.model

        # Install the callback
        def _capture(pred_masks, pix_feat, object_scores):
            self._buffer = {
                "pred_masks": pred_masks,
                "pix_feat": pix_feat,
                "object_scores": object_scores,
            }

        self._inner._tiab_extract_callback = _capture

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Remove callback
        if hasattr(self._inner, '_tiab_extract_callback'):
            del self._inner._tiab_extract_callback
        self.extractor.finalize()

    def on_frame(self, frame_idx, output_masks, obj_ids):
        """Called per frame from the tracking loop after yield.

        Pairs the buffered internal tensors (from _encode_new_memory)
        with the external output masks and saves via the extractor.
        """
        if not self._buffer:
            return

        self.extractor.on_frame(
            frame_idx=frame_idx,
            pred_masks_pre_constraint=self._buffer["pred_masks"],
            pix_feat=self._buffer["pix_feat"],
            object_score_logits=self._buffer["object_scores"],
            output_masks=output_masks,
            obj_ids=list(int(x) for x in obj_ids),
        )

        # Clear buffer for next frame
        self._buffer = {}
