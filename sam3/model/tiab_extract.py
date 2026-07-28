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
        # Keyed by frame so the consumer can take the sample for the frame it is handling, rather
        # than whatever happened to be captured last.
        self._pf_buffer = {}

        # Find the object where _encode_new_memory runs during propagation.
        # Sam3MultiplexPredictorWrapper inherits _encode_new_memory from
        # Sam3TrackerBase via its parent class. When the wrapper calls
        # self._encode_new_memory(), self is the WRAPPER, not the inner model
        # (despite __getattr__ proxying — inheritance takes priority over
        # __getattr__). The callback must be set on the wrapper AND the
        # inner model to cover both code paths (STB and VTM).
        model = predictor.model
        self._targets = []
        if hasattr(model, 'tracker'):
            self._targets.append(model.tracker)  # wrapper (STB path)
            if hasattr(model.tracker, 'model'):
                self._targets.append(model.tracker.model)  # inner (VTM path)
        else:
            self._targets.append(model)
        self._inner = self._targets[0]

        # Install the callback on ALL targets
        def _capture(pred_masks, pix_feat, object_scores):
            self._buffer = {
                "pred_masks": pred_masks,
                "pix_feat": pix_feat,
                "object_scores": object_scores,
            }

        # PER-FRAME capture (blocker 5). The `_capture` callback above rides on
        # `_encode_new_memory`, which fires on only ~6.27% of frames during propagation (measured on
        # mouse045: 170/2713) because propagation runs with run_mem_encoder=False. `_track_step_aux`
        # runs EVERY frame, so this second callback is the one that actually yields a training set.
        # It carries frame_idx, so a sample can never be attributed to the wrong frame.
        def _capture_perframe(frame_idx, pred_masks, pix_feat, object_scores):
            self._pf_buffer[int(frame_idx)] = {
                "pred_masks": pred_masks,
                "pix_feat": pix_feat,
                "object_scores": object_scores,
            }

        for target in self._targets:
            target._tiab_extract_callback = _capture
            target._tiab_perframe_callback = _capture_perframe

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # Remove callback from all targets
        for target in self._targets:
            if hasattr(target, '_tiab_extract_callback'):
                del target._tiab_extract_callback
            if hasattr(target, '_tiab_perframe_callback'):
                del target._tiab_perframe_callback
        self.extractor.finalize()

    def on_frame(self, frame_idx, output_masks, obj_ids):
        """Called per frame from the tracking loop after yield.

        Pairs the buffered internal tensors (from _encode_new_memory)
        with the external output masks and saves via the extractor.
        """
        # PREFER the per-frame capture for THIS frame; fall back to the memory-encoder buffer.
        # The fallback is kept so this class still works against a build without the per-frame hook.
        _pf = self._pf_buffer.pop(int(frame_idx), None)
        if _pf is not None:
            self.extractor.on_frame(
                frame_idx=frame_idx,
                pred_masks_pre_constraint=_pf["pred_masks"],
                pix_feat=_pf["pix_feat"],
                object_score_logits=_pf["object_scores"],
                output_masks=output_masks,
                obj_ids=list(int(x) for x in obj_ids),
            )
            self._buffer = {}
            return

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
