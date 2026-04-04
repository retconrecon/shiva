"""SHIVA Memory Pruning — bounds output dicts for long videos.

Deletes old frame entries from BOTH output_dict (consolidated, holds actual
GPU tensors) and output_dict_per_obj (per-object views) after each yielded
frame to prevent unbounded VRAM growth.

CRITICAL: The per-object entries in output_dict_per_obj are tensor VIEWS
that share storage with the consolidated output_dict. Pruning only the
per-object dict frees ~100 bytes of Python overhead per entry but zero
GPU memory — the consolidated dict still holds the tensors. Both dicts
must be pruned in sync.

SAM3.1's memory retrieval uses dict.get() with None fallback, so pruned
frames are silently skipped. Zero code changes to attention.

Usage:
    from sam3.model.shiva_memory_pruning import prune_output_dict

    for result in predictor.handle_stream_request({...}):
        frame_idx = result["frame_index"]
        # ... process masks ...
        prune_output_dict(inference_state, frame_idx)
"""


def prune_output_dict(inference_state, current_frame_idx,
                      max_recent_frames=500, max_landmark_frames=50):
    """Delete old frame entries from output dicts to bound VRAM.

    Prunes from BOTH:
    - output_dict["non_cond_frame_outputs"] (consolidated, holds GPU tensors)
    - output_dict_per_obj[obj_idx]["non_cond_frame_outputs"] (per-object views)

    Handles both direct inference states (with output_dict at top level)
    and multiplex inference states (with output_dict inside sam2_inference_states).

    Keeps:
    - Last max_recent_frames frames (most recent temporal context)
    - Top max_landmark_frames by score from beyond the recent window
      (temporally bucketed for diversity)
    - All conditioning frames (sacred, never pruned)

    Returns:
        dict with pruning stats, or None if no pruning occurred.
    """
    # Multiplex path: output_dict lives inside sam2_inference_states
    sam2_states = inference_state.get("sam2_inference_states", [])
    if sam2_states:
        total = {"pruned": 0, "kept_landmarks": 0, "retained_total": 0}
        for inner_state in sam2_states:
            result = _prune_single_state(
                inner_state, current_frame_idx,
                max_recent_frames, max_landmark_frames,
            )
            if result:
                total["pruned"] += result["pruned"]
                total["kept_landmarks"] = max(total["kept_landmarks"], result["kept_landmarks"])
                total["retained_total"] = max(total["retained_total"], result["retained_total"])
        # Periodic defragmentation (once per frame, not per inner state)
        if total["pruned"] > 0 and current_frame_idx % 1000 == 0:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return total if total["pruned"] > 0 else None

    # Direct path: output_dict at top level
    return _prune_single_state(
        inference_state, current_frame_idx,
        max_recent_frames, max_landmark_frames,
    )


def _prune_single_state(inference_state, current_frame_idx,
                         max_recent_frames, max_landmark_frames):
    """Prune a single inference state's output dicts."""
    main_non_cond = inference_state.get("output_dict", {}).get(
        "non_cond_frame_outputs", {}
    )

    if len(main_non_cond) <= max_recent_frames + max_landmark_frames:
        return None

    all_frames = sorted(main_non_cond.keys())
    recent_cutoff = current_frame_idx - max_recent_frames

    # Frames beyond the recent window
    old_frames = [f for f in all_frames if f < recent_cutoff]

    if len(old_frames) <= max_landmark_frames:
        return None

    # Score old frames for landmark selection
    scored = []
    for f in old_frames:
        entry = main_non_cond[f]
        score = 0.0
        if "eff_iou_score" in entry:
            s = entry["eff_iou_score"]
            score = float(s.item() if hasattr(s, 'item') else s)
        elif "best_iou_score" in entry:
            s = entry["best_iou_score"]
            score = float(s.item() if hasattr(s, 'item') else s)
        elif "object_score_logits" in entry:
            s = entry["object_score_logits"]
            score = float(s.sigmoid().mean().item() if hasattr(s, 'item') else s)
        scored.append((score, f))

    # Temporal bucketing — one landmark per bucket for diversity
    scored.sort(key=lambda x: -x[0])
    f_min = old_frames[0]
    f_max = old_frames[-1]
    bucket_size = max(1, (f_max - f_min + 1) // max_landmark_frames)
    buckets = {}
    for score, f in scored:
        bid = (f - f_min) // bucket_size
        if bid not in buckets:
            buckets[bid] = f
        if len(buckets) >= max_landmark_frames:
            break

    # If bucketing produced fewer than max_landmark_frames, fill with top-scored
    keep_frames = set(buckets.values())
    if len(keep_frames) < max_landmark_frames:
        for score, f in scored:
            if f not in keep_frames:
                keep_frames.add(f)
            if len(keep_frames) >= max_landmark_frames:
                break

    # Never evict frames protected by confidence injection
    protected = inference_state.get("_shiva_protected_frames", set())
    frames_to_evict = [
        f for f in old_frames if f not in keep_frames and f not in protected
    ]

    if not frames_to_evict:
        return None

    # --- Step 2: Evict from consolidated dict (frees actual GPU tensors) ---
    for f in frames_to_evict:
        main_non_cond.pop(f, None)

    # --- Step 3: Evict from all per-object dicts (removes view references) ---
    for obj_idx in inference_state.get("output_dict_per_obj", {}):
        obj_non_cond = (
            inference_state["output_dict_per_obj"][obj_idx]
            .get("non_cond_frame_outputs", {})
        )
        for f in frames_to_evict:
            obj_non_cond.pop(f, None)

    # --- Step 4: Periodic defragmentation (T6) ---
    # Repeated allocate-delete cycles fragment CUDA memory. gc.collect()
    # breaks reference cycles so empty_cache() can actually return blocks.
    if current_frame_idx % 1000 == 0:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "pruned": len(frames_to_evict),
        "kept_landmarks": len(keep_frames),
        "retained_total": len(main_non_cond),
    }
