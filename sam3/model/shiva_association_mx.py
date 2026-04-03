"""SHIVA BoT-SORT Association for SAM3.1 Multiplex — identity-aware matching.

Replaces SAM3.1's IoU-only threshold matching with BoT-SORT fusion:
- IoU + appearance histogram fused cost matrix
- Hungarian optimal one-to-one assignment
- SENTINEL-adaptive gating (GREEN/YELLOW/RED)

Returns LazyAssociateDetTrkResult compatible with SAM3.1's GPU hotstart
and realize_adt_result() paths.

CRITICAL: det_to_max_iou_trk_idx uses RAW IoU argmax (for reconditioning).
          trk_is_unmatched/is_new_det use HUNGARIAN results (for identity).
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from sam3.model.sam3_video_base import LazyAssociateDetTrkResult


def associate_det_trk_botsort(
    det_masks,          # (N, H, W) float tensor — do NOT binarize input
    det_scores,         # (N,) float tensor — detection scores
    det_keep,           # (N,) bool tensor — which detections to consider
    trk_masks,          # (M, H, W) float tensor — padded to max_num_objects
    num_real_trk,       # int — actual number of tracklets (before padding)
    trk_obj_ids,        # (num_real_trk,) numpy — actual SAM3.1 object IDs (T4)
    appearance_store,   # ShivaAppearanceStore instance
    frame_pixels,       # (H, W, 3) numpy BGR or (H, W) grayscale, or None
    new_det_thresh,     # float
    iou_threshold_trk,  # float
    iou_threshold,      # float
    HIGH_CONF_THRESH,   # float (0.8)
    use_iom,            # bool
    sentinel_status="GREEN",
):
    """
    BoT-SORT IoU+appearance fusion with Hungarian optimal assignment.

    Returns: LazyAssociateDetTrkResult with all 8 GPU tensor fields.
    """
    device = det_masks.device
    N = det_masks.shape[0]
    M = trk_masks.shape[0]  # num_real_trk (no padding in BoT-SORT path)

    # --- 1. Binarize for IoU ---
    det_binary = (det_masks > 0).clone()
    det_binary[~det_keep] = 0
    trk_binary = trk_masks > 0

    # --- 2. Compute raw IoU/IoM matrix ---
    if use_iom:
        from sam3.train.masks_ops import mask_iom
        iou_matrix = mask_iom(det_binary, trk_binary)  # (N, M)
    else:
        from sam3.perflib.masks_ops import mask_iou
        iou_matrix = mask_iou(det_binary, trk_binary)  # (N, M)

    # --- 3. Compute appearance distance matrix ---
    d_reid = torch.ones(N, M, device=device)  # default: no appearance info

    if frame_pixels is not None and appearance_store is not None and num_real_trk > 0:
        # Use actual SAM3.1 object IDs, not positional indices
        real_trk_ids = trk_obj_ids[:num_real_trk].tolist()
        trk_embs = appearance_store.get(real_trk_ids)

        if trk_embs is not None:
            # Batch-transfer all detection masks to CPU in one call
            det_binary_np = det_binary.cpu().numpy()
            embed_dim = trk_embs.shape[1]
            uniform = np.ones(embed_dim, dtype=np.float64) / max(embed_dim, 1)
            det_keep_np = det_keep.cpu().numpy()

            # Extract detection embeddings (works for both histogram and OSNet)
            det_embs = []
            for i in range(N):
                if det_keep_np[i] and det_binary_np[i].sum() > 50:
                    emb = appearance_store.extract_histogram(det_binary_np[i], frame_pixels)
                    det_embs.append(emb if emb is not None else uniform)
                else:
                    det_embs.append(uniform)
            det_embs_np = np.stack(det_embs)  # (N, embed_dim)

            # Distance computation — use store's declared metric
            n_real = min(num_real_trk, M)
            d_reid_np = np.ones((N, M), dtype=np.float64)

            _metric = getattr(appearance_store, 'distance_metric', 'histogram_intersection')
            if _metric == "cosine":
                # Cosine distance for L2-normalized OSNet embeddings
                d_reid_np[:, :n_real] = 1.0 - np.dot(
                    det_embs_np, trk_embs[:n_real].T
                )
            else:
                # Histogram intersection distance
                d_reid_np[:, :n_real] = 1.0 - np.minimum(
                    det_embs_np[:, None, :], trk_embs[None, :n_real, :]
                ).sum(axis=2)

            d_reid = torch.from_numpy(d_reid_np).float().to(device)

    # --- 4. Fuse cost matrix ---
    d_iou = 1.0 - iou_matrix  # (N, M)

    if sentinel_status == "GREEN":
        fused_cost = torch.minimum(d_iou, d_reid)
    elif sentinel_status == "YELLOW":
        fused_cost = 0.3 * d_iou + 0.7 * d_reid
    else:  # RED
        fused_cost = d_iou

    # --- 5. Hungarian assignment on valid detections vs real tracks ---
    # Only match det_keep detections against non-padding tracks
    valid_det_mask = det_keep.cpu().numpy().astype(bool)
    valid_det_indices = np.where(valid_det_mask)[0]
    n_valid_det = len(valid_det_indices)
    n_real_trk = min(num_real_trk, M)

    trk_is_matched = np.zeros(M, dtype=bool)
    det_is_matched = np.zeros(N, dtype=bool)
    hungarian_det_to_trk = {}  # det_idx -> trk_idx

    if n_valid_det > 0 and n_real_trk > 0:
        # Build sub-cost matrix for valid dets × real tracks
        sub_cost = fused_cost[valid_det_indices[:, None],
                              torch.arange(n_real_trk, device=device)].cpu().numpy()

        # Guard against NaN/Inf from degenerate masks (0/0 in IoU)
        if not np.isfinite(sub_cost).all():
            sub_cost = np.nan_to_num(sub_cost, nan=1.0, posinf=1.0, neginf=0.0)

        row_ind, col_ind = linear_sum_assignment(sub_cost)

        for r, c in zip(row_ind, col_ind):
            det_idx = valid_det_indices[r]
            trk_idx = c
            # Accept match only if fused cost is below threshold
            if sub_cost[r, c] < (1.0 - iou_threshold_trk):
                trk_is_matched[trk_idx] = True
                det_is_matched[det_idx] = True
                hungarian_det_to_trk[det_idx] = trk_idx

    # --- 6. Build LazyAssociateDetTrkResult fields ---

    # trk_is_nonempty: which tracks have non-zero area (includes padding)
    trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2))

    # trk_is_unmatched: real tracks that are non-empty but not matched
    trk_is_unmatched_np = np.zeros(M, dtype=bool)
    for j in range(n_real_trk):
        if trk_is_nonempty[j].item() and not trk_is_matched[j]:
            trk_is_unmatched_np[j] = True
    trk_is_unmatched = torch.from_numpy(trk_is_unmatched_np).to(device)

    # is_new_det: unmatched detections with score above threshold
    is_new_det_np = np.zeros(N, dtype=bool)
    det_scores_np = det_scores.cpu().numpy()
    for i in range(N):
        if valid_det_mask[i] and not det_is_matched[i]:
            if det_scores_np[i] >= new_det_thresh:
                is_new_det_np[i] = True
    is_new_det = torch.from_numpy(is_new_det_np).to(device)

    # det_to_max_iou_trk_idx: RAW IoU argmax (NOT Hungarian) — for reconditioning
    det_to_max_iou_trk_idx = torch.argmax(iou_matrix, dim=1)

    # det_is_high_conf: high-confidence detections
    det_is_high_conf = ((det_scores >= HIGH_CONF_THRESH) & det_keep) & ~is_new_det

    # det_is_high_iou: detections with high fused similarity to their best match
    det_is_high_iou = torch.zeros(N, dtype=torch.bool, device=device)
    for det_idx, trk_idx in hungarian_det_to_trk.items():
        if iou_matrix[det_idx, trk_idx] >= iou_threshold:
            det_is_high_iou[det_idx] = True

    # im_mask: (N, M) bool — which (det, trk) pairs are matched by Hungarian
    im_mask = torch.zeros(N, M, dtype=torch.bool, device=device)
    for det_idx, trk_idx in hungarian_det_to_trk.items():
        im_mask[det_idx, trk_idx] = True

    # --- 7. Update appearance store with matched tracks ---
    # Use actual object IDs for store keys, not positional indices
    if frame_pixels is not None and appearance_store is not None:
        for det_idx, trk_idx in hungarian_det_to_trk.items():
            if trk_idx < num_real_trk:
                mask_np = det_binary[det_idx].cpu().numpy()
                if mask_np.sum() > 50:
                    obj_id = int(trk_obj_ids[trk_idx])
                    hist = appearance_store.extract_histogram(mask_np, frame_pixels)
                    appearance_store.update(obj_id, hist)

    # --- 8. Shape/dtype assertions ---
    assert trk_is_unmatched.shape == (M,) and trk_is_unmatched.dtype == torch.bool
    assert trk_is_nonempty.shape == (M,) and trk_is_nonempty.dtype == torch.bool
    assert is_new_det.shape == (N,) and is_new_det.dtype == torch.bool
    assert det_to_max_iou_trk_idx.shape == (N,) and det_to_max_iou_trk_idx.dtype == torch.long
    assert det_is_high_conf.shape == (N,) and det_is_high_conf.dtype == torch.bool
    assert det_is_high_iou.shape == (N,) and det_is_high_iou.dtype == torch.bool
    assert det_keep.shape == (N,) and det_keep.dtype == torch.bool
    assert im_mask.shape == (N, M) and im_mask.dtype == torch.bool

    return LazyAssociateDetTrkResult(
        trk_is_unmatched=trk_is_unmatched,
        trk_is_nonempty=trk_is_nonempty,
        is_new_det=is_new_det,
        det_to_max_iou_trk_idx=det_to_max_iou_trk_idx,
        det_is_high_conf=det_is_high_conf,
        det_is_high_iou=det_is_high_iou,
        det_keep=det_keep,
        im_mask=im_mask,
    )
