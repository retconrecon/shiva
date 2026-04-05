"""TIAB training losses.

Three losses that train the boundary module without pixel-level identity
annotations — only centroid trajectories are needed.

1. Identity consistency: refined mask centroids should track the correct GT trajectory
2. Boundary refinement: focal + dice on boundary pixels where GT is available
3. Gate sparsity: encourage the gate to stay near 0 on non-crossing frames
"""

import torch
import torch.nn.functional as F


def mask_centroid(mask_probs):
    """Compute differentiable centroid matching eval's largest-component behavior.

    Eval uses _largest_component_centroid: binarize, find connected components,
    take centroid of the largest component only. This function replicates that
    by computing connected components on the detached binary mask (non-differentiable)
    and using the largest-component mask to gate the soft probabilities
    (preserving gradient flow through mask_probs).

    Args:
        mask_probs: [B, H, W] sigmoid probabilities in (0, 1)

    Returns:
        centroids: [B, 2] (x, y) normalized to [0, 1]
    """
    from scipy.ndimage import label as ndlabel

    B, H, W = mask_probs.shape
    device = mask_probs.device

    # Find largest connected component per object (non-differentiable)
    binary_np = (mask_probs.detach().cpu().numpy() > 0.5)
    largest_mask = torch.zeros_like(mask_probs)
    for b in range(B):
        labeled, n_comp = ndlabel(binary_np[b])
        if n_comp == 0:
            largest_mask[b] = 0
        elif n_comp == 1:
            largest_mask[b] = torch.from_numpy(binary_np[b].astype(float)).to(device)
        else:
            comp_sizes = [(labeled == c).sum() for c in range(1, n_comp + 1)]
            largest_id = max(range(1, n_comp + 1), key=lambda c: comp_sizes[c - 1])
            largest_mask[b] = torch.from_numpy(
                (labeled == largest_id).astype(float)
            ).to(device)

    # Gate soft probs by largest-component mask (straight-through gradient)
    weighted = largest_mask.detach() * mask_probs

    # Coordinate grids normalized to [0, 1]
    gy = torch.linspace(0, 1, H, device=device).view(1, H, 1).expand(B, H, W)
    gx = torch.linspace(0, 1, W, device=device).view(1, 1, W).expand(B, H, W)

    total_mass = weighted.sum(dim=(1, 2)).clamp(min=1e-6)
    cx = (weighted * gx).sum(dim=(1, 2)) / total_mass
    cy = (weighted * gy).sum(dim=(1, 2)) / total_mass

    return torch.stack([cx, cy], dim=-1)  # [B, 2]


def identity_consistency_loss(
    refined_masks,
    gt_centroids,
    id_map,
    margin=20.0,
    image_size=1008,
):
    """Contrastive centroid loss: each refined mask's centroid should be
    closest to its assigned GT trajectory point.

    Args:
        refined_masks: [B, H, W] mask logits (pre-argmax, post-refinement)
        gt_centroids: [N_animals, 2] GT centroid positions in pixels
        id_map: dict {pred_obj_idx: gt_trajectory_idx}
        margin: pixel margin for contrastive loss
        image_size: for normalizing pixel distances

    Returns:
        loss: scalar
    """
    B = refined_masks.size(0)
    mask_probs = torch.sigmoid(refined_masks)
    pred_cents = mask_centroid(mask_probs)  # [B, 2] normalized

    # Normalize GT centroids to [0, 1]
    gt_norm = gt_centroids.float().to(refined_masks.device) / image_size
    margin_norm = margin / image_size

    if gt_norm.size(0) < 2:
        return torch.tensor(0.0, device=refined_masks.device)

    loss = torch.tensor(0.0, device=refined_masks.device)
    count = 0

    # Vectorized pairwise distances: pred_cents [B, 2] vs gt_norm [N, 2]
    all_dists = torch.cdist(pred_cents, gt_norm)  # [B, N]

    for pred_idx, gt_idx in id_map.items():
        if pred_idx >= B or gt_idx >= gt_norm.size(0):
            continue

        d_pos = all_dists[pred_idx, gt_idx]

        # Nearest wrong GT: mask out the correct one and take min
        dists_row = all_dists[pred_idx].clone()
        dists_row[gt_idx] = float("inf")
        d_neg = dists_row.min()

        loss = loss + F.relu(d_pos - d_neg + margin_norm)
        count += 1

    return loss / max(count, 1)


def boundary_focal_loss(refined_masks, gt_masks, alpha=0.25, gamma=2.0):
    """Focal loss on boundary pixels only.

    Args:
        refined_masks: [B, H, W] mask logits
        gt_masks: [B, H, W] binary GT masks (float 0/1)

    Returns:
        loss: scalar
    """
    # Compute boundary: pixels within 3px of any object edge
    gt_binary = (gt_masks > 0.5).float()
    # Erode and dilate to find boundary
    kernel = torch.ones(1, 1, 7, 7, device=gt_masks.device)
    eroded = F.conv2d(
        gt_binary.unsqueeze(1), kernel, padding=3,
    ).squeeze(1)
    boundary = (eroded > 0) & (eroded < 49)  # not fully interior or exterior
    # Union of all objects' boundaries
    boundary_any = boundary.any(dim=0)  # [H, W]

    if boundary_any.sum() == 0:
        return torch.tensor(0.0, device=refined_masks.device)

    # Extract boundary pixels
    pred_boundary = refined_masks[:, boundary_any]  # [B, n_boundary]
    gt_boundary = gt_masks[:, boundary_any]

    # Focal loss
    p = torch.sigmoid(pred_boundary)
    ce_loss = F.binary_cross_entropy_with_logits(
        pred_boundary, gt_boundary, reduction="none",
    )
    p_t = p * gt_boundary + (1 - p) * (1 - gt_boundary)
    focal_weight = alpha * (1 - p_t) ** gamma
    loss = (focal_weight * ce_loss).mean()

    return loss


def gate_sparsity_loss(gate_values, is_crossing):
    """Encourage gate to be ~0 on non-crossing frames.

    Args:
        gate_values: [B, 1] gate outputs from RefinementGate
        is_crossing: bool — whether this frame is in a crossing zone

    Returns:
        loss: scalar
    """
    if is_crossing:
        # During crossings, don't penalize the gate
        return torch.tensor(0.0, device=gate_values.device)
    else:
        # Outside crossings, push gate toward 0
        return gate_values.mean()


def tiab_combined_loss(
    refined_masks,
    gt_centroids,
    id_map,
    gt_masks=None,
    gate_values=None,
    is_crossing=False,
    image_size=1008,
    lambda_identity=1.0,
    lambda_boundary=0.5,
    lambda_gate=0.1,
):
    """Combined TIAB training loss.

    Args:
        refined_masks: [B, H, W] refined mask logits
        gt_centroids: [N_animals, 2] GT positions in pixels
        id_map: dict {pred_idx: gt_idx}
        gt_masks: [B, H, W] GT masks (optional, for boundary loss)
        gate_values: [B, 1] gate outputs (optional, for sparsity loss)
        is_crossing: whether current frame is in crossing zone
        image_size: for centroid normalization

    Returns:
        total_loss: scalar
        loss_dict: dict of individual loss values for logging
    """
    losses = {}

    # Identity consistency (always)
    l_identity = identity_consistency_loss(
        refined_masks, gt_centroids, id_map, image_size=image_size,
    )
    losses["identity"] = l_identity.item()

    total = lambda_identity * l_identity

    # Boundary refinement (when GT masks available)
    if gt_masks is not None:
        l_boundary = boundary_focal_loss(refined_masks, gt_masks)
        losses["boundary"] = l_boundary.item()
        total = total + lambda_boundary * l_boundary

    # Gate sparsity (when gate values available)
    if gate_values is not None:
        l_gate = gate_sparsity_loss(gate_values, is_crossing)
        losses["gate"] = l_gate.item()
        total = total + lambda_gate * l_gate

    losses["total"] = total.item()
    return total, losses
