# tiab_kp_losses.py - keypoint-supervised objectives for TIAB
#
# WHY THIS FILE EXISTS. The shipped centroid objective in `tiab_losses.py` is degenerate in three
# independent ways, each verified against the code and against the CalMS21 ground truth:
#
#   1. THE TARGET IS SELF-GENERATED. `tiab_train.py:232-244` builds `id_map` by Hungarian-matching
#      the PRE-refinement centroids to GT, and the loss then asks the POST-refinement centroid to
#      stay nearest that same GT. `d_pos < d_neg` therefore holds by construction and the loss is
#      already 0 before the module does anything. Worse, if the tracker has ALREADY swapped, the
#      matching re-derives the swapped assignment and the loss ratifies the swap - it trains the
#      module to preserve the exact failure it exists to prevent.
#
#   2. THE GRADIENT CANNOT REACH THE CONTESTED PIXELS. Two hard gates sit between the loss and the
#      pixels TIAB is supposed to reassign: `_hard_argmax` sends losing pixels through
#      `torch.clamp(x, max=-10.0)` (gradient exactly 0 for x > -10), and `mask_centroid` multiplies
#      by a detached binary `(p > 0.5)` mask (gradient exactly 0 for p <= 0.5). A pixel currently
#      owned by the OTHER animal is, by definition, on the wrong side of both. So the module can
#      only reweight mass it already owns and can never learn to CLAIM a pixel, which is the one
#      behaviour it exists for.
#
#   3. THE HINGE IS INACTIVE ALMOST EVERYWHERE. The effective margin is 20/1008 = 0.0198, so the
#      loss is nonzero only when the predicted centroid lies within ~20 px of the bisector between
#      the two animals. Measured over the real GT: 125 of 85,063 frames (0.147%).
#
# THE FIX IS NOT A BETTER MARGIN, IT IS A BETTER SIGNAL. CalMS21 ships 7 MARS keypoints per animal
# per frame (85,063 frames, all 15 videos, already on disk). A keypoint is an EXTERNAL, per-pixel,
# identity-labelled observation - exactly what a boundary module needs and exactly what a single
# centroid cannot provide. Sampling the mask probability AT a keypoint's pixel puts the gradient
# directly on that pixel with no argmax, no clamp and no detach anywhere in the path.
#
# Two objectives are provided, in decreasing strength of supervision, so the ablation can walk
# BACKWARDS FROM PERFECT: establish that the concept works when given everything, then remove
# advantages one at a time and measure where it breaks.
#
#   voronoi_ownership_loss  - DENSE. Every contested pixel is labelled by nearest GT keypoint.
#                             This is the "perfect" arm: dense per-pixel boundary supervision.
#   keypoint_ownership_loss - SPARSE. Only the 7 keypoints per animal are constrained (each must
#                             lie inside its own mask and outside the other's). 14 constraints per
#                             frame instead of ~10^5, and no Voronoi assumption about the boundary.
#
# COORDINATE SPACE. SAM3.1 resizes every frame to a SQUARE image_size x image_size with
# `v2.Resize(size=(res, res))` (sam3_image_processor.py:24), which does NOT preserve aspect ratio.
# Mask logits therefore live in a stretched square, while GT keypoints are in original video pixels
# (CalMS21: 1024x570). The correct mapping is per-axis: kx * S / W, ky * S / H. Using a single
# scalar (as the shipped `gt_centroids / image_size` does) compresses the y axis by H/S - on
# CalMS21 that is 570/1008, i.e. every target sits 43.5% too high in the frame. Callers pass
# (frame_w, frame_h) here so that error is not expressible.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def keypoints_to_mask_space(
    keypoints: torch.Tensor,
    frame_w: int,
    frame_h: int,
    mask_h: int,
    mask_w: int,
) -> torch.Tensor:
    """Map GT keypoints from original video pixels into mask-tensor pixel coordinates.

    Args:
        keypoints: [B, K, 2] as (x, y) in ORIGINAL video pixels.
        frame_w, frame_h: original video dimensions (e.g. 1024, 570).
        mask_h, mask_w: dimensions of the mask logit tensor (e.g. 1008, 1008).

    Returns:
        [B, K, 2] as (x, y) in mask-tensor pixels.

    The per-axis scaling is the whole point: the resize that produced the mask tensor stretched the
    frame to a square, so x and y carry DIFFERENT scale factors and a single scalar cannot express
    the mapping.
    """
    scaled = keypoints.clone().float()
    scaled[..., 0] = scaled[..., 0] * (mask_w / float(frame_w))
    scaled[..., 1] = scaled[..., 1] * (mask_h / float(frame_h))
    return scaled


def _sample_at(logits: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample [B, H, W] logits at [N, 2] (x, y) mask-space points -> [B, N].

    Bilinear rather than nearest so the gradient reaches the four pixels around each keypoint and
    varies smoothly as the mask boundary moves. `grid_sample` wants normalized [-1, 1] coordinates
    with x first, and align_corners=False matches the half-pixel convention used by the resize that
    produced these logits.
    """
    b, h, w = logits.shape
    gx = (pts[:, 0] / max(w - 1, 1)) * 2.0 - 1.0
    gy = (pts[:, 1] / max(h - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2).expand(b, 1, -1, 2)
    out = F.grid_sample(logits.unsqueeze(1), grid, mode="bilinear",
                        padding_mode="border", align_corners=False)
    return out.view(b, -1)


def keypoint_ownership_loss(
    refined_logits: torch.Tensor,
    keypoints_mask_space: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Each animal's keypoints must lie inside ITS OWN mask and outside every other mask.

    This is a direct statement of the thing identity tracking gets wrong. A swap IS the event where
    animal A's mask covers animal B's keypoints. Penalising exactly that is a far tighter objective
    than asking a centroid to stay nearer one point than another.

    Args:
        refined_logits: [B, H, W] PRE-argmax per-object mask logits. Must be pre-argmax: the
            gradient through `_hard_argmax` is identically zero on the losing side, which is the
            side that needs to move.
        keypoints_mask_space: [B, K, 2] GT keypoints already in mask-tensor coordinates, where row
            b holds the keypoints of the animal that object b is supposed to be.
        valid: optional [B, K] bool; False entries (missing/NaN GT) are ignored.

    Returns:
        (loss, parts) where parts carries the positive/negative terms for logging.

    Formulation: at every keypoint, softmax over the OBJECT axis gives an ownership distribution.
    The correct owner should receive probability 1. That is a cross-entropy over B classes evaluated
    at B*K points, which simultaneously pushes the true owner's logit up and every other owner's
    logit down at that pixel - the positive and negative constraint in one term, with no margin to
    tune and no hinge to go inactive.
    """
    b, _h, _w = refined_logits.shape
    k = keypoints_mask_space.shape[1]
    flat_pts = keypoints_mask_space.reshape(-1, 2)                     # [B*K, 2]
    sampled = _sample_at(refined_logits, flat_pts)                     # [B, B*K]

    # Column j of `sampled` is keypoint j (belonging to animal j // k) evaluated under every
    # object's logits. The target class for that column is exactly j // k.
    target = torch.arange(b, device=refined_logits.device).repeat_interleave(k)   # [B*K]
    logits_t = sampled.transpose(0, 1)                                 # [B*K, B]

    if valid is not None:
        keep = valid.reshape(-1)
        if keep.sum() == 0:
            zero = refined_logits.sum() * 0.0        # keeps grad_fn alive; never a bare tensor
            return zero, {"kp_ce": 0.0, "kp_n": 0}
        logits_t = logits_t[keep]
        target = target[keep]

    ce = F.cross_entropy(logits_t, target)
    with torch.no_grad():
        acc = (logits_t.argmax(dim=1) == target).float().mean().item()
    return ce, {"kp_ce": float(ce.detach()), "kp_acc": acc, "kp_n": int(target.numel())}


def voronoi_ownership_loss(
    refined_logits: torch.Tensor,
    keypoints_mask_space: torch.Tensor,
    contested: Optional[torch.Tensor] = None,
    max_pixels: int = 4096,
    valid: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, dict]:
    """Dense arm: label every contested pixel by its nearest GT keypoint and apply cross-entropy.

    THE "PERFECT" SUPERVISION we walk backwards from. Where `keypoint_ownership_loss` constrains 14
    points, this constrains every pixel the module is actually editing, giving the boundary a shape
    to match rather than a handful of anchors.

    THE ASSUMPTION, STATED PLAINLY: nearest-keypoint is a proxy for true ownership. It is exact away
    from the boundary and approximate within roughly half the inter-keypoint spacing of it - i.e.
    least reliable exactly at the contact line. It is therefore an upper bound on what
    keypoint-derived supervision can teach, NOT a substitute for hand-labelled masks. Report it as
    such: if this arm does not improve tracking, no weaker keypoint-derived signal will either, and
    that is a decisive negative result obtainable tonight with no annotation.

    Args:
        refined_logits: [B, H, W] pre-argmax logits.
        keypoints_mask_space: [B, K, 2] in mask coordinates.
        contested: optional [H, W] bool restricting supervision to contested pixels. Strongly
            recommended - supervising uncontested background teaches nothing and swamps the term.
        max_pixels: cap on supervised pixels per frame, sampled without replacement. Bounds both
            memory and the degree to which one large frame dominates a batch.
        valid: optional [B, K] bool for missing keypoints.
        generator: optional RNG for reproducible subsampling.
    """
    b, h, w = refined_logits.shape

    if contested is None:
        sel = torch.ones(h, w, dtype=torch.bool, device=refined_logits.device)
    else:
        sel = contested.to(refined_logits.device).bool()
    idx = sel.nonzero(as_tuple=False)                                  # [P, 2] as (y, x)
    if idx.numel() == 0:
        zero = refined_logits.sum() * 0.0
        return zero, {"vor_ce": 0.0, "vor_n": 0}
    if idx.shape[0] > max_pixels:
        pick = torch.randperm(idx.shape[0], device=idx.device, generator=generator)[:max_pixels]
        idx = idx[pick]

    pix_xy = torch.stack([idx[:, 1], idx[:, 0]], dim=-1).float()       # [P, 2] as (x, y)

    # Nearest-keypoint assignment. Masked keypoints are pushed to +inf distance so they can never
    # win, rather than being dropped (which would change the per-animal keypoint count and silently
    # bias the assignment toward animals with more valid keypoints).
    kp = keypoints_mask_space.reshape(-1, 2).float()                   # [B*K, 2]
    d = torch.cdist(pix_xy.unsqueeze(0), kp.unsqueeze(0)).squeeze(0)   # [P, B*K]
    if valid is not None:
        d = d.masked_fill(~valid.reshape(1, -1), float("inf"))
    k = keypoints_mask_space.shape[1]
    owner = (d.argmin(dim=1) // k).long()                              # [P] in [0, B)

    logits_p = refined_logits[:, idx[:, 0], idx[:, 1]].transpose(0, 1)  # [P, B]
    ce = F.cross_entropy(logits_p, owner)
    with torch.no_grad():
        acc = (logits_p.argmax(dim=1) == owner).float().mean().item()
    return ce, {"vor_ce": float(ce.detach()), "vor_acc": acc, "vor_n": int(owner.numel())}


def mask_area_regulariser(
    refined_logits: torch.Tensor,
    base_logits: torch.Tensor,
) -> torch.Tensor:
    """Penalise gross area change between the refined and the original masks.

    Closes the shrinkage degeneracy. A centroid- or keypoint-based objective is largely invariant to
    symmetric erosion, so a module can lower its loss by eating the animal's body while keeping the
    anchors satisfied. TIAB is meant to REASSIGN pixels at a boundary, not delete them, so total
    area per object should be roughly conserved. Computed on soft probabilities so it is
    differentiable everywhere and does not reintroduce a hard gate.
    """
    p_new = torch.sigmoid(refined_logits).flatten(1).sum(dim=1)
    p_old = torch.sigmoid(base_logits).flatten(1).sum(dim=1).clamp(min=1.0)
    return ((p_new - p_old) / p_old).pow(2).mean()


def combined_kp_loss(
    refined_logits: torch.Tensor,
    base_logits: torch.Tensor,
    keypoints_mask_space: torch.Tensor,
    contested: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
    lambda_kp: float = 1.0,
    lambda_voronoi: float = 1.0,
    lambda_area: float = 0.1,
    max_pixels: int = 4096,
) -> Tuple[torch.Tensor, dict]:
    """The full objective. Set lambda_voronoi=0 for the sparse-supervision ablation arm.

    Every term is active on every 2-animal frame: there is no margin that can silently switch the
    objective off, which was defect (3) of the shipped loss.
    """
    parts: dict = {}
    total = refined_logits.sum() * 0.0                                 # grad-carrying zero

    if lambda_kp > 0:
        l_kp, p = keypoint_ownership_loss(refined_logits, keypoints_mask_space, valid)
        total = total + lambda_kp * l_kp
        parts.update(p)
    if lambda_voronoi > 0:
        l_v, p = voronoi_ownership_loss(refined_logits, keypoints_mask_space,
                                        contested, max_pixels, valid)
        total = total + lambda_voronoi * l_v
        parts.update(p)
    if lambda_area > 0:
        l_a = mask_area_regulariser(refined_logits, base_logits)
        total = total + lambda_area * l_a
        parts["area"] = float(l_a.detach())

    parts["total"] = float(total.detach())
    return total, parts
