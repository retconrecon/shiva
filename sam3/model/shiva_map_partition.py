"""SHIVA MAP contested-pixel partition — identity-aware replacement for argmax.

THE DEFECT
----------
SAM3 decides which object owns a contested pixel with a pure per-pixel argmax
over object mask logits:

    max_obj_inds = torch.argmax(pred_masks, dim=0)

No position, no velocity, no notion of whose body that pixel was a frame ago.
That is a maximum-likelihood decision with an implicit uniform prior. When two
animals overlap, the pixels over animal A's body can tip to B on nothing more
than a slightly louder logit.

WHY IT COMPOUNDS
----------------
In this fork the partition does NOT reach the emitted mask: `build_outputs`
interpolates raw per-object decoder output and never resolves overlap. The
partition feeds the MEMORY ENCODER (sam3_multiplex_base._tracker_update_memories),
and `object_score_logits` is then derived from mere non-emptiness. So:

    B steals a pixel -> B's memory now contains part of A's body
      -> B's logit there is higher next frame -> the takeover accelerates

which is exactly why the observed failure is gradual, one-directional, and
invisible to presence-based metrics. Fixing the decision here is upstream
prevention; it pairs with the existing memory-purge work, which is downstream
cleanup.

THE FIX
-------
Turn the decision from maximum-likelihood into maximum-a-posteriori by adding
log-priors before the argmax:

    S_o(p) = L_o(p) + lambda(p) * [ w_c * centroid_o(p) + w_p * persist_o(p) ]
    winner(p) = argmax_o S_o(p)

  L_o(p)        SAM mask logit. Unchanged, and still the dominant term.
  centroid_o(p) Gaussian log-prior around object o's KALMAN-PREDICTED centroid.
                Coarse but robust, and it has inertia a slow creep cannot move.
  persist_o(p)  did o own this pixel last frame, after warping by o's predicted
                velocity. Body-shaped by construction, so unlike the centroid
                term it does not penalise an animal's own tail.
  lambda(p)     adaptive gate: ~0 where one logit clearly dominates, large only
                where the top two are close.

lambda is the whole safety argument. A constant heavy prior would glue an
animal to where it is expected and refuse a real fast dart, trading one failure
for another. Gated on the top-two margin, the prior touches only pixels that
were genuine coin-flips, and a frame where every pixel is decisive comes out
bit-identical to today.

CRITICALLY: the prior chooses the ASSIGNMENT, never the VALUES. Retained
logits are the original L_o, so nothing biased is written into memory; we only
change who wins a tie.

STATUS OF THE DEFAULTS
----------------------
The weights below are a starting point, NOT a validated setting. The tuning
set already exists and the objective is stated in identity_swap_fix_design.md:
the correct weights are the ones that flip the known-bad video back to correct
without regressing a single known-clean video. Treat every constant here as a
hypothesis until that A/B has been run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sam3.model.shiva_instrumentation import bump as _shiva_bump

logger = logging.getLogger(__name__)

__all__ = ["MapPartitionConfig", "map_partition_labels", "build_coord_grid"]


@dataclass
class MapPartitionConfig:
    """Weights for the MAP partition. See module docstring on tuning."""

    # Peak strength of the prior, in logits, at a perfectly tied pixel. Mask
    # logits run to roughly +/-20, so 4.0 is decisive on a tie and negligible
    # against a confident pixel.
    lambda_max: float = 4.0
    # Margin (in logits) at which the gate has decayed by 1/e. Small values keep
    # the prior confined to near-exact ties.
    margin_tau: float = 2.0
    # Relative weights of the two priors.
    w_centroid: float = 1.0
    w_persist: float = 0.5
    # Floor on the Gaussian width, in normalized image units. Must be at least
    # animal-body-scale or the prior penalises an animal's own extremities:
    # 0.06 is ~55 px at 912. The Kalman's own position sigma is used when it is
    # larger, which is what makes the prior widen and back off through a long
    # occlusion instead of staying overconfident.
    sigma_floor: float = 0.06
    sigma_scale: float = 2.0
    # Most negative value the centroid log-prior may reach, so a far-away pixel
    # saturates instead of growing without bound.
    centroid_clip: float = 4.0
    # Coast-out: once a filter has missed this many consecutive frames its
    # prediction is no longer trusted and its prior is dropped entirely.
    max_coast_frames: int = 90


_GRID_CACHE: dict = {}


def build_coord_grid(h: int, w: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalized pixel-centre coordinate grids, cached per (h, w, device, dtype)."""
    key = (h, w, str(device), str(dtype))
    got = _GRID_CACHE.get(key)
    if got is None:
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        got = (xs.view(1, 1, w), ys.view(1, h, 1))
        # One entry per resolution actually used; a handful at most.
        if len(_GRID_CACHE) < 16:
            _GRID_CACHE[key] = got
    return got


def _shift_zero_fill(t: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate a [H, W] map by (dy, dx), filling vacated cells with zeros.

    torch.roll wraps, which would teleport an animal's persistence prior from
    one image edge to the opposite one. For an animal swimming near a wall that
    is not a rounding detail, it is a prior asserting it is somewhere it
    provably is not.
    """
    H, W = t.shape
    out = torch.zeros_like(t)
    if abs(dy) >= H or abs(dx) >= W:
        return out
    ys_src = slice(max(0, -dy), H - max(0, dy))
    ys_dst = slice(max(0, dy), H - max(0, -dy))
    xs_src = slice(max(0, -dx), W - max(0, dx))
    xs_dst = slice(max(0, dx), W - max(0, -dx))
    out[ys_dst, xs_dst] = t[ys_src, xs_src]
    return out


def map_partition_labels(
    pred_masks: torch.Tensor,
    centroids: Optional[torch.Tensor],
    sigmas: Optional[torch.Tensor],
    valid: Optional[torch.Tensor],
    prev_labels: Optional[torch.Tensor],
    velocities: Optional[torch.Tensor],
    cfg: MapPartitionConfig,
) -> torch.Tensor:
    """Return per-pixel winning object indices under the MAP score.

    Args:
        pred_masks: [B, 1, H, W] or [B, H, W] object mask logits.
        centroids:  [B, 2] predicted (x, y) in normalized coords, or None.
        sigmas:     [B] predicted positional 1-sigma, normalized, or None.
        valid:      [B] bool, whether object b has a usable prediction.
        prev_labels:[H, W] long, previous frame's winning indices, or None.
        velocities: [B, 2] predicted per-frame (dx, dy) normalized, or None.
        cfg:        weights.

    Returns:
        [1, H, W] long tensor of winning object indices, same convention as
        `torch.argmax(pred_masks, dim=0, keepdim=True)`.
    """
    logits = pred_masks[:, 0] if pred_masks.dim() == 4 else pred_masks
    B, H, W = logits.shape
    if B <= 1:
        return torch.argmax(logits, dim=0, keepdim=True)

    dtype = logits.dtype if logits.is_floating_point() else torch.float32
    score = logits.to(dtype)

    # Adaptive gate from the top-two margin. topk over a small B is cheap.
    top2 = torch.topk(score, k=2, dim=0).values           # [2, H, W]
    margin = (top2[0] - top2[1]).clamp_min(0.0)           # [H, W]
    lam = cfg.lambda_max * torch.exp(-margin / max(cfg.margin_tau, 1e-6))

    prior = torch.zeros_like(score)
    used_any = False

    # --- centroid term ---------------------------------------------------
    if centroids is not None and valid is not None and bool(valid.any()):
        xs, ys = build_coord_grid(H, W, score.device, dtype)
        cx = centroids[:, 0].to(dtype).view(B, 1, 1)
        cy = centroids[:, 1].to(dtype).view(B, 1, 1)
        if sigmas is None:
            sig = torch.full((B, 1, 1), cfg.sigma_floor, device=score.device, dtype=dtype)
        else:
            sig = (sigmas.to(dtype) * cfg.sigma_scale).clamp_min(cfg.sigma_floor)
            sig = sig.view(B, 1, 1)
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2               # [B, H, W]
        cterm = -0.5 * d2 / (sig ** 2)
        # Objects with no usable prediction must not compete in this term at
        # all: push them to -inf BEFORE the relative normalization below, then
        # flatten them to 0 after, so they neither gain nor lose from it.
        neg_inf = torch.finfo(cterm.dtype).min / 4
        cterm = torch.where(valid.view(B, 1, 1), cterm, torch.full_like(cterm, neg_inf))
        # Make the term RELATIVE before clipping. Clipping the absolute
        # log-prior saturates every object at -clip once all of them are far
        # from their prediction, which destroys the ordering the prior exists
        # to provide and hands the tie to the lowest object index. Subtracting
        # the per-pixel best keeps "which object is closest" intact while still
        # bounding the prior's influence to `centroid_clip` logits.
        cterm = cterm - cterm.max(dim=0, keepdim=True).values
        cterm = cterm.clamp_min(-abs(cfg.centroid_clip))
        cterm = torch.where(valid.view(B, 1, 1), cterm, torch.zeros_like(cterm))
        prior = prior + cfg.w_centroid * cterm
        used_any = True

    # --- persistence term ------------------------------------------------
    # "Did object b own this pixel last frame, after moving b by its predicted
    # displacement." Warping matters: without it the term is a brake on motion
    # rather than a prior on identity.
    if prev_labels is not None and cfg.w_persist != 0.0:
        pterm = torch.zeros_like(score)
        for b in range(B):
            owned = prev_labels == b                       # [H, W] bool
            if not bool(owned.any()):
                continue
            if velocities is not None:
                dx = int(round(float(velocities[b, 0]) * W))
                dy = int(round(float(velocities[b, 1]) * H))
                if dx or dy:
                    owned = _shift_zero_fill(owned, dy, dx)
            pterm[b] = owned.to(dtype)
        prior = prior + cfg.w_persist * pterm
        used_any = True

    if not used_any:
        return torch.argmax(logits, dim=0, keepdim=True)

    score = score + lam.unsqueeze(0) * prior

    labels = torch.argmax(score, dim=0, keepdim=True)      # [1, H, W]

    # Count only pixels where the prior actually overturned the appearance
    # argmax. This is the number that says whether the feature is doing
    # anything, and it is the first thing to look at in an A/B.
    if not torch.compiler.is_compiling():
        plain = torch.argmax(logits, dim=0, keepdim=True)
        changed = int((labels != plain).sum())
        if changed:
            _shiva_bump("map_partition_pixels_flipped", changed)
            _shiva_bump("map_partition_frames_active")

    return labels
