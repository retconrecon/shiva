"""TIAB: Temporal Identity-Aware Boundary Module.

Replaces SAM3.1's identity-agnostic argmax in _apply_non_overlapping_constraints
with a learned boundary refinement conditioned on per-object identity signals
(appearance embeddings + trajectory history).

Operates ONLY on contested pixels (where top-2 object scores are within a
margin), leaving uncontested regions untouched. This preserves SAM3.1's strong
single-object segmentation while learning to resolve identity at boundaries.

Insertion point: video_tracking_multiplex.py, inside _encode_new_memory,
replacing the call to _apply_non_overlapping_constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryAttention(nn.Module):
    """Cross-object attention over contested boundary pixels.

    Extracts features at pixels where two or more objects contest ownership,
    runs cross-attention conditioned on identity embeddings, and produces
    per-object logit adjustments for those pixels only.
    """

    def __init__(self, backbone_dim, identity_dim, hidden_dim=64, num_heads=4):
        super().__init__()
        # Project backbone features to hidden dim
        self.pixel_proj = nn.Conv2d(backbone_dim, hidden_dim, 1)
        # Project identity embedding to per-object query
        self.identity_proj = nn.Linear(identity_dim, hidden_dim)
        # Cross-attention: identity queries attend to boundary pixel features
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        # ☢ GEOMETRY CHANNELS. Without these the head CANNOT REPRESENT THE TASK.
        #
        # `pixel_features` arrives as [1, C, H, W] and is expanded across objects, so
        # `contested_feats` is byte-identical for every object; the only per-object variation
        # reaching a pixel is the single 64-d attention `context` vector. The head therefore
        # computes refine_head(shared_field(p) + context_b) - one broadcast vector per object
        # modulating a field that is the same for everyone. To decide "does this contested pixel
        # belong to object 1 or object 2?" it needs the pixel's position RELATIVE TO EACH OBJECT,
        # and it currently has no coordinate, no distance, and no motion input at all.
        #
        # Five cheap channels supply exactly that, per object per pixel:
        #   (x, y)      normalised cell coordinates
        #   (dx, dy)    offset from the cell to THAT object's centroid
        #   d           its magnitude
        # This is the nearly-free prior the GRU was otherwise expected to launder through a 64-d
        # bottleneck. Cost: 5 * hidden_dim extra weights (~320), i.e. capacity is not the point.
        self.n_geom = 5
        self.refine_head = nn.Sequential(
            nn.Linear(hidden_dim + self.n_geom, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    # Max contested pixels before downsampling to prevent OOM.
    # 300K contested pixels would create a 5.76TB attention matrix.
    MAX_CONTESTED = 4096

    def forward(self, pixel_features, identity_embs, contested_mask, obj_centroids=None):
        """
        Args:
            pixel_features: [B, C, H, W] backbone features (stride 16)
            identity_embs: [B, D_identity] per-object identity embeddings
            contested_mask: [H, W] bool — which pixels are contested

        Returns:
            refinement: [B, H, W] logit adjustments (zero at non-contested pixels)
        """
        B, C, H, W = pixel_features.shape
        n_contested = contested_mask.sum().item()

        if n_contested == 0:
            return torch.zeros(B, H, W, device=pixel_features.device)

        # Project pixel features
        pf = self.pixel_proj(pixel_features)  # [B, hidden, H, W]

        # Extract contested pixel features: [B, n_contested, hidden]
        contested_feats = pf[:, :, contested_mask].permute(0, 2, 1)

        # Cap contested pixels to prevent OOM — randomly subsample if too many
        if n_contested > self.MAX_CONTESTED:
            indices = torch.randperm(n_contested, device=pf.device)[:self.MAX_CONTESTED]
            contested_feats = contested_feats[:, indices]

        # Identity query: single vector per object → global context
        # O(n_contested) instead of O(n_contested²)
        id_query = self.identity_proj(identity_embs).unsqueeze(1)  # [B, 1, hidden]

        # Debug: log shapes on first call
        if not getattr(self, '_shape_logged', False):
            import logging
            logging.getLogger(__name__).info(
                f"BoundaryAttention shapes: query={id_query.shape} "
                f"key={contested_feats.shape} pix_feat_in={pixel_features.shape} "
                f"n_contested={n_contested}")
            self._shape_logged = True

        context, _ = self.cross_attn(
            id_query, contested_feats, contested_feats,
        )  # [B, 1, hidden]

        # Combine global context with per-pixel features via pointwise MLP
        # Context is broadcast to all contested pixels
        if n_contested > self.MAX_CONTESTED:
            # Re-extract full contested features for the refinement head
            contested_feats = pf[:, :, contested_mask].permute(0, 2, 1)
        combined = contested_feats + context.expand_as(contested_feats)
        combined = self.norm(combined)

        # Per-object geometry for each contested cell (see n_geom above). Built here rather than
        # upstream so it always matches the exact cells `contested_mask` selected.
        idx = contested_mask.nonzero(as_tuple=False)                  # [n, 2] as (y, x)
        ny = idx[:, 0].float() / max(H - 1, 1)
        nx = idx[:, 1].float() / max(W - 1, 1)
        base = torch.stack([nx, ny], dim=-1).unsqueeze(0).expand(B, -1, -1)   # [B, n, 2]
        if obj_centroids is None:
            # Fall back to frame centre for every object: geometry degrades to absolute position
            # rather than silently mis-informing the head with a wrong per-object reference.
            obj_centroids = torch.full((B, 2), 0.5, device=pixel_features.device)
        off = base - obj_centroids.to(base.device).unsqueeze(1)       # [B, n, 2]
        dist = off.norm(dim=-1, keepdim=True)                         # [B, n, 1]
        combined = torch.cat([combined, base, off, dist], dim=-1)

        # Produce per-pixel adjustment
        adjustments = self.refine_head(combined).squeeze(-1)  # [B, n_contested]

        # Scatter back to full spatial grid
        refinement = torch.zeros(B, H, W, device=pixel_features.device)
        refinement[:, contested_mask] = adjustments

        return refinement


class TemporalIdentityEncoder(nn.Module):
    """Fuses appearance embedding with trajectory history into a
    per-object identity vector for boundary conditioning.

    Appearance comes from the existing SHIVA appearance store (OSNet 512-dim
    or histogram). Trajectory is the last K centroid positions, encoded
    via a small GRU.
    """

    def __init__(self, appearance_dim=512, trajectory_len=16, hidden_dim=128):
        super().__init__()
        self.trajectory_len = trajectory_len
        # ☢ VELOCITY, NOT JUST POSITION. The GRU took (x, y) only - but during a crossing the two
        # animals occupy nearly the SAME coordinates, so their position histories converge and the
        # identity embeddings converge with them, precisely at the moment identity is needed. Heading
        # and speed are what still separate them there. Feeding first differences (dx, dy) alongside
        # position hands the encoder that signal directly instead of asking it to recover derivatives
        # from a raw sequence through a 64-unit bottleneck. Input 2 -> 4.
        self.traj_gru = nn.GRU(4, 64, batch_first=True)
        self.traj_proj = nn.Linear(64, hidden_dim)
        # Appearance projection
        self.appear_proj = nn.Linear(appearance_dim, hidden_dim)
        # Fusion
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        # Phase 1 training: drop appearance entirely so the model
        # learns trajectory-only identity. Set to True during Phase 1
        # training, False during Phase 2+ and inference.
        self.drop_appearance = False

    def forward(self, appearance_emb, centroid_history):
        """
        Args:
            appearance_emb: [B, appearance_dim] — from SHIVA appearance store
            centroid_history: [B, K, 2] — last K normalized (x, y) positions

        Returns:
            identity_emb: [B, hidden_dim]
        """
        # Trajectory: [x, y, dx, dy]. First differences, zero-padded at t=0 so the sequence length
        # is unchanged and a fresh object (whose history is a repeated constant, per the padding rule
        # in shiva_tracker) yields zero velocity rather than a discontinuity.
        _vel = torch.zeros_like(centroid_history)
        _vel[:, 1:] = centroid_history[:, 1:] - centroid_history[:, :-1]
        traj_out, _ = self.traj_gru(torch.cat([centroid_history, _vel], dim=-1))
        traj_feat = self.traj_proj(traj_out[:, -1])  # last hidden state

        # Appearance — zeroed during Phase 1 training so fuse layer
        # learns to rely on trajectory only. appear_proj weights stay
        # at initialization, ready for Phase 2 fine-tuning with real
        # embeddings.
        if self.drop_appearance:
            appear_feat = torch.zeros_like(traj_feat)
        else:
            appear_feat = self.appear_proj(appearance_emb)

        # Fuse
        return self.fuse(torch.cat([appear_feat, traj_feat], dim=-1))


class RefinementGate(nn.Module):
    """Learns when to intervene vs defer to SAM3.1's argmax.

    Outputs a scalar gate in [0, 1] per object:
    - ~0 on non-crossing frames (defer to SAM3.1)
    - ~1 on crossing frames (apply TIAB refinement)
    """

    def __init__(self, identity_dim=128, init_gate_bias=-4.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(identity_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # ☢ INITIALISE THE GATE CLOSED. With PyTorch's default init the final Linear outputs ~0, so
        # Sigmoid gives ~0.5 and the module applies ~55% of a COMPLETELY RANDOM refinement to every
        # object on every frame before it has learned anything (measured: mean gate 0.5453 over 2000
        # random inputs). That is why an untrained TIAB does not merely fail to help, it actively
        # corrupts masks - and why `ZEUS_TIAB=1` has always been worse than leaving TIAB off.
        #
        # Biasing the pre-sigmoid to -4 puts the gate at ~0.02, so at initialisation the module is
        # effectively PASS-THROUGH (refined = pred_masks + 0.02 * refinement) and is therefore
        # equivalent to TIAB=0 up to a negligible perturbation. Training can then only open the gate
        # where the loss says intervening helps. This makes "trained >= untrained" a property of the
        # PARAMETERISATION rather than something we have to hope the optimiser discovers.
        #
        # Same trick as a near-zero-initialised residual branch (ResNet `zero_init_residual`, Fixup)
        # and the standard large-negative bias on a sigmoid gate: start at the identity function.
        nn.init.constant_(self.gate[2].bias, init_gate_bias)

    def forward(self, identity_embs, object_scores):
        """
        Args:
            identity_embs: [B, identity_dim]
            object_scores: [B, 1] — SAM3.1 confidence logits

        Returns:
            gate: [B, 1] — per-object intervention weight
        """
        gate_input = torch.cat([identity_embs, object_scores], dim=-1)
        return self.gate(gate_input)


class TemporalIdentityBoundaryModule(nn.Module):
    """Full TIAB module: replaces argmax with learned identity-aware
    boundary refinement.

    Call flow:
        1. Identify contested pixels (top-2 score margin < threshold)
        2. Encode per-object identity (appearance + trajectory)
        3. Run boundary attention over contested pixels
        4. Gate the refinement
        5. Add gated refinement to original logits
        6. Apply standard argmax on refined logits
    """

    def __init__(
        self,
        backbone_dim=256,
        appearance_dim=512,
        trajectory_len=16,
        identity_dim=128,
        hidden_dim=64,
        num_heads=4,
        contest_margin=2.0,
        contest_fg_threshold=0.0,
    ):
        super().__init__()
        self.contest_margin = contest_margin
        # Logit threshold for 'some object claims this pixel'. 0.0 = the sigmoid midpoint,
        # i.e. the same foreground criterion the tracker itself uses when binarising.
        self.contest_fg_threshold = contest_fg_threshold

        self.identity_encoder = TemporalIdentityEncoder(
            appearance_dim=appearance_dim,
            trajectory_len=trajectory_len,
            hidden_dim=identity_dim,
        )
        self.boundary_attention = BoundaryAttention(
            backbone_dim=backbone_dim,
            identity_dim=identity_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.gate = RefinementGate(identity_dim=identity_dim)

    def forward(
        self,
        pred_masks,
        pix_feat,
        appearance_embs,
        centroid_history,
        object_score_logits,
        return_pre_argmax: bool = False,
    ):
        """
        Args:
            return_pre_argmax: return refined logits BEFORE the non-overlap argmax. Training must
                set this - see the note at the return site. Inference leaves it False.
            pred_masks: [B, H, W] raw mask logits from SAM3.1 decoder
            pix_feat: [B, C, Hf, Wf] backbone features (stride 16)
            appearance_embs: [B, D_appear] per-object appearance embeddings
            centroid_history: [B, K, 2] last K centroid positions (normalized 0-1)
            object_score_logits: [B, 1] SAM3.1 object confidence

        Returns:
            refined_masks: [B, H, W] refined mask logits (non-overlapping)
        """
        B, H, W = pred_masks.shape
        device = pred_masks.device

        # Early exit for single object — no boundary to refine
        if B <= 1:
            return pred_masks

        # Step 1: Find contested pixels
        # Sort scores along object dim, check if top-2 are within margin
        if B == 2:
            score_diff = (pred_masks[0] - pred_masks[1]).abs()
            contested = score_diff < self.contest_margin
            _top1 = torch.maximum(pred_masks[0], pred_masks[1])
        else:
            sorted_scores, _ = pred_masks.sort(dim=0, descending=True)
            score_diff = sorted_scores[0] - sorted_scores[1]
            contested = score_diff < self.contest_margin
            _top1 = sorted_scores[0]

        # ☢ FOREGROUND GATE. Without it the contested set is dominated by BACKGROUND, which is the
        # opposite of what this module is for.
        #
        # `top1 - top2 < margin` is true wherever the objects AGREE - and they agree most strongly on
        # background, where every logit is confidently negative (e.g. -18 vs -19 differ by 1.0 < 2.0).
        # Since background is the large majority of a top-down arena frame, the module was spending
        # its capacity, its attention budget and its MAX_CONTESTED cap on empty bedding rather than on
        # the animal-animal boundary named in the class docstring.
        #
        # Requiring the winning logit to be foreground (> tau) restricts the set to pixels some object
        # actually claims, which is where a boundary can exist at all. This also makes the
        # MAX_CONTESTED=4096 subsample (and its non-deterministic randperm) far less likely to fire.
        contested = contested & (_top1 > self.contest_fg_threshold)

        n_contested = contested.sum().item()

        # If no contested pixels, fall back to standard argmax
        if n_contested == 0:
            # Same reasoning as the main return: training needs pre-argmax logits. A frame with no
            # contested pixels still contributes to the loss (the keypoint terms are always active),
            # and argmax-ing it here would zero the gradient on every losing pixel of that frame.
            return pred_masks if return_pre_argmax else self._hard_argmax(pred_masks)

        # Step 2: Encode identity
        identity_embs = self.identity_encoder(
            appearance_embs, centroid_history,
        )  # [B, identity_dim]

        # Expand pix_feat if shared backbone (B=1) to match per-object batch
        if pix_feat.shape[0] == 1 and B > 1:
            pix_feat = pix_feat.expand(B, -1, -1, -1)

        # Step 3: Operate at feature resolution to avoid 512MB upsample.
        # Downsample contested mask to pix_feat resolution, run attention
        # there (~3MB instead of ~512MB), then bilinear-upsample the
        # refinement back to mask resolution.
        Hf, Wf = pix_feat.shape[-2:]
        # ☢ MAX-POOL, NOT NEAREST. `mode="nearest"` keeps exactly ONE source pixel per (H/Hf x W/Wf)
        # block - at 1152 -> 72 that is 1 of every 16x16 = 256 pixels. A boundary band is only ~2 px
        # wide, so nearest retains it with probability ~2/16 per crossed cell and DROPS ~85% of the
        # cells the band actually touches: the module was mostly not seeing the boundary it exists to
        # refine. Max-pooling marks a cell contested if ANY pixel in it is, which is the correct
        # semantics for "does this cell contain contested boundary?".
        contested_feat_res = F.adaptive_max_pool2d(
            contested.unsqueeze(0).unsqueeze(0).float(), (Hf, Wf),
        ).squeeze(0).squeeze(0).bool()

        # Per-object centroid in NORMALISED feature-grid coordinates, for the geometry channels.
        # Soft (probability-weighted) so it is differentiable and stays defined when a mask is
        # nearly empty; falls back to the frame centre for an object with no mass at all.
        with torch.no_grad():
            _p = torch.sigmoid(pred_masks)                                  # [B, H, W]
            _hh, _ww = _p.shape[-2], _p.shape[-1]
            _ys = torch.linspace(0, 1, _hh, device=_p.device).view(1, _hh, 1)
            _xs = torch.linspace(0, 1, _ww, device=_p.device).view(1, 1, _ww)
            _m = _p.flatten(1).sum(dim=1).clamp(min=1e-6)                   # [B]
            _cy = (_p * _ys).flatten(1).sum(dim=1) / _m
            _cx = (_p * _xs).flatten(1).sum(dim=1) / _m
            _empty = (_p.flatten(1).sum(dim=1) < 1e-5)
            obj_centroids = torch.stack([_cx, _cy], dim=-1)                 # [B, 2]
            obj_centroids[_empty] = 0.5

        # Step 4: Boundary attention at feature resolution
        refinement_feat = self.boundary_attention(
            pix_feat, identity_embs, contested_feat_res, obj_centroids=obj_centroids,
        )  # [B, Hf, Wf]

        # Step 5: Upsample refinement to mask resolution (no learnable params)
        refinement = F.interpolate(
            refinement_feat.unsqueeze(1), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(1)  # [B, H, W]

        # Zero out refinement at non-contested pixels (upsampling may bleed)
        refinement = refinement * contested.float()

        # Step 6: Gate
        gate_val = self.gate(
            identity_embs,
            object_score_logits.view(B, 1) if object_score_logits.dim() == 1
            else object_score_logits[:, :1],
        )  # [B, 1]
        gated_refinement = refinement * gate_val.unsqueeze(-1)

        # Step 7: Add refinement to original logits, then argmax
        refined_masks = pred_masks + gated_refinement

        # ☢ TRAINING MUST SEE THE PRE-ARGMAX LOGITS.
        # `_hard_argmax` routes every LOSING pixel through `torch.clamp(x, max=-10.0)`, whose
        # gradient is exactly 0 for x > -10 - and every contested pixel is within `contest_margin`
        # of the top score by construction, so it is always in that range. Training on the
        # post-argmax tensor therefore gives zero gradient to the losing object at exactly the
        # pixels TIAB exists to reassign: the module can reweight mass it already owns but can never
        # learn to CLAIM a pixel. That single fact is sufficient to explain why TIAB has never
        # produced a measurable improvement.
        #
        # Inference keeps the hard argmax (a tracker needs non-overlapping masks); only the training
        # path takes the soft branch. Default False so the inference contract is unchanged.
        if return_pre_argmax:
            return refined_masks
        return self._hard_argmax(refined_masks)

    @staticmethod
    def _hard_argmax(pred_masks):
        """Standard non-overlapping constraint via argmax on [B, H, W] input."""
        B = pred_masks.size(0)
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)  # [1, H, W]
        batch_obj_inds = torch.arange(B, device=pred_masks.device)[:, None, None]  # [B, 1, 1]
        keep = max_obj_inds == batch_obj_inds  # [B, H, W]
        return torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
