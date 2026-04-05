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
        # Output: per-pixel logit adjustment
        self.refine_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    # Max contested pixels before downsampling to prevent OOM.
    # 300K contested pixels would create a 5.76TB attention matrix.
    MAX_CONTESTED = 4096

    def forward(self, pixel_features, identity_embs, contested_mask):
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
        # Trajectory encoder
        self.traj_gru = nn.GRU(2, 64, batch_first=True)
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
        # Trajectory
        traj_out, _ = self.traj_gru(centroid_history)
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

    def __init__(self, identity_dim=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(identity_dim + 1, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

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
    ):
        super().__init__()
        self.contest_margin = contest_margin

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
    ):
        """
        Args:
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
        else:
            sorted_scores, _ = pred_masks.sort(dim=0, descending=True)
            score_diff = sorted_scores[0] - sorted_scores[1]
            contested = score_diff < self.contest_margin

        n_contested = contested.sum().item()

        # If no contested pixels, fall back to standard argmax
        if n_contested == 0:
            return self._hard_argmax(pred_masks)

        # Step 2: Encode identity
        identity_embs = self.identity_encoder(
            appearance_embs, centroid_history,
        )  # [B, identity_dim]

        # Step 3: Operate at feature resolution to avoid 512MB upsample.
        # Downsample contested mask to pix_feat resolution, run attention
        # there (~3MB instead of ~512MB), then bilinear-upsample the
        # refinement back to mask resolution.
        Hf, Wf = pix_feat.shape[-2:]
        contested_feat_res = F.interpolate(
            contested.unsqueeze(0).unsqueeze(0).float(),
            size=(Hf, Wf), mode="nearest",
        ).squeeze(0).squeeze(0).bool()

        # Step 4: Boundary attention at feature resolution
        refinement_feat = self.boundary_attention(
            pix_feat, identity_embs, contested_feat_res,
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
        return self._hard_argmax(refined_masks)

    @staticmethod
    def _hard_argmax(pred_masks):
        """Standard non-overlapping constraint via argmax on [B, H, W] input."""
        B = pred_masks.size(0)
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)  # [1, H, W]
        batch_obj_inds = torch.arange(B, device=pred_masks.device)[:, None, None]  # [B, 1, 1]
        keep = max_obj_inds == batch_obj_inds  # [B, H, W]
        return torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
