"""TIAB training loop.

Trains the TemporalIdentityBoundaryModule on pre-extracted SAM3.1
frame data. SAM3.1 is NOT loaded during training — only the
pre-extracted tensors (pred_masks, pix_feat, object_scores) are used.

Usage:
    from sam3.model.tiab_train import train_tiab

    train_tiab(
        data_dirs=["outputs/tiab_extract_four_fish_2"],
        output_dir="outputs/tiab_weights",
        num_epochs=10,
    )

Or as standalone:
    python -m sam3.model.tiab_train \\
        --data-dirs outputs/tiab_extract_four_fish_2 outputs/tiab_extract_mice_2_1 \\
        --output-dir outputs/tiab_weights \\
        --epochs 10
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, SubsetRandomSampler

from sam3.model.tiab import TemporalIdentityBoundaryModule
from sam3.model.tiab_dataset import TIABDataset
from sam3.model.tiab_losses import mask_centroid, tiab_combined_loss


def train_tiab(
    data_dirs,
    output_dir,
    # Architecture
    backbone_dim=256,
    appearance_dim=512,
    trajectory_len=16,
    identity_dim=128,
    hidden_dim=64,
    num_heads=4,
    contest_margin=2.0,
    # Training
    num_epochs=10,
    batch_size=8,
    lr=1e-4,
    weight_decay=0.01,
    # Loss weights
    lambda_identity=1.0,
    lambda_boundary=0.5,
    lambda_gate=0.1,
    # Data
    crossing_ratio=0.7,
    image_size=1008,
    # Phase control
    phase=1,  # 1 = trajectory only, 2 = appearance + trajectory
    # Device
    device="cuda",
    # Checkpointing
    save_every=1,
):
    """Train the TIAB module on pre-extracted data.

    Args:
        data_dirs: list of directories containing extracted frame .pt files
        output_dir: where to save model weights and training logs
        backbone_dim: SAM3.1 backbone feature dimension (256 for Hiera)
        appearance_dim: appearance embedding dimension (512 for OSNet)
        num_epochs: training epochs
        batch_size: frames per batch
        lr: learning rate
        device: "cuda" or "cpu"
    """
    os.makedirs(output_dir, exist_ok=True)

    # Reproducibility
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load dataset
    dataset = TIABDataset(
        data_dirs,
        clip_length=1,  # single-frame training for Phase 1
        crossing_ratio=crossing_ratio,
        trajectory_len=trajectory_len,
        image_size=image_size,
    )

    # Create model
    model = TemporalIdentityBoundaryModule(
        backbone_dim=backbone_dim,
        appearance_dim=appearance_dim,
        trajectory_len=trajectory_len,
        identity_dim=identity_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        contest_margin=contest_margin,
    ).to(device)

    if phase == 1:
        # Phase 1: drop appearance so model learns trajectory-only identity.
        model.identity_encoder.drop_appearance = True
        print(f"TIAB parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("Phase 1: appearance dropped — training trajectory-only identity")
        # Exclude appear_proj from optimization (weights stay at init for Phase 2)
        appear_proj_ids = set(id(p) for p in model.identity_encoder.appear_proj.parameters())
        train_params = [p for p in model.parameters() if id(p) not in appear_proj_ids]
    else:
        # Phase 2: appearance + trajectory. All parameters trained.
        model.identity_encoder.drop_appearance = False
        print(f"TIAB parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("Phase 2: appearance + trajectory — full identity signal")
        train_params = list(model.parameters())

    optimizer = torch.optim.AdamW(
        train_params, lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs,
    )

    # Training log
    log = {"epochs": [], "config": {
        "data_dirs": data_dirs,
        "backbone_dim": backbone_dim,
        "appearance_dim": appearance_dim,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "contest_margin": contest_margin,
        "lambda_identity": lambda_identity,
        "lambda_boundary": lambda_boundary,
        "lambda_gate": lambda_gate,
    }}

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()

        # Balanced sampling each epoch
        sampler_indices = dataset.get_balanced_sampler()
        sampler = SubsetRandomSampler(sampler_indices)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

        epoch_losses = {
            "total": 0.0, "identity": 0.0,
            "boundary": 0.0, "gate": 0.0,
        }
        n_batches = 0
        n_contested = 0

        for batch in loader:
            pred_masks = batch["pred_masks"].to(device)
            pix_feat = batch["pix_feat"].to(device)
            object_scores = batch["object_scores"].to(device)
            gt_centroids = batch["gt_centroids"].to(device)
            centroid_history = batch["centroid_history"].to(device)
            is_crossing = batch["is_crossing"]

            # Process each sample in the batch independently
            # (different samples may have different numbers of objects)
            batch_loss = torch.tensor(0.0, device=device)
            batch_count = 0

            for i in range(pred_masks.size(0)):
                pm = pred_masks[i]  # [B_obj, H, W]
                # Convert binary masks (0/1) to logits (-10/+10) if needed.
                # Extraction saves output masks as float 0.0/1.0, but TIAB
                # expects raw logits where the scale determines contested pixels.
                if pm.max() <= 1.0 and pm.min() >= 0.0:
                    pm = pm * 20.0 - 10.0  # 0 → -10, 1 → +10
                pf = pix_feat[i]    # [B_obj, C, Hf, Wf]
                os_ = object_scores[i]  # [B_obj]
                gt_c = gt_centroids[i]  # [N, 2]
                ch = centroid_history[i]  # [B_obj, K, 2]
                crossing = bool(is_crossing[i])

                B_obj = pm.size(0)
                if B_obj <= 1:
                    continue

                # Skip if GT has NaN (incomplete annotation)
                if torch.isnan(gt_c).any():
                    continue

                # Build identity map via Hungarian matching between
                # predicted mask centroids and GT centroids. SAM3.1's object
                # ordering has no guaranteed correspondence to GT animal IDs.
                with torch.no_grad():
                    pred_cents = mask_centroid(torch.sigmoid(pm)).cpu().numpy()
                gt_np = (gt_c / image_size).cpu().numpy()
                n_match = min(B_obj, gt_np.shape[0])
                cost = np.zeros((n_match, n_match))
                for ri in range(n_match):
                    for ci in range(n_match):
                        cost[ri, ci] = np.sqrt(((pred_cents[ri] - gt_np[ci]) ** 2).sum())
                row_ind, col_ind = linear_sum_assignment(cost)
                id_map = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

                # Zero appearance embeddings for Phase 1 so the model learns
                # to rely on trajectory, not noise. Random embeddings would
                # train appear_proj on a distribution that doesn't match
                # inference (where real OSNet/histogram embeddings are used).
                appear_embs = torch.zeros(
                    B_obj, model.identity_encoder.appear_proj.in_features,
                    device=device,
                )

                # Forward
                refined = model(
                    pred_masks=pm,
                    pix_feat=pf,
                    appearance_embs=appear_embs,
                    centroid_history=ch,
                    object_score_logits=os_.unsqueeze(-1) if os_.dim() == 1 else os_,
                )

                # Check if any pixels were contested
                if B_obj == 2:
                    diff = (pm[0] - pm[1]).abs()
                else:
                    sorted_s, _ = pm.sort(dim=0, descending=True)
                    diff = sorted_s[0] - sorted_s[1]
                if (diff < model.contest_margin).sum() > 0:
                    n_contested += 1

                # Compute loss
                loss, loss_dict = tiab_combined_loss(
                    refined_masks=refined,
                    gt_centroids=gt_c,
                    id_map=id_map,
                    gt_masks=None,  # No pixel-level GT in Phase 1
                    gate_values=model.gate(
                        model.identity_encoder(appear_embs, ch),
                        os_.view(B_obj, 1) if os_.dim() == 1 else os_[:, :1],
                    ),
                    is_crossing=crossing,
                    image_size=image_size,
                    lambda_identity=lambda_identity,
                    lambda_boundary=lambda_boundary,
                    lambda_gate=lambda_gate,
                )

                batch_loss = batch_loss + loss
                batch_count += 1

                for k in epoch_losses:
                    if k in loss_dict:
                        epoch_losses[k] += loss_dict[k]

            if batch_count > 0:
                batch_loss = batch_loss / batch_count
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n_batches += 1

        scheduler.step()

        # Compute epoch averages
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches * batch_size, 1)

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"loss={epoch_losses['total']:.4f} "
            f"(id={epoch_losses['identity']:.4f} "
            f"gate={epoch_losses['gate']:.4f}) | "
            f"contested={n_contested}/{n_batches * batch_size} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | "
            f"{elapsed:.1f}s"
        )

        log["epochs"].append({
            "epoch": epoch + 1,
            "losses": dict(epoch_losses),
            "contested_frames": n_contested,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": elapsed,
        })

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            ckpt_path = os.path.join(output_dir, f"tiab_epoch_{epoch+1:03d}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "loss": epoch_losses["total"],
            }, ckpt_path)

        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            best_path = os.path.join(output_dir, "tiab_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "loss": best_loss,
            }, best_path)

    # Save final model and training log
    final_path = os.path.join(output_dir, "tiab_final.pt")
    torch.save({"model_state_dict": model.state_dict()}, final_path)

    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Weights: {final_path}")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TIAB module")
    parser.add_argument("--data-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_tiab(
        data_dirs=args.data_dirs,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
