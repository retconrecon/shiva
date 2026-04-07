"""TIAB training dataset and feature extraction utilities.

Two-step training workflow:
1. Extract: Run SAM3.1 tracking with TIABExtractor hook to save per-frame
   tensors (pred_masks, pix_feat, object_scores, GT centroids) to disk.
2. Train: TIABDataset loads extracted frames, groups into clips centered
   on crossing events, and feeds them to the training loop.

Usage:
    # Step 1: Extract (in experiment script)
    extractor = TIABExtractor(save_dir, gt_centroids, n_animals)
    for result in predictor.handle_stream_request({...}):
        extractor.on_frame(frame_idx, result, inference_state)
    extractor.finalize()

    # Step 2: Train (in training script)
    dataset = TIABDataset(save_dir, clip_length=16, crossing_ratio=0.7)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TIABExtractor:
    """Hooks into SAM3.1 tracking to save per-frame training data for TIAB.

    Captures tensors from inside the tracking pipeline and saves them
    alongside GT centroid positions. Saves only crossing frames +
    a random sample of non-crossing frames to manage disk usage.

    Each frame is saved as a .pt file containing:
        pred_masks: [B, H, W] float16 — mask logits before non-overlap constraint
        pix_feat: [B, C, Hf, Wf] float16 — backbone features at stride 16
        object_scores: [B] float32 — object confidence logits
        gt_centroids: [N, 2] float32 — GT centroid positions in pixels
        is_crossing: bool — whether any pair has IoU > threshold
        obj_ids: list[int] — SAM3.1 object IDs for this frame
    """

    def __init__(
        self,
        save_dir,
        gt_data,
        n_animals,
        crossing_distance_thresh=50.0,
        non_crossing_sample_rate=0.1,
        image_size=1008,
    ):
        """
        Args:
            save_dir: directory to save extracted frames
            gt_data: dict {frame_idx: {animal_id: (cx, cy)}} — GT centroids
            n_animals: number of animals
            crossing_iou_thresh: IoU threshold for crossing detection
            non_crossing_sample_rate: fraction of non-crossing frames to save
            image_size: video resolution for centroid normalization
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.gt_data = gt_data
        self.n_animals = n_animals
        self.crossing_distance_thresh = crossing_distance_thresh
        self.non_crossing_sample_rate = non_crossing_sample_rate
        self.image_size = image_size
        self._frame_count = 0
        self._saved_count = 0
        self._crossing_frames = []
        self._rng = np.random.RandomState(42)

    def on_frame(
        self,
        frame_idx,
        pred_masks_pre_constraint,
        pix_feat,
        object_score_logits,
        output_masks,
        obj_ids,
    ):
        """Called per frame during SAM3.1 tracking.

        Args:
            frame_idx: current frame index
            pred_masks_pre_constraint: [B, 1, H, W] mask logits BEFORE argmax
            pix_feat: [B, C, Hf, Wf] backbone features
            object_score_logits: [B, ...] confidence logits
            output_masks: dict {obj_id: bool_mask} — final output masks
            obj_ids: list of SAM3.1 object IDs
        """
        self._frame_count += 1

        # Determine if crossing via centroid distance (not mask IoU — output
        # masks are non-overlapping by construction after SAM3.1's argmax,
        # so IoU is always 0).
        is_crossing = False
        oids = sorted(output_masks.keys())
        if len(oids) >= 2:
            centroids = {}
            for oid in oids:
                ys, xs = np.where(output_masks[oid])
                if len(xs) > 0:
                    centroids[oid] = (float(xs.mean()), float(ys.mean()))
            clist = list(centroids.values())
            for i in range(len(clist)):
                for j in range(i + 1, len(clist)):
                    dx = clist[i][0] - clist[j][0]
                    dy = clist[i][1] - clist[j][1]
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist < self.crossing_distance_thresh:
                        is_crossing = True
                        break
                if is_crossing:
                    break

        # Decide whether to save this frame
        save = is_crossing or self._rng.random() < self.non_crossing_sample_rate

        if not save:
            return

        # Get GT centroids for this frame (handles both 0-indexed and 1-indexed IDs)
        gt_frame = self.gt_data.get(frame_idx, {})
        gt_centroids = np.full((self.n_animals, 2), np.nan, dtype=np.float32)
        sorted_aids = sorted(gt_frame.keys())
        min_aid = min(sorted_aids) if sorted_aids else 0
        for aid, (cx, cy) in gt_frame.items():
            idx = aid - min_aid
            if 0 <= idx < self.n_animals:
                gt_centroids[idx] = [cx, cy]

        # Save frame data
        frame_data = {
            "pred_masks": pred_masks_pre_constraint.squeeze(1).cpu().half(),
            "pix_feat": pix_feat.cpu().half(),
            "object_scores": object_score_logits.cpu().float().squeeze(),
            "gt_centroids": torch.from_numpy(gt_centroids),
            "is_crossing": is_crossing,
            "obj_ids": list(int(x) for x in obj_ids),
            "frame_idx": frame_idx,
        }
        torch.save(frame_data, self.save_dir / f"frame_{frame_idx:06d}.pt")
        self._saved_count += 1
        if is_crossing:
            self._crossing_frames.append(frame_idx)

    def finalize(self):
        """Save metadata after extraction completes."""
        meta = {
            "total_frames": self._frame_count,
            "saved_frames": self._saved_count,
            "crossing_frames": self._crossing_frames,
            "n_animals": self.n_animals,
            "image_size": self.image_size,
        }
        with open(self.save_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"TIAB extraction: {self._saved_count}/{self._frame_count} frames saved "
              f"({len(self._crossing_frames)} crossings)")


class TIABDataset(Dataset):
    """Dataset of pre-extracted SAM3.1 frames for TIAB training.

    Loads individual frame .pt files and groups them into clips
    of K consecutive frames. Oversamples crossing frames per
    crossing_ratio.
    """

    def __init__(
        self,
        data_dirs,
        clip_length=16,
        crossing_ratio=0.7,
        trajectory_len=16,
        image_size=1008,
    ):
        """
        Args:
            data_dirs: str or list of str — directories with extracted frames
            clip_length: number of frames per training clip
            crossing_ratio: fraction of clips centered on crossings
            trajectory_len: number of centroid positions for trajectory encoding
            image_size: for centroid normalization
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.clip_length = clip_length
        self.crossing_ratio = crossing_ratio
        self.trajectory_len = trajectory_len
        self.image_size = image_size

        # Collect all frame files across videos
        self._all_frames = []  # (dir_path, frame_idx, is_crossing)
        self._frames_by_dir = defaultdict(list)
        self._crossing_indices = []
        self._noncrossing_indices = []

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            meta_path = data_dir / "meta.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                meta = json.load(f)

            # Read crossing labels from per-frame .pt files (authoritative),
            # falling back to meta.json crossing_frames list if .pt unavailable
            crossing_set = set(meta.get("crossing_frames", []))

            frame_files = sorted(data_dir.glob("frame_*.pt"))
            for ff in frame_files:
                fidx = int(ff.stem.split("_")[1])
                # Read is_crossing directly from the .pt file
                try:
                    _fd = torch.load(ff, map_location="cpu", weights_only=False)
                    is_cross = bool(_fd.get("is_crossing", fidx in crossing_set))
                except Exception:
                    is_cross = fidx in crossing_set
                idx = len(self._all_frames)
                self._all_frames.append((str(data_dir), fidx, is_cross))
                self._frames_by_dir[str(data_dir)].append(idx)
                if is_cross:
                    self._crossing_indices.append(idx)
                else:
                    self._noncrossing_indices.append(idx)

        if not self._all_frames:
            raise RuntimeError(f"No frames found in {data_dirs}")

        print(f"TIABDataset: {len(self._all_frames)} frames "
              f"({len(self._crossing_indices)} crossing, "
              f"{len(self._noncrossing_indices)} non-crossing)")

    def __len__(self):
        return len(self._all_frames)

    def __getitem__(self, idx):
        """Returns a single frame's training data.

        Returns dict with:
            pred_masks: [B, H, W] float32
            pix_feat: [B, C, Hf, Wf] float32
            object_scores: [B] float32
            gt_centroids: [N, 2] float32 (pixels)
            gt_centroids_norm: [N, 2] float32 (normalized 0-1)
            is_crossing: bool
            centroid_history: [B, K, 2] float32 (placeholder — filled by collate)
        """
        data_dir, frame_idx, is_crossing = self._all_frames[int(idx)]
        frame_path = Path(data_dir) / f"frame_{frame_idx:06d}.pt"
        frame_data = torch.load(frame_path, map_location="cpu", weights_only=False)

        pred_masks = frame_data["pred_masks"].float()
        pix_feat = frame_data["pix_feat"].float()
        object_scores = frame_data["object_scores"].float()
        gt_centroids = frame_data["gt_centroids"].float()

        # Normalized centroids for loss computation
        # NaN stays in gt_centroids (training loop filters with valid_gt)
        # but centroid_norm must be clean for any downstream use
        gt_centroids_norm = torch.nan_to_num(gt_centroids, nan=0.0) / self.image_size

        B = pred_masks.size(0)

        # Build centroid history from nearby saved frames in same video
        centroid_history = self._build_centroid_history(
            data_dir, frame_idx, B,
        )

        return {
            "pred_masks": pred_masks,
            "pix_feat": pix_feat,
            "object_scores": object_scores,
            "gt_centroids": gt_centroids,
            "gt_centroids_norm": gt_centroids_norm,
            "is_crossing": is_crossing,
            "centroid_history": centroid_history,
            "frame_idx": frame_idx,
        }

    def _build_centroid_history(self, data_dir, frame_idx, n_objects):
        """Build centroid trajectory from nearby saved frames.

        Returns [n_objects, trajectory_len, 2] normalized centroids.
        Fills with the earliest available position if not enough history.
        """
        K = self.trajectory_len
        history = torch.zeros(n_objects, K, 2)

        # Find saved frames in this video before current frame
        dir_frames = self._frames_by_dir[data_dir]
        prior_frames = []
        for gidx in dir_frames:
            _, fidx, _ = self._all_frames[gidx]
            if fidx < frame_idx:
                prior_frames.append((fidx, gidx))
        prior_frames.sort(key=lambda x: x[0], reverse=True)

        # Load centroids from most recent K frames
        loaded = []
        for fidx, gidx in prior_frames[:K]:
            try:
                fd = torch.load(
                    Path(data_dir) / f"frame_{fidx:06d}.pt",
                    map_location="cpu", weights_only=False,
                )
                gt = fd["gt_centroids"].float() / self.image_size
                gt = torch.nan_to_num(gt, nan=0.5)  # replace NaN with center
                loaded.append(gt[:n_objects])
            except Exception:
                continue

        if loaded:
            loaded.reverse()  # chronological order
            # Pad with earliest if not enough
            while len(loaded) < K:
                loaded.insert(0, loaded[0])
            loaded = loaded[-K:]
            history = torch.stack(loaded, dim=1)  # [n_objects, K, 2]
        else:
            # No history — fill with current GT if available
            try:
                fd = torch.load(
                    Path(data_dir) / f"frame_{frame_idx:06d}.pt",
                    map_location="cpu", weights_only=False,
                )
                gt = fd["gt_centroids"].float() / self.image_size
                history = gt[:n_objects].unsqueeze(1).expand(-1, K, -1).clone()
            except Exception:
                pass

        return history

    def get_balanced_sampler(self):
        """Returns indices that balance crossing/non-crossing frames."""
        n_total = len(self)
        n_crossing = int(n_total * self.crossing_ratio)
        n_noncrossing = n_total - n_crossing

        rng = np.random.RandomState(0)
        crossing = rng.choice(
            self._crossing_indices,
            size=min(n_crossing, len(self._crossing_indices)),
            replace=len(self._crossing_indices) < n_crossing,
        )
        noncrossing = rng.choice(
            self._noncrossing_indices,
            size=min(n_noncrossing, len(self._noncrossing_indices)),
            replace=len(self._noncrossing_indices) < n_noncrossing,
        )
        indices = np.concatenate([crossing, noncrossing]).astype(int)
        rng.shuffle(indices)
        return [int(x) for x in indices]
