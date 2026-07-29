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
        contest_margin=2.0,
        min_contested_px=8,
    ):
        """
        Args:
            save_dir: directory to save extracted frames
            gt_data: dict {frame_idx: {animal_id: (cx, cy)}} — GT centroids
            n_animals: number of animals
            crossing_distance_thresh: RETAINED FOR API COMPATIBILITY, no longer used for
                selection. Frame selection is now on the contested-pixel count (see on_frame).
            non_crossing_sample_rate: fraction of uncontested frames to save anyway, so the model
                still sees easy negatives and does not train only on the hard tail.
            image_size: video resolution for centroid normalization
            contest_margin: a foreground pixel is CONTESTED when its top-2 logit gap is below this.
                Must match the margin the training loss uses, or selection and loss disagree about
                what the module is being trained on.
            min_contested_px: a frame is worth keeping when it has at least this many contested
                pixels. 8 is deliberately low: the measured median under the old rule was 2, and the
                point is to stop saving frames with nothing to learn from, not to chase only the
                hardest frames.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.gt_data = gt_data
        self.n_animals = n_animals
        self.crossing_distance_thresh = crossing_distance_thresh
        self.non_crossing_sample_rate = non_crossing_sample_rate
        self.contest_margin = contest_margin
        self.min_contested_px = min_contested_px
        self._contested_hist = []
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

        # ☢ SELECT ON CONTESTED PIXELS DIRECTLY, NOT ON A CENTROID-DISTANCE PROXY.
        #
        # MEASURED FAILURE OF THE OLD RULE (2026-07-28, on this extractor's own output for
        # CalMS21 mouse025, 401 saved shards):
        #   * `is_crossing` fired on 1 of 401 shards (0.2%)
        #   * 42.4% of saved frames contained ZERO contested pixels
        #   * median contested pixels per frame: 2; contested share of foreground: 0.057%
        # TIAB is a contested-pixel refiner, so that training set has essentially no signal. The
        # consequence was mathematical: the identity loss stayed at 8.686e-05 for all 10 epochs,
        # unchanged to four significant figures, while 26 of 28 parameter tensors drifted on the
        # gate regulariser alone. A checkpoint trained that way is the random-weight case.
        #
        # THE CAUSE was `crossing_distance_thresh=50.0`, a pixel constant roughly 4x stricter than
        # the contact definition used everywhere else in this project (one body length, ~177 px for
        # this animal). It also compared MASK-space centroids against a threshold reasoned about in
        # FRAME space, and for CalMS21 the mask is a stretched square (1008x1008 from 1024x570), so
        # the implied threshold was anisotropic and differed by ~1.8x between the axes.
        #
        # THE FIX is to stop proxying. `pred_masks_pre_constraint` is the pre-argmax logit stack, so
        # the contested set can be computed exactly, here, from the same quantity the training loss
        # consumes: a foreground pixel whose top-2 logit gap is below the margin. Selecting on it
        # directly is scale-free (no pixel constant, no species assumption, no coordinate space to
        # get wrong) and it is the definition rather than a correlate of it.
        #
        # The foreground gate matters and is not optional: in a top-down arena the objects agree most
        # strongly on empty bedding, so without `top1 > 0` the "contested" set is dominated by
        # background. Measured on this same data, 98.4% of an ungated contested set was background.
        n_contested = 0
        try:
            lg = pred_masks_pre_constraint.squeeze(1).float()      # [B, H, W] pre-argmax logits
            if lg.shape[0] >= 2:
                top2 = torch.topk(lg, 2, dim=0).values
                fg = top2[0] > 0.0
                n_contested = int((fg & ((top2[0] - top2[1]) < self.contest_margin)).sum().item())
        except Exception:                                          # noqa: BLE001
            n_contested = 0        # never let the selector crash an extraction run

        is_crossing = n_contested >= self.min_contested_px

        # Keep the centroid-distance signal as RECORDED METADATA only. It is still informative for
        # analysis (and for cross-checking against `encounter_coverage.json`), but it no longer
        # decides what gets saved.
        min_pair_dist = float("inf")
        oids = sorted(output_masks.keys())
        if len(oids) >= 2:
            clist = []
            for oid in oids:
                ys, xs = np.where(output_masks[oid])
                if len(xs) > 0:
                    clist.append((float(xs.mean()), float(ys.mean())))
            for i in range(len(clist)):
                for j in range(i + 1, len(clist)):
                    dx = clist[i][0] - clist[j][0]
                    dy = clist[i][1] - clist[j][1]
                    min_pair_dist = min(min_pair_dist, (dx * dx + dy * dy) ** 0.5)

        # Decide whether to save this frame
        save = is_crossing or self._rng.random() < self.non_crossing_sample_rate

        self._contested_hist.append(n_contested)
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
            # Recorded so a later audit can census the training set WITHOUT recomputing logits.
            # The absence of exactly this field is why the empty-contested-set defect went unnoticed
            # through a full training run and into a checkpoint.
            "n_contested": n_contested,
            "min_pair_dist": min_pair_dist,
        }
        torch.save(frame_data, self.save_dir / f"frame_{frame_idx:06d}.pt")
        self._saved_count += 1
        if is_crossing:
            self._crossing_frames.append(frame_idx)

    def finalize(self):
        """Save metadata after extraction completes."""
        hist = np.asarray(self._contested_hist) if self._contested_hist else np.zeros(1)
        saved_zero = int((hist == 0).sum())
        meta = {
            "total_frames": self._frame_count,
            "saved_frames": self._saved_count,
            "crossing_frames": self._crossing_frames,
            "n_animals": self.n_animals,
            "image_size": self.image_size,
            # THE EXTRACT-QUALITY CENSUS. A TIAB training set is only as good as its contested-pixel
            # content, so that content is now a first-class recorded property of the extract rather
            # than something a human has to think to go and measure.
            "contest_margin": self.contest_margin,
            "min_contested_px": self.min_contested_px,
            "contested_mean": float(hist.mean()),
            "contested_median": float(np.median(hist)),
            "contested_max": int(hist.max()),
            "frames_with_zero_contested_pct": float(100.0 * saved_zero / max(len(hist), 1)),
        }
        with open(self.save_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"TIAB extraction: {self._saved_count}/{self._frame_count} frames saved "
              f"({len(self._crossing_frames)} contested)")
        print(f"TIAB extraction: contested px per frame mean {meta['contested_mean']:.1f} "
              f"median {meta['contested_median']:.0f} max {meta['contested_max']} | "
              f"{meta['frames_with_zero_contested_pct']:.1f}% of frames had ZERO")
        # ☢ LOUD REFUSAL, not a silent pass. Training on this is what produced a checkpoint whose
        # loss never moved across 10 epochs, and nothing in the pipeline objected at the time.
        if meta["contested_median"] < 1 or meta["contested_mean"] < 5:
            print("TIAB extraction: ☢ WARNING - this extract has almost no contested pixels. "
                  "TIAB is a contested-pixel refiner, so training on it will produce a checkpoint "
                  "that cannot learn (expect a flat loss). Do NOT use it for an ablation. "
                  "Check the detector/margin before spending GPU on training.")


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
