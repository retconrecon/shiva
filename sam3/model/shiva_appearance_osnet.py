"""SHIVA OSNet-AIN Appearance Store — learned re-ID embeddings.

Drop-in replacement for ShivaAppearanceStore that uses OSNet-AIN (2.2M params)
instead of grayscale histograms. Designed for distinguishing visually similar
individuals — same-species fish, same-strain mice.

Same interface as ShivaAppearanceStore:
    - update(obj_id, embedding)
    - get(obj_ids) -> np.ndarray
    - extract_embedding(mask, frame_pixels) -> np.ndarray or None

Usage:
    from sam3.model.shiva_appearance_osnet import ShivaOSNetAppearanceStore

    store = ShivaOSNetAppearanceStore(device="cuda")
    emb = store.extract_embedding(mask_bool, frame_bgr)
    store.update(obj_id, emb)

Requires: pip install torchreid
"""

import logging

import cv2
import numpy as np
import torch

from sam3.model.shiva_appearance import histogram_intersection_distance

logger = logging.getLogger(__name__)


class ShivaOSNetAppearanceStore:
    """Per-track EMA appearance embeddings using OSNet-AIN backbone.

    Parameters
    ----------
    ema_alpha : float
        EMA decay for embedding updates.
    device : str
        CUDA device string.
    max_consecutive_rejects : int
        Force-reset embedding after this many rejected updates.
    max_update_distance : float
        Maximum cosine distance to accept an EMA update.
    """

    def __init__(self, ema_alpha=0.9, device="cuda",
                 max_consecutive_rejects=50, max_update_distance=0.4):
        self.ema_alpha = ema_alpha
        self.device = device
        self.max_update_distance = max_update_distance
        self._max_consecutive_rejects = max_consecutive_rejects
        self.embeddings = {}  # {obj_id: np.ndarray (512,)}
        self._consecutive_rejects = {}
        self.embed_dim = 512

        self.model = self._load_osnet()
        self.model.eval()

        # ImageNet normalization constants for OSNet
        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)

        logger.info("OSNet-AIN appearance store loaded (%.1fM params)",
                     sum(p.numel() for p in self.model.parameters()) / 1e6)

    def _load_osnet(self):
        """Load OSNet-AIN model from torchreid."""
        try:
            from torchreid.models import build_model
            model = build_model(
                name="osnet_ain_x1_0",
                num_classes=1000,
                pretrained=True,
            )
            # Remove classifier — we want feature embeddings
            model.classifier = torch.nn.Identity()
            return model.to(self.device)
        except ImportError:
            raise ImportError(
                "OSNet requires torchreid: pip install torchreid\n"
                "Or download osnet_ain_x1_0 weights manually from:\n"
                "https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html"
            )

    def extract_embedding(self, mask, frame_pixels):
        """Extract OSNet embedding from masked region of frame.

        Args:
            mask: (H, W) bool — the object's mask
            frame_pixels: (H, W, 3) BGR uint8 frame

        Returns:
            (512,) L2-normalized float64 embedding, or None if crop is too small.
        """
        ys, xs = np.where(mask)
        if len(xs) < 16 or len(ys) < 16:
            return None

        H, W = frame_pixels.shape[:2]
        x1 = max(0, int(xs.min()) - 5)
        y1 = max(0, int(ys.min()) - 5)
        x2 = min(W, int(xs.max()) + 6)
        y2 = min(H, int(ys.max()) + 6)

        if (x2 - x1) < 16 or (y2 - y1) < 16:
            return None

        crop = frame_pixels[y1:y2, x1:x2].copy()

        # Zero pixels outside the mask to prevent background shortcuts
        crop_mask = mask[y1:y2, x1:x2]
        crop[~crop_mask] = 0

        # BGR → RGB, then directly to tensor (no PIL round-trip)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        import torchvision.transforms.functional as F
        tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
        tensor = F.resize(tensor, [256, 128], antialias=True)
        tensor = F.normalize(tensor, mean=self._mean.tolist(), std=self._std.tolist())
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy().flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        else:
            return None

        return embedding.astype(np.float64)

    def update(self, obj_id, embedding, max_update_distance=None,
               crossing_active=False):
        """Update track's appearance model with EMA.

        Skips update if new embedding diverges too much (possible swap).
        Force-resets after too many consecutive rejections — but only when
        no crossing is active (during crossings, current embedding is
        contaminated by overlapping animals).
        """
        if embedding is None:
            return
        if max_update_distance is None:
            max_update_distance = self.max_update_distance

        if obj_id not in self.embeddings:
            self.embeddings[obj_id] = embedding.copy()
            self._consecutive_rejects[obj_id] = 0
        else:
            # Cosine distance (embeddings are L2-normalized)
            dist = 1.0 - float(np.dot(self.embeddings[obj_id], embedding))
            if dist < max_update_distance:
                self.embeddings[obj_id] = (
                    self.ema_alpha * self.embeddings[obj_id]
                    + (1 - self.ema_alpha) * embedding
                )
                # Re-normalize after EMA
                norm = np.linalg.norm(self.embeddings[obj_id])
                if norm > 1e-8:
                    self.embeddings[obj_id] /= norm
                self._consecutive_rejects[obj_id] = 0
            else:
                self._consecutive_rejects[obj_id] = (
                    self._consecutive_rejects.get(obj_id, 0) + 1
                )
                if (self._consecutive_rejects[obj_id] >= self._max_consecutive_rejects
                        and not crossing_active):
                    self.embeddings[obj_id] = embedding.copy()
                    self._consecutive_rejects[obj_id] = 0
                    logger.info(
                        "OSNet embedding reset for obj %s after %d "
                        "consecutive rejected updates",
                        obj_id, self._max_consecutive_rejects,
                    )

    def get(self, obj_ids):
        """Get appearance embeddings for a list of track IDs.

        Returns uniform fallback for tracks without embeddings.

        Returns:
            (M, 512) float64 array. Never returns None.
        """
        uniform = np.ones(self.embed_dim, dtype=np.float64) / np.sqrt(self.embed_dim)
        embs = []
        for oid in obj_ids:
            if oid in self.embeddings:
                embs.append(self.embeddings[oid])
            else:
                embs.append(uniform)
        if not embs:
            return np.zeros((0, self.embed_dim), dtype=np.float64)
        return np.stack(embs)

    def remove(self, obj_id):
        """Remove a track's embedding."""
        self.embeddings.pop(obj_id, None)
        self._consecutive_rejects.pop(obj_id, None)

    @staticmethod
    def cosine_distance(a, b):
        """Cosine distance between two L2-normalized vectors."""
        return 1.0 - float(np.dot(a, b))

    def extract_histogram(self, mask, frame_pixels):
        """Interface-compatible wrapper that returns uniform fallback on failure.

        Matches ShivaAppearanceStore.extract_histogram contract: never returns None.
        """
        emb = self.extract_embedding(mask, frame_pixels)
        if emb is None:
            return np.ones(self.embed_dim, dtype=np.float64) / np.sqrt(self.embed_dim)
        return emb

    n_bins = 512  # compatibility attribute
    distance_metric = "cosine"  # explicit metric declaration
