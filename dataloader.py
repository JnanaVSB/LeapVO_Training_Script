"""
dataloader.py — LEAP-VO Training Dataloader
=============================================
Loads processed TAP-Vid-Kubric (MOVI-F) sequences from disk.

Expected dataset layout (produced by process_one.py / prepare_dataset.sh):

    training_dataset/
    ├── 000000/
    │   ├── 000000.npy        ← annotation file
    │   └── frames/
    │       ├── 000.png       ← 256×256 RGB frames
    │       ├── 001.png
    │       └── ... (023.png)
    ├── 000001/
    └── ...

Each .npy file contains a dict with:
    'coords'       : (N, T, 2)  float32  — [x, y] pixel trajectories (256×256 space)
    'visibility'   : (N, T)     bool     — True = visible, False = occluded
    'query_points' : (N, 3)     float32  — [t, y, x] query location
    'is_dynamic'   : (N,)       bool     — True = point on a dynamic object

LeapKernel forward() expects:
    rgbs      : [B, T, 3, H, W]  float32  in [0, 255]
    queries   : [B, N, 3]        float32  in [frame_idx, x, y]  ← x before y

Notes:
    - query_points in .npy is stored as [t, y, x] (TAP-Vid convention)
      but LeapKernel queries are [t, x, y] — we swap the last two dims.
    - coords in .npy is (N, T, 2) but train.py expects [T, N, 2] — we transpose.
    - visibility in .npy is (N, T) but train.py expects [T, N] — we transpose.
    - We subsample num_tracks tracks per sequence (paper: N=256, .npy has 256).
    - LeapKernel internally resizes frames to (384, 512) via F.interpolate,
      so we pass the raw 256×256 frames — no resizing needed here.
    - If a sequence has fewer than min_frames frames it is skipped.
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LeapKubricDataset(Dataset):
    """
    PyTorch Dataset for processed MOVI-F sequences.

    Loads whatever sequences are present under data_root at construction time.
    As more sequences finish processing, re-instantiate (or restart training)
    to pick them up — no code changes needed.

    Args:
        data_root   : path to training_dataset/ folder
        num_tracks  : number of tracks to sample per sequence (paper: 256)
        min_frames  : skip sequences with fewer than this many frames
        augment     : apply colour jitter augmentation to frames
        seed        : random seed for track subsampling (None = non-deterministic)
    """

    def __init__(
        self,
        data_root: str,
        num_tracks: int = 256,
        min_frames: int = 8,
        augment: bool = True,
        seed: int = None,
    ):
        self.data_root  = Path(data_root)
        self.num_tracks = num_tracks
        self.min_frames = min_frames
        self.augment    = augment
        self.rng        = random.Random(seed)

        # Discover all valid sequence directories
        self.sequences = self._find_sequences()
        if len(self.sequences) == 0:
            raise RuntimeError(
                f"No valid sequences found under {self.data_root}. "
                "Check that prepare_dataset.sh has finished at least one sequence."
            )
        logger.info(f"Dataset: {len(self.sequences)} sequences found under {self.data_root}")

    def _find_sequences(self):
        """
        Walk data_root and collect directories that have both
        a .npy annotation file and at least one frame PNG.
        """
        sequences = []
        if not self.data_root.exists():
            raise RuntimeError(f"data_root does not exist: {self.data_root}")

        for seq_dir in sorted(self.data_root.iterdir()):
            if not seq_dir.is_dir():
                continue

            seq_id   = seq_dir.name
            npy_path = seq_dir / f"{seq_id}.npy"
            frame_dir = seq_dir / "frames"

            if not npy_path.exists():
                continue
            if not frame_dir.exists() or not any(frame_dir.glob("*.png")):
                continue

            # Count frames
            n_frames = len(list(frame_dir.glob("*.png")))
            if n_frames < self.min_frames:
                logger.debug(f"Skipping {seq_id}: only {n_frames} frames < {self.min_frames}")
                continue

            sequences.append({
                "seq_id":    seq_id,
                "npy_path":  npy_path,
                "frame_dir": frame_dir,
                "n_frames":  n_frames,
            })

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # ---- Load annotations ----
        annot = np.load(seq["npy_path"], allow_pickle=True).item()

        # coords      : (N, T, 2)  float32  [x, y]
        # visibility  : (N, T)     bool
        # query_points: (N, 3)     float32  [t, y, x]  ← TAP-Vid convention
        # is_dynamic  : (N,)       bool
        coords       = annot["coords"]        # (N, T, 2)
        visibility   = annot["visibility"]    # (N, T)
        query_points = annot["query_points"]  # (N, 3)  [t, y, x]
        is_dynamic   = annot["is_dynamic"]    # (N,)

        N_total, T, _ = coords.shape

        # ---- Subsample tracks ----
        # If dataset has exactly num_tracks, this is a no-op.
        # If it has more (e.g. future datasets), we randomly subsample.
        if N_total >= self.num_tracks:
            chosen = sorted(self.rng.sample(range(N_total), self.num_tracks))
        else:
            # Repeat-sample up to num_tracks (shouldn't happen with MOVI-F N=256)
            chosen = self.rng.choices(range(N_total), k=self.num_tracks)
            chosen = sorted(chosen)

        coords       = coords[chosen]        # (N, T, 2)
        visibility   = visibility[chosen]    # (N, T)
        query_points = query_points[chosen]  # (N, 3)
        is_dynamic   = is_dynamic[chosen]    # (N,)

        # ---- Load frames ----
        frame_dir = seq["frame_dir"]
        frame_files = sorted(frame_dir.glob("*.png"))[:T]   # cap at annotation T

        frames = []
        for fp in frame_files:
            img = Image.open(fp).convert("RGB")
            frames.append(np.array(img, dtype=np.float32))   # (H, W, 3)

        # Stack → (T, H, W, 3) then → (T, 3, H, W)
        frames = np.stack(frames, axis=0)                     # (T, H, W, 3)

        # ---- Colour augmentation ----
        if self.augment:
            frames = self._color_jitter(frames)

        # (T, H, W, 3) → (T, 3, H, W), values in [0, 255]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # [T, 3, H, W]

        # ---- Convert annotations to tensors ----
        # coords: (N, T, 2) → transpose → (T, N, 2)
        coords = torch.from_numpy(coords).permute(1, 0, 2).float()      # [T, N, 2]

        # visibility: (N, T) → transpose → (T, N)
        visibility = torch.from_numpy(visibility).permute(1, 0).bool()  # [T, N]

        # is_dynamic: (N,)
        is_dynamic = torch.from_numpy(is_dynamic.astype(np.bool_))      # [N]

        # ---- Build queries: [N, 3] = [frame_idx, x, y] ----
        # query_points from .npy: [t, y, x]  (TAP-Vid convention: row=y first)
        # LeapKernel expects:     [t, x, y]  (x=column first, matching coords)
        qt  = torch.from_numpy(query_points[:, 0:1]).float()   # [N, 1]  frame index
        qy  = torch.from_numpy(query_points[:, 1:2]).float()   # [N, 1]  y (row)
        qx  = torch.from_numpy(query_points[:, 2:3]).float()   # [N, 1]  x (col)
        queries = torch.cat([qt, qx, qy], dim=1)               # [N, 3]  [t, x, y]

        # ---- Valid mask ----
        # A frame is valid for a track once the query frame has been reached.
        # This matches how LeapKernel builds track_mask internally.
        query_frames = queries[:, 0].long()                     # [N]
        t_idx        = torch.arange(T).unsqueeze(1)             # [T, 1]
        valid        = (t_idx >= query_frames.unsqueeze(0))     # [T, N]  bool

        return {
            "rgbs":       frames,      # [T, 3, H, W]  float32  [0,255]
            "coords":     coords,      # [T, N, 2]     float32  pixel coords
            "visibility": visibility,  # [T, N]        bool
            "is_dynamic": is_dynamic,  # [N]           bool
            "queries":    queries,     # [N, 3]        float32  [t, x, y]
            "valid":      valid,       # [T, N]        bool
        }

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _color_jitter(self, frames, brightness=0.3, contrast=0.3,
                      saturation=0.2, hue=0.05):
        """
        Apply random colour jitter to all frames in a sequence.
        Same transform applied consistently across the whole sequence
        (don't jitter each frame independently — that would break temporal
        consistency which the tracker relies on).

        frames : (T, H, W, 3)  float32  in [0, 255]
        returns: (T, H, W, 3)  float32  in [0, 255]
        """
        # Brightness
        b_factor = 1.0 + random.uniform(-brightness, brightness)
        frames   = frames * b_factor

        # Contrast — per-channel mean
        means  = frames.mean(axis=(0, 1, 2), keepdims=True)
        c_factor = 1.0 + random.uniform(-contrast, contrast)
        frames   = (frames - means) * c_factor + means

        # Saturation — convert to grayscale mix
        gray     = frames.mean(axis=3, keepdims=True)
        s_factor = 1.0 + random.uniform(-saturation, saturation)
        frames   = gray + (frames - gray) * s_factor

        # Hue shift — simple channel roll (approximation, avoids cv2 dep)
        if random.random() < 0.3:
            h_shift = random.uniform(-hue, hue)
            frames[..., 0] = frames[..., 0] + h_shift * 255

        return np.clip(frames, 0.0, 255.0)


# ===========================================================================
# Smoke test — run this file directly to verify the dataset loads correctly
# ===========================================================================

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Path to training_dataset/")
    parser.add_argument("--num_tracks", type=int, default=256)
    parser.add_argument("--num_check",  type=int, default=3,
                        help="Number of samples to check")
    args = parser.parse_args()

    dataset = LeapKubricDataset(
        data_root=args.data_root,
        num_tracks=args.num_tracks,
        augment=False,
    )
    print(f"\n✓ Dataset loaded: {len(dataset)} sequences\n")

    for i in range(min(args.num_check, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}:")
        for k, v in sample.items():
            print(f"  {k:12s}: shape={tuple(v.shape)}  dtype={v.dtype}  "
                  f"range=[{v.float().min():.2f}, {v.float().max():.2f}]")

        # Verify queries are in bounds
        T = sample["rgbs"].shape[0]
        N = sample["queries"].shape[0]
        qt = sample["queries"][:, 0]
        assert qt.min() >= 0 and qt.max() < T, \
            f"Query frame index out of range: {qt.min()}-{qt.max()} vs T={T}"

        # Verify valid mask matches query frames
        qf    = sample["queries"][:, 0].long()
        valid = sample["valid"]
        for n in range(min(5, N)):
            assert valid[qf[n]:, n].all(), \
                f"Track {n}: valid should be True from frame {qf[n]} onward"
            if qf[n] > 0:
                assert not valid[:qf[n], n].any(), \
                    f"Track {n}: valid should be False before frame {qf[n]}"

        print(f"  ✓ shapes OK, queries in-bounds, valid mask consistent\n")

    print("All checks passed.")