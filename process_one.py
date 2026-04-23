"""
Process a SINGLE MOVI-F sequence. Called by prepare_dataset.sh for each index.
Each invocation is a fresh process, so TF memory is fully reclaimed.

Usage:
    python process_one.py --idx 0 --output_dir ./training_dataset
"""
import os
import sys
import shutil
import numpy as np
from pathlib import Path
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

sys.path.insert(0, "/home/jnana/kubric/kubric/challenges/point_tracking")
sys.path.insert(0, "/home/jnana/kubric/kubric/challenges/movi")

import tensorflow as tf
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

from PIL import Image
from dataset import add_tracks
import movi_f


TRAIN_SIZE = (256, 256)
TRACKS_PER_SEQ = 256
SAMPLING_STRIDE = 4


def downsample_raw(raw_example):
    h, w = 256, 256
    video = tf.cast(tf.image.resize(raw_example["video"], [h, w], method="bilinear"), tf.uint8)
    seg = tf.cast(tf.image.resize(raw_example["segmentations"], [h, w], method="nearest"), tf.uint8)
    depth = tf.cast(tf.image.resize(tf.cast(raw_example["depth"], tf.float32), [h, w], method="nearest"), tf.uint16)
    obj_coords = tf.cast(tf.image.resize(tf.cast(raw_example["object_coordinates"], tf.float32), [h, w], method="nearest"), tf.uint16)
    normal = tf.cast(tf.image.resize(tf.cast(raw_example["normal"], tf.float32), [h, w], method="nearest"), tf.uint16)

    return {
        "video": video,
        "segmentations": seg,
        "depth": depth,
        "object_coordinates": obj_coords,
        "normal": normal,
        "forward_flow": raw_example["forward_flow"],
        "backward_flow": raw_example["backward_flow"],
        "camera": raw_example["camera"],
        "instances": raw_example["instances"],
        "metadata": raw_example["metadata"],
        "events": raw_example["events"],
        "background": raw_example["background"],
    }


def extract_is_dynamic(raw_example, query_points_np, seg_256):
    inst_dyn = raw_example["instances"]["is_dynamic"]
    if hasattr(inst_dyn, "numpy"):
        inst_dyn = inst_dyn.numpy()
    if hasattr(seg_256, "numpy"):
        seg_256 = seg_256.numpy()

    h, w = seg_256.shape[1], seg_256.shape[2]
    is_dyn = np.zeros(query_points_np.shape[0], dtype=bool)

    for i in range(query_points_np.shape[0]):
        t = int(np.clip(round(query_points_np[i, 0]), 0, seg_256.shape[0] - 1))
        y = int(np.clip(round(query_points_np[i, 1] - 0.5), 0, h - 1))
        x = int(np.clip(round(query_points_np[i, 2] - 0.5), 0, w - 1))
        seg_id = int(seg_256[t, y, x, 0])
        if seg_id > 0:
            inst_idx = seg_id - 1
            if inst_idx < len(inst_dyn):
                is_dyn[i] = bool(inst_dyn[inst_idx])
    return is_dyn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    seq_id = f"{args.idx:06d}"
    output_dir = Path(args.output_dir)
    seq_dir = output_dir / seq_id
    tmp_dir = output_dir / f"{seq_id}_tmp"

    # Check if already done
    npy_path = seq_dir / f"{seq_id}.npy"
    frames_dir = seq_dir / "frames"
    if npy_path.exists() and frames_dir.is_dir():
        try:
            data = np.load(npy_path, allow_pickle=True).item()
            if all(k in data for k in ["coords", "visibility", "query_points", "is_dynamic"]):
                print(f"SKIP {seq_id}")
                sys.exit(0)
        except Exception:
            pass

    # Load one example from GCS
    builder = movi_f.MoviF(data_dir="gs://kubric-public/tfds")
    ds = builder.as_dataset(split=f"train[{args.idx}:{args.idx + 1}]")

    raw_example = next(iter(ds))

    # Downsample
    small = downsample_raw(raw_example)

    # Run annotation
    processed = add_tracks(
        small,
        train_size=TRAIN_SIZE,
        vflip=False,
        random_crop=False,
        tracks_to_sample=TRACKS_PER_SEQ,
        sampling_stride=SAMPLING_STRIDE,
        max_seg_id=25,
        max_sampled_frac=0.1,
        snap_to_occluder=False,
    )

    query_points = processed["query_points"].numpy()
    target_points = processed["target_points"].numpy()
    occluded = processed["occluded"].numpy()
    video = processed["video"].numpy()

    video_uint8 = ((video + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    is_dynamic = extract_is_dynamic(raw_example, query_points, small["segmentations"])

    # Save
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    fdir = tmp_dir / "frames"
    fdir.mkdir()
    for t in range(video_uint8.shape[0]):
        Image.fromarray(video_uint8[t]).save(fdir / f"{t:03d}.png")

    annot = {
        "coords": target_points.astype(np.float32),
        "visibility": (~occluded).astype(bool),
        "query_points": query_points.astype(np.float32),
        "is_dynamic": is_dynamic.astype(bool),
    }
    np.save(tmp_dir / f"{seq_id}.npy", annot, allow_pickle=True)

    os.rename(tmp_dir, seq_dir)
    print(f"DONE {seq_id}")


if __name__ == "__main__":
    main()