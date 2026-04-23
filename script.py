import os
import sys
import time
import gc
import shutil
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ===== PATHS =====
sys.path.insert(0, "/home/jnana/kubric/kubric/challenges/point_tracking")
sys.path.insert(0, "/home/jnana/kubric/kubric/challenges/movi")

import tensorflow as tf
from PIL import Image
from dataset import add_tracks
import movi_f

# ===== TF SETTINGS =====
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# ===== CONFIG =====
OUTPUT_DIR = Path("/home/jnana/ASU/PIR/leapvo_training/training_dataset")
NUM_SEQUENCES = 2000
TRAIN_SIZE = (256, 256)
TRACKS_PER_SEQ = 256
SAMPLING_STRIDE = 4


def is_valid(seq_dir: Path) -> bool:
    if not seq_dir.is_dir():
        return False
    npy = seq_dir / f"{seq_dir.name}.npy"
    frames = seq_dir / "frames"
    if not npy.exists() or not frames.is_dir():
        return False
    try:
        data = np.load(npy, allow_pickle=True).item()
        needed = ["coords", "visibility", "query_points", "is_dynamic"]
        return all(k in data for k in needed)
    except Exception:
        return False


def downsample_raw(raw_example):
    """
    Downsample raw MOVI-F data from 512x512 to 256x256 BEFORE running
    track_points. This cuts memory usage by ~4x.

    Segmentation and object_coordinates use nearest interpolation to
    preserve integer IDs and quantized coords. Video uses bilinear.
    Depth and normals use nearest.
    """
    h, w = 256, 256

    video = tf.cast(
        tf.image.resize(raw_example["video"], [h, w], method="bilinear"),
        tf.uint8,
    )

    seg = tf.cast(
        tf.image.resize(raw_example["segmentations"], [h, w], method="nearest"),
        tf.uint8,
    )

    depth = tf.cast(
        tf.image.resize(
            tf.cast(raw_example["depth"], tf.float32), [h, w], method="nearest"
        ),
        tf.uint16,
    )

    obj_coords = tf.cast(
        tf.image.resize(
            tf.cast(raw_example["object_coordinates"], tf.float32),
            [h, w],
            method="nearest",
        ),
        tf.uint16,
    )

    normal = tf.cast(
        tf.image.resize(
            tf.cast(raw_example["normal"], tf.float32), [h, w], method="nearest"
        ),
        tf.uint16,
    )

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
    """
    For each query point, look up segmentation ID at its location and
    check is_dynamic from instance metadata.
    """
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


def process_sequence(seq_id, raw_example):
    seq_dir = OUTPUT_DIR / seq_id
    tmp_dir = OUTPUT_DIR / f"{seq_id}_tmp"

    if is_valid(seq_dir):
        return "skip"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    # Downsample 512->256 BEFORE annotation to save memory
    small = downsample_raw(raw_example)

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

    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir()
    for t in range(video_uint8.shape[0]):
        Image.fromarray(video_uint8[t]).save(frames_dir / f"{t:03d}.png")

    annot = {
        "coords": target_points.astype(np.float32),
        "visibility": (~occluded).astype(bool),
        "query_points": query_points.astype(np.float32),
        "is_dynamic": is_dynamic.astype(bool),
    }
    np.save(tmp_dir / f"{seq_id}.npy", annot, allow_pickle=True)

    os.rename(tmp_dir, seq_dir)

    del processed, small
    del query_points, target_points, occluded, video, video_uint8, is_dynamic

    return "done"


def count_done():
    if not OUTPUT_DIR.exists():
        return 0
    return sum(1 for d in OUTPUT_DIR.iterdir() if is_valid(d))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    done = count_done()
    print(f"Already processed: {done}")
    if done >= NUM_SEQUENCES:
        print(f"Already have {done} >= {NUM_SEQUENCES}. Done.")
        return

    print("Loading MOVI-F from GCS...")
    builder = movi_f.MoviF(data_dir="gs://kubric-public/tfds")
    ds = builder.as_dataset(split="train")

    processed_now = 0
    failed = 0
    start = time.time()

    for idx, raw_example in enumerate(ds):
        if done + processed_now >= NUM_SEQUENCES:
            break

        seq_id = f"{idx:06d}"

        try:
            t0 = time.time()
            status = process_sequence(seq_id, raw_example)

            if status == "done":
                processed_now += 1

            dt = time.time() - t0
            total = done + processed_now
            rate = processed_now / max(time.time() - start, 1e-6)
            remaining = NUM_SEQUENCES - total
            eta = remaining / rate / 60 if rate > 0 else 0

            print(
                f"[{total}/{NUM_SEQUENCES}] {seq_id}"
                f" | {status} | {dt:.1f}s"
                f" | {rate:.2f}/s | ETA {eta:.0f}m"
            )

        except Exception as e:
            failed += 1
            print(f"[FAIL] {seq_id}: {type(e).__name__}: {e}")
            tmp_dir = OUTPUT_DIR / f"{seq_id}_tmp"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

        finally:
            del raw_example
            tf.keras.backend.clear_session()
            gc.collect()

    elapsed = time.time() - start
    total = done + processed_now
    print(f"\n{'='*60}")
    print(f"DONE: {total} sequences ({processed_now} new, {failed} failed)")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()