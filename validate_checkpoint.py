"""
validate_checkpoint.py — Retroactively run validation on saved checkpoints.

Walks every leap_*.pth in a run directory, loads the model state, runs
validation, and writes/appends to that run's val_losses.csv.

Use this for runs that were trained before validation was added (baseline,
gce_both) so their plots can include val curves alongside trunc_both's.

Usage:
    python validate_checkpoint.py \\
        --run_dir ~/ASU/PIR/leapvo_training/checkpoints/baseline \\
        --val_dataset_root ~/ASU/PIR/leapvo_training/validation_dataset \\
        --num_tracks 128 --num_anchors 64 \\
        --loss_vis bce --loss_dyn bce

The --loss_vis / --loss_dyn flags must match what the run was trained with,
or the val numbers will be mis-attributed. We default to BCE.

Existing val_losses.csv rows are preserved; this script only appends rows
for checkpoint steps that aren't already in the file (so it's safe to re-run
or interrupt).
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Reach into train.py for the building blocks. PYTHONPATH must include
# the leapvo dir, or run this script from there.
import train as T   # noqa: E402  (pulls in build_leap_cfg, run_validation,
                    #              compute_batch_loss, LeapKernel,
                    #              LeapKubricDataset, collate_fn)


CKPT_RE = re.compile(r"leap_(\d{6})\.pth$")


def find_checkpoints(run_dir):
    """Return sorted list of (step, path) for every leap_NNNNNN.pth in run_dir."""
    found = []
    for p in sorted(Path(run_dir).iterdir()):
        m = CKPT_RE.match(p.name)
        if m:
            step = int(m.group(1))
            found.append((step, p))
    return found


def already_validated(val_csv_path):
    """Return set of step numbers that already have a row in val_losses.csv."""
    if not val_csv_path.exists():
        return set()
    done = set()
    with open(val_csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["step"]))
            except (KeyError, ValueError):
                continue
    return done


def build_model(args, device):
    """Construct LeapKernel using the same config build path train.py uses."""
    cfg = T.build_leap_cfg(args)
    model = T.LeapKernel(cfg=cfg, stride=4).to(device)
    return model


def load_state_into_model(model, ckpt_path, device):
    """Load model weights from a training checkpoint (saved by save_checkpoint)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        state = ckpt["model"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        logging.warning(f"  unexpected keys in ckpt: {len(unexpected)}")
    if missing:
        logging.warning(f"  missing keys in model:   {len(missing)}")


def main():
    p = argparse.ArgumentParser()

    # Run + dataset paths
    p.add_argument("--run_dir", required=True,
                   help="Path to a single run directory (contains leap_*.pth)")
    p.add_argument("--val_dataset_root", required=True,
                   help="Validation dataset path (same format as training_dataset)")

    # Loss config — must match what the run was trained with
    p.add_argument("--loss_vis",   default="bce", choices=["bce", "gce", "trunc_gce"])
    p.add_argument("--loss_dyn",   default="bce", choices=["bce", "gce", "trunc_gce"])
    p.add_argument("--gce_q",      type=float, default=0.7)
    p.add_argument("--trunc_frac", type=float, default=0.7)
    p.add_argument("--w1", type=float, default=1.0)
    p.add_argument("--w2", type=float, default=0.5)
    p.add_argument("--w3", type=float, default=0.5)

    # Model / data — must match training to keep loss values comparable
    p.add_argument("--window_size",  type=int, default=8)
    p.add_argument("--train_iters",  type=int, default=4)
    p.add_argument("--num_tracks",   type=int, default=128)
    p.add_argument("--num_anchors",  type=int, default=64)

    # Validation knobs
    p.add_argument("--val_max_batches", type=int, default=200)
    p.add_argument("--val_workers",     type=int, default=2)
    p.add_argument("--force",  action="store_true",
                   help="Re-validate every checkpoint, ignoring existing rows.")
    p.add_argument("--checkpoints", type=int, nargs="*", default=None,
                   help="Only validate these specific step numbers (e.g. 2000 4000).")

    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    run_dir = Path(args.run_dir).expanduser()
    val_root = Path(args.val_dataset_root).expanduser()
    val_csv_path = run_dir / "val_losses.csv"

    if not run_dir.exists():
        logging.error(f"run_dir does not exist: {run_dir}")
        sys.exit(1)
    if not val_root.exists():
        logging.error(f"val_dataset_root does not exist: {val_root}")
        sys.exit(1)

    # Find checkpoints
    ckpts = find_checkpoints(run_dir)
    if not ckpts:
        logging.error(f"No leap_*.pth checkpoints found in {run_dir}")
        sys.exit(1)
    logging.info(f"Found {len(ckpts)} checkpoints in {run_dir.name}")

    if args.checkpoints is not None:
        wanted = set(args.checkpoints)
        ckpts = [(s, p) for s, p in ckpts if s in wanted]
        logging.info(f"Filtering to requested steps: {sorted(wanted)} → {len(ckpts)} ckpts")

    done = set() if args.force else already_validated(val_csv_path)
    if done:
        logging.info(f"  {len(done)} step(s) already validated; will skip those.")

    todo = [(s, p) for s, p in ckpts if s not in done]
    if not todo:
        logging.info("Nothing to do — all checkpoints already validated.")
        return

    logging.info(f"Will validate {len(todo)} checkpoint(s).")

    # ---- Build val loader once ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    val_dataset = T.LeapKubricDataset(
        data_root=str(val_root),
        num_tracks=args.num_tracks,
    )
    logging.info(f"Validation dataset: {len(val_dataset)} sequences")
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.val_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=T.collate_fn,
    )

    # ---- Build model once; reload weights per checkpoint ----
    model = build_model(args, device)

    # ---- Open val CSV (append if exists, else fresh + header) ----
    if val_csv_path.exists():
        val_csv_file = open(val_csv_path, "a", newline="")
        val_csv_writer = csv.writer(val_csv_file)
        logging.info(f"Appending to existing {val_csv_path.name}")
    else:
        val_csv_file = open(val_csv_path, "w", newline="")
        val_csv_writer = csv.writer(val_csv_file)
        val_csv_writer.writerow([
            "step", "val_loss_total", "val_loss_main",
            "val_loss_vis", "val_loss_dyn", "n_batches",
        ])
        logging.info(f"Created {val_csv_path}")

    try:
        for step, ckpt_path in todo:
            logging.info(f"--- step {step} ({ckpt_path.name}) ---")
            load_state_into_model(model, ckpt_path, device)

            val = T.run_validation(
                model, val_loader, args, device,
                max_batches=args.val_max_batches,
            )

            val_csv_writer.writerow([
                step,
                val["total"], val["main"], val["vis"], val["dyn"],
                val["n_batches"],
            ])
            val_csv_file.flush()

            logging.info(
                f"  step {step:6d} | total={val['total']:.4f}  "
                f"main={val['main']:.4f}  vis={val['vis']:.4f}  "
                f"dyn={val['dyn']:.4f}  ({val['n_batches']} batches)"
            )
    finally:
        val_csv_file.close()
        logging.info(f"Wrote results → {val_csv_path}")


if __name__ == "__main__":
    main()