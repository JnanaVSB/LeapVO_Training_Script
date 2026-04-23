#!/usr/bin/env python3
"""
plot_losses.py — Plot LEAP-VO training losses as INDIVIDUAL PNG FILES.

Reads losses.csv from each run directory.

Outputs:

(A) Comparison plots (all runs overlaid, one metric per PNG):
    comparison_loss_main.png
    comparison_loss_vis.png
    comparison_loss_dyn.png
    comparison_loss_total.png
    comparison_grad_norm.png
    comparison_delta_abs.png
    comparison_sigma_diag_mean.png

(B) Individual per-run plots (one metric per PNG):
    {run}/{run}_loss_main.png
    {run}/{run}_loss_vis.png
    {run}/{run}_loss_dyn.png
    {run}/{run}_loss_total.png
    {run}/{run}_grad_norm.png
    {run}/{run}_delta_abs.png
    {run}/{run}_sigma_diag_mean.png

Usage:
    python plot_losses.py
    python plot_losses.py --ckpt_root ~/my/checkpoints
    python plot_losses.py --runs baseline gce_both
    python plot_losses.py --no_individual
    python plot_losses.py --no_comparison
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUN_STYLES = {
    "baseline":   {"color": "#1f77b4", "label": "baseline (BCE)"},
    "gce_both":   {"color": "#d62728", "label": "gce_both (GCE, q=0.7)"},
    "trunc_both": {"color": "#2ca02c", "label": "trunc_both (Trunc-GCE)"},
}


def load_csv(path):
    """Load a losses.csv into a dict of column-name -> numpy array."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    columns = {}
    for key in rows[0].keys():
        vals = []
        for row in rows:
            try:
                vals.append(float(row[key]))
            except (ValueError, TypeError):
                vals.append(np.nan)
        columns[key] = np.array(vals)

    return columns


def find_runs(ckpt_root, run_names):
    """Return dict of {run_name: csv_data} for runs that have a losses.csv."""
    found = {}
    for name in run_names:
        csv_path = Path(ckpt_root).expanduser() / name / "losses.csv"
        if not csv_path.exists():
            print(f"  [skip] {name}: {csv_path} not found")
            continue

        data = load_csv(csv_path)
        if data is None:
            print(f"  [skip] {name}: CSV is empty")
            continue

        if "step" not in data:
            print(f"  [skip] {name}: missing 'step' column")
            continue

        print(
            f"  [ok]   {name}: {len(data['step'])} log entries, "
            f"final step = {int(data['step'][-1])}"
        )
        found[name] = data

    return found


def smooth(y, window=10):
    """Simple moving average for visual smoothing. NaN-safe enough for plotting."""
    if len(y) < window or window <= 1:
        return y

    y = np.asarray(y, dtype=float)
    kernel = np.ones(window) / window

    # crude NaN handling: interpolate only for smoothing visualization
    x = np.arange(len(y))
    finite = np.isfinite(y)
    if finite.sum() == 0:
        return y
    if finite.sum() == 1:
        return np.full_like(y, y[finite][0], dtype=float)

    y_filled = np.interp(x, x[finite], y[finite])
    smoothed = np.convolve(y_filled, kernel, mode="valid")

    pad_left = (len(y) - len(smoothed)) // 2
    pad_right = len(y) - len(smoothed) - pad_left

    return np.concatenate([
        np.full(pad_left, smoothed[0]),
        smoothed,
        np.full(pad_right, smoothed[-1]),
    ])


def plot_single_metric(
    runs_to_plot,
    out_path,
    title,
    y_label,
    value_getter,
    smooth_window=10,
    yscale="linear",
):
    """
    Plot one metric per figure.

    runs_to_plot: dict of {run_name: data}
    value_getter: function(data) -> y array or None
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    plotted_any = False

    for name, data in runs_to_plot.items():
        style = RUN_STYLES.get(name, {"color": "gray", "label": name})

        if "step" not in data:
            continue

        y = value_getter(data)
        if y is None:
            continue

        y = np.asarray(y, dtype=float)
        steps = np.asarray(data["step"], dtype=float)

        if len(steps) == 0 or len(y) == 0 or len(steps) != len(y):
            continue

        ax.plot(steps, y, color=style["color"], alpha=0.25, linewidth=0.8)
        ax.plot(
            steps,
            smooth(y, smooth_window),
            color=style["color"],
            linewidth=2.0,
            label=style["label"],
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(f"  [skip] {out_path} (no valid data)")
        return

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("training step")
    ax.set_ylabel(y_label)
    ax.set_yscale(yscale)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def plot_comparison(runs, out_dir, smooth_window=10):
    """All runs overlaid — one PNG per metric."""
    print("\nComparison plots:")

    specs = [
        {
            "filename": "comparison_loss_main.png",
            "title": "L_main (Cauchy NLL) — comparison across runs",
            "ylabel": "loss",
            "getter": lambda d: d.get("loss_main"),
            "yscale": "linear",
        },
        {
            "filename": "comparison_loss_vis.png",
            "title": "L_vis (visibility) — comparison across runs",
            "ylabel": "loss",
            "getter": lambda d: d.get("loss_vis"),
            "yscale": "linear",
        },
        {
            "filename": "comparison_loss_dyn.png",
            "title": "L_dyn (dynamic label) — comparison across runs",
            "ylabel": "loss",
            "getter": lambda d: d.get("loss_dyn"),
            "yscale": "linear",
        },
        {
            "filename": "comparison_loss_total.png",
            "title": "L_total — comparison across runs",
            "ylabel": "loss",
            "getter": lambda d: d.get("loss_total"),
            "yscale": "linear",
        },
        {
            "filename": "comparison_grad_norm.png",
            "title": "gradient norm (pre-clip) — comparison across runs",
            "ylabel": "grad norm",
            "getter": lambda d: d.get("grad_norm"),
            "yscale": "log",
        },
        {
            "filename": "comparison_delta_abs.png",
            "title": "|delta| (stride space) — comparison across runs",
            "ylabel": "|delta|",
            "getter": lambda d: d.get("delta_abs"),
            "yscale": "linear",
        },
        {
            "filename": "comparison_sigma_diag_mean.png",
            "title": "mean diag(Sigma) — comparison across runs",
            "ylabel": "mean diag(Sigma)",
            "getter": lambda d: (
                0.5 * (d["sigma_diag_x"] + d["sigma_diag_y"])
                if "sigma_diag_x" in d and "sigma_diag_y" in d
                else None
            ),
            "yscale": "log",
        },
    ]

    for spec in specs:
        plot_single_metric(
            runs_to_plot=runs,
            out_path=out_dir / spec["filename"],
            title=spec["title"],
            y_label=spec["ylabel"],
            value_getter=spec["getter"],
            smooth_window=smooth_window,
            yscale=spec["yscale"],
        )


def plot_individual(runs, out_dir, smooth_window=10):
    """One PNG per metric per run."""
    print("\nPer-run individual plots:")

    for name, data in runs.items():
        label = RUN_STYLES.get(name, {}).get("label", name)
        run_out = out_dir / name
        run_out.mkdir(parents=True, exist_ok=True)

        specs = [
            {
                "filename": f"{name}_loss_main.png",
                "title": f"L_main (Cauchy NLL) — {label}",
                "ylabel": "loss",
                "getter": lambda d: d.get("loss_main"),
                "yscale": "linear",
            },
            {
                "filename": f"{name}_loss_vis.png",
                "title": f"L_vis (visibility) — {label}",
                "ylabel": "loss",
                "getter": lambda d: d.get("loss_vis"),
                "yscale": "linear",
            },
            {
                "filename": f"{name}_loss_dyn.png",
                "title": f"L_dyn (dynamic label) — {label}",
                "ylabel": "loss",
                "getter": lambda d: d.get("loss_dyn"),
                "yscale": "linear",
            },
            {
                "filename": f"{name}_loss_total.png",
                "title": f"L_total — {label}",
                "ylabel": "loss",
                "getter": lambda d: d.get("loss_total"),
                "yscale": "linear",
            },
            {
                "filename": f"{name}_grad_norm.png",
                "title": f"gradient norm (pre-clip) — {label}",
                "ylabel": "grad norm",
                "getter": lambda d: d.get("grad_norm"),
                "yscale": "log",
            },
            {
                "filename": f"{name}_delta_abs.png",
                "title": f"|delta| (stride space) — {label}",
                "ylabel": "|delta|",
                "getter": lambda d: d.get("delta_abs"),
                "yscale": "linear",
            },
            {
                "filename": f"{name}_sigma_diag_mean.png",
                "title": f"mean diag(Sigma) — {label}",
                "ylabel": "mean diag(Sigma)",
                "getter": lambda d: (
                    0.5 * (d["sigma_diag_x"] + d["sigma_diag_y"])
                    if "sigma_diag_x" in d and "sigma_diag_y" in d
                    else None
                ),
                "yscale": "log",
            },
        ]

        for spec in specs:
            plot_single_metric(
                runs_to_plot={name: data},
                out_path=run_out / spec["filename"],
                title=spec["title"],
                y_label=spec["ylabel"],
                value_getter=spec["getter"],
                smooth_window=smooth_window,
                yscale=spec["yscale"],
            )


def print_summary_table(runs):
    """Print end-of-training numbers side by side."""
    print()
    print("=" * 78)
    print("FINAL VALUES (last logged step per run)")
    print("=" * 78)
    header = (
        f"{'run':<14} {'step':>7} {'main':>10} {'vis':>8} "
        f"{'dyn':>8} {'total':>10} {'gnorm':>10}"
    )
    print(header)
    print("-" * len(header))

    for name, data in runs.items():
        i = -1

        def v(col):
            arr = data.get(col)
            if arr is None or len(arr) == 0:
                return np.nan
            return arr[i]

        print(
            f"{name:<14} {int(data['step'][i]):>7} "
            f"{v('loss_main'):>10.3f} "
            f"{v('loss_vis'):>8.4f} "
            f"{v('loss_dyn'):>8.4f} "
            f"{v('loss_total'):>10.3f} "
            f"{v('grad_norm'):>10.1f}"
        )
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt_root",
        default="~/ASU/PIR/leapvo_training/checkpoints",
        help="Root directory containing per-run subdirs",
    )
    p.add_argument(
        "--runs",
        nargs="+",
        default=["baseline", "gce_both", "trunc_both"],
        help="Run names to plot (must match subdirectory names)",
    )
    p.add_argument(
        "--out_dir",
        default=".",
        help="Where to write the PNG files",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=10,
        help="Moving-average window (in log intervals)",
    )
    p.add_argument(
        "--no_individual",
        action="store_true",
        help="Skip per-run individual plots",
    )
    p.add_argument(
        "--no_comparison",
        action="store_true",
        help="Skip overlaid comparison plots",
    )
    args = p.parse_args()

    print(f"Looking for runs under: {Path(args.ckpt_root).expanduser()}")
    runs = find_runs(args.ckpt_root, args.runs)

    if not runs:
        print("\nNo runs found with a losses.csv — nothing to plot.")
        return

    print(f"\nFound {len(runs)} run(s): {list(runs.keys())}")
    print_summary_table(runs)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_comparison:
        plot_comparison(runs, out_dir, args.smooth)

    if not args.no_individual:
        plot_individual(runs, out_dir, args.smooth)

    print("\nDone.")


if __name__ == "__main__":
    main()