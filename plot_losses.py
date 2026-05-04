"""
plot_losses.py — Plot LEAP-VO training losses, individually and comparatively.

Reads losses.csv (always) and val_losses.csv (if present) from each run dir.
Validation curves are drawn as dashed lines in the same color as the train
curve. Generates:

  (A) Comparison plots (ALL runs overlaid):
        loss_comparison.png   — L_main, L_vis, L_dyn, L_total  (2x2 grid)
        diagnostics.png       — gnorm, |delta|, diag(Sigma)    (1x3 row)

  (B) Individual per-run plots (one pair per run):
        {run}/{run}_losses.png        — L_main, L_vis, L_dyn, L_total (2x2)
        {run}/{run}_diagnostics.png   — gnorm, |delta|, diag(Sigma)  (1x3)

Auto-skips any run whose CSV is missing — so this works with 1, 2, or 3 runs.

Usage:
    python plot_losses.py                                  # default paths
    python plot_losses.py --ckpt_root ~/my/checkpoints     # custom root
    python plot_losses.py --runs baseline gce_both         # subset
    python plot_losses.py --no_individual                  # only comparisons
    python plot_losses.py --no_comparison                  # only per-run plots
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Default colors per run — consistent across all plots
RUN_STYLES = {
    "baseline":   {"color": "#1f77b4", "label": "baseline (BCE)"},
    "gce_both":   {"color": "#d62728", "label": "gce_both (GCE, q=0.7)"},
    "trunc_both": {"color": "#2ca02c", "label": "trunc_both (Trunc-GCE)"},
}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load a CSV into a dict of column-name -> numpy array."""
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
    """
    Return dict of {run_name: {'train': train_csv, 'val': val_csv_or_None}}
    for runs that have at least a losses.csv.

    Sorts val by step to handle retroactive validation runs cleanly.
    """
    found = {}
    for name in run_names:
        run_dir = Path(ckpt_root).expanduser() / name
        train_csv = run_dir / "losses.csv"
        if not train_csv.exists():
            print(f"  [skip] {name}: {train_csv} not found")
            continue
        train_data = load_csv(train_csv)
        if train_data is None:
            print(f"  [skip] {name}: train CSV is empty")
            continue

        val_data = None
        val_csv = run_dir / "val_losses.csv"
        if val_csv.exists():
            val_data = load_csv(val_csv)
            if val_data is not None and len(val_data["step"]) > 0:
                # Sort by step ascending (retroactive runs may write
                # checkpoints out of order)
                order = np.argsort(val_data["step"])
                val_data = {k: v[order] for k, v in val_data.items()}
            else:
                val_data = None

        n_train = len(train_data["step"])
        n_val   = len(val_data["step"]) if val_data is not None else 0
        print(f"  [ok]   {name}: {n_train} train rows, {n_val} val rows, "
              f"final train step = {int(train_data['step'][-1])}")
        found[name] = {"train": train_data, "val": val_data}
    return found


def smooth(y, window=10):
    """Simple moving average for visual smoothing. NaN-safe."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    smoothed = np.convolve(y, kernel, mode="valid")
    pad_left = (len(y) - len(smoothed)) // 2
    pad_right = len(y) - len(smoothed) - pad_left
    return np.concatenate([
        np.full(pad_left, smoothed[0]),
        smoothed,
        np.full(pad_right, smoothed[-1]),
    ])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

# Map between train CSV column names and val CSV column names.
# Both use the same shorthand on the y-axis.
TRAIN_COLS = {
    "main":  "loss_main",
    "vis":   "loss_vis",
    "dyn":   "loss_dyn",
    "total": "loss_total",
}
VAL_COLS = {
    "main":  "val_loss_main",
    "vis":   "val_loss_vis",
    "dyn":   "val_loss_dyn",
    "total": "val_loss_total",
}


def _plot_losses_2x2(runs_to_plot, title, out_path, smooth_window=10):
    """
    runs_to_plot: dict of {run_name: {'train': data, 'val': data_or_None}}
    Draws main / vis / dyn / total in a 2x2 grid.
    Train: solid line. Val: dashed line + circle markers, same color.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    panels = [
        ("main",  "L_main (Cauchy NLL)",                     axes[0]),
        ("vis",   "L_vis (visibility)",                      axes[1]),
        ("dyn",   "L_dyn (dynamic label)",                   axes[2]),
        ("total", "L_total = 1.0·main + 0.5·vis + 0.5·dyn",  axes[3]),
    ]

    for short_key, panel_title, ax in panels:
        train_col = TRAIN_COLS[short_key]
        val_col   = VAL_COLS[short_key]
        for name, sources in runs_to_plot.items():
            style = RUN_STYLES.get(name, {"color": "gray", "label": name})
            tdata = sources.get("train")
            vdata = sources.get("val")

            # Train: faint raw + solid smoothed
            if tdata is not None and train_col in tdata:
                steps = tdata["step"]
                y = tdata[train_col]
                ax.plot(steps, y, color=style["color"],
                        alpha=0.20, linewidth=0.7)
                ax.plot(steps, smooth(y, smooth_window),
                        color=style["color"], linewidth=2.0,
                        label=f"{style['label']} (train)")

            # Val: dashed + markers
            if vdata is not None and val_col in vdata:
                ax.plot(vdata["step"], vdata[val_col],
                        color=style["color"], linewidth=1.8,
                        linestyle="--", marker="o", markersize=4,
                        label=f"{style['label']} (val)")

        ax.set_title(panel_title, fontsize=12)
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def _plot_diagnostics_1x3(runs_to_plot, title, out_path, smooth_window=10):
    """gnorm (log), |delta|, mean diag(Sigma) (log) — train-only data."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    panels = [
        ("grad_norm",  "gradient norm (pre-clip)",    "log",    axes[0]),
        ("delta_abs",  "|delta| (stride space)",      "linear", axes[1]),
    ]
    for col, panel_title, yscale, ax in panels:
        for name, sources in runs_to_plot.items():
            style = RUN_STYLES.get(name, {"color": "gray", "label": name})
            tdata = sources.get("train")
            if tdata is None or col not in tdata:
                continue
            steps = tdata["step"]
            y = tdata[col]
            ax.plot(steps, y, color=style["color"], alpha=0.25, linewidth=0.8)
            ax.plot(steps, smooth(y, smooth_window),
                    color=style["color"], linewidth=2.0, label=style["label"])
        ax.set_title(panel_title, fontsize=12)
        ax.set_xlabel("training step")
        ax.set_yscale(yscale)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best", fontsize=9)

    # Panel 3: mean sigma_diag, log scale
    ax = axes[2]
    for name, sources in runs_to_plot.items():
        style = RUN_STYLES.get(name, {"color": "gray", "label": name})
        tdata = sources.get("train")
        if (tdata is None
            or "sigma_diag_x" not in tdata or "sigma_diag_y" not in tdata):
            continue
        steps = tdata["step"]
        y = 0.5 * (tdata["sigma_diag_x"] + tdata["sigma_diag_y"])
        ax.plot(steps, y, color=style["color"], alpha=0.25, linewidth=0.8)
        ax.plot(steps, smooth(y, smooth_window),
                color=style["color"], linewidth=2.0, label=style["label"])
    ax.set_title("mean diag(Sigma)  [lower = more confident model]", fontsize=12)
    ax.set_xlabel("training step")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Public plot drivers
# ---------------------------------------------------------------------------

def plot_comparison(runs, out_dir, smooth_window=10):
    """All runs overlaid — the summary plots."""
    print("\nComparison plots:")
    _plot_losses_2x2(
        runs,
        title="LEAP-VO loss comparison "
              "(faint=raw train, solid=smoothed train, dashed=val)",
        out_path=out_dir / "loss_comparison.png",
        smooth_window=smooth_window,
    )
    _plot_diagnostics_1x3(
        runs,
        title="Training diagnostics — comparison across runs",
        out_path=out_dir / "diagnostics.png",
        smooth_window=smooth_window,
    )


def plot_individual(runs, out_dir, smooth_window=10):
    """One pair of plots per run, saved in a per-run subdirectory."""
    print("\nPer-run individual plots:")
    for name, sources in runs.items():
        label = RUN_STYLES.get(name, {}).get("label", name)
        run_out = out_dir / name
        run_out.mkdir(parents=True, exist_ok=True)
        _plot_losses_2x2(
            {name: sources},
            title=f"Training losses — {label}",
            out_path=run_out / f"{name}_losses.png",
            smooth_window=smooth_window,
        )
        _plot_diagnostics_1x3(
            {name: sources},
            title=f"Diagnostics — {label}",
            out_path=run_out / f"{name}_diagnostics.png",
            smooth_window=smooth_window,
        )


def print_summary_table(runs):
    """Print end-of-training numbers side by side."""
    print()
    print("=" * 92)
    print("FINAL VALUES (last logged step per run)")
    print("=" * 92)
    header = (f"{'run':<14} {'split':<6} {'step':>7} {'main':>10} "
              f"{'vis':>8} {'dyn':>8} {'total':>10}")
    print(header)
    print("-" * len(header))
    for name, sources in runs.items():
        tdata = sources.get("train")
        vdata = sources.get("val")
        if tdata is not None:
            i = -1
            print(f"{name:<14} {'train':<6} {int(tdata['step'][i]):>7} "
                  f"{tdata['loss_main'][i]:>10.3f} "
                  f"{tdata['loss_vis'][i]:>8.4f} "
                  f"{tdata['loss_dyn'][i]:>8.4f} "
                  f"{tdata['loss_total'][i]:>10.3f}")
        if vdata is not None and len(vdata["step"]) > 0:
            i = -1
            print(f"{'':<14} {'val':<6} {int(vdata['step'][i]):>7} "
                  f"{vdata['val_loss_main'][i]:>10.3f} "
                  f"{vdata['val_loss_vis'][i]:>8.4f} "
                  f"{vdata['val_loss_dyn'][i]:>8.4f} "
                  f"{vdata['val_loss_total'][i]:>10.3f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_root",
                   default="~/ASU/PIR/leapvo_training/checkpoints",
                   help="Root directory containing per-run subdirs")
    p.add_argument("--runs", nargs="+",
                   default=["baseline", "gce_both", "trunc_both"],
                   help="Run names to plot (must match subdirectory names)")
    p.add_argument("--out_dir", default=".",
                   help="Where to write the PNG files")
    p.add_argument("--smooth", type=int, default=10,
                   help="Moving-average window (in log intervals)")
    p.add_argument("--no_individual", action="store_true",
                   help="Skip per-run individual plots")
    p.add_argument("--no_comparison", action="store_true",
                   help="Skip overlaid comparison plots")
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

    


# #!/usr/bin/env python3
# """
# plot_losses.py — Plot LEAP-VO training losses as INDIVIDUAL PNG FILES.

# Reads losses.csv from each run directory.

# Outputs:

# (A) Comparison plots (all runs overlaid, one metric per PNG):
#     comparison_loss_main.png
#     comparison_loss_vis.png
#     comparison_loss_dyn.png
#     comparison_loss_total.png
#     comparison_grad_norm.png
#     comparison_delta_abs.png
#     comparison_sigma_diag_mean.png

# (B) Individual per-run plots (one metric per PNG):
#     {run}/{run}_loss_main.png
#     {run}/{run}_loss_vis.png
#     {run}/{run}_loss_dyn.png
#     {run}/{run}_loss_total.png
#     {run}/{run}_grad_norm.png
#     {run}/{run}_delta_abs.png
#     {run}/{run}_sigma_diag_mean.png

# Usage:
#     python plot_losses.py
#     python plot_losses.py --ckpt_root ~/my/checkpoints
#     python plot_losses.py --runs baseline gce_both
#     python plot_losses.py --no_individual
#     python plot_losses.py --no_comparison
# """

# import argparse
# import csv
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np


# RUN_STYLES = {
#     "baseline":   {"color": "#1f77b4", "label": "baseline (BCE)"},
#     "gce_both":   {"color": "#d62728", "label": "gce_both (GCE, q=0.7)"},
#     "trunc_both": {"color": "#2ca02c", "label": "trunc_both (Trunc-GCE)"},
# }


# def load_csv(path):
#     """Load a losses.csv into a dict of column-name -> numpy array."""
#     rows = []
#     with open(path) as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             rows.append(row)

#     if not rows:
#         return None

#     columns = {}
#     for key in rows[0].keys():
#         vals = []
#         for row in rows:
#             try:
#                 vals.append(float(row[key]))
#             except (ValueError, TypeError):
#                 vals.append(np.nan)
#         columns[key] = np.array(vals)

#     return columns


# def find_runs(ckpt_root, run_names):
#     """Return dict of {run_name: csv_data} for runs that have a losses.csv."""
#     found = {}
#     for name in run_names:
#         csv_path = Path(ckpt_root).expanduser() / name / "losses.csv"
#         if not csv_path.exists():
#             print(f"  [skip] {name}: {csv_path} not found")
#             continue

#         data = load_csv(csv_path)
#         if data is None:
#             print(f"  [skip] {name}: CSV is empty")
#             continue

#         if "step" not in data:
#             print(f"  [skip] {name}: missing 'step' column")
#             continue

#         print(
#             f"  [ok]   {name}: {len(data['step'])} log entries, "
#             f"final step = {int(data['step'][-1])}"
#         )
#         found[name] = data

#     return found


# def smooth(y, window=10):
#     """Simple moving average for visual smoothing. NaN-safe enough for plotting."""
#     if len(y) < window or window <= 1:
#         return y

#     y = np.asarray(y, dtype=float)
#     kernel = np.ones(window) / window

#     # crude NaN handling: interpolate only for smoothing visualization
#     x = np.arange(len(y))
#     finite = np.isfinite(y)
#     if finite.sum() == 0:
#         return y
#     if finite.sum() == 1:
#         return np.full_like(y, y[finite][0], dtype=float)

#     y_filled = np.interp(x, x[finite], y[finite])
#     smoothed = np.convolve(y_filled, kernel, mode="valid")

#     pad_left = (len(y) - len(smoothed)) // 2
#     pad_right = len(y) - len(smoothed) - pad_left

#     return np.concatenate([
#         np.full(pad_left, smoothed[0]),
#         smoothed,
#         np.full(pad_right, smoothed[-1]),
#     ])


# def plot_single_metric(
#     runs_to_plot,
#     out_path,
#     title,
#     y_label,
#     value_getter,
#     smooth_window=10,
#     yscale="linear",
# ):
#     """
#     Plot one metric per figure.

#     runs_to_plot: dict of {run_name: data}
#     value_getter: function(data) -> y array or None
#     """
#     fig, ax = plt.subplots(figsize=(10, 5.5))

#     plotted_any = False

#     for name, data in runs_to_plot.items():
#         style = RUN_STYLES.get(name, {"color": "gray", "label": name})

#         if "step" not in data:
#             continue

#         y = value_getter(data)
#         if y is None:
#             continue

#         y = np.asarray(y, dtype=float)
#         steps = np.asarray(data["step"], dtype=float)

#         if len(steps) == 0 or len(y) == 0 or len(steps) != len(y):
#             continue

#         ax.plot(steps, y, color=style["color"], alpha=0.25, linewidth=0.8)
#         ax.plot(
#             steps,
#             smooth(y, smooth_window),
#             color=style["color"],
#             linewidth=2.0,
#             label=style["label"],
#         )
#         plotted_any = True

#     if not plotted_any:
#         plt.close(fig)
#         print(f"  [skip] {out_path} (no valid data)")
#         return

#     ax.set_title(title, fontsize=12)
#     ax.set_xlabel("training step")
#     ax.set_ylabel(y_label)
#     ax.set_yscale(yscale)
#     ax.grid(alpha=0.3, which="both")
#     ax.legend(loc="best", fontsize=9)

#     fig.tight_layout()
#     fig.savefig(out_path, dpi=130, bbox_inches="tight")
#     plt.close(fig)
#     print(f"  → {out_path}")


# def plot_comparison(runs, out_dir, smooth_window=10):
#     """All runs overlaid — one PNG per metric."""
#     print("\nComparison plots:")

#     specs = [
#         {
#             "filename": "comparison_loss_main.png",
#             "title": "L_main (Cauchy NLL) — comparison across runs",
#             "ylabel": "loss",
#             "getter": lambda d: d.get("loss_main"),
#             "yscale": "linear",
#         },
#         {
#             "filename": "comparison_loss_vis.png",
#             "title": "L_vis (visibility) — comparison across runs",
#             "ylabel": "loss",
#             "getter": lambda d: d.get("loss_vis"),
#             "yscale": "linear",
#         },
#         {
#             "filename": "comparison_loss_dyn.png",
#             "title": "L_dyn (dynamic label) — comparison across runs",
#             "ylabel": "loss",
#             "getter": lambda d: d.get("loss_dyn"),
#             "yscale": "linear",
#         },
#         {
#             "filename": "comparison_loss_total.png",
#             "title": "L_total — comparison across runs",
#             "ylabel": "loss",
#             "getter": lambda d: d.get("loss_total"),
#             "yscale": "linear",
#         },
#         {
#             "filename": "comparison_grad_norm.png",
#             "title": "gradient norm (pre-clip) — comparison across runs",
#             "ylabel": "grad norm",
#             "getter": lambda d: d.get("grad_norm"),
#             "yscale": "log",
#         },
#         {
#             "filename": "comparison_delta_abs.png",
#             "title": "|delta| (stride space) — comparison across runs",
#             "ylabel": "|delta|",
#             "getter": lambda d: d.get("delta_abs"),
#             "yscale": "linear",
#         },
#         {
#             "filename": "comparison_sigma_diag_mean.png",
#             "title": "mean diag(Sigma) — comparison across runs",
#             "ylabel": "mean diag(Sigma)",
#             "getter": lambda d: (
#                 0.5 * (d["sigma_diag_x"] + d["sigma_diag_y"])
#                 if "sigma_diag_x" in d and "sigma_diag_y" in d
#                 else None
#             ),
#             "yscale": "log",
#         },
#     ]

#     for spec in specs:
#         plot_single_metric(
#             runs_to_plot=runs,
#             out_path=out_dir / spec["filename"],
#             title=spec["title"],
#             y_label=spec["ylabel"],
#             value_getter=spec["getter"],
#             smooth_window=smooth_window,
#             yscale=spec["yscale"],
#         )


# def plot_individual(runs, out_dir, smooth_window=10):
#     """One PNG per metric per run."""
#     print("\nPer-run individual plots:")

#     for name, data in runs.items():
#         label = RUN_STYLES.get(name, {}).get("label", name)
#         run_out = out_dir / name
#         run_out.mkdir(parents=True, exist_ok=True)

#         specs = [
#             {
#                 "filename": f"{name}_loss_main.png",
#                 "title": f"L_main (Cauchy NLL) — {label}",
#                 "ylabel": "loss",
#                 "getter": lambda d: d.get("loss_main"),
#                 "yscale": "linear",
#             },
#             {
#                 "filename": f"{name}_loss_vis.png",
#                 "title": f"L_vis (visibility) — {label}",
#                 "ylabel": "loss",
#                 "getter": lambda d: d.get("loss_vis"),
#                 "yscale": "linear",
#             },
#             {
#                 "filename": f"{name}_loss_dyn.png",
#                 "title": f"L_dyn (dynamic label) — {label}",
#                 "ylabel": "loss",
#                 "getter": lambda d: d.get("loss_dyn"),
#                 "yscale": "linear",
#             },
#             {
#                 "filename": f"{name}_loss_total.png",
#                 "title": f"L_total — {label}",
#                 "ylabel": "loss",
#                 "getter": lambda d: d.get("loss_total"),
#                 "yscale": "linear",
#             },
#             {
#                 "filename": f"{name}_grad_norm.png",
#                 "title": f"gradient norm (pre-clip) — {label}",
#                 "ylabel": "grad norm",
#                 "getter": lambda d: d.get("grad_norm"),
#                 "yscale": "log",
#             },
#             {
#                 "filename": f"{name}_delta_abs.png",
#                 "title": f"|delta| (stride space) — {label}",
#                 "ylabel": "|delta|",
#                 "getter": lambda d: d.get("delta_abs"),
#                 "yscale": "linear",
#             },
#             {
#                 "filename": f"{name}_sigma_diag_mean.png",
#                 "title": f"mean diag(Sigma) — {label}",
#                 "ylabel": "mean diag(Sigma)",
#                 "getter": lambda d: (
#                     0.5 * (d["sigma_diag_x"] + d["sigma_diag_y"])
#                     if "sigma_diag_x" in d and "sigma_diag_y" in d
#                     else None
#                 ),
#                 "yscale": "log",
#             },
#         ]

#         for spec in specs:
#             plot_single_metric(
#                 runs_to_plot={name: data},
#                 out_path=run_out / spec["filename"],
#                 title=spec["title"],
#                 y_label=spec["ylabel"],
#                 value_getter=spec["getter"],
#                 smooth_window=smooth_window,
#                 yscale=spec["yscale"],
#             )


# def print_summary_table(runs):
#     """Print end-of-training numbers side by side."""
#     print()
#     print("=" * 78)
#     print("FINAL VALUES (last logged step per run)")
#     print("=" * 78)
#     header = (
#         f"{'run':<14} {'step':>7} {'main':>10} {'vis':>8} "
#         f"{'dyn':>8} {'total':>10} {'gnorm':>10}"
#     )
#     print(header)
#     print("-" * len(header))

#     for name, data in runs.items():
#         i = -1

#         def v(col):
#             arr = data.get(col)
#             if arr is None or len(arr) == 0:
#                 return np.nan
#             return arr[i]

#         print(
#             f"{name:<14} {int(data['step'][i]):>7} "
#             f"{v('loss_main'):>10.3f} "
#             f"{v('loss_vis'):>8.4f} "
#             f"{v('loss_dyn'):>8.4f} "
#             f"{v('loss_total'):>10.3f} "
#             f"{v('grad_norm'):>10.1f}"
#         )
#     print()


# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument(
#         "--ckpt_root",
#         default="~/ASU/PIR/leapvo_training/checkpoints",
#         help="Root directory containing per-run subdirs",
#     )
#     p.add_argument(
#         "--runs",
#         nargs="+",
#         default=["baseline", "gce_both", "trunc_both"],
#         help="Run names to plot (must match subdirectory names)",
#     )
#     p.add_argument(
#         "--out_dir",
#         default=".",
#         help="Where to write the PNG files",
#     )
#     p.add_argument(
#         "--smooth",
#         type=int,
#         default=10,
#         help="Moving-average window (in log intervals)",
#     )
#     p.add_argument(
#         "--no_individual",
#         action="store_true",
#         help="Skip per-run individual plots",
#     )
#     p.add_argument(
#         "--no_comparison",
#         action="store_true",
#         help="Skip overlaid comparison plots",
#     )
#     args = p.parse_args()

#     print(f"Looking for runs under: {Path(args.ckpt_root).expanduser()}")
#     runs = find_runs(args.ckpt_root, args.runs)

#     if not runs:
#         print("\nNo runs found with a losses.csv — nothing to plot.")
#         return

#     print(f"\nFound {len(runs)} run(s): {list(runs.keys())}")
#     print_summary_table(runs)

#     out_dir = Path(args.out_dir).expanduser()
#     out_dir.mkdir(parents=True, exist_ok=True)

#     if not args.no_comparison:
#         plot_comparison(runs, out_dir, args.smooth)

#     if not args.no_individual:
#         plot_individual(runs, out_dir, args.smooth)

#     print("\nDone.")


# if __name__ == "__main__":
#     main()