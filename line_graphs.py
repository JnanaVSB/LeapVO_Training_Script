#!/usr/bin/env python3
"""
generate_line_graphs.py
Generates line plots with shaded std bands showing how metrics
vary across multiple runs. Useful for demonstrating run-to-run stability.

Usage: python generate_line_graphs.py

Output: logs/figures/ directory with line plot PDFs and PNGs
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict

NUM_RUNS = 5
DATASETS = {
    "sintel": {"dir": "logs_run/sintel", "title": "MPI-Sintel"},
    "replica": {"dir": "logs_run/replica", "title": "Replica"},
    "shibuya": {"dir": "logs_run/shibuya", "title": "TartanAir-Shibuya"},
}
FIG_DIR = "logs_run/figures_line"

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def parse_error_sum(filepath):
    results = {}
    if not os.path.exists(filepath):
        return results
    with open(filepath, "r") as f:
        lines = f.readlines()
    for line in lines:
        match = re.match(
            r"\s*(\S+)\s*\|\s*ATE:\s*([\d.]+),\s*RPE trans:\s*([\d.]+),\s*RPE rot:\s*([\d.]+)",
            line,
        )
        if match:
            scene = match.group(1)
            ate = float(match.group(2))
            rpe_t = float(match.group(3))
            rpe_r = float(match.group(4))
            results[scene] = (ate, rpe_t, rpe_r)
    return results


def collect_per_run(base_dir):
    """Collect results organized per run.
    Returns: dict[run_i] -> dict[scene] -> (ate, rpe_t, rpe_r)
    """
    per_run = OrderedDict()
    for run_i in range(1, NUM_RUNS + 1):
        error_file = os.path.join(base_dir, f"run_{run_i}", "error_sum.txt")
        results = parse_error_sum(error_file)
        if results:
            per_run[run_i] = results
    return per_run


def short_scene_name(scene):
    for prefix in ["sintel-", "replica-", "shibuya-"]:
        if scene.startswith(prefix):
            scene = scene[len(prefix):]
    scene = scene.replace("-Sequence_1", "")
    return scene


def plot_per_scene_lines(dataset_name, dataset_title, per_run):
    """Line plot: one line per scene, x=run number, y=ATE, with markers."""
    if not per_run:
        return

    # Find scenes present in all runs
    all_scenes = set()
    for run_results in per_run.values():
        all_scenes.update(run_results.keys())

    runs = sorted(per_run.keys())
    scenes = sorted(all_scenes)

    # Build matrix: (scenes, runs) — use NaN for missing
    ate_matrix = np.full((len(scenes), len(runs)), np.nan)
    for j, run_i in enumerate(runs):
        for i, scene in enumerate(scenes):
            if scene in per_run[run_i]:
                ate_matrix[i, j] = per_run[run_i][scene][0]  # ATE

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = plt.cm.tab20(np.linspace(0, 1, len(scenes)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'X', 'h', '+', 'x', '1', '2', '3']

    for i, scene in enumerate(scenes):
        vals = ate_matrix[i]
        valid = ~np.isnan(vals)
        if valid.sum() == 0:
            continue
        ax.plot(np.array(runs)[valid], vals[valid],
                marker=markers[i % len(markers)], color=colors[i],
                linewidth=1.5, markersize=5, label=short_scene_name(scene))

    ax.set_xlabel("Run")
    ax.set_ylabel("ATE (RMSE)")
    ax.set_title(f"LEAP-VO: Per-Scene ATE Across Runs — {dataset_title}")
    ax.set_xticks(runs)
    ax.set_xticklabels([str(r) for r in runs])

    # Put legend outside if many scenes
    if len(scenes) > 6:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0,
                  fontsize=8, ncol=1)
        fig.subplots_adjust(right=0.72)
    else:
        ax.legend(fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"{dataset_name}_per_scene_lines.{ext}")
        fig.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_aggregate_line_with_band(dataset_name, dataset_title, per_run):
    """Line plot with shaded band: x=run, y=mean ATE across scenes, band=±std across scenes."""
    if not per_run:
        return

    runs = sorted(per_run.keys())

    # For each run, compute mean and std of ATE across all scenes in that run
    run_means = []
    run_stds = []
    for run_i in runs:
        ates = [v[0] for v in per_run[run_i].values()]
        run_means.append(np.mean(ates))
        run_stds.append(np.std(ates))

    run_means = np.array(run_means)
    run_stds = np.array(run_stds)
    x = np.array(runs)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(x, run_means, 'o-', color='#4878CF', linewidth=2, markersize=7, label='Mean ATE', zorder=3)
    ax.fill_between(x, run_means - run_stds, run_means + run_stds,
                    alpha=0.25, color='#4878CF', label='± 1 std (across scenes)', zorder=2)

    ax.set_xlabel("Run")
    ax.set_ylabel("ATE (RMSE)")
    ax.set_title(f"LEAP-VO: Mean ATE Across Runs — {dataset_title}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in runs])
    ax.legend()

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"{dataset_name}_aggregate_line.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_three_metrics_lines(dataset_name, dataset_title, per_run):
    """3-panel line plot with shaded bands for ATE, RPE_t, RPE_r."""
    if not per_run:
        return

    runs = sorted(per_run.keys())
    x = np.array(runs)

    metrics = {
        "ATE (RMSE)": 0,
        "RPE Trans (RMSE)": 1,
        "RPE Rot (deg, RMSE)": 2,
    }
    colors = ['#4878CF', '#6ACC65', '#D65F5F']

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    fig.suptitle(f"LEAP-VO: Metric Stability Across Runs — {dataset_title}",
                 fontsize=13, fontweight='bold', y=1.02)

    for ax, (metric_name, metric_idx), color in zip(axes, metrics.items(), colors):
        run_means = []
        run_stds = []
        for run_i in runs:
            vals = [v[metric_idx] for v in per_run[run_i].values()]
            run_means.append(np.mean(vals))
            run_stds.append(np.std(vals))

        run_means = np.array(run_means)
        run_stds = np.array(run_stds)

        ax.plot(x, run_means, 'o-', color=color, linewidth=2, markersize=6, zorder=3)
        ax.fill_between(x, run_means - run_stds, run_means + run_stds,
                        alpha=0.25, color=color, zorder=2)

        ax.set_xlabel("Run")
        ax.set_ylabel(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in runs])

        # Annotate values
        for xi, m in zip(x, run_means):
            ax.annotate(f'{m:.4f}', (xi, m), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=7, color='#333333')

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"{dataset_name}_stability_3metrics.{ext}")
        fig.savefig(path, bbox_inches='tight')
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_combined_all_datasets(all_per_run):
    """Single figure: one line per dataset, shaded band, showing ATE stability."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    colors = {'sintel': '#4878CF', 'replica': '#6ACC65', 'shibuya': '#D65F5F'}

    for dname, per_run in all_per_run.items():
        if not per_run:
            continue
        runs = sorted(per_run.keys())
        x = np.array(runs)

        run_means = []
        run_stds = []
        for run_i in runs:
            ates = [v[0] for v in per_run[run_i].values()]
            run_means.append(np.mean(ates))
            run_stds.append(np.std(ates))

        run_means = np.array(run_means)
        run_stds = np.array(run_stds)

        label = DATASETS[dname]["title"]
        c = colors.get(dname, '#333333')

        ax.plot(x, run_means, 'o-', color=c, linewidth=2, markersize=6,
                label=label, zorder=3)
        ax.fill_between(x, run_means - run_stds, run_means + run_stds,
                        alpha=0.2, color=c, zorder=2)

    ax.set_xlabel("Run")
    ax.set_ylabel("Mean ATE (RMSE)")
    ax.set_title("LEAP-VO: Run-to-Run Stability Across Datasets")
    ax.set_xticks(range(1, NUM_RUNS + 1))
    ax.legend()

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"combined_stability_all_datasets.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    all_per_run = {}

    for dname, dinfo in DATASETS.items():
        print(f"\n=== {dinfo['title']} ===")
        per_run = collect_per_run(dinfo["dir"])

        if not per_run:
            print(f"  No results found, skipping.")
            continue

        all_per_run[dname] = per_run

        # Per-scene lines (each scene is a line)
        plot_per_scene_lines(dname, dinfo["title"], per_run)

        # Aggregate line with shaded band
        plot_aggregate_line_with_band(dname, dinfo["title"], per_run)

        # 3-metric stability panel
        plot_three_metrics_lines(dname, dinfo["title"], per_run)

    # Combined all datasets in one plot
    print(f"\n=== Combined ===")
    plot_combined_all_datasets(all_per_run)

    print(f"\nAll line graphs saved to: {FIG_DIR}/")


if __name__ == "__main__":
    main()