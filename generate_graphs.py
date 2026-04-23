#!/usr/bin/env python3
"""
generate_graphs.py
Generates publication-ready graphs from LEAP-VO multi-run results.
Produces bar charts with error bars (mean ± std) for each dataset.

Usage: python generate_graphs.py

Output: logs/figures/ directory with PDF and PNG graphs
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

NUM_RUNS = 4  
DATASETS = {
    "sintel": {"dir": "logs_SAM/sintel", "title": "MPI-Sintel"},
    "replica": {"dir": "logs_SAM/replica", "title": "Replica"},
    "shibuya": {"dir": "logs_SAM/shibuya", "title": "TartanAir-Shibuya"},
}
FIG_DIR = "logs_SAM/figures"

# Use a clean style suitable for printing
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
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


def collect_results(base_dir):
    all_results = defaultdict(lambda: {"ate": [], "rpe_trans": [], "rpe_rot": []})
    runs_found = 0
    for run_i in range(1, NUM_RUNS + 1):
        error_file = os.path.join(base_dir, f"run_{run_i}", "error_sum.txt")
        run_results = parse_error_sum(error_file)
        if run_results:
            runs_found += 1
            for scene, (ate, rpe_t, rpe_r) in run_results.items():
                all_results[scene]["ate"].append(ate)
                all_results[scene]["rpe_trans"].append(rpe_t)
                all_results[scene]["rpe_rot"].append(rpe_r)
    return dict(all_results), runs_found


def short_scene_name(scene):
    """Remove dataset prefix for cleaner axis labels."""
    for prefix in ["sintel-", "replica-", "shibuya-"]:
        if scene.startswith(prefix):
            scene = scene[len(prefix):]
    # Remove -Sequence_1 suffix
    scene = scene.replace("-Sequence_1", "")
    return scene


def plot_per_dataset_bars(dataset_name, dataset_title, all_results, runs_found):
    """Generate a 3-subplot figure: ATE, RPE_t, RPE_r for one dataset."""
    scenes = sorted(all_results.keys())
    if not scenes:
        return

    short_names = [short_scene_name(s) for s in scenes]

    ate_means = [np.mean(all_results[s]["ate"]) for s in scenes]
    ate_stds = [np.std(all_results[s]["ate"]) for s in scenes]
    rpe_t_means = [np.mean(all_results[s]["rpe_trans"]) for s in scenes]
    rpe_t_stds = [np.std(all_results[s]["rpe_trans"]) for s in scenes]
    rpe_r_means = [np.mean(all_results[s]["rpe_rot"]) for s in scenes]
    rpe_r_stds = [np.std(all_results[s]["rpe_rot"]) for s in scenes]

    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(scenes) * 0.7), 10))
    fig.suptitle(f"LEAP-VO Results on {dataset_title} ({runs_found} runs)", fontsize=14, fontweight='bold')

    x = np.arange(len(scenes))
    bar_color = '#4878CF'
    err_color = '#222222'

    # ATE
    axes[0].bar(x, ate_means, yerr=ate_stds, capsize=4, color=bar_color,
                edgecolor='white', linewidth=0.5, error_kw={'color': err_color, 'linewidth': 1.5})
    axes[0].set_ylabel("ATE (RMSE)")
    axes[0].set_title("Absolute Trajectory Error")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=45, ha='right')

    # RPE Translation
    axes[1].bar(x, rpe_t_means, yerr=rpe_t_stds, capsize=4, color='#6ACC65',
                edgecolor='white', linewidth=0.5, error_kw={'color': err_color, 'linewidth': 1.5})
    axes[1].set_ylabel("RPE Trans (RMSE)")
    axes[1].set_title("Relative Pose Error - Translation")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=45, ha='right')

    # RPE Rotation
    axes[2].bar(x, rpe_r_means, yerr=rpe_r_stds, capsize=4, color='#D65F5F',
                edgecolor='white', linewidth=0.5, error_kw={'color': err_color, 'linewidth': 1.5})
    axes[2].set_ylabel("RPE Rot (deg, RMSE)")
    axes[2].set_title("Relative Pose Error - Rotation")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names, rotation=45, ha='right')

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"{dataset_name}_per_scene.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_ate_only(dataset_name, dataset_title, all_results, runs_found):
    """Generate a single clean ATE bar chart — best for fitting in the report."""
    scenes = sorted(all_results.keys())
    if not scenes:
        return

    short_names = [short_scene_name(s) for s in scenes]

    ate_means = [np.mean(all_results[s]["ate"]) for s in scenes]
    ate_stds = [np.std(all_results[s]["ate"]) for s in scenes]

    fig, ax = plt.subplots(figsize=(max(6, len(scenes) * 0.6), 3.5))

    x = np.arange(len(scenes))
    bars = ax.bar(x, ate_means, yerr=ate_stds, capsize=3, color='#4878CF',
                  edgecolor='white', linewidth=0.5,
                  error_kw={'color': '#222222', 'linewidth': 1.2})

    ax.set_ylabel("ATE (RMSE)")
    ax.set_title(f"LEAP-VO: Absolute Trajectory Error on {dataset_title} (n={runs_found})")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right')

    # Add value labels on top of bars
    for i, (m, s) in enumerate(zip(ate_means, ate_stds)):
        if m + s < ax.get_ylim()[1] * 0.85:
            ax.text(i, m + s + ax.get_ylim()[1] * 0.02, f'{m:.4f}',
                    ha='center', va='bottom', fontsize=7, color='#333333')

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"{dataset_name}_ate.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_summary_across_datasets(all_dataset_results):
    """Bar chart comparing average ATE across all 3 datasets."""
    dataset_names = []
    avg_ates = []
    avg_stds = []

    for dname, (results, runs_found) in all_dataset_results.items():
        if not results or runs_found == 0:
            continue

        # Get scenes with all runs
        complete_ates = []
        for scene, d in results.items():
            if len(d["ate"]) == runs_found:
                complete_ates.append(np.array(d["ate"]))

        if not complete_ates:
            continue

        ate_matrix = np.stack(complete_ates)  # (scenes, runs)
        per_run_avg = ate_matrix.mean(axis=0)  # (runs,)

        dataset_names.append(DATASETS[dname]["title"])
        avg_ates.append(per_run_avg.mean())
        avg_stds.append(per_run_avg.std())

    if not dataset_names:
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(len(dataset_names))
    colors = ['#4878CF', '#6ACC65', '#D65F5F']

    bars = ax.bar(x, avg_ates, yerr=avg_stds, capsize=5,
                  color=colors[:len(dataset_names)],
                  edgecolor='white', linewidth=0.5,
                  error_kw={'color': '#222222', 'linewidth': 1.5})

    # Value labels
    for i, (m, s) in enumerate(zip(avg_ates, avg_stds)):
        ax.text(i, m + s + ax.get_ylim()[1] * 0.03, f'{m:.5f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel("Average ATE (RMSE)")
    ax.set_title("LEAP-VO: Average ATE Across Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names)

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"summary_ate_all_datasets.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_boxplot_per_dataset(dataset_name, dataset_title, all_results, runs_found):
    """Box plot showing distribution across runs per scene."""
    scenes = sorted(all_results.keys())
    if not scenes:
        return

    short_names = [short_scene_name(s) for s in scenes]
    ate_data = [all_results[s]["ate"] for s in scenes]

    fig, ax = plt.subplots(figsize=(max(6, len(scenes) * 0.6), 3.5))

    bp = ax.boxplot(ate_data, patch_artist=True, labels=short_names,
                    boxprops=dict(facecolor='#4878CF', alpha=0.7),
                    medianprops=dict(color='#222222', linewidth=1.5),
                    whiskerprops=dict(color='#666666'),
                    capprops=dict(color='#666666'),
                    flierprops=dict(marker='o', markersize=4, markerfacecolor='#D65F5F'))

    ax.set_ylabel("ATE (RMSE)")
    ax.set_title(f"LEAP-VO: ATE Distribution on {dataset_title} (n={runs_found})")
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"{dataset_name}_ate_boxplot.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    all_dataset_results = {}

    for dname, dinfo in DATASETS.items():
        print(f"\n=== {dinfo['title']} ===")
        results, runs_found = collect_results(dinfo["dir"])

        if runs_found == 0:
            print(f"  No results found, skipping.")
            continue

        all_dataset_results[dname] = (results, runs_found)

        # Per-scene bar charts (3 metrics)
        plot_per_dataset_bars(dname, dinfo["title"], results, runs_found)

        # ATE-only bar chart (compact, good for report)
        plot_ate_only(dname, dinfo["title"], results, runs_found)

        # Box plot
        plot_boxplot_per_dataset(dname, dinfo["title"], results, runs_found)

    # Summary across datasets
    print(f"\n=== Summary ===")
    plot_summary_across_datasets(all_dataset_results)

    print(f"\nAll figures saved to: {FIG_DIR}/")
    print(f"Files generated:")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()