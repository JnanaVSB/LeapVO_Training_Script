#!/usr/bin/env python3
"""
generate_graphs.py
Generates publication-ready graphs from LEAP-VO multi-run results.
Includes the key report figure: grouped bar chart of all 3 metrics across all 3 datasets.

Usage: python generate_graphs.py
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

NUM_RUNS = 5
DATASETS = {
    "sintel": {"dir": "logs_run/sintel", "title": "MPI Sintel"},
    "replica": {"dir": "logs_run/replica", "title": "Replica"},
    "shibuya": {"dir": "logs_run/shibuya", "title": "TartanAir Shibuya"},
}
FIG_DIR = "logs_one/figures"

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
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


def get_per_run_averages(results, runs_found):
    """For each run, compute average metric across all complete scenes.
    Returns (per_run_ate, per_run_rpe_t, per_run_rpe_r) each shape (runs,)"""
    complete_scenes = [s for s in results if len(results[s]["ate"]) == runs_found]
    if not complete_scenes:
        return None, None, None

    ate_matrix = np.array([results[s]["ate"] for s in complete_scenes])
    rpe_t_matrix = np.array([results[s]["rpe_trans"] for s in complete_scenes])
    rpe_r_matrix = np.array([results[s]["rpe_rot"] for s in complete_scenes])

    return ate_matrix.mean(axis=0), rpe_t_matrix.mean(axis=0), rpe_r_matrix.mean(axis=0)


def short_scene_name(scene):
    for prefix in ["sintel-", "replica-", "shibuya-"]:
        if scene.startswith(prefix):
            scene = scene[len(prefix):]
    scene = scene.replace("-Sequence_1", "")
    return scene


# =========================================================================
# GRAPH 1: Overall dataset comparison - grouped bars for all 3 metrics
# THIS IS THE KEY REPORT FIGURE
# =========================================================================
def plot_overall_dataset_comparison(all_dataset_stats):
    """Grouped bar chart: 3 datasets x 3 metrics, with error bars."""
    datasets = []
    ate_means, ate_stds = [], []
    rpe_t_means, rpe_t_stds = [], []
    rpe_r_means, rpe_r_stds = [], []

    for dname in ["sintel", "replica", "shibuya"]:
        if dname not in all_dataset_stats:
            continue
        stats = all_dataset_stats[dname]
        datasets.append(DATASETS[dname]["title"])
        ate_means.append(stats["ate_mean"])
        ate_stds.append(stats["ate_std"])
        rpe_t_means.append(stats["rpe_t_mean"])
        rpe_t_stds.append(stats["rpe_t_std"])
        rpe_r_means.append(stats["rpe_r_mean"])
        rpe_r_stds.append(stats["rpe_r_std"])

    if not datasets:
        return

    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars1 = ax.bar(x - width, ate_means, width, yerr=ate_stds, capsize=4,
                   label='ATE (m)', color='#4878CF', edgecolor='white',
                   error_kw={'linewidth': 1.3})
    bars2 = ax.bar(x, rpe_t_means, width, yerr=rpe_t_stds, capsize=4,
                   label='RPE$_t$ (m)', color='#6ACC65', edgecolor='white',
                   error_kw={'linewidth': 1.3})
    bars3 = ax.bar(x + width, rpe_r_means, width, yerr=rpe_r_stds, capsize=4,
                   label='RPE$_r$ (deg)', color='#D65F5F', edgecolor='white',
                   error_kw={'linewidth': 1.3})

    # Value labels on top
    for bars, means, stds in [(bars1, ate_means, ate_stds),
                               (bars2, rpe_t_means, rpe_t_stds),
                               (bars3, rpe_r_means, rpe_r_stds)]:
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel("Error")
    ax.set_title(f"LEAP-VO: Overall Performance Across Datasets (n={NUM_RUNS} runs)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"overall_dataset_comparison.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


# =========================================================================
# GRAPH 1b: Same but separate subplots (better if scales differ a lot)
# =========================================================================
def plot_overall_dataset_comparison_subplots(all_dataset_stats):
    """3 separate subplots: one per metric, bars for each dataset."""
    datasets = []
    ate_means, ate_stds = [], []
    rpe_t_means, rpe_t_stds = [], []
    rpe_r_means, rpe_r_stds = [], []

    for dname in ["sintel", "replica", "shibuya"]:
        if dname not in all_dataset_stats:
            continue
        stats = all_dataset_stats[dname]
        datasets.append(DATASETS[dname]["title"])
        ate_means.append(stats["ate_mean"])
        ate_stds.append(stats["ate_std"])
        rpe_t_means.append(stats["rpe_t_mean"])
        rpe_t_stds.append(stats["rpe_t_std"])
        rpe_r_means.append(stats["rpe_r_mean"])
        rpe_r_stds.append(stats["rpe_r_std"])

    if not datasets:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle(f"LEAP-VO: Overall Performance (mean ± std, n={NUM_RUNS} runs)",
                 fontsize=13, fontweight='bold')

    metrics = [
        ("ATE (m)", ate_means, ate_stds, '#4878CF'),
        ("RPE$_t$ (m)", rpe_t_means, rpe_t_stds, '#6ACC65'),
        ("RPE$_r$ (deg)", rpe_r_means, rpe_r_stds, '#D65F5F'),
    ]

    x = np.arange(len(datasets))
    for ax, (ylabel, means, stds, color) in zip(axes, metrics):
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=color,
                      edgecolor='white', linewidth=0.5,
                      error_kw={'color': '#222222', 'linewidth': 1.5})

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + ax.get_ylim()[1] * 0.01, f'{m:.4f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=9)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"overall_dataset_comparison_3panel.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


# =========================================================================
# GRAPH 2: Per-scene bar charts per dataset
# =========================================================================
def plot_per_scene_bars(dataset_name, dataset_title, all_results, runs_found):
    scenes = sorted(all_results.keys())
    if not scenes:
        return

    short_names = [short_scene_name(s) for s in scenes]

    ate_means = [np.mean(all_results[s]["ate"]) for s in scenes]
    ate_stds = [np.std(all_results[s]["ate"]) for s in scenes]

    fig, ax = plt.subplots(figsize=(max(6, len(scenes) * 0.6), 3.5))

    x = np.arange(len(scenes))
    ax.bar(x, ate_means, yerr=ate_stds, capsize=3, color='#4878CF',
           edgecolor='white', linewidth=0.5,
           error_kw={'color': '#222222', 'linewidth': 1.2})

    ax.set_ylabel("ATE (RMSE)")
    ax.set_title(f"LEAP-VO: Per-Scene ATE on {dataset_title} (n={runs_found})")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right')

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"{dataset_name}_ate_per_scene.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


# =========================================================================
# GRAPH 3: Per-scene bar charts - all 3 metrics
# =========================================================================
def plot_per_scene_all_metrics(dataset_name, dataset_title, all_results, runs_found):
    scenes = sorted(all_results.keys())
    if not scenes:
        return

    short_names = [short_scene_name(s) for s in scenes]

    fig, axes = plt.subplots(3, 1, figsize=(max(8, len(scenes) * 0.7), 10))
    fig.suptitle(f"LEAP-VO: Per-Scene Results on {dataset_title} (n={runs_found})",
                 fontsize=14, fontweight='bold')

    x = np.arange(len(scenes))
    metric_info = [
        ("ATE (RMSE)", "ate", '#4878CF'),
        ("RPE Trans (RMSE)", "rpe_trans", '#6ACC65'),
        ("RPE Rot (deg, RMSE)", "rpe_rot", '#D65F5F'),
    ]

    for ax, (ylabel, key, color) in zip(axes, metric_info):
        means = [np.mean(all_results[s][key]) for s in scenes]
        stds = [np.std(all_results[s][key]) for s in scenes]

        ax.bar(x, means, yerr=stds, capsize=4, color=color,
               edgecolor='white', linewidth=0.5,
               error_kw={'color': '#222222', 'linewidth': 1.5})
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha='right')

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        path = os.path.join(FIG_DIR, f"{dataset_name}_per_scene_3metrics.{ext}")
        fig.savefig(path)
        print(f"  Saved: {path}")
    plt.close(fig)


# =========================================================================
# MAIN
# =========================================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    all_dataset_stats = {}

    for dname, dinfo in DATASETS.items():
        print(f"\n=== {dinfo['title']} ===")
        results, runs_found = collect_results(dinfo["dir"])

        if runs_found == 0:
            print(f"  No results found, skipping.")
            continue

        # Compute overall dataset stats
        per_run_ate, per_run_rpe_t, per_run_rpe_r = get_per_run_averages(results, runs_found)

        if per_run_ate is not None:
            all_dataset_stats[dname] = {
                "ate_mean": per_run_ate.mean(),
                "ate_std": per_run_ate.std(),
                "rpe_t_mean": per_run_rpe_t.mean(),
                "rpe_t_std": per_run_rpe_t.std(),
                "rpe_r_mean": per_run_rpe_r.mean(),
                "rpe_r_std": per_run_rpe_r.std(),
            }

        # Per-scene bar charts
        plot_per_scene_bars(dname, dinfo["title"], results, runs_found)
        plot_per_scene_all_metrics(dname, dinfo["title"], results, runs_found)

    # Overall comparison figures
    print(f"\n=== Overall Comparison ===")
    plot_overall_dataset_comparison(all_dataset_stats)
    plot_overall_dataset_comparison_subplots(all_dataset_stats)

    print(f"\nAll figures saved to: {FIG_DIR}/")
    for f in sorted(os.listdir(FIG_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()