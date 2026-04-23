#!/usr/bin/env python3
"""
parse_results.py
Parses LEAP-VO multi-run results and computes:
  1. Per-scene mean +/- std across runs
  2. Overall per-dataset mean +/- std across runs (what goes in the report table)

Usage: python parse_results.py
"""

import os
import re
import numpy as np
from collections import defaultdict

NUM_RUNS = 5
DATASETS = {
    "sintel": {"dir": "logs_run/sintel", "title": "MPI Sintel"},
    "replica": {"dir": "logs_run/replica", "title": "Replica"},
    "shibuya": {"dir": "logs_run/shibuya", "title": "TartanAir Shibuya"},
}


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


def main():
    # =========================================================================
    # Collect all data
    # =========================================================================
    all_data = {}  # dataset_name -> {scene -> {metric -> [values across runs]}}

    for dname, dinfo in DATASETS.items():
        base_dir = dinfo["dir"]
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

        all_data[dname] = {"results": dict(all_results), "runs_found": runs_found}

    # =========================================================================
    # TABLE 1: Overall dataset-level summary (THIS GOES IN THE REPORT)
    # For each run, compute average ATE/RPE_t/RPE_r across all scenes in that
    # run, giving one number per run. Then report mean +/- std of those numbers
    # across the 5 runs.
    # =========================================================================
    print("\n" + "=" * 90)
    print("  TABLE FOR REPORT: Overall Dataset Metrics (mean +/- std across runs)")
    print("=" * 90)
    print(f"  {'Dataset':<25} {'# Scenes':<10} {'ATE (m)':<22} {'RPE_t (m)':<22} {'RPE_r (deg)':<22}")
    print(f"  {'-'*97}")

    report_csv_lines = ["dataset,num_scenes,num_runs,ate_mean,ate_std,rpe_t_mean,rpe_t_std,rpe_r_mean,rpe_r_std"]

    for dname, dinfo in DATASETS.items():
        d = all_data[dname]
        results = d["results"]
        runs_found = d["runs_found"]

        if runs_found == 0:
            print(f"  {dinfo['title']:<25} {'N/A':<10} {'No results':<22}")
            continue

        # Find scenes that have results in ALL runs
        complete_scenes = [s for s in results if len(results[s]["ate"]) == runs_found]

        if not complete_scenes:
            print(f"  {dinfo['title']:<25} {0:<10} {'No complete scenes':<22}")
            continue

        # For each run, compute average across all complete scenes
        # Build matrix: (num_scenes, num_runs)
        ate_matrix = np.array([results[s]["ate"] for s in complete_scenes])
        rpe_t_matrix = np.array([results[s]["rpe_trans"] for s in complete_scenes])
        rpe_r_matrix = np.array([results[s]["rpe_rot"] for s in complete_scenes])

        # Average across scenes for each run -> shape (num_runs,)
        per_run_ate = ate_matrix.mean(axis=0)
        per_run_rpe_t = rpe_t_matrix.mean(axis=0)
        per_run_rpe_r = rpe_r_matrix.mean(axis=0)

        # Mean and std across runs
        ate_str = f"{per_run_ate.mean():.4f} +/- {per_run_ate.std():.4f}"
        rpe_t_str = f"{per_run_rpe_t.mean():.4f} +/- {per_run_rpe_t.std():.4f}"
        rpe_r_str = f"{per_run_rpe_r.mean():.4f} +/- {per_run_rpe_r.std():.4f}"

        n_scenes = len(complete_scenes)
        print(f"  {dinfo['title']:<25} {n_scenes:<10} {ate_str:<22} {rpe_t_str:<22} {rpe_r_str:<22}")

        report_csv_lines.append(
            f"{dinfo['title']},{n_scenes},{runs_found},"
            f"{per_run_ate.mean():.6f},{per_run_ate.std():.6f},"
            f"{per_run_rpe_t.mean():.6f},{per_run_rpe_t.std():.6f},"
            f"{per_run_rpe_r.mean():.6f},{per_run_rpe_r.std():.6f}"
        )

    print(f"  {'-'*97}")
    print(f"  (Computed as: for each run, average metric across scenes; then mean +/- std across {NUM_RUNS} runs)")

    # Save report-level CSV
    report_csv_path = "logs_run/report_summary.csv"
    os.makedirs("logs", exist_ok=True)
    with open(report_csv_path, "w") as f:
        f.write("\n".join(report_csv_lines) + "\n")
    print(f"\n  Report CSV saved to: {report_csv_path}")

    # =========================================================================
    # Also print per-run breakdown so you can verify
    # =========================================================================
    print("\n\n" + "=" * 90)
    print("  PER-RUN BREAKDOWN (average across scenes per run)")
    print("=" * 90)

    for dname, dinfo in DATASETS.items():
        d = all_data[dname]
        results = d["results"]
        runs_found = d["runs_found"]

        if runs_found == 0:
            continue

        complete_scenes = [s for s in results if len(results[s]["ate"]) == runs_found]
        if not complete_scenes:
            continue

        ate_matrix = np.array([results[s]["ate"] for s in complete_scenes])
        rpe_t_matrix = np.array([results[s]["rpe_trans"] for s in complete_scenes])
        rpe_r_matrix = np.array([results[s]["rpe_rot"] for s in complete_scenes])

        per_run_ate = ate_matrix.mean(axis=0)
        per_run_rpe_t = rpe_t_matrix.mean(axis=0)
        per_run_rpe_r = rpe_r_matrix.mean(axis=0)

        print(f"\n  {dinfo['title']} ({len(complete_scenes)} scenes):")
        print(f"    {'Run':<6} {'ATE':<12} {'RPE_t':<12} {'RPE_r':<12}")
        for i in range(runs_found):
            print(f"    {i+1:<6} {per_run_ate[i]:<12.5f} {per_run_rpe_t[i]:<12.5f} {per_run_rpe_r[i]:<12.5f}")
        print(f"    {'Mean':<6} {per_run_ate.mean():<12.5f} {per_run_rpe_t.mean():<12.5f} {per_run_rpe_r.mean():<12.5f}")
        print(f"    {'Std':<6} {per_run_ate.std():<12.5f} {per_run_rpe_t.std():<12.5f} {per_run_rpe_r.std():<12.5f}")

    # =========================================================================
    # TABLE 2: Per-scene details (for appendix or detailed reference)
    # =========================================================================
    for dname, dinfo in DATASETS.items():
        d = all_data[dname]
        results = d["results"]
        runs_found = d["runs_found"]

        if runs_found == 0:
            continue

        print(f"\n\n{'='*90}")
        print(f"  PER-SCENE DETAILS: {dinfo['title']}")
        print(f"{'='*90}")
        print(f"  {'Scene':<30} {'Runs':<6} {'ATE (mean +/- std)':<24} {'RPE_t (mean +/- std)':<24} {'RPE_r (mean +/- std)':<24}")
        print(f"  {'-'*108}")

        for scene in sorted(results.keys()):
            sd = results[scene]
            n = len(sd["ate"])
            ate_arr = np.array(sd["ate"])
            rpe_t_arr = np.array(sd["rpe_trans"])
            rpe_r_arr = np.array(sd["rpe_rot"])

            ate_str = f"{ate_arr.mean():.5f} +/- {ate_arr.std():.5f}"
            rpe_t_str = f"{rpe_t_arr.mean():.5f} +/- {rpe_t_arr.std():.5f}"
            rpe_r_str = f"{rpe_r_arr.mean():.5f} +/- {rpe_r_arr.std():.5f}"

            flag = "" if n == runs_found else " *"
            print(f"  {scene:<30} {n:<6} {ate_str:<24} {rpe_t_str:<24} {rpe_r_str:<24}{flag}")

        incomplete = [s for s in results if len(results[s]["ate"]) < runs_found]
        if incomplete:
            print(f"\n  * Incomplete: {', '.join(incomplete)}")

        # Per-scene CSV
        csv_path = os.path.join(dinfo["dir"], "per_scene_stats.csv")
        with open(csv_path, "w") as f:
            f.write("scene,num_runs,ate_mean,ate_std,rpe_trans_mean,rpe_trans_std,rpe_rot_mean,rpe_rot_std\n")
            for scene in sorted(results.keys()):
                sd = results[scene]
                n = len(sd["ate"])
                a = np.array(sd["ate"])
                rt = np.array(sd["rpe_trans"])
                rr = np.array(sd["rpe_rot"])
                f.write(f"{scene},{n},{a.mean():.6f},{a.std():.6f},{rt.mean():.6f},{rt.std():.6f},{rr.mean():.6f},{rr.std():.6f}\n")
        print(f"  CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()