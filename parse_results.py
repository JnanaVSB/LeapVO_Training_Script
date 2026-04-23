#!/usr/bin/env python3
"""
parse_results.py
Parses LEAP-VO multi-run results and computes mean +/- std for each scene and dataset.
Outputs tables suitable for the report.
 
Usage: python parse_results.py
"""
 
import os
import re
import numpy as np
from collections import defaultdict
 
NUM_RUNS = 5
DATASETS = {
    "sintel": "logs_run/sintel",
    "replica": "logs_run/replica",
    "shibuya": "logs_run/shibuya",
}
 
def parse_error_sum(filepath):
    """Parse an error_sum.txt file, return dict of scene -> (ate, rpe_trans, rpe_rot)"""
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
    for dataset_name, base_dir in DATASETS.items():
        print(f"\n{'='*80}")
        print(f"  DATASET: {dataset_name.upper()}")
        print(f"{'='*80}")
 
        # Collect results across runs
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
 
        if runs_found == 0:
            print(f"  No results found in {base_dir}/run_*/error_sum.txt")
            continue
 
        print(f"  Found {runs_found} runs\n")
 
        # Print per-scene table
        header = f"  {'Scene':<30} {'Runs':<6} {'ATE (mean +/- std)':<22} {'RPE_t (mean +/- std)':<22} {'RPE_r (mean +/- std)':<22}"
        print(header)
        print(f"  {'-'*102}")
 
        # For computing dataset average: only use scenes present in ALL runs
        complete_scenes_ate = []
        complete_scenes_rpe_t = []
        complete_scenes_rpe_r = []
 
        for scene in sorted(all_results.keys()):
            d = all_results[scene]
            n = len(d["ate"])
            ate_arr = np.array(d["ate"])
            rpe_t_arr = np.array(d["rpe_trans"])
            rpe_r_arr = np.array(d["rpe_rot"])
 
            ate_str = f"{ate_arr.mean():.5f} +/- {ate_arr.std():.5f}"
            rpe_t_str = f"{rpe_t_arr.mean():.5f} +/- {rpe_t_arr.std():.5f}"
            rpe_r_str = f"{rpe_r_arr.mean():.5f} +/- {rpe_r_arr.std():.5f}"
 
            flag = "" if n == runs_found else " *"
            print(f"  {scene:<30} {n:<6} {ate_str:<22} {rpe_t_str:<22} {rpe_r_str:<22}{flag}")
 
            if n == runs_found:
                complete_scenes_ate.append(ate_arr)
                complete_scenes_rpe_t.append(rpe_t_arr)
                complete_scenes_rpe_r.append(rpe_r_arr)
 
        # Dataset average across complete scenes
        print(f"  {'-'*102}")
        if complete_scenes_ate:
            ate_matrix = np.stack(complete_scenes_ate)
            rpe_t_matrix = np.stack(complete_scenes_rpe_t)
            rpe_r_matrix = np.stack(complete_scenes_rpe_r)
 
            per_run_ate = ate_matrix.mean(axis=0)
            per_run_rpe_t = rpe_t_matrix.mean(axis=0)
            per_run_rpe_r = rpe_r_matrix.mean(axis=0)
 
            n_complete = len(complete_scenes_ate)
            avg_ate_str = f"{per_run_ate.mean():.5f} +/- {per_run_ate.std():.5f}"
            avg_rpe_t_str = f"{per_run_rpe_t.mean():.5f} +/- {per_run_rpe_t.std():.5f}"
            avg_rpe_r_str = f"{per_run_rpe_r.mean():.5f} +/- {per_run_rpe_r.std():.5f}"
            print(f"  {'AVERAGE (' + str(n_complete) + ' scenes)':<30} {runs_found:<6} {avg_ate_str:<22} {avg_rpe_t_str:<22} {avg_rpe_r_str:<22}")
        else:
            print(f"  No scenes had results in all {runs_found} runs")
 
        incomplete = [s for s in all_results if len(all_results[s]["ate"]) < runs_found]
        if incomplete:
            print(f"\n  * Scenes with incomplete runs: {', '.join(incomplete)}")
 
        # Save CSV
        csv_path = os.path.join(base_dir, "summary_stats.csv")
        with open(csv_path, "w") as f:
            f.write("scene,num_runs,ate_mean,ate_std,rpe_trans_mean,rpe_trans_std,rpe_rot_mean,rpe_rot_std\n")
            for scene in sorted(all_results.keys()):
                d = all_results[scene]
                n = len(d["ate"])
                a = np.array(d["ate"])
                rt = np.array(d["rpe_trans"])
                rr = np.array(d["rpe_rot"])
                f.write(f"{scene},{n},{a.mean():.6f},{a.std():.6f},{rt.mean():.6f},{rt.std():.6f},{rr.mean():.6f},{rr.std():.6f}\n")
            if complete_scenes_ate:
                f.write(f"AVERAGE,{runs_found},{per_run_ate.mean():.6f},{per_run_ate.std():.6f},{per_run_rpe_t.mean():.6f},{per_run_rpe_t.std():.6f},{per_run_rpe_r.mean():.6f},{per_run_rpe_r.std():.6f}\n")
        print(f"\n  CSV saved to: {csv_path}")
 
 
if __name__ == "__main__":
    main()
 