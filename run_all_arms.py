"""Runs every decoding arm listed in config/arms.yaml, back-to-back, in one
unattended process. Each arm is a call to main.py (unchanged), which already
knows how to resume a partially-completed arm (see main.py's
is_generation_complete()). This script adds arm-level skipping on top of
that: if an arm's last iteration is already done, it's skipped without
spawning a subprocess at all, so re-running this script after a Colab
disconnect picks up exactly where the whole sweep left off -- no manual
bookkeeping.

Usage (from the repo root, after `cd` into it on Colab):
    python run_all_arms.py
    python run_all_arms.py --drive-root /content/drive/MyDrive/FinalProject
    python run_all_arms.py --arms minp_005 topk_10   # run only these arms
"""
import argparse
import os
import subprocess
import sys

import yaml

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/FinalProject"


def is_arm_complete(run_dir, num_iterations):
    final_model_dir = os.path.join(run_dir, str(num_iterations), "model", "final_model")
    if not os.path.isdir(final_model_dir):
        return False
    metrics_path = os.path.join(run_dir, str(num_iterations), "data_metrics.json")
    return os.path.isfile(metrics_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-root", default=os.environ.get("DRIVE_ROOT", DEFAULT_DRIVE_ROOT))
    parser.add_argument("--arms-manifest", default=os.path.join(os.path.dirname(__file__), "config", "arms.yaml"))
    parser.add_argument("--arms", nargs="*", default=None, help="Only run these arm names (default: all)")
    args = parser.parse_args()

    with open(args.arms_manifest) as f:
        manifest = yaml.safe_load(f)

    num_iterations = manifest["num_iterations"]
    human_data_alpha = manifest["contamination"]["human_data_alpha"]
    ai_beta = manifest["contamination"]["ai_beta"]
    arms = manifest["arms"]
    if args.arms:
        selected = set(args.arms)
        arms = [a for a in arms if a["name"] in selected]
        missing = selected - {a["name"] for a in arms}
        if missing:
            sys.exit(f"Unknown arm name(s): {sorted(missing)}")

    dataset_path = os.path.join(args.drive_root, "data", "wikitext103")
    experiments_root = os.path.join(args.drive_root, "experiments")

    if not os.path.isfile(os.path.join(dataset_path, "train.json")):
        sys.exit(
            f"Expected WikiText-103 data at {dataset_path}/train.json but it "
            "wasn't found. Load it first (see README.md's Colab setup)."
        )

    print(f"=== {len(arms)} arm(s) queued, {num_iterations} iterations each, "
          f"human_data_alpha={human_data_alpha} ai_beta={ai_beta} "
          f"({manifest['contamination']['target_synthetic_pct']}% synthetic) ===")

    for i, arm in enumerate(arms, start=1):
        run_dir = os.path.join(experiments_root, arm["name"])
        print(f"\n--- Arm {i}/{len(arms)}: {arm['name']} -> {run_dir} ---")

        if is_arm_complete(run_dir, num_iterations):
            print(f"Skipping {arm['name']}: already completed (all {num_iterations} iterations present).")
            continue

        command = [
            "python", "main.py",
            "--config-name=ours",
            f"decoding={arm['decoding']}",
            *arm.get("overrides", []),
            f"human_data_alpha={human_data_alpha}",
            f"ai_beta={ai_beta}",
            f"num_iterations={num_iterations}",
            f"dataset.path={dataset_path}",
            f"hydra.run.dir={run_dir}",
        ]
        print("Running:", " ".join(command))
        result = subprocess.run(command)
        if result.returncode != 0:
            sys.exit(
                f"Arm {arm['name']} failed (exit code {result.returncode}). "
                "Re-run this script to resume from here once the issue is fixed."
            )

    print("\nAll requested arms complete (or already were, and were skipped).")


if __name__ == "__main__":
    main()
