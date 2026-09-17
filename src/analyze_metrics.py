"""Fit and plot a chosen metric's decay across a run's generations.

Usage: python src/analyze_metrics.py --experiment_path <path> --metric <field_name> [--plot]

Fits an exponential decay (y = a * exp(-b * x)) via linear regression on
log(y) vs. generation index (numpy.polyfit) -- no scipy dependency needed
(constitution Principle V; see research.md item 6).
"""
import argparse
import glob
import json
import os

import numpy as np


def load_metric_series(experiment_path, metric):
    """Return (generation_indices, values) for every {experiment_path}/{i}/data_metrics.json
    that both exists and contains `metric`, sorted by generation index."""
    points = []
    for path in glob.glob(os.path.join(str(experiment_path), "*", "data_metrics.json")):
        iteration_dir = os.path.basename(os.path.dirname(path))
        if not iteration_dir.isdigit():
            continue
        with open(path, "r") as f:
            metrics = json.load(f)
        if metric in metrics and metrics[metric] is not None:
            points.append((int(iteration_dir), float(metrics[metric])))

    points.sort(key=lambda p: p[0])
    if not points:
        return np.array([]), np.array([])
    generations, values = zip(*points)
    return np.array(generations, dtype=np.float64), np.array(values, dtype=np.float64)


def fit_exponential_decay(generations, values):
    """Fit y = a * exp(-b * x) via linear regression on log(y) vs x.
    Returns (a, b, r_squared). Raises ValueError if any value is <= 0
    (log-space fit requires strictly positive values)."""
    if np.any(values <= 0):
        raise ValueError(
            "Exponential decay fit requires strictly positive metric values; "
            "found a zero or negative value in the series."
        )

    log_values = np.log(values)
    slope, intercept = np.polyfit(generations, log_values, 1)
    b = -slope
    a = np.exp(intercept)

    predicted_log = intercept + slope * generations
    ss_res = np.sum((log_values - predicted_log) ** 2)
    ss_tot = np.sum((log_values - np.mean(log_values)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

    return a, b, r_squared


def main():
    parser = argparse.ArgumentParser(description="Fit and plot a metric's decay across generations.")
    parser.add_argument("--experiment_path", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    generations, values = load_metric_series(args.experiment_path, args.metric)

    if len(generations) < 3:
        raise SystemExit(
            f"Need at least 3 completed generations with metric '{args.metric}' "
            f"to fit a decay curve; found {len(generations)}. Let more generations "
            "complete before running this analysis."
        )

    a, b, r_squared = fit_exponential_decay(generations, values)
    print(f"Fit: {args.metric} ~= {a:.6f} * exp(-{b:.6f} * generation)")
    print(f"R^2 = {r_squared:.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fitted = a * np.exp(-b * generations)
        plt.figure()
        plt.scatter(generations, values, label="observed")
        plt.plot(generations, fitted, label=f"fit (R^2={r_squared:.3f})", color="orange")
        plt.xlabel("Generation")
        plt.ylabel(args.metric)
        plt.title(f"{args.metric} decay across generations")
        plt.legend()

        out_path = os.path.join(str(args.experiment_path), f"{args.metric}_decay.png")
        plt.savefig(out_path)
        print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
