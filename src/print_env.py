"""Environment snapshot: reports the actual installed library versions and
the actual detected compute device, so runs can be explained after the fact
even when a library version can't be hard-pinned (e.g. Colab's preinstalled
torch build) -- constitution Principle III.

Usage: python src/print_env.py --save <path.json>
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from importlib import metadata

import torch


# Package names as they appear in requirements.txt / requirements-colab.txt.
# Kept as an explicit list (rather than parsing the requirements file) so
# this script has no dependency on being run from any particular directory.
TRACKED_PACKAGES = [
    "datasets", "evaluate", "accelerate", "torch", "transformers",
    "sentencepiece", "protobuf", "scikit-learn", "wandb", "tqdm",
    "hydra-core", "matplotlib", "nltk", "textstat", "mauve-text",
    "seaborn", "numpy",
]


def detect_device():
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    if cuda_available:
        device_detected = "cuda"
    elif mps_available:
        device_detected = "mps"
    else:
        device_detected = "cpu"
    return device_detected, cuda_available, mps_available


def build_snapshot():
    installed_packages = {}
    for name in TRACKED_PACKAGES:
        try:
            installed_packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed_packages[name] = None

    device_detected, cuda_available, mps_available = detect_device()

    return {
        "python_version": sys.version,
        "installed_packages": installed_packages,
        "device_detected": device_detected,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description="Print/save an environment snapshot.")
    parser.add_argument("--save", type=str, default=None, help="Path to write the snapshot JSON to.")
    args = parser.parse_args()

    snapshot = build_snapshot()

    print(f"Python: {snapshot['python_version'].splitlines()[0]}")
    print(f"Device detected: {snapshot['device_detected']} "
          f"(cuda_available={snapshot['cuda_available']}, mps_available={snapshot['mps_available']})")
    print("Installed packages:")
    for name, version in snapshot["installed_packages"].items():
        print(f"  {name}: {version if version is not None else 'NOT INSTALLED'}")

    if args.save:
        with open(args.save, "w") as f:
            json.dump(snapshot, f, indent=4)
        print(f"Snapshot saved to {args.save}")


if __name__ == "__main__":
    main()
