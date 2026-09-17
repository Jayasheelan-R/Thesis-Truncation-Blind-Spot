"""Defaults-audit recording: capture the fully-resolved config and the
actual generate-step CLI args that produced a given generation, so results
can be explained and reproduced later (constitution Principle III).
"""
import json
import os
from datetime import datetime, timezone

from omegaconf import OmegaConf


def _command_to_dict(command):
    """Convert a ["python", "src/generate.py", "--flag", "value", ...] list
    into {"flag": "value", ...}, skipping the leading interpreter/script
    entries. Assumes every flag takes exactly one value (true for every
    flag main.py currently builds for generate.py)."""
    args = {}
    i = 2  # skip "python", "src/generate.py"
    while i < len(command):
        token = command[i]
        if token.startswith("--"):
            key = token[2:]
            value = command[i + 1] if i + 1 < len(command) else None
            args[key] = value
            i += 2
        else:
            i += 1
    return args


def write_generation_config_audit(experiment_path, iteration, cfg, generate_command):
    """Write generation_config_audit.json to {experiment_path}/{iteration}/.

    - resolved_config: the full, fully-resolved Hydra config actually used.
    - resolved_generate_args: the actual CLI args passed to generate.py.
    - seed: duplicated at top level for quick inspection.
    - timestamp: ISO 8601, when this record was written.
    """
    resolved_config = OmegaConf.to_container(cfg, resolve=True)

    record = {
        "iteration": iteration,
        "resolved_config": resolved_config,
        "resolved_generate_args": _command_to_dict(generate_command),
        "seed": resolved_config.get("seed"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    out_dir = os.path.join(str(experiment_path), str(iteration))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "generation_config_audit.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=4)
    return out_path
