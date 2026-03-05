#!/usr/bin/env python3
"""Generate sweep config YAML files from the base template.

Produces one YAML per (dimension, value) pair, overriding the relevant
parameter and setting output_dir / results_file / wandb_run_name accordingly.

Usage:
    python scripts/generate_sweep_configs.py
    python scripts/generate_sweep_configs.py --output-dir configs/sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Sweep grid definition
# ---------------------------------------------------------------------------

SWEEP_GRID: dict[str, dict[str, list]] = {
    "size": {
        "reservoir_size": [500, 2000, 10000, 50000],
    },
    "spectral_radius": {
        "spectral_radius": [0.5, 0.9, 0.99, 1.1],
    },
    "leak_rate": {
        "leak_rate": [0.1, 0.3, 0.7, 1.0],
    },
    "topology": {
        "topology": ["erdos_renyi", "small_world"],
    },
}


def _config_name(dimension: str, value: object) -> str:
    """Generate a filesystem-safe config name from dimension + value."""
    val_str = str(value).replace(".", "p")
    return f"{dimension}_{val_str}"


def generate(base_path: Path, output_dir: Path) -> list[Path]:
    """Generate sweep configs and return paths of created files."""
    with base_path.open() as f:
        base = yaml.safe_load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    run_id = 0

    for dimension, param_values in SWEEP_GRID.items():
        for param, values in param_values.items():
            for value in values:
                name = _config_name(dimension, value)
                config = dict(base)
                config[param] = value
                config["output_dir"] = f"checkpoints/track_a/sweep/{name}"
                config["results_file"] = f"results/track_a/sweep/{name}/train_metrics.json"
                config["wandb_run_name"] = f"sweep/{name}"
                config["_sweep_dimension"] = dimension
                config["_sweep_param"] = param
                config["_sweep_value"] = value
                config["_run_id"] = run_id

                out_path = output_dir / f"{name}.yaml"
                with out_path.open("w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                created.append(out_path)
                run_id += 1

    # Also generate the best_combo config (uses base defaults)
    best = dict(base)
    best["output_dir"] = "checkpoints/track_a/sweep/best_combo"
    best["results_file"] = "results/track_a/sweep/best_combo/train_metrics.json"
    best["wandb_run_name"] = "sweep/best_combo"
    best["_sweep_dimension"] = "best_combo"
    best["_sweep_param"] = ""
    best["_sweep_value"] = None
    best["_run_id"] = run_id

    out_path = output_dir / "best_combo.yaml"
    with out_path.open("w") as f:
        yaml.dump(best, f, default_flow_style=False, sort_keys=False)
    created.append(out_path)

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep config YAML files")
    parser.add_argument(
        "--output-dir",
        default="configs/sweep",
        help="Directory to write generated configs (default: configs/sweep)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.resolve()
    base_path = repo_root / "configs" / "sweep" / "base.yaml"
    output_dir = repo_root / args.output_dir

    created = generate(base_path, output_dir)
    print(f"Generated {len(created)} sweep configs in {output_dir}/")
    for p in created:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
