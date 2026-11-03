#!/usr/bin/env python3
"""Train the shipped classification experiment.

Set DATA_ROOT to the ROI dataset root, then:

    python examples/train_classifier.py
"""

from pathlib import Path

from wildtrain.harness.config import load_experiment
from wildtrain.harness.runner import run_experiment

CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "classification"
    / "classification_train.yaml"
)


def main() -> None:
    experiment = load_experiment(CONFIG)
    result = run_experiment(experiment, debug=True)
    print(result)


if __name__ == "__main__":
    main()
