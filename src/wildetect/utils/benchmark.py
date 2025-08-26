import os
import time
from typing import List

import optuna
import torch

from ..core.config import LoaderConfig, PredictionConfig
from ..core.detection_pipeline import DetectionPipeline


class BenchmarkPipeline(object):
    def __init__(
        self,
        image_paths: List[str],
        prediction_config: PredictionConfig,
        loader_config: LoaderConfig,
    ):
        self.image_paths = image_paths
        self.prediction_config = prediction_config
        self.loader_config = loader_config

    def run(self, n_trials: int = 30, timeout: int = 3600, direction: str = "minimize"):
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        # Start the optimization process
        print("Starting Optuna optimization for inference throughput...")
        study.optimize(self, n_trials=n_trials, timeout=timeout)

        # Output the best result
        best_trial = study.best_trial
        print("\n" + "=" * 50)
        print("Optimization completed!")
        print(f"Best trial: #{best_trial.number}")
        print(f"Best Value (Throughput): {best_trial.value:.2f} img/sec")
        print("Best Params:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")

    def _run_benchmark(self) -> float:
        """Runs a short benchmark and returns the achieved throughput (img/sec)."""
        detection_pipeline = DetectionPipeline(
            config=self.prediction_config, loader_config=self.loader_config
        )

        start_time = time.perf_counter()
        torch.cuda.synchronize()
        detection_pipeline.run_detection(
            image_paths=self.image_paths, override_loading_config=False
        )
        torch.cuda.synchronize()
        end_time = time.perf_counter()

        return len(self.image_paths) / (end_time - start_time)

    def __call__(self, trial: optuna.Trial):
        # Suggest values for the hyperparameters
        batch_size = trial.suggest_categorical(
            "batch_size", [8, 16, 32, 64, 128, 256, 512]
        )
        num_workers = trial.suggest_int(
            "num_workers", 0, os.cpu_count() - 3
        )  # 0 to 16 workers

        self.prediction_config.batch_size = batch_size
        self.loader_config.batch_size = batch_size
        self.loader_config.num_workers = num_workers

        return self._run_benchmark()
