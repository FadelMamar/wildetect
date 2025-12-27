import csv
import json
from typing import Any
import optuna
from abc import ABC, abstractmethod
from pathlib import Path

from .classification_trainer import ClassifierTrainer
from .detection_trainer import UltralyticsDetectionTrainer
from ..shared.models import (
    ClassificationSweepConfig, 
    ClassificationConfig,
    DetectionSweepConfig,
    SweepObjectiveTypes,
    DetectionConfig
)
from ..shared.config_loader import ConfigLoader
from ..utils.logging import get_logger

logger = get_logger(__name__)

class Sweeper(ABC):
    
    @abstractmethod
    def __call__(self, trial: optuna.Trial) -> Any:
        pass

    @abstractmethod
    def run(self):
        pass


class ClassifierSweeper(Sweeper):
    def __init__(self, sweep_config_path: str, debug: bool = False):
        self.sweep_config_path = sweep_config_path
        self.sweep_cfg: ClassificationSweepConfig = ClassificationSweepConfig.from_yaml(self.sweep_config_path)
        self.base_cfg: ClassificationConfig = ConfigLoader.load_classification_config(self.sweep_cfg.base_config)
        self.counter = 0
        self.debug = debug
        self.study = None
        
    def __call__(self, trial: optuna.Trial):
        self.counter += 1

        try:
            # Model params
            model_params = self.sweep_cfg.parameters.model
            self.base_cfg.model.backbone = trial.suggest_categorical("backbone", model_params.backbone)
            self.base_cfg.model.dropout = trial.suggest_categorical("dropout", model_params.dropout)
            
            # Train params
            train_params = self.sweep_cfg.parameters.train
            lr = trial.suggest_categorical("lr", train_params.lr)
            lrf = trial.suggest_categorical("lrf", train_params.lrf)
            label_smoothing = trial.suggest_categorical("label_smoothing", train_params.label_smoothing)
            weight_decay = trial.suggest_categorical("weight_decay", train_params.weight_decay)
            batch_size = trial.suggest_categorical("batch_size", train_params.batch_size)
            epochs = trial.suggest_categorical("epochs", train_params.epochs)

            self.base_cfg.train.lr = lr
            self.base_cfg.train.lrf = lrf
            self.base_cfg.train.label_smoothing = label_smoothing
            self.base_cfg.train.weight_decay = weight_decay
            self.base_cfg.train.batch_size = batch_size
            self.base_cfg.train.epochs = epochs

            self.base_cfg.mlflow.run_name = f"trial_{self.counter}_{self.base_cfg.model.backbone}"
            self.base_cfg.mlflow.experiment_name = self.sweep_cfg.sweep_name
            self.base_cfg.checkpoint.dirpath = f"checkpoints/classification_sweeps/{self.sweep_cfg.sweep_name}"

            logger.info(
                "Running trial %d with params: backbone=%s, lr=%s, batch_size=%s",
                self.counter,
                self.base_cfg.model.backbone,
                lr,
                batch_size,
            )
            
            # Train
            trainer = ClassifierTrainer(self.base_cfg)
            trainer.run(debug=self.debug)

            # Handle cases where best_model_score might be None
            if trainer.best_model_score is None:
                logger.warning("Trial %d completed but best_model_score is None. Pruning trial.", self.counter)
                raise optuna.TrialPruned("Training completed but best_model_score is None")

            return trainer.best_model_score

        except optuna.TrialPruned:
            # Re-raise pruning exceptions
            raise
        except Exception as e:
            logger.error("Trial %d failed with error: %s", self.counter, e)
            logger.exception("Full traceback:")
            # Raise to mark trial as failed
            raise

    def run(self):
        study = optuna.create_study(
            direction="maximize",
            study_name=self.sweep_cfg.sweep_name,
            storage="sqlite:///{}.db".format(self.sweep_cfg.sweep_name),
            sampler=optuna.samplers.TPESampler(seed=self.sweep_cfg.seed),
            load_if_exists=True
        )
        
        # Store study for later access (e.g., for saving results)
        self.study = study
                
        # Start the optimization process
        logger.info("Starting Optuna optimization for hyperparameter sweep: %s", self.sweep_cfg.sweep_name)
        study.optimize(
            self,
            n_trials=self.sweep_cfg.n_trials,
            timeout=self.sweep_cfg.timeout,
        )
        
        # Output the best result
        best_trial = study.best_trial
        logger.info("\n" + "=" * 50)
        logger.info("Optimization completed!")
        logger.info("Best trial: #%d", best_trial.number)
        logger.info("Best Value (Score): %.6f", best_trial.value)
        logger.info("Best Params:")
        for key, value in best_trial.params.items():
            logger.info("  %s: %s", key, value)
        logger.info("Total trials completed: %d", len(study.trials))
        logger.info("=" * 50)
        
        # Save benchmark results
        if self.sweep_cfg.output.save_results:
            self._save_sweep_results()

    def _save_sweep_results(self):
        """Save sweep results to the specified output directory."""
        try:
            # Ensure output directory exists
            output_path = Path(self.sweep_cfg.output.directory)
            output_path.mkdir(parents=True, exist_ok=True)

            # Check if study is available
            if not hasattr(self, "study") or self.study is None:
                logger.warning(
                    "No study found in sweep pipeline. Results cannot be saved."
                )
                return

            study = self.study
            output_format = self.sweep_cfg.output.format

            # Extract best trial data
            best_trial = study.best_trial
            best_trial_data = {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
            }

            # Extract all trials data if requested
            all_trials_data = []
            if self.sweep_cfg.output.include_optimization_history:
                for trial in study.trials:
                    trial_data = {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                        "state": trial.state.name if trial.state else None,
                    }
                    all_trials_data.append(trial_data)

            # Study metadata
            study_metadata = {
                "direction": study.direction.name,
                "n_trials": len(study.trials),
                "best_trial_number": best_trial.number,
                "best_value": best_trial.value,
            }

            # Save JSON format if requested
            if output_format in ("json", "both"):
                try:
                    json_data = {
                        "metadata": study_metadata,
                        "best_trial": best_trial_data,
                    }
                    if self.sweep_cfg.output.include_optimization_history:
                        json_data["all_trials"] = all_trials_data

                    json_file = output_path / "sweep_results.json"
                    with open(json_file, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)

                    logger.info("Saved JSON results to: %s", json_file)
                except Exception as e:
                    logger.warning("Failed to save JSON results: %s", e)

            # Save CSV format if requested (CSV always includes all trials as it's tabular data)
            if output_format in ("csv", "both"):
                try:
                    csv_file = output_path / "sweep_results.csv"
                    with open(csv_file, "w", newline="", encoding="utf-8") as f:
                        all_trials_for_csv = []
                        for trial in study.trials:
                            trial_data = {
                                "number": trial.number,
                                "value": trial.value,
                                "params": trial.params,
                            }
                            all_trials_for_csv.append(trial_data)

                        if all_trials_for_csv:
                            # Write header
                            fieldnames = ["trial_number", "value"]
                            # Add param columns (use first trial to get param names)
                            param_names = list(all_trials_for_csv[0]["params"].keys())
                            fieldnames.extend(param_names)

                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()

                            # Write trial data
                            for trial_data in all_trials_for_csv:
                                row = {
                                    "trial_number": trial_data["number"],
                                    "value": trial_data["value"],
                                }
                                row.update(trial_data["params"])
                                writer.writerow(row)
                        else:
                            # Fallback: just write best trial if no trials available
                            fieldnames = ["trial_number", "value"]
                            param_names = list(best_trial_data["params"].keys())
                            fieldnames.extend(param_names)

                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            row = {
                                "trial_number": best_trial_data["number"],
                                "value": best_trial_data["value"],
                            }
                            row.update(best_trial_data["params"])
                            writer.writerow(row)

                    logger.info("Saved CSV results to: %s", csv_file)
                except Exception as e:
                    logger.warning("Failed to save CSV results: %s", e)

            # Generate performance plots if requested
            if self.sweep_cfg.output.save_plots:
                try:
                    # Optimization history plot
                    try:
                        fig = optuna.visualization.plot_optimization_history(study)
                        plot_file = output_path / "optimization_history.html"
                        fig.write_html(str(plot_file))
                        logger.info("Saved optimization history plot to: %s", plot_file)
                    except Exception as e:
                        logger.warning("Failed to generate optimization history plot: %s", e)

                    # Parallel coordinate plot
                    try:
                        fig = optuna.visualization.plot_parallel_coordinate(study)
                        plot_file = output_path / "parallel_coordinate.html"
                        fig.write_html(str(plot_file))
                        logger.info("Saved parallel coordinate plot to: %s", plot_file)
                    except Exception as e:
                        logger.warning("Failed to generate parallel coordinate plot: %s", e)

                    # Parameter importance plot (may fail if not enough trials)
                    try:
                        if len(study.trials) > 1:
                            fig = optuna.visualization.plot_param_importances(study)
                            plot_file = output_path / "param_importances.html"
                            fig.write_html(str(plot_file))
                            logger.info("Saved parameter importance plot to: %s", plot_file)
                    except Exception as e:
                        logger.warning("Failed to generate parameter importance plot (may need more trials): %s", e)
                except Exception as e:
                    logger.warning("Failed to generate plots: %s", e)

            logger.info("Sweep results saved successfully to: %s", output_path)

        except Exception as e:
            logger.warning("Failed to save sweep results: %s", e)


class DetectionSweeper(Sweeper):
    def __init__(self, sweep_config_path: str, debug: bool = False):
        self.sweep_config_path = sweep_config_path
        self.sweep_cfg: DetectionSweepConfig = DetectionSweepConfig.from_yaml(self.sweep_config_path)
        self.base_cfg: DetectionConfig = ConfigLoader.load_detection_config(self.sweep_cfg.base_config)
        self.counter = 0
        self.debug = debug
        self.study = None
        
    def __call__(self, trial: optuna.Trial):
        self.counter += 1

        try:
            # Model params (optional - only if provided in sweep config)
            if self.sweep_cfg.parameters.model is not None:
                model_params = self.sweep_cfg.parameters.model
                if model_params.architecture_file is not None:
                    self.base_cfg.model.architecture_file = trial.suggest_categorical("architecture_file", model_params.architecture_file)
                if model_params.weights is not None:
                    self.base_cfg.model.weights = trial.suggest_categorical("weights", model_params.weights)
            
            # Train params
            train_params = self.sweep_cfg.parameters.train
            lr0 = trial.suggest_categorical("lr0", train_params.lr0)
            lrf = trial.suggest_categorical("lrf", train_params.lrf)
            batch = trial.suggest_categorical("batch", train_params.batch)
            epochs = trial.suggest_categorical("epochs", train_params.epochs)
            imgsz = trial.suggest_categorical("imgsz", train_params.imgsz)
            optimizer = trial.suggest_categorical("optimizer", train_params.optimizer)
            weight_decay = trial.suggest_categorical("weight_decay", train_params.weight_decay)

            self.base_cfg.train.lr0 = lr0
            self.base_cfg.train.lrf = lrf
            self.base_cfg.train.batch = batch
            self.base_cfg.train.epochs = epochs
            self.base_cfg.train.imgsz = imgsz
            self.base_cfg.train.optimizer = optimizer
            self.base_cfg.train.weight_decay = weight_decay

            # Loss weights (optional)
            if train_params.box is not None:
                box = trial.suggest_categorical("box", train_params.box)
                self.base_cfg.train.box = box
            if train_params.cls is not None:
                cls = trial.suggest_categorical("cls", train_params.cls)
                self.base_cfg.train.cls = cls
            if train_params.dfl is not None:
                dfl = trial.suggest_categorical("dfl", train_params.dfl)
                self.base_cfg.train.dfl = dfl

            # Set MLflow experiment name and run name
            self.base_cfg.mlflow.experiment_name = self.sweep_cfg.sweep_name
            self.base_cfg.mlflow.run_name = f"trial_{self.counter}_lr0_{lr0}_batch_{batch}"

            logger.info(
                "Running trial %d with params: lr0=%s, batch=%s, epochs=%s, imgsz=%s, optimizer=%s",
                self.counter,
                lr0,
                batch,
                epochs,
                imgsz,
                optimizer,
            )
            
            # Train
            trainer = UltralyticsDetectionTrainer(self.base_cfg)
            trainer.run(debug=self.debug)

            if self.sweep_cfg.objective == SweepObjectiveTypes.FITNESS:
                if trainer.best_fitness is None:
                    logger.warning("Trial %d completed but best_fitness is None. Pruning trial.", self.counter)
                    raise optuna.TrialPruned("Training completed but best_fitness is None")
                return trainer.best_fitness
            else:
                if self.sweep_cfg.objective.value in trainer.metrics:
                    return trainer.metrics[self.sweep_cfg.objective.value]
                else:
                    logger.warning("Trial %d completed but %s is not in metrics. Pruning trial.", self.counter, self.sweep_cfg.objective)
                    raise optuna.TrialPruned(f"Training completed but {self.sweep_cfg.objective.value} is not in metrics keys (available keys: {trainer.metrics.keys()})")

        except optuna.TrialPruned as e:
            raise e
        except Exception as e:
            logger.error("Trial %d failed with error: %s", self.counter, e)
            logger.exception("Full traceback:")
            raise e

    def run(self):
        study = optuna.create_study(
            direction=self.sweep_cfg.direction.value,
            study_name=self.sweep_cfg.sweep_name,
            storage="sqlite:///{}.db".format(self.sweep_cfg.sweep_name),
            sampler=optuna.samplers.TPESampler(seed=self.sweep_cfg.seed),
            load_if_exists=True
        )
        
        # Store study for later access (e.g., for saving results)
        self.study = study
                
        # Start the optimization process
        logger.info("Starting Optuna optimization for hyperparameter sweep: %s", self.sweep_cfg.sweep_name)
        study.optimize(
            self,
            n_trials=self.sweep_cfg.n_trials,
            timeout=self.sweep_cfg.timeout,
        )
        
        # Output the best result
        best_trial = study.best_trial
        logger.info("\n" + "=" * 50)
        logger.info("Optimization completed!")
        logger.info("Best trial: #%d", best_trial.number)
        logger.info("Best Value (Fitness): %.6f", best_trial.value)
        logger.info("Best Params:")
        for key, value in best_trial.params.items():
            logger.info("  %s: %s", key, value)
        logger.info("Total trials completed: %d", len(study.trials))
        logger.info("=" * 50)
        
        # Save benchmark results
        if self.sweep_cfg.output and self.sweep_cfg.output.save_results:
            self._save_sweep_results()

    def _save_sweep_results(self):
        """Save sweep results to the specified output directory."""
        try:
            # Ensure output directory exists
            output_path = Path(self.sweep_cfg.output.directory) if self.sweep_cfg.output.directory else Path(f"results/sweeps/{self.sweep_cfg.sweep_name}")
            output_path.mkdir(parents=True, exist_ok=True)

            # Check if study is available
            if not hasattr(self, "study") or self.study is None:
                logger.warning(
                    "No study found in sweep pipeline. Results cannot be saved."
                )
                return

            study = self.study
            output_format = self.sweep_cfg.output.format if self.sweep_cfg.output else "json"

            # Extract best trial data
            best_trial = study.best_trial
            best_trial_data = {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
            }

            # Extract all trials data if requested
            all_trials_data = []
            include_history = self.sweep_cfg.output.include_optimization_history if self.sweep_cfg.output else True
            if include_history:
                for trial in study.trials:
                    trial_data = {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                        "state": trial.state.name if trial.state else None,
                    }
                    all_trials_data.append(trial_data)

            # Study metadata
            study_metadata = {
                "direction": study.direction.name,
                "n_trials": len(study.trials),
                "best_trial_number": best_trial.number,
                "best_value": best_trial.value,
            }

            # Save JSON format if requested
            if output_format in ("json", "both"):
                try:
                    json_data = {
                        "metadata": study_metadata,
                        "best_trial": best_trial_data,
                    }
                    if include_history:
                        json_data["all_trials"] = all_trials_data

                    json_file = output_path / "sweep_results.json"
                    with open(json_file, "w", encoding="utf-8") as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)

                    logger.info("Saved JSON results to: %s", json_file)
                except Exception as e:
                    logger.warning("Failed to save JSON results: %s", e)

            # Save CSV format if requested (CSV always includes all trials as it's tabular data)
            if output_format in ("csv", "both"):
                try:
                    csv_file = output_path / "sweep_results.csv"
                    with open(csv_file, "w", newline="", encoding="utf-8") as f:
                        all_trials_for_csv = []
                        for trial in study.trials:
                            trial_data = {
                                "number": trial.number,
                                "value": trial.value,
                                "params": trial.params,
                            }
                            all_trials_for_csv.append(trial_data)

                        if all_trials_for_csv:
                            # Write header
                            fieldnames = ["trial_number", "value"]
                            # Add param columns (use first trial to get param names)
                            param_names = list(all_trials_for_csv[0]["params"].keys())
                            fieldnames.extend(param_names)

                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()

                            # Write trial data
                            for trial_data in all_trials_for_csv:
                                row = {
                                    "trial_number": trial_data["number"],
                                    "value": trial_data["value"],
                                }
                                row.update(trial_data["params"])
                                writer.writerow(row)
                        else:
                            # Fallback: just write best trial if no trials available
                            fieldnames = ["trial_number", "value"]
                            param_names = list(best_trial_data["params"].keys())
                            fieldnames.extend(param_names)

                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            row = {
                                "trial_number": best_trial_data["number"],
                                "value": best_trial_data["value"],
                            }
                            row.update(best_trial_data["params"])
                            writer.writerow(row)

                    logger.info("Saved CSV results to: %s", csv_file)
                except Exception as e:
                    logger.warning("Failed to save CSV results: %s", e)

            # Generate performance plots if requested
            save_plots = self.sweep_cfg.output.save_plots if self.sweep_cfg.output else True
            if save_plots:
                try:
                    # Optimization history plot
                    try:
                        fig = optuna.visualization.plot_optimization_history(study)
                        plot_file = output_path / "optimization_history.html"
                        fig.write_html(str(plot_file))
                        logger.info("Saved optimization history plot to: %s", plot_file)
                    except Exception as e:
                        logger.warning("Failed to generate optimization history plot: %s", e)

                    # Parallel coordinate plot
                    try:
                        fig = optuna.visualization.plot_parallel_coordinate(study)
                        plot_file = output_path / "parallel_coordinate.html"
                        fig.write_html(str(plot_file))
                        logger.info("Saved parallel coordinate plot to: %s", plot_file)
                    except Exception as e:
                        logger.warning("Failed to generate parallel coordinate plot: %s", e)

                    # Parameter importance plot (may fail if not enough trials)
                    try:
                        if len(study.trials) > 1:
                            fig = optuna.visualization.plot_param_importances(study)
                            plot_file = output_path / "param_importances.html"
                            fig.write_html(str(plot_file))
                            logger.info("Saved parameter importance plot to: %s", plot_file)
                    except Exception as e:
                        logger.warning("Failed to generate parameter importance plot (may need more trials): %s", e)
                except Exception as e:
                    logger.warning("Failed to generate plots: %s", e)

            logger.info("Sweep results saved successfully to: %s", output_path)

        except Exception as e:
            logger.warning("Failed to save sweep results: %s", e)
