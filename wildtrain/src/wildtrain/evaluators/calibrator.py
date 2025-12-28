"""Detection model calibrator for inference hyperparameter optimization.

This module provides the DetectionCalibrator class that uses Optuna to find
optimal inference hyperparameters (conf_thres, iou_thres, max_det) for 
detection models by evaluating on a validation dataset.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional, Generator, List
import supervision as sv
from supervision.metrics.detection import ConfusionMatrix

import optuna

from ..shared.sweeper import Sweeper
from ..shared.schemas import (
    CalibrationConfig,
    DetectionEvalConfig,
    SweepObjectiveTypes,
    OverlapMetricConfig,
    MergingMethodConfig,
)
from ..utils.logging import get_logger
from .ultralytics import UltralyticsEvaluator

logger = get_logger(__name__)


class DetectionCalibrator(Sweeper):
    """Calibrator for detection model inference hyperparameters.
    
    Uses Optuna optimization to find the best combination of conf_thres,
    iou_thres, and optionally max_det that maximizes the chosen objective
    metric (e.g., F1 score).
    
    Args:
        calibration_config_path: Path to the calibration YAML config file.
        debug: If True, limits evaluation iterations for faster testing.
    
    Example:
        >>> calibrator = DetectionCalibrator("calibration_config.yaml")
        >>> calibrator.run()
    """
    
    def __init__(self, calibration_config_path: str, debug: bool = False,):
        self.calibration_config_path = calibration_config_path
        self.calibration_cfg = CalibrationConfig.from_yaml(calibration_config_path)
        self.base_cfg = DetectionEvalConfig.from_yaml(self.calibration_cfg.base_config)
        self.counter = 0
        self.debug = debug
        self.study: Optional[optuna.Study] = None
        self.overlap_metrics = {OverlapMetricConfig.IOU:sv.detection.utils.iou_and_nms.OverlapMetric.IOU,
                   OverlapMetricConfig.IOS:sv.detection.utils.iou_and_nms.OverlapMetric.IOS
                   }
        self._gt_and_preds: List[dict[str, List[sv.Detections]]] = None

        assert self.base_cfg.dataset.load_as_single_class, "Single class must be True for calibration"
    
    def get_gt_and_preds(self,) -> Generator[Dict[str, List[sv.Detections]], None, None]:
        if self._gt_and_preds is not None:
            return self._gt_and_preds
        logger.info("Running inference to collect predictions")
        evaluator = UltralyticsEvaluator(self.base_cfg,disable_detection_filtering=True)
        self._gt_and_preds = list(evaluator._run_inference())
        return self._gt_and_preds
    
    def _apply_filtering(
        self,
        detections: sv.Detections,
        iou_threshold: float,
        merging_method: MergingMethodConfig = MergingMethodConfig.NMS,
        overlap_metric: OverlapMetricConfig = OverlapMetricConfig.IOU,
    ) -> sv.Detections:
        """Apply filtering to remove duplicate predictions."""
        overlap_metric = self.overlap_metrics[overlap_metric]
        if merging_method == MergingMethodConfig.NMS:
            return detections.with_nms(
                threshold=iou_threshold,
                class_agnostic=self.class_agnostic,
                overlap_metric=overlap_metric,
            )
        elif merging_method == MergingMethodConfig.NMM:
            return detections.with_nmm(
                threshold=iou_threshold,
                class_agnostic=self.class_agnostic,
                overlap_metric=overlap_metric,
            )
        else:
            raise ValueError(f"Invalid merging method: {merging_method}")

    def _compute_metrics(
        self,
        iou_threshold: float,
        match_iou_threshold: float,
        conf_threshold: float = 0.0,
        merging_method: str = "nms",
        overlap_metric='iou',
    ) -> Dict[str, float]:
        """Compute precision, recall, F1 for given thresholds.

        Handles duplicate predictions across images by collapsing them into groups.
        If any prediction in a duplicate group matches GT, the entire group is
        considered one true positive.

        Args:
            iou_threshold: IoU threshold for NMS filtering
            conf_threshold: IoU threshold for pred-to-GT matching
            merging_method: Method for merging duplicate predictions

        Returns:
            Dictionary with precision, recall, f1 metrics
        """

        preds = []
        gts = []
        for values in self.get_gt_and_preds():
            preds.append(values["predictions"])
            gts.append(values["ground_truth"])

        preds = [
            self._apply_filtering(
                pred,
                iou_threshold=iou_threshold,
                overlap_metric=OverlapMetricConfig(overlap_metric),
                merging_method=MergingMethodConfig(merging_method),
            )
            for pred in preds
        ]
        confusion_matrix = ConfusionMatrix.from_detections(
            predictions=preds,
            targets=gts,
            classes=self.labels,
            iou_threshold=match_iou_threshold,
            conf_threshold=conf_threshold,
        )

        tp = confusion_matrix.matrix[0, 0]
        fp = confusion_matrix.matrix[1, 0]
        fn = confusion_matrix.matrix[0, 1]

        e = 1e-6
        precision = tp / (tp + fp + e)
        recall = tp / (tp + fn + e)
        f1 = 2 * precision * recall / (precision + recall + e)
        # print("tn:",tn)

        return {"precision": precision, "recall": recall, "f1": f1}
     
    def __call__(self, trial: optuna.Trial) -> float:
        """Objective function for a single Optuna trial.
        
        Suggests hyperparameters, updates the evaluation config, runs
        evaluation, and returns the objective metric value.
        
        Args:
            trial: Optuna trial object.
            
        Returns:
            The objective metric value (e.g., best F1 score).
        """
        self.counter += 1
        
        try:            
            # Suggest hyperparameters
            params = self.calibration_cfg.parameters
            conf_thres = trial.suggest_categorical(
                "conf_thres", 
                params.conf_thres
            )
            iou_thres = trial.suggest_categorical(
                "iou_thres", 
                params.iou_thres
            )
            matching_iou_thres = trial.suggest_categorical(
                "matching_iou_thres", 
                params.matching_iou_thres
            )
            merging_method = trial.suggest_categorical(
                "merging_method", 
                [m.value for m in params.merging_method]
            )
            overlap_metric = trial.suggest_categorical(
                "overlap_metric", 
                [m.value for m in params.overlap_metrics]
            )
            
            logger.info(
                "Running trial %d with params: conf_thres=%.3f, iou_thres=%.3f, overlap_metric=%s",
                self.counter,
                conf_thres,
                iou_thres,
                overlap_metric,
            )
            
            # Create evaluator and run evaluation
            metrics = self._compute_metrics(
                iou_threshold=iou_thres,
                match_iou_threshold=matching_iou_thres,
                conf_threshold=conf_thres,
                merging_method=merging_method,
                overlap_metric=overlap_metric,
            )
            
            # Extract objective metric
            objective_value = self._extract_objective(metrics)
            
            if objective_value is None:
                logger.warning(
                    "Trial %d completed but objective metric is None. Pruning trial.",
                    self.counter
                )
                raise optuna.TrialPruned("Evaluation completed but objective metric is None")
            
            logger.info("Trial %d result: %.6f", self.counter, objective_value)
            return objective_value
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error("Trial %d failed with error: %s", self.counter, e)
            logger.exception("Full traceback:")
            raise
    
    def _extract_objective(self, metrics: Dict[str, float]) -> Optional[float]:
        """Extract the objective metric value from the evaluation report.
        
        Args:
            metrics: Evaluation metrics dictionary from the evaluator.
            
        Returns:
            The objective metric value, or None if not found.
        """
        objective = self.calibration_cfg.objective
        
        if objective == SweepObjectiveTypes.F1_SCORE:
            return metrics['f1']
        
        elif objective == SweepObjectiveTypes.PRECISION:
            return metrics['precision']
        
        elif objective == SweepObjectiveTypes.RECALL:
            return metrics['recall']
                
        else:
            logger.warning("Unknown objective type: %s", objective)
            return None
    
    def run(self):
        """Run the calibration optimization process."""
        study = optuna.create_study(
            direction=self.calibration_cfg.direction.value,
            study_name=self.calibration_cfg.calibration_name,
            storage=f"sqlite:///{self.calibration_cfg.calibration_name}.db",
            sampler=optuna.samplers.TPESampler(seed=self.calibration_cfg.seed),
            load_if_exists=True
        )
        
        self.study = study
        
        logger.info(
            "Starting Optuna calibration: %s", 
            self.calibration_cfg.calibration_name
        )
        
        study.optimize(
            self,
            n_trials=self.calibration_cfg.n_trials,
            timeout=self.calibration_cfg.timeout,
        )
        
        # Output the best result
        best_trial = study.best_trial
        logger.info("\n" + "=" * 50)
        logger.info("Calibration completed!")
        logger.info("Best trial: #%d", best_trial.number)
        logger.info("Best Value (%s): %.6f", self.calibration_cfg.objective, best_trial.value)
        logger.info("Best Params:")
        for key, value in best_trial.params.items():
            logger.info("  %s: %s", key, value)
        logger.info("Total trials completed: %d", len(study.trials))
        logger.info("=" * 50)
        
        # Save results if configured
        if self.calibration_cfg.output and self.calibration_cfg.output.save_results:
            self._save_calibration_results()
    
    def _save_calibration_results(self):
        """Save calibration results to the specified output directory."""
        try:
            output_cfg = self.calibration_cfg.output
            output_path = Path(output_cfg.directory) if output_cfg.directory else Path(
                f"results/calibrations/{self.calibration_cfg.calibration_name}"
            )
            output_path.mkdir(parents=True, exist_ok=True)
            
            if self.study is None:
                logger.warning("No study found. Results cannot be saved.")
                return
            
            study = self.study
            output_format = output_cfg.format if output_cfg else "json"
            
            # Extract best trial data
            best_trial = study.best_trial
            best_trial_data = {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
            }
            
            # Extract all trials data
            all_trials_data = []
            include_history = output_cfg.include_optimization_history if output_cfg else True
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
                "objective": self.calibration_cfg.objective.value,
            }
            
            # Save JSON
            if output_format in ("json", "both"):
                json_data = {
                    "metadata": study_metadata,
                    "best_trial": best_trial_data,
                }
                if include_history:
                    json_data["all_trials"] = all_trials_data
                
                json_file = output_path / "calibration_results.json"
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                logger.info("Saved JSON results to: %s", json_file)
            
            # Save CSV
            if output_format in ("csv", "both"):
                csv_file = output_path / "calibration_results.csv"
                with open(csv_file, "w", newline="", encoding="utf-8") as f:
                    trials_for_csv = []
                    for trial in study.trials:
                        trials_for_csv.append({
                            "number": trial.number,
                            "value": trial.value,
                            "params": trial.params,
                        })
                    
                    if trials_for_csv:
                        fieldnames = ["trial_number", "value"]
                        param_names = list(trials_for_csv[0]["params"].keys())
                        fieldnames.extend(param_names)
                        
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for trial_data in trials_for_csv:
                            row = {
                                "trial_number": trial_data["number"],
                                "value": trial_data["value"],
                            }
                            row.update(trial_data["params"])
                            writer.writerow(row)
                    
                logger.info("Saved CSV results to: %s", csv_file)
            
            # Generate plots if requested
            save_plots = output_cfg.save_plots if output_cfg else True
            if save_plots:
                try:
                    fig = optuna.visualization.plot_optimization_history(study)
                    plot_file = output_path / "optimization_history.html"
                    fig.write_html(str(plot_file))
                    logger.info("Saved optimization history plot to: %s", plot_file)
                except Exception as e:
                    logger.warning("Failed to generate optimization history plot: %s", e)
                
                try:
                    fig = optuna.visualization.plot_parallel_coordinate(study)
                    plot_file = output_path / "parallel_coordinate.html"
                    fig.write_html(str(plot_file))
                    logger.info("Saved parallel coordinate plot to: %s", plot_file)
                except Exception as e:
                    logger.warning("Failed to generate parallel coordinate plot: %s", e)
                
                try:
                    if len(study.trials) > 1:
                        fig = optuna.visualization.plot_param_importances(study)
                        plot_file = output_path / "param_importances.html"
                        fig.write_html(str(plot_file))
                        logger.info("Saved parameter importance plot to: %s", plot_file)
                except Exception as e:
                    logger.warning("Failed to generate parameter importance plot: %s", e)
            
            logger.info("Calibration results saved to: %s", output_path)
            
        except Exception as e:
            logger.warning("Failed to save calibration results: %s", e)
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best hyperparameters found during calibration.
        
        Returns:
            Dictionary of best parameters, or None if no study exists.
        """
        if self.study is None:
            return None
        return self.study.best_trial.params
