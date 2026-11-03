"""One Optuna sweep for either harness task.

Legacy classification and detection sweep YAML (``base_config`` plus a nested
``parameters`` tree) converts into ``ExperimentConfig.sweep.parameters``, a
dict of dotted paths. The trial writes those choices onto a copy of the
experiment and scores it through ``run_experiment``.
"""

import csv
import json
from pathlib import Path
from typing import Any, Callable, Optional

import optuna
from omegaconf import OmegaConf

from ..shared.schemas.sweep import ClassificationSweepConfig, DetectionSweepConfig
from ..utils.logging import get_logger
from .config import (
    ExperimentConfig,
    SweepBlock,
    experiment_from_mapping,
    load_experiment,
)
from .result import FitResult
from .runner import run_experiment

logger = get_logger(__name__)

_EXPERIMENT_ROOTS = {"tracking", "run", "classification", "detection", "task"}
_SCORE_FROM_RESULT = {"fitness", "f1_score", "val_f1score"}


def apply_dotted(config: ExperimentConfig, path: str, value: Any) -> None:
    """Set ``path`` on the experiment or on its task body."""
    if not path:
        raise ValueError("sweep parameter path is empty")
    parts = path.split(".")
    target: Any = config if parts[0] in _EXPERIMENT_ROOTS else config.body
    cursor = target
    for part in parts[:-1]:
        if not hasattr(cursor, part):
            raise AttributeError(f"sweep path {path!r} has no attribute {part!r}")
        cursor = getattr(cursor, part)
    leaf = parts[-1]
    if not hasattr(cursor, leaf):
        raise AttributeError(f"sweep path {path!r} has no attribute {leaf!r}")
    setattr(cursor, leaf, value)


def score_from_result(result: FitResult, objective: Optional[str]) -> Optional[float]:
    """Pick the trial score.

    Classification sweeps and ``fitness`` use ``FitResult.best_score``. Any
    other name is looked up on ``FitResult.metrics``, which is where detection
    keeps the Ultralytics metrics dict.
    """
    if objective is None or objective in _SCORE_FROM_RESULT:
        return _as_float(result.best_score)
    metrics = result.metrics or {}
    if objective not in metrics:
        return None
    return _as_float(metrics[objective])


def parameters_from_legacy_classification(sweep: ClassificationSweepConfig) -> dict:
    """Dotted categorical grid matching ``ClassifierSweeper``."""
    model = sweep.parameters.model
    train = sweep.parameters.train
    return {
        "model.backbone": list(model.backbone),
        "model.dropout": list(model.dropout),
        "train.lr": list(train.lr),
        "train.lrf": list(train.lrf),
        "train.label_smoothing": list(train.label_smoothing),
        "train.weight_decay": list(train.weight_decay),
        "train.batch_size": list(train.batch_size),
        "train.epochs": list(train.epochs),
    }


def parameters_from_legacy_detection(sweep: DetectionSweepConfig) -> dict:
    """Dotted categorical grid matching ``DetectionSweeper``."""
    parameters: dict[str, list] = {}
    model = sweep.parameters.model
    if model is not None:
        if model.architecture_file:
            parameters["model.architecture_file"] = list(model.architecture_file)
        if model.weights:
            parameters["model.weights"] = list(model.weights)
    train = sweep.parameters.train
    for field in (
        "lr0",
        "lrf",
        "batch",
        "epochs",
        "imgsz",
        "optimizer",
        "weight_decay",
        "box",
        "cls",
        "dfl",
    ):
        values = getattr(train, field)
        if values:
            parameters[f"train.{field}"] = list(values)
    return parameters


def experiment_from_legacy_sweep(
    sweep: ClassificationSweepConfig | DetectionSweepConfig,
) -> ExperimentConfig:
    """Wrap a legacy sweep file around its base experiment."""
    experiment = load_experiment(sweep.base_config)
    if isinstance(sweep, ClassificationSweepConfig):
        parameters = parameters_from_legacy_classification(sweep)
        direction = "maximize"
        objective = "f1_score"
    else:
        parameters = parameters_from_legacy_detection(sweep)
        direction = sweep.direction.value
        objective = sweep.objective.value
    experiment.sweep = SweepBlock(
        n_trials=sweep.n_trials,
        direction=direction,
        objective=objective,
        parameters=parameters,
        storage=f"sqlite:///{sweep.sweep_name}.db",
        seed=sweep.seed,
        name=sweep.sweep_name,
        timeout=sweep.timeout,
    )
    return experiment


def load_sweep(path: Path | str) -> ExperimentConfig:
    """Load a canonical experiment that contains ``sweep``, or a legacy sweep YAML."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Sweep file not found: {config_path}")
    loaded = OmegaConf.load(config_path)
    payload = OmegaConf.to_container(loaded, resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Sweep file must be a mapping: {config_path}")
    if (
        "base_config" in payload
        and "parameters" in payload
        and not _is_experiment_document(payload)
    ):
        return experiment_from_legacy_sweep(_legacy_sweep_model(payload))
    experiment = experiment_from_mapping(payload)
    if experiment.sweep is None:
        raise ValueError(f"Sweep file has no sweep block: {config_path}")
    return experiment


class ExperimentSweeper:
    """Optuna study that runs the registered task, or a supplied objective."""

    def __init__(
        self,
        config: ExperimentConfig,
        objective: Optional[Callable[[ExperimentConfig], float]] = None,
        debug: bool = False,
    ) -> None:
        if config.sweep is None:
            raise ValueError("ExperimentConfig.sweep is required")
        self.config = config
        self.objective = objective
        self.debug = debug
        self.study: Optional[optuna.Study] = None

    def __call__(self, trial: optuna.Trial) -> float:
        sweep = self.config.sweep
        assert sweep is not None
        trial_config = self.config.model_copy(deep=True)
        for name, choices in sweep.parameters.items():
            value = trial.suggest_categorical(name, list(choices))
            apply_dotted(trial_config, name, value)
        if self.objective is not None:
            return float(self.objective(trial_config))

        trial_config.run.name = f"{trial_config.run.name or 'trial'}_{trial.number}"
        if sweep.name:
            trial_config.tracking.mlflow.experiment_name = sweep.name
        trial_config.tracking.mlflow.run_name = trial_config.run.name
        if trial_config.task == "classification" and trial_config.run.output_dir:
            trial_config.run.output_dir = str(
                Path(trial_config.run.output_dir) / f"trial_{trial.number}"
            )
        result = run_experiment(
            trial_config, debug=self.debug or trial_config.run.debug
        )
        score = score_from_result(result, sweep.objective)
        if score is None:
            raise optuna.TrialPruned(
                f"trial {trial.number} produced no score for {sweep.objective!r}"
            )
        return score

    def run(self) -> optuna.Study:
        sweep = self.config.sweep
        assert sweep is not None
        study = optuna.create_study(
            direction=sweep.direction,
            study_name=sweep.name or f"{self.config.task}-sweep",
            storage=sweep.storage,
            sampler=optuna.samplers.TPESampler(seed=sweep.seed),
            load_if_exists=sweep.storage is not None,
        )
        self.study = study
        logger.info("Starting sweep %s for %s", study.study_name, self.config.task)
        study.optimize(self, n_trials=sweep.n_trials, timeout=sweep.timeout)
        return study


def save_study(
    study: optuna.Study,
    directory: str | Path,
    output_format: str = "json",
    include_history: bool = True,
    save_plots: bool = False,
) -> None:
    """Write sweep results. Plot files are best-effort."""
    output_path = Path(directory)
    output_path.mkdir(parents=True, exist_ok=True)
    best = study.best_trial
    history = [
        {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": trial.state.name if trial.state else None,
        }
        for trial in study.trials
    ]
    metadata = {
        "direction": study.direction.name,
        "n_trials": len(study.trials),
        "best_trial_number": best.number,
        "best_value": best.value,
    }
    if output_format in {"json", "both"}:
        payload: dict[str, Any] = {
            "metadata": metadata,
            "best_trial": {
                "number": best.number,
                "value": best.value,
                "params": best.params,
            },
        }
        if include_history:
            payload["all_trials"] = history
        (output_path / "sweep_results.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    if output_format in {"csv", "both"}:
        _write_csv(output_path / "sweep_results.csv", history or [
            {"number": best.number, "value": best.value, "params": best.params}
        ])
    if save_plots:
        _write_plots(study, output_path)


def _write_csv(path: Path, trials: list[dict]) -> None:
    param_names = list(trials[0]["params"].keys()) if trials else []
    fieldnames = ["trial_number", "value", *param_names]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            row = {"trial_number": trial["number"], "value": trial["value"]}
            row.update(trial["params"])
            writer.writerow(row)


def _write_plots(study: optuna.Study, output_path: Path) -> None:
    plots = (
        ("plot_optimization_history", "optimization_history.html"),
        ("plot_parallel_coordinate", "parallel_coordinate.html"),
        ("plot_param_importances", "param_importances.html"),
    )
    for name, filename in plots:
        try:
            if name == "plot_param_importances" and len(study.trials) < 2:
                continue
            figure = getattr(optuna.visualization, name)(study)
            figure.write_html(str(output_path / filename))
        except Exception as exc:
            logger.warning("Failed to write %s: %s", filename, exc)


def _legacy_sweep_model(payload: dict):
    train = (payload.get("parameters") or {}).get("train") or {}
    if "lr0" in train or "imgsz" in train:
        return DetectionSweepConfig.model_validate(payload)
    return ClassificationSweepConfig.model_validate(payload)


def _is_experiment_document(payload: dict) -> bool:
    return isinstance(payload.get("tracking"), dict) or "classification" in payload


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
