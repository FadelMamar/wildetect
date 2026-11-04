"""Experiment seed is applied before training and before a sweep."""

from pathlib import Path

from wildtrain.harness.config import SweepBlock, load_experiment
from wildtrain.harness.result import FitResult
from wildtrain.harness.runner import apply_experiment_seed, run_experiment
from wildtrain.harness.sweep import ExperimentSweeper

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_classification_seed_uses_lightning(monkeypatch) -> None:
    seen = {}

    def fake_seed_everything(seed, workers=True):
        seen["seed"] = seed
        seen["workers"] = workers

    monkeypatch.setattr("lightning.seed_everything", fake_seed_everything)
    path = CONFIGS / "classification" / "classification_train.yaml"
    experiment = load_experiment(path)
    experiment.seed = 17
    apply_experiment_seed(experiment)
    assert seen == {"seed": 17, "workers": True}


def test_detection_seed_reaches_ultralytics_config(monkeypatch) -> None:
    seen = {}

    def fake_set_seed(seed):
        seen["python"] = seed

    monkeypatch.setattr("wildtrain.utils.misc.set_seed", fake_set_seed)
    experiment = load_experiment(CONFIGS / "detection" / "yolo_configs" / "yolo.yaml")
    experiment.seed = 23
    apply_experiment_seed(experiment)
    assert seen["python"] == 23
    assert experiment.trainer_config().train.seed == 23


def test_run_experiment_seeds_before_the_task(monkeypatch) -> None:
    order = []
    path = CONFIGS / "classification" / "classification_train.yaml"
    experiment = load_experiment(path)
    experiment.seed = 5

    def fake_apply(config):
        order.append(("seed", config.seed))

    class FakeTask:
        name = "classification"

        def run(self, config, debug: bool = False):
            order.append(("run", config.seed))
            return FitResult(best_path="best.ckpt", best_score=0.1, score_name="f1")

    monkeypatch.setattr("wildtrain.harness.runner.apply_experiment_seed", fake_apply)
    run_experiment(
        experiment,
        registry=type("R", (), {"get": lambda self, name: FakeTask()})(),
    )
    assert order == [("seed", 5), ("run", 5)]


def test_sweep_seeds_before_trials(monkeypatch) -> None:
    path = CONFIGS / "classification" / "classification_train.yaml"
    experiment = load_experiment(path)
    experiment.seed = 9
    experiment.sweep = SweepBlock(
        n_trials=1,
        objective="f1_score",
        parameters={"train.lr": [0.001]},
        seed=3,
    )
    seen = []

    def fake_apply(config):
        seen.append(config.seed)

    def fake_run(config, debug=False):
        return FitResult(best_path="best.ckpt", best_score=0.2, score_name="f1")

    monkeypatch.setattr("wildtrain.harness.sweep.apply_experiment_seed", fake_apply)
    monkeypatch.setattr("wildtrain.harness.sweep.run_experiment", fake_run)
    monkeypatch.setattr(
        "wildtrain.harness.runner.apply_experiment_seed", lambda config: None
    )
    ExperimentSweeper(experiment).run()
    assert seen[0] == 9
