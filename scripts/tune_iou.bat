call cd /d %~dp0 && cd ..

@REM Tunes duplicate-removal thresholds with Optuna.
@REM
@REM Arguments:
@REM   --csv_path: Input CSV containing candidate duplicates (here: "animal-duplicates.csv").
@REM   --verbose: Enable/disable verbose logging output (False keeps it quieter).
@REM
@REM   --iou_min/--iou_max: Search bounds for the IoU threshold parameter Optuna will tune.
@REM     Note: This script currently allows [-1.0, 1.0]. If your IoU is the standard [0, 1],
@REM     this wide range may be intentional (e.g., using a signed/shifted metric) or just permissive.
@REM
@REM   --overlap_min/--overlap_max: Search bounds for an "overlap" threshold for two images to be considered overlapping.
@REM     (Typically overlap is a fraction in [0, 1]; here you're tuning between 0.0 and 0.1.)
@REM
@REM   --sensor_height: Camera sensor height used by the geometry model (units? often mm).
@REM   --focal_length: Camera focal length used by the geometry model (units? often mm).
@REM   --flight_height: Platform/flight altitude used by the geometry model (units? often meters).
@REM
@REM   --n_trials: Number of Optuna trials to run (higher = more search, slower).
@REM   --optuna_load_if_exists: Resume an existing Optuna study if present in the storage.
@REM   --run_name: Study name (used to identify the run in Optuna storage).
@REM   --optuna_storage: Optuna storage backend (SQLite DB file in repo root here).

call uv run scripts\custom_iou_tunner.py run_duplicate_tuner --csv_path="animal-duplicates.csv" --verbose=False ^
        --iou_min=-1.0 --iou_max=1.0 ^
        --overlap_min=0.0 --overlap_max=0.1 ^
        --sensor_height=24 --focal_length=35 --flight_height=180 ^
        --n_trials=150 --optuna_load_if_exists=True --run_name="duplicate-tuner-strict" --optuna_storage="sqlite:///duplicate-tuner.db"

echo "To visualize tuning, run: `uv run optuna-dashboard sqlite:///duplicate-tuner.db --port 8044`"


pause