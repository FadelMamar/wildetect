call cd /d "%~dp0" && cd ..

call set MLFLOW_TRACKING_URI=http://127.0.0.1:5000

call uv run wildtrain run-server --config configs/inference.yaml

