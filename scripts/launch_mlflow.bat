call cd /d %~dp0 && cd ..

call deactivate

call uv run  mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

call pause
@REM call deactivate
