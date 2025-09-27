call cd /d %~dp0 && cd ..\..\wildtrain

@REM call cd /d %~dp0 && cd ..

call uv run mlflow server --backend-store-uri runs\mlflow --host 0.0.0.0 --port 5000

call pause
@REM call deactivate
