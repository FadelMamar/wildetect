:: call cd /d "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

:: call cd /d "D:\datalabeling"
::call deactivate

:: call .venv-mlflow\Scripts\activate

call cd /d %~dp0 && cd ..\..\wildtrain

call uv run mlflow server --backend-store-uri runs\mlflow --host 0.0.0.0 --port 5000

call pause
@REM call deactivate
