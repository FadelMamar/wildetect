call cd /d "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

::call cd /d "D:\datalabeling"
::call deactivate

call .venv-mlflow\Scripts\activate

call mlflow server --backend-store-uri runs\mlflow

@REM call deactivate
