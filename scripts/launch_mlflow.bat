call c: 

call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call dir

@REM cd "D:\datalabeling"

@REM call deactivate

call .venv-mlflow\Scripts\activate

call mlflow server --backend-store-uri runs\mlflow

@REM call deactivate
