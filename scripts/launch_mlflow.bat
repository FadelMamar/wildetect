call cd "C:\Users\Machine Learning\Desktop\workspace-wildAI\datalabeling"

call cd "D:\datalabeling"

call deactivate

call .venv-mlflow\Scripts\activate

call mlflow server --backend-store-uri runs\mlflow

call deactivate
