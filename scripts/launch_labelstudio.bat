call cd /d %~dp0 && cd ..

call deactivate

call .venv-ls\Scripts\activate

call uv run --active --env-file .env label-studio start -p 8080

call deactivate
