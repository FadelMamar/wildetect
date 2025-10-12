call cd /d %~dp0

call deactivate

call ..\.venv-ls\Scripts\activate

call uv run --active --env-file ..\.env --no-sync label-studio start -p 8080

call deactivate
