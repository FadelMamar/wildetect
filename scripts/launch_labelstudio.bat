call cd /d %~dp0

call deactivate
call deactivate

call ..\.venv-ls\Scripts\activate
call uv pip install label-studio==1.18.0
call uv run --active --no-sync --env-file ..\.env label-studio start -p 8080

call deactivate
