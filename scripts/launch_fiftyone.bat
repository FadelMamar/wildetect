call cd /d %~dp0 && cd ..

call uv run --no-sync --env-file .env fiftyone app launch

call pause