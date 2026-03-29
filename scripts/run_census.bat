call cd /d "%~dp0" && cd ..

call uv run --env-file .env --no-sync wildetect detection census -c config/census.yaml

call pause