call cd /d "%~dp0" && cd ..


call uv run --env-file .env wildetect detection detect  -c config/detection.yaml

call pause
