call cd /d "%~dp0" && cd ..

call uv run wildetect detection census -c config/census.yaml --profile --gpu-profile --line-profile

call pause