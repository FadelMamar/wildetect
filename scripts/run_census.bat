call cd /d "%~dp0" && cd ..
call .venv\Scripts\activate

call uv run wildetect detection census -c config/census.yaml

call pause