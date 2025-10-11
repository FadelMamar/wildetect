call cd /d %~dp0 && cd ..

call uv run wildetect services inference-server --port 4141 --workers 2

call pause