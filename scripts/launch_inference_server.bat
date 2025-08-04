call cd /d %~dp0 
call cd ..

call uv run wildetect inference-server --port 4141 --workers 2