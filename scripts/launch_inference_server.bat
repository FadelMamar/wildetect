call cd /d %~dp0 
call cd ..

call uv run wildetect services inference-server --port 4141 --workers 4 --max-batch-size 2

call pause