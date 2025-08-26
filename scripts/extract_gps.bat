
call cd /d %~dp0 && cd ..

call uv run wildtect visualization extract-gps-coordinates -c config/visualization.yaml