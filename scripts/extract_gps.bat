
call cd /d %~dp0 && cd ..

call uv run wildetect visualization extract-gps-coordinates -c config/extract-gps.yaml
