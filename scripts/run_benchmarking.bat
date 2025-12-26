
call cd /d "%~dp0" && cd ..

call uv run wildetect benchmarking detection -c config/benchmark.yaml

call pause