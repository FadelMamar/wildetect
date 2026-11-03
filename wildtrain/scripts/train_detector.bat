call cd /d %~dp0 && cd ..

call set CONFIG_FILE=configs\detection\yolo_configs\yolo.yaml

call uv run wildtrain run -c %CONFIG_FILE%
