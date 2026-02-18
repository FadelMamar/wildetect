call cd /d %~dp0 && cd ..

call set CONFIG_FILE=configs\detection\yolo_configs\yolo.yaml

call uv run wildtrain train detector -c %CONFIG_FILE%