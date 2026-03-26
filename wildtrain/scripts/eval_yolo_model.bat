call cd /d "%~dp0" && cd ..

call uv run wildtrain evaluate yolo-model -c configs\detection\yolo_configs\eval_config.yaml