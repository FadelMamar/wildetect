call cd /d "%~dp0" && cd ..

REM yolo-model was removed. This file is the same detection eval document.
call uv run wildtrain evaluate -c configs\detection\yolo_configs\eval_config.yaml
