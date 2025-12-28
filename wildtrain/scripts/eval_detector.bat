call cd /d "%~dp0" && cd ..

call uv run wildtrain evaluate detector -c wildtrain\configs\detection\yolo_configs\yolo_eval.yaml 