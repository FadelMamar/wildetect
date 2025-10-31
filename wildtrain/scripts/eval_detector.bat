call cd /d "%~dp0" && cd ..

call uv run wildtrain evaluate detector -c configs\detection\yolo_configs\yolo_eval.yaml --type yolo