@echo off
REM Example script to run the detection pipeline using WildTrain CLI

call cd /d %~dp0 && cd ..

REM Set example config path
set CONFIG_PATH=configs/detection/yolo_configs/yolo_pipeline_config.yaml

REM Run the run_detection_pipeline CLI command
call uv run wildtrain pipeline detection -c %CONFIG_PATH% 
