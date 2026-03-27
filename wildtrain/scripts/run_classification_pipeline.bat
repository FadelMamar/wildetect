@echo off
REM Example script to run the classification pipeline using WildTrain CLI
call cd /d %~dp0 && cd ..

REM Set example config path
set CONFIG_PATH=configs\classification\classification_pipeline_config.yaml

REM Run the run_classification_pipeline CLI command
call uv run wildtrain pipeline classification ^
    -c %CONFIG_PATH% 