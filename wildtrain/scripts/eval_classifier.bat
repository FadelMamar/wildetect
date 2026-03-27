@echo off
REM Example script to evaluate a classifier using WildTrain CLI

call cd /d "%~dp0" && cd ..

REM Set example config path
set CONFIG_PATH=configs\classification\classification_eval.yaml

REM Run the evaluate-classifier CLI command
call uv run wildtrain evaluate classifier -c %CONFIG_PATH% 