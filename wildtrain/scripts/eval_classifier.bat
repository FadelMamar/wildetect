@echo off
REM Evaluate a classifier from the new eval document.

call cd /d "%~dp0" && cd ..

set CONFIG_PATH=configs\classification\classification_eval.yaml

call uv run wildtrain evaluate -c %CONFIG_PATH%
