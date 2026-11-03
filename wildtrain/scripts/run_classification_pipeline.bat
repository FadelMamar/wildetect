@echo off
REM Train then evaluate one experiment. Set eval.enabled: true in the file.
call cd /d %~dp0 && cd ..

set CONFIG_PATH=configs\classification\classification_train.yaml

call uv run wildtrain run -c %CONFIG_PATH%
