call cd /d %~dp0 && cd ..

call set CONFIG_FILE=configs\classification\classification_train.yaml

call uv run wildtrain run -c %CONFIG_FILE%
