call cd /d %~dp0 && cd ..

call set CONFIG_FILE=wildtrain\configs\classification\classification_train.yaml

call uv run wildtrain train classifier -c %CONFIG_FILE%