@echo off
REM Example: Import a dataset using the WildTrain CLI with a YAML config file

call cd /d "%~dp0" && cd ..

REM Run the import command using only the config file
call uv run wildata create-roi-dataset --config configs\roi-create-config.yaml -v

REM Set the path to your config file (edit as needed)
:: set CONFIG_FILE=configs\bulk-roi-create-config.yaml
:: call uv run wildata bulk-create-roi-datasets --config %CONFIG_FILE% -n 2 -v

call pause
