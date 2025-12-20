@echo off
REM Example: Import a dataset using the WildTrain CLI with a YAML config file

call cd /d "%~dp0" && cd ..

REM Set the path to your config file (edit as needed)
set CONFIG_FILE=configs\bulk-import-config-example.yaml

REM Run the import command using only the config file
call uv run wildata bulk-import-datasets --config %CONFIG_FILE% -n 2

call pause