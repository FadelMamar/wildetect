@echo off
REM Example script for updating GPS data from CSV
REM This script demonstrates how to use the update_gps_from_csv command

echo Updating GPS data from CSV...

call cd /d "%~dp0" && cd ..

REM Using config file
uv run wildata update-gps-from-csv --config configs/gps-update-config-example.yaml --verbose

echo GPS update complete!
pause 