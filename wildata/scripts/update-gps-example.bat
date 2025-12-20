@echo off
REM Example script for updating GPS data from CSV
REM This script demonstrates how to use the update_gps_from_csv command

echo Updating GPS data from CSV...

call cd /d "%~dp0" && cd ..

REM Using config file
uv run wildata update-gps-from-csv --config configs/gps-update-config-example.yaml --verbose

REM Or using command line arguments
REM python -m wildata update-gps-from-csv ^
REM     --image-folder "path/to/images" ^
REM     --csv "path/to/gps_coordinates.csv" ^
REM     --output "path/to/output" ^
REM     --skip-rows 1 ^
REM     --filename-col "filename" ^
REM     --lat-col "latitude" ^
REM     --lon-col "longitude" ^
REM     --alt-col "altitude" ^
REM     --verbose

echo GPS update complete!
pause 