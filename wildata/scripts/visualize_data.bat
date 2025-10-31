@echo off
REM Visualize datasets using the wildata CLI
REM This script provides examples for both classification and detection visualization

call cd /d "%~dp0" && cd ..
echo ========================================
echo WildData Visualization Examples
echo ========================================

REM Set the dataset name and root directory (edit as needed)
set DATASET_NAME=savmap
set ROOT_DIR=D:\workspace\data\demo-dataset

REM Optionally set other parameters (uncomment and edit as needed)
REM set KEEP_CLASSES=lion,elephant
REM set DISCARD_CLASSES=termite mound,rocks
REM set SPLIT=val

echo.
echo Example 1: Visualize Classification Dataset
echo -------------------------------------------
echo Dataset: %DATASET_NAME%
echo Root Directory: %ROOT_DIR%
echo Split: train
echo.

echo call uv run --no-sync wildata visualize-classification %DATASET_NAME% --root %ROOT_DIR% --split train

echo.
echo Example 2: Visualize Classification Dataset with Class Filtering
echo ----------------------------------------------------------------
echo Dataset: %DATASET_NAME%
echo Root Directory: %ROOT_DIR%
echo Split: val
echo Keep Classes: 
echo Discard Classes: termite mound,rocks
echo.

REM Example with keep/discard classes (uncomment to use)
echo call uv run --no-sync wildata visualize-classification %DATASET_NAME% --root %ROOT_DIR% --split val --keep-classes %KEEP_CLASSES% --discard-classes %DISCARD_CLASSES%

echo.
echo Example 3: Visualize Detection Dataset
echo --------------------------------------
echo Dataset: %DATASET_NAME%
echo Root Directory: %ROOT_DIR%
echo Split: train
echo.

REM Run the visualize_detection command
call uv run --no-sync wildata visualize-detection %DATASET_NAME% --root %ROOT_DIR% --split train
call uv run --no-sync fiftyone datasets stats "%DATASET_NAME%-train"


call pause