@echo off
REM Example: Convert a Label Studio JSON export to COCO JSON (wildata convert-ls-to-coco)
call cd /d "%~dp0" && cd ..

REM Edit: path to your Label Studio export (.json)
set LS_JSON=D:\PhD\Data per camp\Exported annotations and labels\Harvard data\200_Sabie_granite.json

REM Default output is <input_stem>_coco.json beside the input. Uncomment to set output explicitly:
::set OUT_OPT=--out-file "path\to\annotations_coco.json"

call uv run --env-file ..\.env wildata convert-ls-to-coco "%LS_JSON%" --verbose --parse-ls-config

REM Optional: category IDs from Label Studio XML labeling config
::call uv run --env-file ..\.env wildata convert-ls-to-coco "%LS_JSON%" %OUT_OPT% --ls-xml-config "configs\label_studio_config.xml" --verbose

REM Optional: derive categories from Label Studio API (LABEL_STUDIO_URL / LABEL_STUDIO_API_KEY in ..\.env)
::call uv run --env-file ..\.env wildata convert-ls-to-coco "%LS_JSON%" %OUT_OPT% --parse-ls-config --verbose

call pause
