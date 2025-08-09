call cd /d "%~dp0" && cd ..
call .venv\Scripts\activate

call wildetect detection census -c config/census.yaml -m weights\best.pt ^
                                 --roi-weights weights\roi_classifier.torchscript ^

call pause

@REM --profile --gpu-profile