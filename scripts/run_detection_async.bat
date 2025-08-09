call cd /d "%~dp0" && cd ..


call uv run wildetect detection detect --pipeline-type async -c config/detection.yaml -m weights\best.pt ^
                                 --roi-weights weights\roi_classifier.torchscript ^

call pause

  
@REM --profile --gpu-profile