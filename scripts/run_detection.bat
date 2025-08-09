call cd /d "%~dp0" && cd ..


call wildetect detection detect  -c config/detection.yaml -m weights\best.pt ^
                                 --roi-weights weights\roi_classifier.torchscript ^

call pause

  
@REM --profile --gpu-profile