call cd /d "%~dp0" && cd ..


call uv run wildetect detection detect  -c config/detection.yaml

call pause

  
@REM --profile --gpu-profile