call cd /d "%~dp0" && cd ..

@REM Do not forget to launch Fiftyone if you want to visualize the results

call uv run wildetect detection detect  -c config/detection.yaml

call pause
  
@REM --profile --gpu-profile