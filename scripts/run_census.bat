call cd /d "%~dp0" && cd ..
call .venv\Scripts\activate

call wildetect detection census -c config/census.yaml

call pause

@REM --profile --gpu-profile