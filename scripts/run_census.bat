call cd /d "%~dp0" && cd ..
call .venv\Scripts\activate

@REM Do not forget to launch Label Studio and Fiftyone if you want to visualize the results

if exist .env (
    call .\scripts\load_env.bat
    call uv run wildetect detection census -c config/census.yaml
    pause

) else (
    echo .env file not found. Please provide it in the root directory.
    pause
)

call pause