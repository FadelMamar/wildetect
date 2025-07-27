call cd /d %~dp0
call cd ..

call .venv-ls\Scripts\activate && ^
(
if exist .env (
    echo Loading .env file...
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        REM Skip lines starting with # (comments)
        echo %%a | findstr /r "^#" >nul
        if errorlevel 1 (
            REM Skip empty lines
            if not "%%a"=="" (
                set "%%a=%%b"
                @REM echo Set %%a=%%b
            )
        )
    )
) else (
    echo .env file not found
)

call set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%LOCAL_FILES_DOCUMENT_ROOT%
call set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
call set LOCAL_FILES_DOCUMENT_ROOT=%LOCAL_FILES_DOCUMENT_ROOT%

call label-studio start -p 8080

call deactivate
) || (
    echo Failed to start Label Studio. Virtual environment could not be activated. &&
    pause
)