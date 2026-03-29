if exist .env (
    echo Loading .env file...
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        REM Skip lines starting with # (comments)
        @REM echo %%a | findstr /r "^#" >nul
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
    exit /b 1
)