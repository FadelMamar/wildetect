@echo off
@REM WildDetect CLI wrapper for Windows
call cd /d %~dp0 && cd ..
call uv run wildetect %*