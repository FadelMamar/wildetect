@echo off
@REM WildDetect CLI wrapper for Windows
call cd /d %~dp0
call uv run wildetect %*