@echo off
@REM WildDetect CLI wrapper for Windows
@REM This script runs wildetect as a Python module to avoid entry point issues

uv run python -m wildetect.cli %*