call cd /d "%~dp0" && cd ..

@REM Running API tests
call uv run pytest tests/api/test_basic.py -v -s
call uv run pytest tests/api/test_endpoints.py -v -s
