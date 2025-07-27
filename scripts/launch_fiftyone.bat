call cd /d %~dp0
call cd ..

call .venv\Scripts\activate.bat
call uv run --no-sync fiftyone app launch
call pause