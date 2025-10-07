call cd /d %~dp0
call cd ..

call .venv-ls\Scripts\activate && ^
(
call load_env.bat
call set LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%LOCAL_FILES_DOCUMENT_ROOT%
call set LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
call set LOCAL_FILES_DOCUMENT_ROOT=%LOCAL_FILES_DOCUMENT_ROOT%

call label-studio start -p 8080

call deactivate
) || (
    echo Failed to start Label Studio. Virtual environment could not be activated. &&
    pause
)