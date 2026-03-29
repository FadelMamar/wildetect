@echo off
REM Script to run the WildData Streamlit UI

echo Starting WildData Streamlit UI...

REM Install UI dependencies if needed

REM Run the Streamlit app
call uv run streamlit run src/wildata/ui.py --server.port 8544 --server.address localhost

call pause
