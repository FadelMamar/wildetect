REM Repo root; Fire subcommand is "run" before flags.
call cd /d "%~dp0" && cd ..

call uv run scripts/tile_gps_matching.py run --config=config/tile-gps-matching.yaml

call pause
