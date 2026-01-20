call cd /d %~dp0 && cd ..
call uv run scripts\custom_iou_tunner.py run_duplicate_tuner --csv_path="animal-duplicates.csv"^ 
        --n_trials=150 --iou_min=0.1 --iou_max=0.9^
        --overlap_min=0.0 --overlap_max=0.5