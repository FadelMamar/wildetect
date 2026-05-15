#!/bin/bash
# Download data from Dryad

uv run scripts/upload_coco_dataset_hf.py upload_all --images_dir="data/Dry_Leopard_rock" \
    --json_dir="data/Annotations/Dry season - Rep 1/all/coco-format" \
    --repository="fadel841/wildetect"