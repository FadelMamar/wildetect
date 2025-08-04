call cd /d "%~dp0" && cd ..
call .venv\Scripts\activate


call set "model_path=D:\workspace\repos\wildetect\weights\best.pt"
call set "roi_weights_path=D:\workspace\repos\wildetect\weights\best.pt"
call set "inference_service_url=http://localhost:4141/predict"
call set "IMAGE_DIR=D:\workspace\data\savmap_dataset_v2\raw\tmp"
@REM --inference-service-url %inference_service_url%^

call wildetect detect %IMAGE_DIR%^
  --model %model_path%^
  --output "results"^
  --device "auto"^
  --roi-weights %roi_weights_path%^
  --pipeline-type "single" --queue-size 10 ^
  --overlap-ratio 0.2^
  --batch-size 32^
  --tile-size 800^
  --cls-imgsz 128^
  --sensor-height 24.0^
  --focal-length 35.0^
  --flight-height 180.0^

call pause

  
@REM --profile --gpu-profile