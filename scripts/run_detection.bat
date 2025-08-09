call cd /d "%~dp0" && cd ..


call set "model_path=D:\PhD\workspace\wildetect\models\labeler\9\artifacts\best.pt"
call set "roi_weights_path=D:\PhD\workspace\wildetect\models\classifier\2\artifacts\best.ckpt-v6.torchscript"
@REM call set "inference_service_url=http://localhost:4141/predict"
call set IMAGE_DIR="D:\PhD\Data per camp\Dry season\Kapiri\Farm\DJI_202310040946_001_KapiriFarm1"
@REM --inference-service-url %inference_service_url%^

call uv run wildetect detect %IMAGE_DIR%^
  --model %model_path%^
  --output "results"^
  --device "auto"^
  --roi-weights %roi_weights_path%^
  --pipeline-type "single" --queue-size 128 ^
  --overlap-ratio 0.2^
  --batch-size 32^
  --tile-size 800^
  --cls-imgsz 128^
  --sensor-height 24.0^
  --focal-length 35.0^
  --flight-height 180.0^

call pause

  
@REM --profile --gpu-profile