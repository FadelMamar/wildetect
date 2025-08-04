call cd /d "%~dp0" && cd ..
call .venv\Scripts\activate

@REM --roi-weights "D:\PhD\workspace\wildetect\models\classifier\2\artifacts\best.ckpt-v6.torchscript"^
@REM --inference-service-url "http://localhost:4141/predict"^

call wildetect detect "D:\workspace\data\savmap_dataset_v2\raw\images"^
  --model "D:\workspace\repos\wildetect\weights\best.pt"^
  --output "results"^
  --device "auto"^
  --inference-service-url "http://localhost:4141/predict"^
  --pipeline-type "single" --queue-size 10 ^
  --overlap-ratio 0.2^
  --batch-size 32^
  --tile-size 800^
  --cls-imgsz 128^
  --sensor-height 24.0^
  --focal-length 35.0^
  --flight-height 180.0^

  
@REM --profile --gpu-profile