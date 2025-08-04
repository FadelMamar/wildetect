call cd /d "%~dp0" && cd ..
call .venv\Scripts\activate

@REM --roi-weights "D:\PhD\workspace\wildetect\models\classifier\2\artifacts\best.ckpt-v6.torchscript"^

call wildetect detect "D:\PhD\Data per camp\Dry season\Kapiri\Farm\DJI_202310040946_001_KapiriFarm1"^
  --model "D:\PhD\workspace\wildetect\models\labeler\9\artifacts\best.pt"^
  --roi-weights "D:\PhD\workspace\wildetect\models\classifier\2\artifacts\best.ckpt-v6.torchscript"^
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

call pause

  
@REM --profile --gpu-profile