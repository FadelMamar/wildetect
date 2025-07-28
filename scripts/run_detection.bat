call cd /d "%~dp0" && cd ..
call .venv\Scripts\activate

@REM --roi-weights "D:\PhD\workspace\wildetect\models\classifier\2\artifacts\best.ckpt-v6.torchscript"^

call wildetect detect "D:\PhD\Data per camp\tmp"^
  --model "D:\PhD\workspace\wildetect\models\labeler\9\artifacts\best.pt"^
  --output "results"^
  --device "auto" 
  
@REM --profile --gpu-profile