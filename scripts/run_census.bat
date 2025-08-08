call cd /d "%~dp0" && cd ..
call .venv\Scripts\activate

@REM --roi-weights "D:\PhD\workspace\wildetect\models\classifier\2\artifacts\best.ckpt-v6.torchscript"^

call wildetect detection census -c config/census.yaml -m weights\best.pt ^
                                 --roi-weights weights\roi_classifier.torchscript ^
                                 --images D:\workspace\data\savmap_dataset_v2\raw\tmp
  
@REM --profile --gpu-profile