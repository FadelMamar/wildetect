call cd /d "%~dp0" && cd ..
call .venv\Scripts\activate

call wildetect detection census -c config/census.yaml -m weights\best.pt ^
                                 --roi-weights weights\roi_classifier.torchscript ^
                                 --images D:\workspace\data\savmap_dataset_v2\raw\tmp
  
@REM --profile --gpu-profile