call cd /d "%~dp0" && cd ..


call uv run  wildtrain register classifier --config wildtrain/configs/registration/classifier_registration_example.yaml

call uv run  wildtrain register detector wildtrain/configs/registration/detector_registration_example.yaml

