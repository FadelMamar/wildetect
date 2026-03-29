call cd /d "%~dp0" && cd ..


@REM call uv run  wildtrain register classifier --config wildtrain/configs/registration/classifier_registration_example.yaml

call uv run  wildtrain register detector configs\registration\detector_registration_example.yaml

