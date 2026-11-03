call cd /d "%~dp0" && cd ..

call uv run wildtrain register -c configs/registration/classifier_registration_example.yaml
call uv run wildtrain register -c configs/registration/detector_registration_example.yaml
