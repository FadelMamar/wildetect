call cd /d "%~dp0" && cd ..

:: Detection pipeline
call uv run pytest tests/test_detection_pipeline.py::TestDetectionPipeline::test_detection_pipeline_with_real_images -v

:: Data loading
call uv run pytest tests/test_data_loading.py -v

