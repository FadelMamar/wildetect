call .venv\Scripts\activate
call dvc init
call dvc remote add -d dataregistry s3://dvc-storage
call dvc remote modify dataregistry endpointurl http://localhost:9000
call dvc remote modify --local dataregistry access_key_id minioadmin
call dvc remote modify --local dataregistry secret_access_key minioadmin

:: Disable SSL (for local development)
call dvc remote modify dataregistry ssl_verify false
