from wildetect.core.visualization import LabelStudioManager
from dotenv import load_dotenv
import os

load_dotenv(r'D:\workspace\repos\wildetect\.env')

cfg = dict(url=os.getenv('LABEL_STUDIO_URL'), 
                            api_key=os.getenv('LABEL_STUDIO_API_KEY'
                                              ))
ls_mgr = LabelStudioManager(**cfg)

ls_client = ls_mgr.client

label_config = ls_mgr.label_config

project_name = "example-full-savmap"
resp = ls_client.projects.create(title=project_name,label_config=label_config)

project_id = resp.id
resp_storage = ls_client.import_storage.local.create(project=project_id,
                                      path=r"D:\workspace\data\savmap_dataset_v2\raw\images",
                                      use_blob_urls=True,
                                      # synchronizable=True
                                     )

resp_sync = ls_client.import_storage.local.sync(id=resp_storage.id)

tasks = ls_client.tasks.list(project=project_id,)
mapping = {task.storage_filename: task.id for task in tasks}

