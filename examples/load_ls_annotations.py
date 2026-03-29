import os
from wildetect.core.visualization import LabelStudioManager
from dotenv import load_dotenv

load_dotenv('../.env',override=True)

client = LabelStudioManager(url=os.environ["LABEL_STUDIO_URL"], 
                            api_key=os.environ["LABEL_STUDIO_API_KEY"],
                            )
for i in range(141,196):
    try:
        #client.delete_project(i)
        pass
    except Exception as e:
        print(f"Error deleting project {i}: {e}")

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

