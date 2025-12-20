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


