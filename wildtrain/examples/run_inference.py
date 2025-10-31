from wildtrain.models.detector import Detector
import torch
import json
import requests

url = "http://localhost:4141/predict"
data = torch.rand(1,3,800,800)
out = Detector.predict_inference_service(data,url=url)
print(out)


#payload = json.dumps({"instances": torch.rand(1,3,800,800).tolist()})
#response = requests.post(
#    url=f"http://localhost:4141/invocations",
#    data=payload,
#    headers={"Content-Type": "application/json"},
#)
#print(response.json())








