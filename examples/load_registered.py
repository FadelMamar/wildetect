from wildtrain.utils.mlflow import load_registered_model
from wildetect.core.data import Detection
import torch

model,metadata = load_registered_model(alias='demo',name='detector',load_unwrapped=True)

print(metadata)
print(model.classifier)

batch = metadata['batch']
imgsz = metadata['imgsz']

detections = model.predict(torch.rand(1,3,imgsz,imgsz),return_as_dict=False)

print(detections)

print(Detection.from_supervision(detections[0]))

