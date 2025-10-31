from wildtrain.utils.mlflow import load_registered_model
from wildtrain.models.detector import Detector
import torch
from omegaconf import OmegaConf


def load_detector():
    model = load_registered_model(alias='demo',name='detector',load_unwrapped=True)

    print(model.metadata)
    #print(model.localizer is None,model.classifier is None)

    print(model.predict(torch.rand(1,3,640,640),return_as_dict=True))
    

def load_from_config():
    localizer_cfg = OmegaConf.load("models-registry/detector/10/artifacts/localizer_config.yaml")
    localizer_cfg.weights = "models-registry/detector/10/artifacts/best.pt"
    model = Detector.from_config(localizer_config=localizer_cfg,
                                        classifier_ckpt="models-registry/detector/10/artifacts/best.ckpt")
    print(model.localizer is None,model.classifier is None)

    print(model.predict(torch.rand(1,3,640,640),return_as_dict=False))

#load_from_config()

load_detector()