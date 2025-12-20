# from mmengine.registry import MODELS,DATASETS,RUNNERS

from mmengine.config import Config
from mmdet.registry import *
from mmdet.models.data_preprocessors import DetDataPreprocessor


def build_model(config_file: str):
    mmdet_cfg = Config.fromfile(config_file)
    model_cfg = mmdet_cfg.model
    model = MODELS.build(model_cfg)
    return model
