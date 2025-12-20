import fiftyone as fo
import sys
import os
from wildtrain.visualization import add_predictions_from_detector
from wildtrain.models.localizer import UltralyticsLocalizer
from wildtrain.models.classifier import GenericClassifier
from wildtrain.models.detector import Detector
from omegaconf import OmegaConf

def main(config: str):
    # Load configuration
    cfg = OmegaConf.load(config)
        
    print(OmegaConf.to_yaml(cfg))
    
    # Extract configuration values
    dataset_name = cfg.fiftyone.dataset_name
    prediction_field = cfg.fiftyone.prediction_field
    
    localizer_cfg = cfg.model.localizer
    classifier_cfg = cfg.model.classifier
    processing_cfg = cfg.processing
    
    print(f"[bold green]Uploading detector predictions to FiftyOne dataset:[/bold green] {dataset_name}")
    
    # Create localizer with config
    localizer = UltralyticsLocalizer(
        weights=localizer_cfg.weights,
        imgsz=localizer_cfg.imgsz,
        device=localizer_cfg.device,
        conf_thres=localizer_cfg.conf_thres,
        iou_thres=localizer_cfg.iou_thres,
        max_det=localizer_cfg.max_det,
        overlap_metric=localizer_cfg.overlap_metric
    )
    
    # Create classifier if checkpoint provided
    classifier = None
    if classifier_cfg.checkpoint is not None:
        print(f"[bold blue]Loading classifier from:[/bold blue] {classifier_cfg.checkpoint}")
        classifier = GenericClassifier.load_from_checkpoint(str(classifier_cfg.checkpoint))
    
    # Create detector
    detector = Detector(localizer=localizer, classifier=classifier)
    
    add_predictions_from_detector(
        dataset_name=dataset_name,
        detector=detector,
        imgsz=localizer_cfg.imgsz,
        prediction_field=prediction_field,
        batch_size=processing_cfg.batch_size,
        debug=processing_cfg.debug,
    )

if __name__ == "__main__":
    main(config=r"D:\workspace\repos\wildtrain\configs\detection\visualization.yaml")