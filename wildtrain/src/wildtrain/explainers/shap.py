from timm.models.nfnet import model_cfgs
import torch
import shap
from typing import Optional
from wildtrain.models.classifier import GenericClassifier
from wildtrain.data.classification_datamodule import ClassificationDataModule

class ClassifierSHAPExplainer:
    """
    SHAP DeepExplainer for GenericClassifier.
    
    Args:
        model: Trained GenericClassifier instance.
        data_module: Initialized ClassificationDataModule (setup must be called).
        background_loader: 'train' or 'val' (which dataloader to use for background).
        background_samples: Number of background samples to use for SHAP.
        device: 'cpu' or 'cuda'.

    """
    def __init__(self,
                 checkpoint_path: str,
                 data_module: ClassificationDataModule,
                 background_loader: str = 'train',
                 background_samples: int = 100,
                 model: Optional[GenericClassifier] = None,
                 device: Optional[str] = None):

        self.data_module = data_module
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path=checkpoint_path,model=model)
        self.background = self._sample_background(background_loader, background_samples)
        self.explainer = shap.GradientExplainer(self.model, self.background)

    def _load_model(self, checkpoint_path: str, model: Optional[GenericClassifier] = None) -> torch.nn.Module:
        if model is None:
            model  = GenericClassifier.load_from_checkpoint(checkpoint_path,map_location=self.device)

        model = torch.nn.Sequential(
                                    model,
                                    torch.nn.Softmax(dim=1)
                )
        model = model.to(self.device)
        model.eval()
        model.to(self.device)
        return model

    def _sample_background(self, loader_type: str, n: int=100) -> torch.Tensor:
        if loader_type == 'train':
            loader = self.data_module.train_dataset
        elif loader_type == 'val':
            loader = self.data_module.val_dataset
        else:
            raise ValueError("loader_type must be 'train' or 'val'")
        images = []
        for i in range(n):
            imgs, _ = loader[i]
            images.append(imgs)
        images = torch.stack(images)
        return images.to(self.device)

    def explain(self, images: torch.Tensor):
        images = images.to(self.device)
        shap_values = self.explainer.shap_values(images)
        return shap_values
