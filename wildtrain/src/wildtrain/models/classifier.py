import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torch.nn.functional as F
from torch.export import Dim
from pathlib import Path
import timm
from typing import Optional,Tuple,Union,Dict,Any
import traceback
import json
import onnxruntime as ort
import tempfile

from ..utils.logging import get_logger
from ..utils.mlflow import load_registered_model

logger = get_logger(__name__)

class ResizeNormalize(nn.Module):
    def __init__(self, input_size: int, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer('mean', mean.view(1, 3, 1, 1))
        self.register_buffer('std', std.view(1, 3, 1, 1))
        self.input_size = input_size
        assert isinstance(self.input_size,int), f"input_size must be an integer, received {type(self.input_size)}"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(
            x, 
            size=(self.input_size, self.input_size), 
            mode='bicubic', 
            align_corners=False
        )
        # Normalize using registered buffers
        x = (x - self.mean) / self.std
        return x

class BackboneNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', mean.view(1, 3, 1, 1))
        self.register_buffer('std', std.view(1, 3, 1, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.2,hidden_dim:int=128,num_layers:int=2):
        super().__init__()
        layers = []
        for i in range(num_layers-1):
            if i == 0:
                layers.append(nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout)
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout)
                ))
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class GenericClassifier(nn.Module):
    """
    Generic classifier model supporting timm and MMPretrain backbones.
    """

    def __init__(
        self,
        label_to_class_map: dict,
        backbone: str = "timm/vit_base_patch14_reg4_dinov2.lvd142m",
        backbone_source: str = "timm",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = True,
        input_size: int = 224,
        mean: list[float] = [0.554, 0.469, 0.348],
        std: list[float] = [0.203, 0.173, 0.144],
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.label_to_class_map = label_to_class_map
        self.metadata: Optional[Dict[str,Any]] = None


        # register buffers
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.register_buffer("input_size", torch.tensor([input_size]).int())
        self.register_buffer(
            "num_classes", torch.tensor([len(label_to_class_map)]).int()
        )
        self.register_buffer("num_layers", torch.tensor([num_layers]).int())
        self.register_buffer("hidden_dim", torch.tensor([hidden_dim]).int())
        self.register_buffer("dropout", torch.tensor([dropout]).float())

        self._check_arguments()

        self.backbone = backbone
        self.backbone_source = backbone_source
        self.pretrained = pretrained

        self.backbone_trfs,self.feature_extractor = self._get_backbone()
        self.fc = ClassifierHead(input_dim=self.feature_extractor.num_features, 
                                num_classes=self.num_classes.item(), 
                                dropout=self.dropout.item(), 
                                hidden_dim=self.hidden_dim.item(), 
                                num_layers=self.num_layers.item())

        # Initialize preprocessing
        self.preprocessing = ResizeNormalize(input_size=self.input_size.item(), mean=self.mean, std=self.std)

        self._onnx_program: Optional[ort.InferenceSession] = None
        self._onnx_providers=[]
        if torch.cuda.is_available():
            self._onnx_providers=['CUDAExecutionProvider']
        self._onnx_providers.append('CPUExecutionProvider')
        return None
    
    def set_onnx_program(self,
                    onnx_program:Optional[ort.InferenceSession]=None):
        self._onnx_program = onnx_program

    def _check_arguments(
        self,
    ):
        if self.num_classes is None:
            raise ValueError("num_classes must be provided")
        if not isinstance(self.input_size, torch.Tensor):
            raise ValueError("input_size must be a tensor")
        if not isinstance(self.mean, torch.Tensor):
            raise ValueError("mean must be a tensor")
        if not isinstance(self.std, torch.Tensor):
            raise ValueError("std must be a tensor")

    def _get_backbone(
        self,
    ) -> Tuple[nn.Module,nn.Module]:

        if self.backbone_source != "timm":
            ValueError(f"Unsupported backbone source: {self.backbone_source}")

        # Create timm model
        model = timm.create_model(
            self.backbone, pretrained=self.pretrained, num_classes=0,global_pool=''
        )
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        norm_mean = torch.tensor(data_cfg.get('mean', self.mean))
        norm_std = torch.tensor(data_cfg.get('std', self.std))
        trfs = BackboneNormalize(norm_mean, norm_std)

        # Set input size    
        try:
            model.set_input_size((self.input_size.item(),self.input_size.item()))
        except Exception:
            logger.info(f"Backbone {self.backbone} does not support setting input size")

        # Freeze backbone
        if self.freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            logger.info(f"Backbone {self.backbone} frozen")
        return trfs,model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocessing(x.float())
        x = self.backbone_trfs(x)
        if "vit" in self.backbone: # get CLS token for ViT models
            x = self.feature_extractor(x)[:,0]
        else:
            x = self.feature_extractor(x)
        return self.fc(x)
    
    def export(self,mode:str,output_path:str,batch_size:int=8)->"GenericClassifier":
        if mode == "torchscript":
            return self.to_torchscript(output_path=output_path)
        elif mode == "onnx":
            return self.to_onnx(output_path=output_path,batch_size=batch_size)
        else:
            raise ValueError(f"Unsupported export mode: {mode}")

    def to_torchscript(self,output_path:Optional[str]=None)->"GenericClassifier":
        self.eval()
        for module in [self.feature_extractor, self.fc, self.preprocessing, self.backbone_trfs]:
            if isinstance(module, nn.Module):
                module = torch.jit.script(module)
        if output_path is not None:
            torch.jit.save(module,output_path)
        return self
    
    def to_onnx(self, 
                output_path:str,
                report:bool=False,
                batch_size:int=8,
    )->"GenericClassifier":
        self.eval()
        b,c,h,w = batch_size,3,self.input_size.item(),self.input_size.item()
        x = torch.rand(b,c,h,w,dtype=torch.float32)
        self.forward(x)
        
        # export program
        dynamic_shapes={"x": {0: Dim("batch_size",min=1,max=b),
                                1:c,
                                2:h,
                                3:w
                            }
                        }
        #exported_program = torch.export.export(self,(x,),dynamic_shapes=dynamic_shapes)

        # export onnx
        onnx_program = self._export_onnx(shape=(b,c,h,w),report=report)
        self._save_onnx_program(onnx_program,output_path=output_path)
        self._onnx_program = ort.InferenceSession(output_path,providers=self._onnx_providers)
        return self

    def _export_onnx(self,shape:Tuple[int,int,int,int],
                    report:bool=False)->torch.onnx.ONNXProgram:
        b,c,h,w = shape
        x = torch.rand(b,c,h,w,dtype=torch.float32)
        cfg = dict(dynamic_shapes={"input": {0: Dim("batch_size",min=1,max=b)},
                   "output": {0: Dim("batch_size",min=1,max=b)}})
        onnx_program = torch.onnx.export(self, x, input_names=["input"],
                        output_names=["output"],
                        dynamo=True,
                        report=report,
                        **cfg
                        )
        onnx_program.optimize()
        self.load_as_onnx = True
        self._onnx_batch_size = b
        return onnx_program
        
    def _save_onnx_program(self,onnx_program:torch.onnx.ONNXProgram,
                        #exported_program:torch.export.ExportedProgram,
                        #shape:Tuple[int,int,int,int],
                        output_path:Union[str,Path]):
        try:
            onnx_program.save(Path(output_path).with_suffix(".onnx"))
            #exported_program_path = Path(output_path).with_suffix(".exported_program.pt")
            #torch.export.save(exported_program, exported_program_path,extra_files={"shape":json.dumps(list(shape))})
        except Exception:
            logger.error(f"Failed to save ONNX program to {output_path}. {traceback.format_exc()}")
    
    def _predict_onnx(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(self._onnx_program, ort.InferenceSession), "ONNX program not created. Call to_onnx() first."
        out = self._onnx_program.run(['output'], {'input': x.cpu().numpy()})
        return torch.from_numpy(out[0])

    def predict(self, x: torch.Tensor) -> list[dict]:
        if self._onnx_program is not None:
            logits = self._predict_onnx(x)
        else:
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
        
        probs = torch.softmax(logits, dim=1)
        labels = probs.argmax(dim=1).cpu().tolist()
        classes = [
            self.label_to_class_map.get(i,"None") for i in labels
        ]
        scores = probs.max(dim=1).values.cpu().tolist()
        return [{"class": c, "score": s, "class_id":l} for c, s, l in zip(classes, scores, labels)]
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, map_location: str = "cpu",label_to_class_map:Optional[dict]=None):
        if str(checkpoint_path).endswith(".onnx"):
            assert isinstance(label_to_class_map,dict), "label_to_class_map must be provided if checkpoint_path is an ONNX file"
            return cls._load_onnx(onnx_path=checkpoint_path,label_to_class_map=label_to_class_map)
        try:
            return cls._load_from_lightning_ckpt(checkpoint_path=checkpoint_path, map_location=map_location)
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {checkpoint_path} using lightning. {e}")
            return cls._load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=map_location,)

    @classmethod
    def _load_from_lightning_ckpt(cls, checkpoint_path: str, map_location: str = "cpu"):
        """Load from a PyTorch Lightning checkpoint by extracting the underlying model."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Check if this is a Lightning checkpoint
        if 'state_dict' in checkpoint and 'hyper_parameters' in checkpoint:
            # Extract the model state dict from the Lightning checkpoint
            state_dict = checkpoint['state_dict']
            
            # Remove the 'model.' prefix from state dict keys
            model_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.')}
            
            # Get hyperparameters from the Lightning module
            hparams = checkpoint['hyper_parameters']
            
            # Create model instance using saved hyperparameters
            model = cls(
                label_to_class_map=hparams['label_to_class_map'],
                backbone=hparams['backbone'],
                backbone_source=hparams['backbone_source'],
                input_size=hparams['input_size'],
                mean=hparams['mean'],
                std=hparams['std'],
                dropout=hparams['dropout'],
                freeze_backbone=hparams['freeze_backbone'],
                num_layers=hparams['num_layers'],
                hidden_dim=hparams['hidden_dim'],
            )
            # Initialize the model
            model(torch.rand(1, 3, hparams['input_size'], hparams['input_size']))  
            
            # Load the model weights
            model.load_state_dict(model_state_dict)
            return model
        else:
            raise ValueError("Checkpoint does not appear to be a valid PyTorch Lightning checkpoint")

    @classmethod
    def _load_from_checkpoint(cls, checkpoint_path: Optional[str]=None, map_location: str = "cpu",state_dict:Optional[dict]=None) -> 'GenericClassifier':
        
        if state_dict is None:
            assert checkpoint_path is not None, "checkpoint_path must be provided if state_dict is not provided"
            state_dict = torch.load(checkpoint_path, map_location=map_location)

        label_to_class_map = state_dict["label_to_class_map"]
        backbone = state_dict["backbone"]
        backbone_source = state_dict["backbone_source"]
        input_size = state_dict["input_size"].item()
        model = cls(
            label_to_class_map=label_to_class_map,
            backbone=backbone,
            backbone_source=backbone_source,
            input_size=input_size,
        )

        # Initialize the model
        model(torch.rand(1, 3,input_size, input_size))

        model.load_state_dict(state_dict)

        return model
    
    @classmethod
    def _load_onnx(cls,onnx_path:str,label_to_class_map: dict)->"GenericClassifier":
        model = cls(label_to_class_map=label_to_class_map,pretrained=False)
        ort_sess = ort.InferenceSession(onnx_path,providers=model._onnx_providers)
        model._onnx_program = ort_sess
        return model
    
    @classmethod
    def from_mlflow(cls,name:str,alias:str,dwnd_location:Optional[str]=None,mlflow_tracking_uri:str="http://localhost:5000")->'GenericClassifier':
        model = load_registered_model(alias=alias,name=name,
                                    load_unwrapped=True,
                                    dwnd_location=dwnd_location,
                                    mlflow_tracking_url=mlflow_tracking_uri)

        export_format = model.metadata.get("cls_export_format")
        if export_format is not None:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "classifier.onnx"
                model.export(mode=export_format,
                                        batch_size=model.metadata["batch"],
                                        output_path=temp_path)
        return model