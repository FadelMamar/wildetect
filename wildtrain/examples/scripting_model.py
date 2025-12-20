from wildtrain.models.classifier import GenericClassifier
import torch

model = GenericClassifier(
    label_to_class_map={0: "dog", 1: "cat"},
    backbone="timm/vit_base_patch14_reg4_dinov2.lvd142m",
    backbone_source="timm",
    pretrained=True,
    dropout=0.2,
    freeze_backbone=True,
    input_size=224,
)

model(torch.randn(1, 3, 224, 224))

# model = torch.jit.script(model)

#torch.save(model, "models/classifier_demo.pt")