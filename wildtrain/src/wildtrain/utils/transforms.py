"""
Transform utilities for creating torchvision transforms from configuration.
"""

import traceback
from typing import Any
from omegaconf import DictConfig


def create_transforms(transforms: dict[str, Any]) -> dict[str, Any]:
    """
    Create transforms for training and validation from a dictionary.

    Args:
        transforms: Dictionary with 'train' and 'val' keys, each containing a list of
                   transformation configurations. Each config should have:
                   - 'name': torchvision transformation class name (e.g., 'RandomResizedCrop')
                   - 'params': dictionary of parameters for the transformation

    Returns:
        Dictionary with 'train' and 'val' keys containing composed transforms
    """
    import torchvision.transforms.v2 as T
    import torchvision.transforms.functional as F

    def create_transform_list(transform_configs: list) -> T.Compose:
        """Create a list of transforms from configuration."""
        transform_list = []

        for config in transform_configs:
            if isinstance(config, str):
                # Simple case: just the transform name
                transform_name = config
                params = {}
            elif isinstance(config, (dict, DictConfig)):
                # Full configuration with parameters - convert DictConfig to dict if needed
                if isinstance(config, DictConfig):
                    config = dict(config)

                transform_name = config.get("name", config.get("type"))
                params = config.get("params", config.get("kwargs", {}))

                # Convert DictConfig params to regular dict if needed
                if isinstance(params, DictConfig):
                    params = dict(params)

                # Ensure params is a proper dictionary with string keys
                if not isinstance(params, dict):
                    params = {}
                else:
                    # Convert any non-string keys to strings
                    params = {str(k): v for k, v in params.items()}

                if transform_name is None:
                    raise ValueError(
                        f"Transform config missing 'name' or 'type' key: {config}"
                    )
            else:
                raise ValueError(f"Invalid transform config: {config}")

            # Get the transform class from torchvision
            if hasattr(T, transform_name):
                transform_class = getattr(T, transform_name)
            elif hasattr(F, transform_name):
                # For functional transforms, we need to handle them differently
                raise ValueError(
                    f"Functional transforms like {transform_name} are not supported yet"
                )
            else:
                raise ValueError(f"Unknown transform: {transform_name}")

            # Create the transform instance with parameters
            for key, value in params.items():
                if isinstance(value, (list, tuple)) and not isinstance(value, str):
                    params[key] = list(value)
            try:
                transform_instance = transform_class(**params)
                transform_list.append(transform_instance)
            except Exception as e:
                raise ValueError(
                    f"Failed to create transform {transform_name} with params {params}: {traceback.format_exc()}"
                )

        return T.Compose(transform_list)

    result = {}

    # Create train transforms
    if "train" in transforms:
        result["train"] = create_transform_list(transforms["train"])
    else:
        # Default train transforms
        result["train"] = T.Compose([T.ToTensor()])

    # Create validation transforms
    if "val" in transforms:
        result["val"] = create_transform_list(transforms["val"])
    else:
        # Default validation transforms
        result["val"] = T.Compose([T.ToTensor()])

    return result 