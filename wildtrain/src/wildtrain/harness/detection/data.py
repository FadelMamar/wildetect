"""Dataset YAML preparation for detection training.

This is the same merge-and-filter sequence the trainer used to inline.
"""

from pathlib import Path

from ...data.filters.algorithms import FilterDataCfg
from ...shared.schemas.detection import DetectionConfig
from ...utils.io import merge_data_cfg


def resolve_detection_data_cfg(
    config: DetectionConfig, save_dir: Path
) -> FilterDataCfg:
    """Point ``config.dataset.data_cfg`` at one YOLO yaml and return its filter.

    Exactly one of ``root_data_directory`` or ``data_cfg`` must already be set.
    A root is merged into ``save_dir`` before filtering.
    """
    has_cfg = config.dataset.data_cfg is not None
    has_root = config.dataset.root_data_directory is not None
    if has_cfg == has_root:
        raise ValueError("Either root_data_directory or data_cfg must be provided")

    if config.dataset.root_data_directory is not None:
        data_cfg = save_dir / f"{config.name}_merged.yaml"
        merge_data_cfg(
            root_data_directory=config.dataset.root_data_directory,
            output_path=data_cfg,
            force_merge=config.dataset.force_merge,
        )
        config.dataset.data_cfg = data_cfg.as_posix()

    return FilterDataCfg(
        data_config_yaml=config.dataset.data_cfg,
        keep_classes=config.dataset.keep_classes,
        discard_classes=config.dataset.discard_classes,
    )
