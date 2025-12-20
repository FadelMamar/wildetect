"""
Simplified Curriculum Learning Interface for WildTrain.

This module provides a unified interface for curriculum learning strategies:
- Difficulty-based curriculum learning
- Multi-scale curriculum learning
- Combined strategies

Usage:
    from wildtrain.data.curriculum import CurriculumConfig, CurriculumDataModuleMixin
    
    # Configure curriculum
    config = CurriculumConfig(
        enabled=True,
        type="difficulty",
        difficulty_strategy="linear",
        start_difficulty=0.0,
        end_difficulty=1.0
    )
    
    # Add to data module
    class MyDataModule(CurriculumDataModuleMixin, L.LightningDataModule):
        def __init__(self, curriculum_config=None, **kwargs):
            CurriculumDataModuleMixin.__init__(self, curriculum_config)
            L.LightningDataModule.__init__(self)
            # ... rest of initialization
"""

from ...shared.models import CurriculumConfig
from .manager import CurriculumManager
from .mixins import CurriculumDataModuleMixin
from .callback import CurriculumCallback
from .dataset import (
    CurriculumDetectionDataset, 
    PatchDataset
)

__all__ = [
    "CurriculumConfig",
    "CurriculumManager", 
    "CurriculumDataModuleMixin",
    "CurriculumCallback",
    "CurriculumDetectionDataset",
    "PatchDataset"
]
