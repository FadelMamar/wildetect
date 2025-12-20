"""
Curriculum learning mixins for data modules.

This module provides mixin classes that add curriculum learning capabilities
to existing data modules.
"""

from typing import Optional, Dict, Any
import lightning as L

from .manager import CurriculumManager
from ...shared.models import CurriculumConfig


class CurriculumDataModuleMixin:
    """
    Mixin to add difficulty-based curriculum learning to any Lightning DataModule.
    
    Usage:
        class MyDataModule(CurriculumDataModuleMixin, L.LightningDataModule):
            def __init__(self, curriculum_config=None, **kwargs):
                CurriculumDataModuleMixin.__init__(self, curriculum_config)
                L.LightningDataModule.__init__(self)
                # ... rest of initialization
    """
    
    def __init__(self, curriculum_config: Optional[CurriculumConfig] = None):
        """
        Initialize curriculum learning mixin.
        
        Args:
            curriculum_config: Configuration for curriculum learning. If None, curriculum is disabled.
        """
        self.curriculum_config = curriculum_config
        self.curriculum_manager: Optional[CurriculumManager] = None
        
        if curriculum_config and curriculum_config.enabled:
            self.curriculum_manager = CurriculumManager(curriculum_config)
    
    def setup_curriculum(self, max_epochs: int):
        """
        Setup curriculum after data module is initialized.
        
        Args:
            max_epochs: Maximum number of training epochs
        """
        if self.curriculum_manager:
            self.curriculum_manager.set_max_epochs(max_epochs)
    
    def update_curriculum_epoch(self, epoch: int) -> Optional[Dict[str, Any]]:
        """
        Update curriculum state for new epoch.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Curriculum state dictionary or None if curriculum is disabled
        """
        if self.curriculum_manager:
            return self.curriculum_manager.update_epoch(epoch)
        return None
    
    def should_include_sample(self, sample_difficulty: float) -> bool:
        """
        Check if a sample should be included based on current curriculum.
        
        Args:
            sample_difficulty: Difficulty score of the sample (0.0 = easiest, 1.0 = hardest)
            
        Returns:
            True if sample should be included, False otherwise
        """
        if self.curriculum_manager:
            return self.curriculum_manager.should_include_sample(sample_difficulty)
        return True  # Include all samples if curriculum is disabled
    
    def get_current_difficulty(self) -> float:
        """
        Get current difficulty level.
        
        Returns:
            Current difficulty level (0.0 = easiest, 1.0 = hardest)
        """
        if self.curriculum_manager:
            return self.curriculum_manager.get_current_difficulty()
        return 1.0  # Max difficulty if curriculum is disabled
    
    def get_curriculum_state(self) -> Optional[Dict[str, Any]]:
        """
        Get current curriculum state.
        
        Returns:
            Complete curriculum state dictionary or None if curriculum is disabled
        """
        if self.curriculum_manager:
            return self.curriculum_manager.get_state()
        return None
    
    def is_curriculum_enabled(self) -> bool:
        """
        Check if curriculum learning is enabled.
        
        Returns:
            True if curriculum learning is enabled, False otherwise
        """
        return self.curriculum_manager is not None 