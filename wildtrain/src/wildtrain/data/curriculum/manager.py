"""
Core Curriculum Learning Manager.

This module provides a unified interface for difficulty-based curriculum learning strategies.
"""

import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
import lightning as L

# Import the Pydantic model for type hints
from ...shared.models import CurriculumConfig


class DifficultyStrategy(ABC):
    """Abstract base class for difficulty progression strategies."""
    
    @abstractmethod
    def get_difficulty(self, epoch: int, max_epochs: int, config: CurriculumConfig) -> float:
        """Return difficulty level (0.0 = easiest, 1.0 = hardest) for given epoch."""
        pass


class LinearDifficultyStrategy(DifficultyStrategy):
    """Linear progression from easy to hard."""
    
    def get_difficulty(self, epoch: int, max_epochs: int, config: CurriculumConfig) -> float:
        effective_epoch = max(0, epoch - config.warmup_epochs)
        effective_max_epochs = max_epochs - config.warmup_epochs
        
        if effective_max_epochs <= 0:
            return config.end_difficulty
            
        progress = min(1.0, effective_epoch / effective_max_epochs)
        return config.start_difficulty + progress * (config.end_difficulty - config.start_difficulty)


class ExponentialDifficultyStrategy(DifficultyStrategy):
    """Exponential progression from easy to hard."""
    
    def get_difficulty(self, epoch: int, max_epochs: int, config: CurriculumConfig) -> float:
        effective_epoch = max(0, epoch - config.warmup_epochs)
        effective_max_epochs = max_epochs - config.warmup_epochs
        
        if effective_max_epochs <= 0:
            return config.end_difficulty
            
        progress = min(1.0, effective_epoch / effective_max_epochs)
        return 1.0 - np.exp(-3.0 * progress)  # Growth rate of 3.0


class RandomDifficultyStrategy(DifficultyStrategy):
    """No curriculum - random sampling."""
    
    def get_difficulty(self, epoch: int, max_epochs: int, config: CurriculumConfig) -> float:
        return config.end_difficulty  # Always max difficulty


class CurriculumManager:
    """Unified manager for difficulty-based curriculum learning strategies."""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_epoch = 0
        self.current_difficulty = 0.0
        self.max_epochs = 100
        
        # Initialize difficulty strategies
        self.difficulty_strategies = {
            "linear": LinearDifficultyStrategy(),
            "exponential": ExponentialDifficultyStrategy(),
            "random": RandomDifficultyStrategy(),
        }
        
    def set_max_epochs(self, max_epochs: int):
        """Set the maximum number of epochs for curriculum planning."""
        self.max_epochs = max_epochs
    
    def update_epoch(self, epoch: int) -> Dict[str, Any]:
        """Update curriculum state for new epoch."""
        self.current_epoch = epoch
        
        # Update difficulty if enabled
        if self.config.enabled and self.config.type == "difficulty":
            strategy = self.difficulty_strategies.get(self.config.difficulty_strategy)
            if strategy:
                self.current_difficulty = strategy.get_difficulty(
                    epoch, self.max_epochs, self.config
                )
        
        return {
            'epoch': epoch,
            'difficulty': self.current_difficulty,
        }
    
    def should_include_sample(self, sample_difficulty: float) -> bool:
        """Check if sample should be included based on current curriculum."""
        if not self.config.enabled or self.config.type != "difficulty":
            return True
            
        # Include samples that are not harder than current difficulty level
        return sample_difficulty <= self.current_difficulty + 0.1  # Small buffer
    
    def get_current_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty
    
    def get_state(self) -> Dict[str, Any]:
        """Get current curriculum state."""
        return {
            'epoch': self.current_epoch,
            'difficulty': self.current_difficulty,
            'config': self.config
        } 