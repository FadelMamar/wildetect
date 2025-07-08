"""
Dataset management for drone image analysis campaigns.
Provides backward compatibility for CensusData class.
"""

from typing import Dict, Optional

from ..config import LoaderConfig
from .census import CampaignMetadata, CensusDataManager, DetectionResults

# Alias for backward compatibility
CensusData = CensusDataManager

__all__ = ["CensusData", "CensusDataManager", "CampaignMetadata", "DetectionResults"]
