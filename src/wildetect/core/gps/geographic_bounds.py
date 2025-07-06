"""
Geographic bounds utilities for wildlife detection.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from wildetect.utils.utils import compute_iou

logger = logging.getLogger(__name__)


@dataclass
class GeographicBounds:
    """Geographic bounding box for image footprint in UTM coordinates"""

    north: float  # Max latitude
    south: float  # Min latitude
    east: float  # Max longitude
    west: float  # Min longitude

    @property
    def area(self) -> float:
        """Calculate area in square degrees covered by the bounding box."""
        return (self.east - self.west) * (self.north - self.south)

    @property
    def box(
        self,
    ) -> List[float]:
        return [self.west, self.south, self.east, self.north]

    def overlap_ratio(self, other: "GeographicBounds") -> float:
        """Calculate overlap ratio (IoU) with another bounds using torchmetrics IntersectionOverUnion.
        Args:
            other (GeographicBounds): Another geographic bounds object.
        Returns:
            float: Overlap ratio (IoU) between the two bounds.
        """
        box_self = self.box
        box_other = other.box

        return compute_iou(box_self, box_other)

    def expand(self, margin: float) -> "GeographicBounds":
        """Expand the bounds by a margin.

        Args:
            margin (float): Margin to expand by

        Returns:
            GeographicBounds: Expanded bounds
        """
        return GeographicBounds(
            north=self.north + margin,
            south=self.south - margin,
            east=self.east + margin,
            west=self.west - margin,
        )
