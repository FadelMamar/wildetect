"""Sweep output settings shared with the detection calibrator.

Training sweeps live on the experiment document (``sweep:``). The old
``base_config`` sweep models have been removed.
"""

from typing import Literal, Optional

from pydantic import Field

from .base import BaseConfig


class SweepOutputConfig(BaseConfig):
    """Output configuration for hyperparameter sweep results."""

    directory: Optional[str] = Field(
        default=None,
        description="Directory for sweep results",
    )
    save_results: bool = Field(default=True, description="Save detailed results")
    save_plots: bool = Field(default=True, description="Generate visualization plots")
    format: Literal["json", "csv", "both"] = Field(
        default="json", description="Output format"
    )
    include_optimization_history: bool = Field(
        default=True, description="Include all trials in output"
    )
