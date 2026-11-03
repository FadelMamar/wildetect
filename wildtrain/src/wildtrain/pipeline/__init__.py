"""Train-then-eval pipeline.

The implementation is :class:`wildtrain.harness.pipeline.ExperimentPipeline`.
The per-task pipeline modules are gone.
"""

from ..harness.pipeline import ExperimentPipeline

__all__ = ["ExperimentPipeline"]
