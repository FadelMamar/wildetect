"""Label policy for ROI classification.

``collapse`` is the product default: every class other than the background
becomes one class (``wildlife`` unless renamed). ``identity`` keeps species
labels. Existing configs set this with ``single_class.enable``.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional

from omegaconf import DictConfig, OmegaConf


def as_plain(value: Any) -> Any:
    """Convert a config section to plain Python containers."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


@dataclass(frozen=True)
class LabelPolicy:
    """How ROI class ids are presented to the classifier."""

    mode: Literal["collapse", "identity"] = "collapse"
    background_class_name: str = "background"
    single_class_name: str = "wildlife"
    keep_classes: Optional[list[str]] = None
    discard_classes: Optional[list[str]] = None

    @property
    def collapse(self) -> bool:
        return self.mode == "collapse"


def resolve_label_policy(
    single_class: Any = None,
    label_policy: Any = None,
) -> LabelPolicy:
    """Resolve the policy from ``label_policy``, else ``single_class``.

    When neither block is present the policy is collapse, which is the
    behavior existing single-class wildlife configs and registered
    classifiers already use. ``mode`` wins over ``enable`` when both are set.
    ``enable: true`` is collapse and ``enable: false`` is identity.
    """
    source = label_policy if label_policy is not None else single_class
    if source is None:
        return LabelPolicy()

    data = as_plain(source)
    if not isinstance(data, dict):
        raise TypeError(f"Label policy must be a mapping, got {type(source)}")

    mode = data.get("mode")
    if mode is None:
        mode = "collapse" if data.get("enable", True) else "identity"
    if mode not in ("collapse", "identity"):
        raise ValueError(
            f"Label policy mode must be 'collapse' or 'identity', got {mode!r}"
        )

    keep = data.get("keep_classes")
    discard = data.get("discard_classes")
    return LabelPolicy(
        mode=mode,
        background_class_name=data.get("background_class_name") or "background",
        single_class_name=data.get("single_class_name") or "wildlife",
        keep_classes=list(keep) if keep else None,
        discard_classes=list(discard) if discard else None,
    )
