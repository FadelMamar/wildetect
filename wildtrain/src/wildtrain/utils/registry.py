from typing import Callable, Dict


class Registry:
    """
    Simple registry for factories (models, datasets, etc.).
    """

    def __init__(self):
        self._registry: Dict[str, Callable[..., object]] = {}

    def register(self, name: str, fn: Callable[..., object]) -> None:
        self._registry[name] = fn

    def get(self, name: str) -> Callable[..., object]:
        if name not in self._registry:
            raise KeyError(f"{name} not found in registry")
        return self._registry[name]
