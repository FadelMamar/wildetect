"""In-process registry of harness tasks."""

from typing import Dict

from .task import Task


class TaskRegistry:
    """Map a task name to the object that runs it."""

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}

    def register(self, task: Task) -> None:
        if task.name in self._tasks:
            raise ValueError(f"Task {task.name!r} is already registered")
        self._tasks[task.name] = task

    def get(self, name: str) -> Task:
        try:
            return self._tasks[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._tasks)) or "(none)"
            raise KeyError(f"Unknown task {name!r}. Known tasks: {known}") from exc

    def names(self) -> list[str]:
        return sorted(self._tasks)
