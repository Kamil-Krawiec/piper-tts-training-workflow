"""Prerequisites for the six steps, shared by navigation and step availability."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowProgress:
    project: bool = False
    prompts: int = 0
    accepted: int = 0
    datasets: int = 0
    runs: int = 0
    models: int = 0
    training: bool = False

    def blocked_reason(self, step: int) -> str | None:
        if step not in range(1, 7):
            raise ValueError("Workflow step must be between 1 and 6")
        if step == 1:
            return None
        if not self.project:
            return "Choose or create a project in Step 1 first."
        if step == 3 and not (self.prompts or self.datasets):
            return "Prepare and save your prompt queue in Step 2 first."
        if step == 5 and not self.datasets:
            return "Create or import a dataset in Step 4 first."
        if step == 6 and not (self.datasets or self.runs or self.models):
            return "Start a training run in Step 5 first. A saved checkpoint is needed to export a voice."
        return None

    @property
    def next_step(self) -> int:
        if not self.project:
            return 1
        if self.training:
            return 5
        if self.runs or self.models:
            return 6
        if self.datasets:
            return 5
        if self.accepted:
            return 4
        return 3 if self.prompts else 2
