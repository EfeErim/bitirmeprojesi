"""Canonical application workflows."""

from .inference import InferenceWorkflow
from .training import TrainingWorkflow, TrainingWorkflowResult

__all__ = ["InferenceWorkflow", "TrainingWorkflow", "TrainingWorkflowResult"]
