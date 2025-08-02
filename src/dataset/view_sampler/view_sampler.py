from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Tuple

import torch
from jaxtyping import Float, Int64
from torch import Tensor
from src.utils.step_tracker import StepTracker


T = TypeVar("T")


class ViewSampler(ABC, Generic[T]):
    cfg: T
    stage: str
    # is_overfitting: bool
    # cameras_are_circular: bool
    step_tracker: StepTracker

    def __init__(
        self,
        cfg: T,
        stage: str = "train",
        is_overfitting: bool = False,
        cameras_are_circular: bool = False,
        step_tracker: StepTracker = None,
    ) -> None:
        assert stage in ["train", "val", "test"], f"Invalid stage: {stage}"
        self.cfg = cfg
        self.stage = stage
        self.is_overfitting = is_overfitting
        self.cameras_are_circular = cameras_are_circular
        self.step_tracker = step_tracker

    @abstractmethod
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> Tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        pass

    @property
    @abstractmethod
    def num_target_views(self) -> int:
        pass

    @property
    @abstractmethod
    def num_context_views(self) -> int:
        pass

    @property
    def global_step(self) -> int:
        return 0 if self.step_tracker is None else self.step_tracker.get_step()