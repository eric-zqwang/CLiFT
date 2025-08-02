import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from src.evaluation.evaluation_index_generator import IndexEntry
from src.utils.step_tracker import StepTracker
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage: str,
        step_tracker: StepTracker = None,
    ) -> None:
        super().__init__(cfg, stage, step_tracker)

        dacite_config = Config(cast=[tuple])
        with Path(cfg.index_path).open("r") as f:
            data = json.load(f)
    
        data = {k: v for k, v in data.items() if k is not None and v is not None}

        self.index = {
            k: from_dict(IndexEntry, v, dacite_config)
            for k, v in data.items()
        }

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)
        return context_indices, target_indices

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0