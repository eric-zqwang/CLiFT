import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from tqdm import tqdm


@dataclass
class EvaluationIndexGeneratorCfg:
    num_target_views: int
    min_distance: int
    max_distance: int
    min_overlap: float
    max_overlap: float
    output_path: Path
    save_previews: bool
    seed: int


@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]


class EvaluationIndexGenerator(LightningModule):
    generator: torch.Generator
    cfg: EvaluationIndexGeneratorCfg
    index: dict[str, IndexEntry]

    def __init__(self, cfg: EvaluationIndexGeneratorCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)
        self.index = {}


    def save_index(self) -> None:
        self.cfg.output_path.mkdir(exist_ok=True, parents=True)
        with (self.cfg.output_path / "evaluation_index.json").open("w") as f:
            json.dump(
                {k: None if v is None else asdict(v) for k, v in self.index.items()}, f
            )