import os

import hydra
import lightning.pytorch as pl
import torch

from src.dataset.dl3dv import build_dl3dv_annotation_dataloader
from src.dataset.re10k import build_re10k_annotation_dataloader
from src.utils.step_tracker import StepTracker


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    # fix the seed (run several passes with different test_seed values to get
    # multiple K-means entries per scene)
    pl.seed_everything(cfg.test_seed, workers=True)

    assert cfg.data.kmeans_dir is not None, "Error: data.kmeans_dir must be set."
    os.makedirs(cfg.data.kmeans_dir, exist_ok=True)

    step_tracker = StepTracker()

    # initialize data
    if cfg.data.name == 're10k':
        loader = build_re10k_annotation_dataloader(cfg, step_tracker=step_tracker)
    elif cfg.data.name == 'dl3dv':
        loader = build_dl3dv_annotation_dataloader(cfg, step_tracker=step_tracker)
    else:
        raise ValueError(f"Invalid data: {cfg.data.name}")

    # initialize model and load the first-stage encoder weights
    model = hydra.utils.instantiate(cfg.base_model, cfg, step_tracker=step_tracker)

    weights = torch.load(cfg.model.ckpt_path, map_location='cpu', weights_only=False)['state_dict']
    model.transformer.load_state_dict(
        {k.replace('transformer.', ''): v for k, v in weights.items()
         if k.startswith('transformer.')}, strict=False
    )

    # initialize trainer. Single device: concurrent DDP ranks would race on the
    # per-scene metadata.json read-modify-write.
    trainer = pl.Trainer(accelerator=cfg.trainer.accelerator, devices=1, max_epochs=1, logger=False)

    # run one annotation pass over the training scenes
    trainer.test(model=model, dataloaders=loader)


if __name__ == '__main__':
    main()
