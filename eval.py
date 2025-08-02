import os
import hydra
import lightning.pytorch as pl
from src.dataset.dl3dv import build_dl3dv_test_dataloader, build_dl3dv_dataloader
from src.dataset.re10k import build_re10k_test_dataloader
from src.utils.step_tracker import StepTracker
import torch

@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.test_seed, workers=True)

    # create directories for inference outputs
    inference_dir = os.path.join(cfg.experiment_output_path, "inference", cfg.inference_dir)
    os.makedirs(inference_dir, exist_ok=True)

    step_tracker = StepTracker()

    # initialize data
    if cfg.data.name == 're10k':
        test_loader = build_re10k_test_dataloader(cfg)
    elif cfg.data.name == 'dl3dv':
        test_loader = build_dl3dv_test_dataloader(cfg)
    else:
        raise ValueError(f"Invalid data: {cfg.data.name}")    

    # initialize model
    model = hydra.utils.instantiate(cfg.base_model, cfg, step_tracker=None)

    # initialize trainer
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator, 
        precision=cfg.trainer.precision,
        max_epochs=1, 
        logger=False
    )

    # start inference
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=cfg.ckpt_path)

if __name__ == '__main__':
    main()