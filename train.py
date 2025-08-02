import os
import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
# from src.dataset.dl3dv import build_dl3dv_dataloader
from src.dataset.re10k import build_re10k_dataloader
from src.utils.step_tracker import StepTracker
import torch


def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    last_checkpoint = hydra.utils.instantiate(cfg.last_checkpoint)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    # print_callback = PrintCallback()
    return [checkpoint_monitor, last_checkpoint, lr_monitor]


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    # import pdb; pdb.set_trace()
    # fix the seed
    pl.seed_everything(cfg.train_seed, workers=True)

    # create directories for training outputs
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)
    step_tracker = StepTracker()

    # initialize data
    if cfg.data.name == 're10k':
        train_loader, val_loader = build_re10k_dataloader(cfg, step_tracker=step_tracker)
    # elif cfg.data.name == 'dl3dv':
    #     train_loader, val_loader = build_dl3dv_dataloader(cfg, step_tracker=step_tracker)
    else:
        raise ValueError(f"Invalid data: {cfg.data.name}")

    # initialize model
    model = hydra.utils.instantiate(cfg.base_model, cfg, step_tracker=step_tracker)
        
    # initialize logger
    logger = hydra.utils.instantiate(cfg.logger)

    # initialize callbacks
    callbacks = init_callbacks(cfg)
    
    # initialize trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        **cfg.trainer
    )

    # check the checkpoint
    if cfg.ckpt_path is not None:
        assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exist."


    # start training
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.ckpt_path
    )


if __name__ == '__main__':
    main()