#!/bin/bash

# DL3DV first stage: fine-tune the RE10K-pretrained encoder-decoder (LVSM-style,
# random token-drop ratio) on DL3DV with 4-6 context views per scene.
# Reproduces the released DL3DV model's first stage: effective batch 24
# (4 GPUs x batch 6), lr 2e-4, 100k steps.
#
# NOTE: batch 6 with up to 6 context views (10752 tokens) needs high-memory
# GPUs (e.g. 80GB). On smaller cards reduce data.batch_size and scale
# +trainer.devices / +trainer.accumulate_grad_batches to keep the effective
# batch at 24.

python finetune.py \
    experiment_name=dl3dv_first_stage \
    data=dl3dv \
    model=encoder_decoder \
    model.model_name._target_=src.model.encoder_decoder_dl3dv.DL3DVTransformer \
    view_sampler=bounded_v2 \
    data.batch_size=6 \
    data.val_batch_size=1 \
    trainer.precision=16-mixed \
    model.optimizer.lr=2e-4 \
    model.lr_scheduler.warmup_iters=2500 \
    model.ckpt_path=output/re10k_first_stage/training/last.ckpt \
    trainer.max_epochs=-1 \
    +trainer.max_steps=100000 \
    trainer.check_val_every_n_epoch=10 \
    +trainer.gradient_clip_val=1.0 \
    +trainer.gradient_clip_algorithm=norm \
    +trainer.devices=4 \
    +trainer.strategy=ddp
