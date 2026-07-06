#!/bin/bash

# DL3DV second stage: train the condenser (squeezer) on precomputed K-means
# assignments, starting from the first-stage encoder-decoder. The encoder is
# frozen; the decoder side trains at squeezer.decoder_lr_scale x lr (0.1) and
# the squeezer at the full lr. Reproduces the released epoch129 checkpoint's
# schedule: effective batch 32 (4 GPUs x batch 8), lr 3e-4, ~40k steps.
#
# NOTE: the condenser's cluster attention mask is large (max 10752 tokens per
# sample), so batch 8 needs high-memory GPUs (e.g. 80GB). On 24GB cards use
# data.batch_size=4 with 8 GPUs, or add +trainer.accumulate_grad_batches to
# keep the effective batch at 32.

python finetune.py \
    experiment_name=dl3dv_second_stage \
    data=dl3dv \
    data.kmeans_dir=../Dataset/dl3dv_kmeans_faiss_merged/ \
    base_model._target_=src.lightning_clift.LightningCLiFTWrapper \
    model=encoder_decoder \
    model.model_name._target_=src.model.squeezer_decoder_dl3dv.SqueezerDecoder \
    view_sampler=bounded_v2 \
    data.batch_size=8 \
    data.val_batch_size=1 \
    trainer.precision=16-mixed \
    model.optimizer.lr=3e-4 \
    model.lr_scheduler.warmup_iters=2500 \
    model.ckpt_path=output/dl3dv_first_stage/training/last.ckpt \
    trainer.max_epochs=140 \
    trainer.check_val_every_n_epoch=5 \
    +trainer.gradient_clip_val=1.0 \
    +trainer.gradient_clip_algorithm=norm \
    +trainer.devices=4 \
    +trainer.strategy=ddp
