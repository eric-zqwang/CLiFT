#!/bin/bash

# Evaluate CLiFT on DL3DV-10K (compute-adaptive rendering, paper Fig. 2 right /
# supplementary). Sweeps the storage-token ratio (Ns = seq_len * ratio) with a
# single trained network and reports PSNR / SSIM / LPIPS per ratio.
#
# Usage:
#   bash script/eval/dl3dv/clift.sh       # 4 context views (default)
#   bash script/eval/dl3dv/clift.sh 6     # 6 context views
#
# The evaluation index covers 140 scenes (132 pass the data-integrity filters);
# each scene renders 50 target frames.

num_context_views=${1:-4}
index_path=assets/dl3dv_start_0_distance_50_ctx_${num_context_views}v_video_0_50.json
ckpt_path=output/dl3dv_second_stage/training/last.ckpt
val_batch_size=1

token_ratios=(0.0625 0.125 0.25 0.5 0.75 1.0)

for ratio in "${token_ratios[@]}"
do
    echo "Running DL3DV evaluation for ${num_context_views} context views, token ratio: $ratio"
    python eval.py \
        experiment_name=dl3dv_clift \
        data=dl3dv \
        base_model._target_=src.lightning_clift.LightningCLiFTWrapper \
        model=encoder_decoder \
        model.model_name._target_=src.model.squeezer_decoder_dl3dv.SqueezerDecoder \
        model.token_ratio=$ratio \
        view_sampler=evaluation \
        view_sampler.cfg.num_context_views=$num_context_views \
        view_sampler.cfg.index_path=$index_path \
        data.val_batch_size=$val_batch_size \
        model.encoder.num_layers=6 \
        model.decoder.num_layers=6 \
        model.encoder.hidden_dim=768 \
        model.decoder.hidden_dim=768 \
        trainer.precision=16-mixed \
        ckpt_path=$ckpt_path \
        inference_dir=dl3dv_${num_context_views}v_ratio_${ratio} \
        save_images=false \
        save_videos=false
done
