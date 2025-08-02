#!/bin/bash

# Global variables
index_path=assets/evaluation_index_re10k_video_context4_data_100.json
val_batch_size=1

token_ratios=(0.03125 0.0625 0.125 0.25 0.5)

for ratio in "${token_ratios[@]}"
do
    echo "Running evaluation for token ratio: $ratio"
    python eval.py \
        experiment_name=re10k_first_stage \
        data=re10k \
        model=encoder_decoder \
        model.model_name._target_=src.model.encoder_decoder_kmeans.LiFTnvsKmeans \
        model.token_ratio=$ratio \
        view_sampler=evaluation \
        view_sampler.cfg.num_context_views=4 \
        view_sampler.cfg.index_path=$index_path \
        data.val_batch_size=$val_batch_size \
        trainer.precision=16-mixed \
        ckpt_path=output/re10k_first_stage/training/last.ckpt \
        inference_dir=dummy
done