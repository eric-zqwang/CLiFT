#!/bin/bash

# Global variables
index_path=assets/evaluation_index_re10k_context4.json
val_batch_size=1

token_ratios=(0.0625 0.125 0.25 0.5 0.75 1.0)

for ratio in "${token_ratios[@]}"
do
    echo "Running evaluation for token ratio: $ratio"
    python eval.py \
        experiment_name=re10k_second_stage \
        data=re10k \
        base_model._target_=src.lightning_clift.LightningCLiFTWrapper \
        model=encoder_decoder \
        model.model_name._target_=src.model.squeezer_decoder.CLiFTnvs \
        model.token_ratio=$ratio \
        view_sampler=evaluation \
        view_sampler.cfg.num_context_views=4 \
        view_sampler.cfg.index_path=$index_path \
        data.val_batch_size=$val_batch_size \
        model.encoder.num_layers=6 \
        model.decoder.num_layers=6 \
        model.encoder.hidden_dim=768 \
        model.decoder.hidden_dim=768 \
        trainer.precision=16-mixed \
        ckpt_path=output/re10k_second_stage/training/last.ckpt \
        inference_dir=dummy \
        save_images=false \
        save_videos=false
done
