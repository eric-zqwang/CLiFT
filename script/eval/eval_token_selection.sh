#!/bin/bash

# Reproduce Table 1: compute-adaptive rendering via render-time token selection.
# Fix the storage budget (Ns = 4096, i.e. storage_token_ratio 0.5 over 8*32*32
# tokens) and sweep the render budget Nr. Uses the large-scene test set
# (50 scenes, 8 context views, >200 target frames per scene).

index_path=assets/evaluation_index_re10k_video_context_min200_50_scenes.json
ckpt_path=output/re10k_second_stage/training/last.ckpt
val_batch_size=1

num_render_tokens=(4096 3072 2048 1024 512)

for nr in "${num_render_tokens[@]}"
do
    echo "Running token-selection evaluation for Nr=$nr"
    python eval.py \
        experiment_name=re10k_token_selection \
        data=re10k \
        base_model._target_=src.lightning_clift_token_selection.LightningCLiFTTokenSelection \
        model=encoder_decoder \
        model.model_name._target_=src.model.clift_token_selection.CLiFTnvsTokenSelection \
        model.render_selection.num_render_tokens=$nr \
        model.render_selection.storage_token_ratio=0.5 \
        view_sampler=evaluation \
        view_sampler.cfg.num_context_views=8 \
        view_sampler.cfg.index_path=$index_path \
        data.val_batch_size=$val_batch_size \
        model.encoder.num_layers=6 \
        model.decoder.num_layers=6 \
        model.encoder.hidden_dim=768 \
        model.decoder.hidden_dim=768 \
        trainer.precision=16-mixed \
        ckpt_path=$ckpt_path \
        inference_dir=Nr${nr} \
        save_images=false \
        save_videos=false
done
