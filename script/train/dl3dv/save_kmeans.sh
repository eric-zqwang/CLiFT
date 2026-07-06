#!/bin/bash

# Precompute K-means assignments for DL3DV second-stage (condenser) training.
# Each pass samples one context set (4-6 views) per training scene, clusters the
# first-stage encoder tokens at a random token ratio in [0.05, 1.0], and appends
# one entry to <kmeans_dir>/<scene>/metadata.json. Run multiple passes with
# different seeds so every scene gets several context sets / token budgets.

kmeans_dir=../Dataset/dl3dv_kmeans_faiss_merged/

for seed in 0 1 2
do
    echo "Running K-means annotation pass with seed $seed"
    python save_kmeans.py \
        experiment_name=dl3dv_kmeans_annotation \
        data=dl3dv \
        data.kmeans_dir=$kmeans_dir \
        base_model._target_=src.lightning_save_kmeans.KmeansAnnotator \
        model=encoder_decoder \
        model.model_name._target_=src.model.squeezer_decoder_dl3dv.SqueezerDecoder \
        view_sampler=bounded_v2 \
        model.ckpt_path=output/dl3dv_first_stage/training/last.ckpt \
        test_seed=$seed
done
