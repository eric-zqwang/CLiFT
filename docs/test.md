## Inference

Pretrained checkpoints for both datasets are available on [Hugging Face](https://huggingface.co/EricW123456/CLiFT). Place them as follows:

| Hugging Face file | Local path |
|---|---|
| `re10k/first_stage.ckpt` | `output/re10k_first_stage/training/last.ckpt` |
| `re10k/second_stage.ckpt` | `output/re10k_second_stage/training/last.ckpt` |
| `dl3dv/first_stage.ckpt` | `output/dl3dv_first_stage/training/last.ckpt` |
| `dl3dv/second_stage.ckpt` | `output/dl3dv_second_stage/training/last.ckpt` |

### RE10K Evaluation
Reproduced RE10K PSNR for Figure 1:
```
bash script/eval/re10k/main.sh
```

Reproduced rendering metrics (PSNR, LPIPS, SSIM) for Figure 2:
```
# Ours w/o Condenser and K-means
bash script/eval/re10k/random.sh
# Ours w/o Condenser
bash script/eval/re10k/kmeans.sh
# Ours
bash script/eval/re10k/clift.sh
```

Reproduced compute-adaptive rendering for Table 1 (render-time token selection). We fix the storage budget (Ns = 4096) and sweep the render budget Nr over {4096, 3072, 2048, 1024, 512} on the large-scene test set (50 scenes with more than 200 frames, 8 context views per scene), reporting PSNR/SSIM/LPIPS together with rendering and token-selection FPS:
```
bash script/eval/re10k/token_selection.sh
```

### DL3DV Evaluation

The evaluation protocol is adapted from [DepthSplat](https://github.com/cvg/depthsplat); the context/target frame indices for 4 and 6 context views ship with the repo under `assets/dl3dv_start_0_distance_50_ctx_{4,6}v_video_0_50.json` (140 scenes, 50 target frames each), and only these 140 scenes need to be downloaded for evaluation. Each script takes the number of context views (4 or 6) as an argument.

Reproduced rendering metrics (PSNR, LPIPS, SSIM) sweeping the storage-token ratio:
```
# Ours w/o Condenser and K-means
bash script/eval/dl3dv/random.sh 6
# Ours w/o Condenser
bash script/eval/dl3dv/kmeans.sh 6
# Ours
bash script/eval/dl3dv/clift.sh 6
```

### Visualization

To generate visual outputs during evaluation, you can enable the following options in your evaluation scripts:
- Set `save_images=True` to save rendered images
- Set `save_videos=True` to save rendered videos
