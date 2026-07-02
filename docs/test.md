## Inference
You can download the pretrained checkpoints for RE10K [here](https://drive.google.com/file/d/1jRMxtBa-16zHwUr9zfEuiusb9DLqpQbr/view?usp=sharing).


### Evaluation
Reproduced RE10K PSNR for Figure 1:
```
bash script/eval/eval_main.sh
```

Reproduced rendering metrics (PSNR, LPIPS, SSIM) for Figure 2:
```
# Ours w/o Condenser and K-means
bash script/eval/eval_random.sh
# Ours w/o Condenser
bash script/eval/eval_kmeans.sh
# Ours
bash script/eval/eval_clift.sh
```

Reproduced compute-adaptive rendering for Table 1 (render-time token selection). We fix the storage budget (Ns = 4096) and sweep the render budget Nr over {4096, 3072, 2048, 1024, 512} on the large-scene test set (50 scenes with more than 200 frames, 8 context views per scene), reporting PSNR/SSIM/LPIPS together with rendering and token-selection FPS:
```
bash script/eval/eval_token_selection.sh
```

### Visualization

To generate visual outputs during evaluation, you can enable the following options in your evaluation scripts:
- Set `save_images=True` to save rendered images
- Set `save_videos=True` to save rendered videos
