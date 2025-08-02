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

### Visualization

To generate visual outputs during evaluation, you can enable the following options in your evaluation scripts:
- Set `save_images=True` to save rendered images
- Set `save_videos=True` to save rendered videos
