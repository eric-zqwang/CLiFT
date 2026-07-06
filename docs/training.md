## Training

CLiFT is trained in two stages on both datasets:

1. **First stage (encoder-decoder):** an LVSM-style encoder-decoder is trained
   for novel view synthesis with a random token-drop ratio, so the decoder
   learns to render from a variable number of tokens.
2. **Second stage (condenser):** K-means assignments of the encoder tokens are
   precomputed, then the condenser (squeezer) is trained to aggregate each
   cluster into its anchor token. The encoder is frozen and the decoder trains
   at `squeezer.decoder_lr_scale` (0.1) x the base learning rate.

Checkpoints and logs are written to `output/<experiment_name>/training/`;
metrics are logged with Weights & Biases (set `WANDB_MODE=offline` to disable
syncing).

### RealEstate10K

```bash
# Stage 1: encoder-decoder with random token ratio
bash script/train/re10k/first_stage.sh

# (Optional) Precompute K-means assignments yourself instead of downloading
# them (see the data preparation guide):
bash script/train/re10k/save_kmeans.sh

# Stage 2: condenser, initialized from the first-stage checkpoint. Uses the
# precomputed K-means assignments.
bash script/train/re10k/second_stage.sh
```

### DL3DV

The DL3DV model starts from the RE10K first-stage checkpoint and fine-tunes on
DL3DV scenes with 4-6 context views per scene (padded views are masked).

```bash
# Stage 1: fine-tune the encoder-decoder on DL3DV
# (expects output/re10k_first_stage/training/last.ckpt)
bash script/train/dl3dv/first_stage.sh

# Precompute K-means assignments with the first-stage encoder (or download
# ours; see the data preparation guide). Each pass adds one context set +
# clustering per scene at a random token ratio; run several passes (the
# script loops over seeds) so scenes get multiple entries.
bash script/train/dl3dv/save_kmeans.sh

# Stage 2: condenser on the precomputed assignments
bash script/train/dl3dv/second_stage.sh
```

Both DL3DV stages default to an effective batch size that reproduces the
released checkpoint (24 for stage 1, 32 for stage 2) on 4 high-memory GPUs;
adjust `data.batch_size` / `+trainer.devices` /
`+trainer.accumulate_grad_batches` for your hardware (see the notes in each
script).
