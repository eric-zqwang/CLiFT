python train.py \
    experiment_name=re10k_first_stage \
    data=re10k \
    model=encoder_decoder \
    model.model_name._target_=src.model.encoder_decoder.Transformer \
    model.lr_scheduler.warmup_iters=2500 \
    view_sampler=bounded_v2 \
    data.batch_size=16 \
    data.val_batch_size=16 \
    trainer.precision=16-mixed \
    data.num_workers=4 \
    trainer.max_epochs=100 \
    trainer.check_val_every_n_epoch=10 \
    +trainer.gradient_clip_val=1.0 \
    +trainer.gradient_clip_algorithm="norm" \
    +trainer.devices=4 \
    +trainer.strategy=ddp
