python train.py \
    experiment_name=overfit_re10k \
    data=re10k \
    model=encoder_decoder \
    view_sampler=bounded_v2 \
    view_sampler.cfg.num_context_views=4 \
    data.batch_size=4 \
    data.val_batch_size=1 \
    data.num_workers=4 \
    data.overfit=1 \
    trainer.precision=16-mixed