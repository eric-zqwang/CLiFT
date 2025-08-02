from src.lightning_lift import LightningLiFTWrapper
import torch
import hydra

class LightningCLiFTWrapper(LightningLiFTWrapper):
    def __init__(self, cfg, step_tracker):
        super().__init__(cfg, step_tracker)

    def forward(self, data_dict):
        if data_dict.get('features') is None:
            features = self.transformer.encode_scene(
                data_dict['condition_views'],
                data_dict['condition_views_plucker_coords']
            )
            data_dict['features'] = features

        output_dict = self.transformer(
            data_dict['features'],
            data_dict['anchor_idx'],
            data_dict['labels'],
            data_dict['num_tokens'],
            data_dict['sampled_views_plucker_coords']
        )

        return output_dict

    def validation_step(self, data_dict, idx):
        self._get_plucker_coords(data_dict)

        features, anchor_idx, labels, num_tokens = self.transformer.encode_and_kmeans(
            data_dict['condition_views'],
            data_dict['condition_views_plucker_coords']
        )
        data_dict['features'] = features
        data_dict['anchor_idx'] = anchor_idx
        data_dict['labels'] = labels
        data_dict['num_tokens'] = num_tokens

        output_dict = self(data_dict)
        if 'loss' not in output_dict:
            loss_dict = self._calc_loss(data_dict, output_dict)
        else:
            loss_dict = {"mse_loss": output_dict['loss']}

        metric_dict, psnr_raw = self._calc_metric(data_dict, output_dict)

        for metric_name, metric in metric_dict.items():
            self.log(f'val_metric/{metric_name}', metric, on_step=False, on_epoch=True, sync_dist=True)

        total_loss = 0
        for loss_name, loss in loss_dict.items():
            self.log(f'val_loss/{loss_name}', loss, on_step=False, on_epoch=True, sync_dist=True)
            total_loss += loss
        
        self.log('val_loss/total_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)


        sampled_views_plucker_coords = data_dict['sampled_views_plucker_coords']
        n_target_views = sampled_views_plucker_coords.shape[1]
        is_video = n_target_views > 8
        if (is_video and idx < 4) or (not is_video and idx == 0):
            self._vis_samples(data_dict, output_dict, is_video)

        return total_loss



    def test_step(self, data_dict, idx):
        self._get_plucker_coords(data_dict)

        features, anchor_idx, labels, num_tokens = self.transformer.encode_and_kmeans(
            data_dict['condition_views'],
            data_dict['condition_views_plucker_coords'],
            self.cfg.model.token_ratio
        )
        data_dict['features'] = features
        data_dict['anchor_idx'] = anchor_idx
        data_dict['labels'] = labels
        data_dict['num_tokens'] = num_tokens

        output_dict = self(data_dict)

        metric_dict, psnr_raw = self._calc_metric(data_dict, output_dict)

        for metric_name, metric in metric_dict.items():
            self.log(f'test_metric/{metric_name}', metric, on_step=False, on_epoch=True, sync_dist=True)

        self._save_test_samples(data_dict, output_dict, psnr_raw)


    def freeze_encoder(self):
        # 1) Freeze encoder
        for name, param in self.transformer.encoder.named_parameters():
            param.requires_grad = False

        for name, param in self.transformer.linear_input.named_parameters():
            param.requires_grad = False


    def freeze_encoder_decoder(self):
        for name, param in self.transformer.named_parameters():
            param.requires_grad = False

        for name, param in self.transformer.squeezer.named_parameters():
            param.requires_grad = True


    def configure_optimizers(self):

        self.freeze_encoder()

        lr = self.cfg.model.optimizer.lr
        special_lr = lr * self.cfg.model.squeezer.decoder_lr_scale

        special_names = [
            "transformer.decoder",
            "transformer.linear_target",
            "transformer.patch_to_image",
            "transformer.output_norm"
        ]

        param_groups = [
            {
                "params": [],
                "lr": special_lr,
            },
            {
                "params": [],
                "lr": lr,
            }
        ]

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(name.startswith(sn) for sn in special_names):
                param_groups[0]["params"].append(param)
            else:
                param_groups[1]["params"].append(param)

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=self.cfg.model.optimizer.betas,
            weight_decay=self.cfg.model.optimizer.weight_decay,
            eps=self.cfg.model.optimizer.eps,
        )

        max_iters = self.trainer.estimated_stepping_batches
        scheduler = hydra.utils.instantiate(self.cfg.model.lr_scheduler, optimizer=optimizer, max_iters=max_iters)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
