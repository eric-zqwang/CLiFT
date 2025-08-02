import lightning.pytorch as pl
import torch
import numpy as np
from src.utils.visualizer import log_images
import hydra
from src.utils.visualizer import save_images
from src.utils.visualizer import log_videos, save_videos, vis_policy_per_scene
import os
import wandb
from src.utils.camera import ray_condition
from src.loss.perceptual import PerceptualLoss
from src.utils.metric_utils import MetricComputer

class LightningLiFTWrapper(pl.LightningModule):
    def __init__(self, cfg, step_tracker):
        super().__init__()
        self.cfg = cfg
        self.step_tracker = step_tracker
        self.trainable_parameters = []

        self._load_modules()
    

    def _load_modules(self):
        if hasattr(self.cfg, 'pruning') and self.cfg.pruning.name == 'predictor':
            self.predictor = hydra.utils.instantiate(self.cfg.model.pruning, _recursive_=False)
            
        self.transformer = hydra.utils.instantiate(
            self.cfg.model.model_name,
            cfg=self.cfg.model,
            _recursive_=False
        )

        if hasattr(self.transformer, 'trainable_params'):
            self.trainable_parameters = self.transformer.trainable_params
        else:
            self.trainable_parameters.append([list(self.transformer.parameters()), 1.0])


    def setup(self, stage=None):
        self.perceptual_loss = PerceptualLoss(
            device=self.device
        )
        self.metric_computer = MetricComputer(device=self.device)

    def padding_plucker_coords(self, plucker_coords, num_context_views):
        for i in range(len(plucker_coords)):
            if num_context_views[i] < plucker_coords[i].shape[0]:
                plucker_coords[i][num_context_views[i]:] = 0
        return plucker_coords

    def forward(self, data_dict):
        condition_views = data_dict['condition_views']
        condition_views_plucker_coords = data_dict['condition_views_plucker_coords']

        sampled_views = data_dict['sampled_views']
        sampled_views_plucker_coords = data_dict['sampled_views_plucker_coords']

        if self.training and self.cfg.data.name == 'dl3dv':
            condition_views_plucker_coords = self.padding_plucker_coords(
                condition_views_plucker_coords, data_dict['num_context_views']
            )
        
        if self.cfg.data.name == 'dl3dv':
            output_dict = self.transformer(
                condition_views,
                condition_views_plucker_coords,
                sampled_views_plucker_coords,
                data_dict['num_context_views']
            )
        else:
            output_dict = self.transformer(
                condition_views,
                condition_views_plucker_coords,
                sampled_views_plucker_coords,
            )

        output_dict.update({
            'gt_sampled_views': sampled_views
        })

        return output_dict

    def _calc_loss(self, data_dict, output_dict):

        pred_images = output_dict['pred_sampled_views'].permute(0, 1, 4, 2, 3)
        gt_images = data_dict['sampled_views'].permute(0, 1, 4, 2, 3)

        loss_dict = {}

        mse_loss = torch.nn.functional.mse_loss(
            pred_images,
            gt_images,
        )
        loss_dict['mse_loss'] = mse_loss

        if self.cfg.loss.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(
                pred_images.flatten(0, 1),
                gt_images.flatten(0, 1)
            )
            loss_dict['perceptual_loss'] = perceptual_loss * self.cfg.loss.perceptual_weight

        return loss_dict
    
    def _get_plucker_coords(self, data_dict):
        if data_dict.get('condition_views_plucker_coords') is None:
            condition_views_plucker_coords = ray_condition(
                data_dict['condition_views_intrinsics'],
                data_dict['condition_views_extrinsics'],
                self.cfg.data.image_size[0],
                self.cfg.data.image_size[1],
                self.device
            )
            sampled_views_plucker_coords = ray_condition(
                data_dict['sampled_views_intrinsics'],
                data_dict['sampled_views_extrinsics'],
                self.cfg.data.image_size[0],
                self.cfg.data.image_size[1],
                self.device
            )
            data_dict['condition_views_plucker_coords'] = condition_views_plucker_coords
            data_dict['sampled_views_plucker_coords'] = sampled_views_plucker_coords
        

    def training_step(self, data_dict, idx):
        self._get_plucker_coords(data_dict)

        output_dict = self(data_dict)

        if 'loss' not in output_dict:
            loss_dict = self._calc_loss(data_dict, output_dict)
        else:
            loss_dict = {"mse_loss": output_dict['loss']}

        total_loss = 0
        for loss_name, loss in loss_dict.items():
            self.log(f'train_loss/{loss_name}', loss, on_step=True, on_epoch=False)
            total_loss += loss

        self.log('train_loss/total_loss', total_loss, on_step=True, on_epoch=False)

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
            
        return total_loss


    def calculate_psnr(self, pred_sampled_views, gt_sampled_views):
        """Calculate PSNR for each view.
        Args:
            pred_sampled_views: [B, N, C, H, W] numpy array in range [0, 255]
            gt_sampled_views: [B, N, C, H, W] numpy array in range [0, 255]
        Returns:
            numpy array of shape [B, N] containing PSNR values
        """        
        psnr_values = np.zeros((pred_sampled_views.shape[0], pred_sampled_views.shape[1]))
        
        for i in range(pred_sampled_views.shape[0]):
            for j in range(pred_sampled_views.shape[1]):
                mse = np.mean((np.array(pred_sampled_views[i, j], dtype=np.float32) - np.array(gt_sampled_views[i, j], dtype=np.float32)) ** 2)
                if mse == 0:
                    psnr_values[i, j] = 100
                else:
                    psnr_values[i, j] = 20 * np.log10(255 / np.sqrt(mse))
                
        return psnr_values
    

    def to_numpy_image(self, image):
        image = image.clamp(0, 1).float()
        image = image.cpu().numpy()
        image = (image * 255).round().astype('uint8')
        return image


    def _calc_metric(self, data_dict, output_dict):
        bs, v = output_dict['pred_sampled_views'].shape[:2]
        pred_sampled_views = output_dict['pred_sampled_views'].flatten(0, 1).permute(0, 3, 1, 2)
        gt_sampled_views = data_dict['sampled_views'].flatten(0, 1).permute(0, 3, 1, 2)

        metric_dict, psnr_raw = self.metric_computer(pred_sampled_views, gt_sampled_views)
        return metric_dict, psnr_raw.reshape(bs, v)
    
    def _vis_samples(self, data_dict, output_dict, is_video):

        pred_images = self.to_numpy_image(output_dict['pred_sampled_views'])
        gt_images = self.to_numpy_image(data_dict['sampled_views'])
        condition_images = self.to_numpy_image(data_dict['condition_views'])

        max_samples = min(12, pred_images.shape[0])
        pred_images = pred_images[:max_samples]
        gt_images = gt_images[:max_samples]
        condition_images = condition_images[:max_samples]

        n_context_views = condition_images.shape[1]
        if is_video:
            log_videos(pred_images, gt_images, condition_images, self.logger, self.global_step, data_dict)
        else:
            log_images(pred_images, gt_images, condition_images, self.logger, self.global_step, data_dict)
        if 'hard_keep_decision' in output_dict and output_dict['hard_keep_decision'] is not None:
            bs = min(max_samples, output_dict['hard_keep_decision'].shape[0])
            v = pred_images.shape[1]
            policy = output_dict['hard_keep_decision'].to(torch.int32)
            for i in range(bs):
                policy_visualization = vis_policy_per_scene(
                    condition_images[i], 
                    policy[i], 
                    num_context_views=n_context_views,
                    patch_size=self.cfg.model.patch_size, 
                    log=True
                )
                vis = wandb.Image(
                    policy_visualization,
                    caption=f"Policy Visualization"
                )
                self.logger.experiment.log({
                    f"policy_visualizations/{data_dict['data_id'][i]}": vis,
                }, step=self.global_step)
        
    
    def validation_step(self, data_dict, idx):
        self._get_plucker_coords(data_dict)

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


    def _save_test_samples(self, data_dict, output_dict, psnr_raw):
        pred_images = self.to_numpy_image(output_dict['pred_sampled_views'])
        gt_images = self.to_numpy_image(data_dict['sampled_views'])
        condition_images = self.to_numpy_image(data_dict['condition_views'])

        if self.cfg.save_videos:
            save_dir = os.path.join(
                self.cfg.experiment_output_path,
                "inference",
                self.cfg.inference_dir,
                data_dict['data_id'][0]
            )
            
            # Assume always one batch
            save_images(
                timestamps=None,
                condition_images=condition_images[0],
                save_dir=save_dir,
            )

            save_videos(pred_images, save_dir + "/pred.mp4", fps=30)
            save_videos(gt_images, save_dir + "/gt.mp4", fps=30)


        if self.cfg.save_images:
            bs = pred_images.shape[0]
            v = pred_images.shape[1]

            if 'hard_keep_decision' in output_dict and output_dict['hard_keep_decision'] is not None:
                if output_dict['hard_keep_decision'].ndim == 3:
                    policy = output_dict['hard_keep_decision'].reshape(bs, v, -1)
                    policy = policy[:, 0, :]
                else:
                    policy = output_dict['hard_keep_decision']


            for i in range(bs):
                save_dir = os.path.join(
                    self.cfg.experiment_output_path,
                    "inference", 
                    self.cfg.inference_dir, 
                    data_dict['data_id'][i]
                )

                os.makedirs(save_dir, exist_ok=True)
                save_images(
                    timestamps=None,
                    condition_images=condition_images[i],
                    pred_images=pred_images[i],
                    gt_images=gt_images[i],
                    save_dir=save_dir,
                    psnr=psnr_raw[i]
                )

                if 'hard_keep_decision' in output_dict and output_dict['hard_keep_decision'] is not None:
                    n_context_views = condition_images.shape[1]
                    vis_policy_per_scene(
                        condition_images[i], 
                        policy[i], 
                        num_context_views=n_context_views,
                        patch_size=self.cfg.model.patch_size, 
                        save_path=save_dir+f"/policy",
                        log=False
                    )

    def test_step(self, data_dict, idx):
        self._get_plucker_coords(data_dict)

        output_dict = self(data_dict)
        metric_dict, psnr_raw = self._calc_metric(data_dict, output_dict)
        for metric_name, metric in metric_dict.items():
            self.log(f'test_metric/{metric_name}', metric, on_step=False, on_epoch=True, sync_dist=True)

        if self.cfg.save_images or self.cfg.save_videos:
            self._save_test_samples(data_dict, output_dict, psnr_raw)
        


    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_parameters:
            param_groups.append({"params": params, "lr": self.cfg.model.optimizer.lr * lr_scale})

        optimizer = torch.optim.AdamW(
            param_groups,
            betas=self.cfg.model.optimizer.betas,
            weight_decay=self.cfg.model.optimizer.weight_decay,
            eps=self.cfg.model.optimizer.eps
        )
        max_iters = self.trainer.estimated_stepping_batches
        scheduler = hydra.utils.instantiate(self.cfg.model.lr_scheduler, optimizer=optimizer, max_iters=max_iters)

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
  
