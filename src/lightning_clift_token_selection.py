import time

import torch

from src.lightning_clift import LightningCLiFTWrapper
from src.utils.camera import ray_condition, get_downsample_camera_ray


class LightningCLiFTTokenSelection(LightningCLiFTWrapper):
    """Inference wrapper for compute-adaptive CLiFT rendering (paper Algorithm 1).

    For each scene we encode + condense the context views into Ns storage CLiFTs
    once, then walk the target views in temporal order. Every frame selects the
    Nr storage tokens closest to that view and renders from them only. We report
    PSNR/SSIM/LPIPS together with rendering and token-selection FPS. Sweeping
    ``model.render_selection.num_render_tokens`` reproduces the paper's Table 1.
    """

    def __init__(self, cfg, step_tracker):
        super().__init__(cfg, step_tracker)
        rs = cfg.model.render_selection
        self.target_range = rs.target_range
        self.context_downsample = rs.context_downsample
        self.target_downsample = rs.target_downsample
        self.storage_token_ratio = rs.storage_token_ratio

    def _prepare_geometry(self, data_dict):
        """Compute Plücker coords (for rendering) and camera rays (for selection)."""
        if data_dict.get("condition_views_plucker_coords") is None:
            data_dict["condition_views_plucker_coords"] = ray_condition(
                data_dict["condition_views_intrinsics"],
                data_dict["condition_views_extrinsics"],
                self.cfg.data.image_size[0],
                self.cfg.data.image_size[1],
                self.device,
            )
            data_dict["sampled_views_plucker_coords"] = ray_condition(
                data_dict["sampled_views_intrinsics"],
                data_dict["sampled_views_extrinsics"],
                self.cfg.data.image_size[0],
                self.cfg.data.image_size[1],
                self.device,
            )

        # Context rays live on the token grid (image_size / patch_size).
        data_dict["condition_views_camera_ray"] = get_downsample_camera_ray(
            data_dict["condition_views_intrinsics"],
            data_dict["condition_views_extrinsics"],
            self.cfg.data.image_size[0],
            self.cfg.data.image_size[1],
            self.device,
            downsample_factor=self.context_downsample,
        )
        # Target rays use a coarser, symmetrically expanded grid (patch expansion)
        # so a token is scored against a slightly larger frustum than the visible one.
        data_dict["sampled_views_camera_ray"] = get_downsample_camera_ray(
            data_dict["sampled_views_intrinsics"],
            data_dict["sampled_views_extrinsics"],
            self.cfg.data.image_size[0],
            self.cfg.data.image_size[1],
            self.device,
            downsample_factor=self.target_downsample,
            extended_range=(self.target_range, self.target_range),
        )

    def forward(self, data_dict):
        transformer = self.transformer
        condition_views = data_dict["condition_views"]
        condition_plucker = data_dict["condition_views_plucker_coords"]
        sampled_plucker = data_dict["sampled_views_plucker_coords"]
        num_view = sampled_plucker.shape[1]
        bs = condition_views.shape[0]

        # 1) Encode the scene and condense into Ns storage CLiFTs. The condenser's
        # cluster-masked cross-attention can underflow to NaN under fp16, so we run
        # this once-per-scene step in fp32. It is outside the per-frame render loop,
        # so it does not affect the reported rendering FPS.
        with torch.autocast(device_type=self.device.type, enabled=False):
            features, anchor_idx, labels, num_tokens = transformer.encode_and_kmeans(
                condition_views, condition_plucker, token_ratio=self.storage_token_ratio
            )
            num_storage = int(num_tokens[0].item())
            center_idx = anchor_idx[0, :num_storage]
            squeezed_features = transformer.squeezer(features, anchor_idx, labels, num_tokens)
            squeezed_features = squeezed_features[:, :num_storage]  # [bs, Ns, C]

        # 2) Geometry of the storage tokens: source ray + owning context view.
        context_ray = data_dict["condition_views_camera_ray"]  # [1, V, h, w, 6]
        _, num_context_view, grid_h, grid_w, _ = context_ray.shape
        context_ray_flat = context_ray.reshape(-1, 6)          # [V*h*w, 6]
        center_rays = context_ray_flat[center_idx]             # [Ns, 6]
        center_view = torch.div(center_idx, grid_h * grid_w, rounding_mode="floor")  # [Ns]

        target_ray = data_dict["sampled_views_camera_ray"]     # [1, num_view, H', W', 6]
        context_extrinsics = data_dict["condition_views_extrinsics"][0]  # [V, 4, 4]
        target_extrinsics = data_dict["sampled_views_extrinsics"][0]     # [num_view, 4, 4]

        # 3) Walk target views in order, selecting Nr tokens per frame.
        last_combined_distances = None
        previous_selected = None
        predictions = []
        render_time = 0.0
        selection_time = 0.0

        for i in range(num_view):
            cam_center_dist = transformer.camera_center_distance(
                context_extrinsics, target_extrinsics[i]
            )  # [V]
            center_cam_center_dist = cam_center_dist[center_view]  # [Ns]
            exist_same_ray = bool(cam_center_dist.min().item() == 0)

            torch.cuda.synchronize()
            selection_start = time.time()
            selected_idx, selected_mask, combined_distances = transformer.select_tokens(
                center_rays,
                center_cam_center_dist,
                target_ray[0, i],
                exist_same_ray,
                last_combined_distances,
                previous_selected,
            )
            torch.cuda.synchronize()
            selection_time += time.time() - selection_start

            out, frame_render_time = transformer.render_selected(
                squeezed_features, selected_idx, sampled_plucker[:, i : i + 1], bs
            )
            render_time += frame_render_time
            predictions.append(out)

            last_combined_distances = combined_distances
            previous_selected = selected_mask

        pred_sampled_views = torch.cat(predictions, dim=1)  # [bs, num_view, H, W, 3]

        return {
            "pred_sampled_views": pred_sampled_views,
            "render_fps": num_view / render_time,
            "token_selection_fps": num_view / selection_time,
            "total_fps": num_view / (render_time + selection_time),
            "num_render_tokens": int(transformer.num_render_tokens),
            "num_storage_tokens": num_storage,
        }

    def test_step(self, data_dict, idx):
        self._prepare_geometry(data_dict)

        output_dict = self(data_dict)

        metric_dict, psnr_raw = self._calc_metric(data_dict, output_dict)
        for metric_name, metric in metric_dict.items():
            self.log(f"test_metric/{metric_name}", metric, on_step=False, on_epoch=True, sync_dist=True)

        self.log("test_render_fps", output_dict["render_fps"], on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "test_token_selection_fps",
            output_dict["token_selection_fps"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log("test_total_fps", output_dict["total_fps"], on_step=False, on_epoch=True, sync_dist=True)

        if self.cfg.save_images or self.cfg.save_videos:
            self._save_test_samples(data_dict, output_dict, psnr_raw)
