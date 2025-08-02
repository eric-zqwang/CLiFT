import torch
import numpy as np
import wandb
import os
from PIL import Image, ImageDraw
import wandb


def log_images(
        pred_images, 
        gt_images, 
        condition_images, 
        logger,
        global_step,
        data_dict
    ):
    bs = pred_images.shape[0]

    for i in range(bs):
        sampled_views_comparison = []
        condition_views = []

        # Sampled views comparison
        pred_row = np.concatenate([
            pred_images[i, j] for j in range(pred_images.shape[1])
        ], axis=1)

        gt_row = np.concatenate([
            gt_images[i, j] for j in range(gt_images.shape[1])
        ], axis=1)
        
        combined_sampled_image = np.concatenate([pred_row, gt_row], axis=0)

        # Log sampled views comparison
        sampled_views_comparison.append(wandb.Image(
            combined_sampled_image,
            caption=f"Sampled Views"
        ))

        # Add condition images visualization
        condition_row = np.concatenate([
            condition_images[i, j] for j in range(condition_images.shape[1])
        ], axis=1)

        # Log condition views
        condition_views.append(wandb.Image(
            condition_row,
            caption=f"Condition Views"
        ))

        logger.experiment.log({
            f"sampled_views/{data_dict['data_id'][i]}": sampled_views_comparison,
            f"condition_views/{data_dict['data_id'][i]}": condition_views,
        }, step=global_step)


def _save(images, save_path, index, psnr=None):
    if psnr is not None:
        suffix = f"_psnr_{psnr:.2f}"
    else:
        suffix = ""

    if images.ndim == 3:
        image = Image.fromarray(images)
        image.save(f"{save_path}_{index}{suffix}.png")
        return

    for i in range(images.shape[0]):
        image = Image.fromarray(images[i])  # Convert to PIL image
        image.save(f"{save_path}_{index}{suffix}.png")  # Save image


def save_images(
        timestamps,
        save_dir,
        condition_images=None,
        pred_images=None,
        gt_images=None,
        psnr=None
    ):
    # bs = pred_images.shape[0]
    
    os.makedirs(save_dir, exist_ok=True)
    
    if condition_images is not None:
        for i in range(condition_images.shape[0]):
            _save(condition_images[i], save_dir+"/condition", i)
    
    if pred_images is not None:
        for i in range(pred_images.shape[0]):
            _save(pred_images[i], save_dir+"/rendered", i, psnr[i] if psnr is not None else None)
    
    if gt_images is not None:
        for i in range(gt_images.shape[0]):
            _save(gt_images[i], save_dir+"/gt", i)

import skvideo.io
from pathlib import Path

def save_videos(images, save_dir, fps=30, condition_images=None):
    # Image.fromarray(prep_image(image)).save(path)
    path = Path(save_dir)
    path.parent.mkdir(exist_ok=True, parents=True)
    
    frames = [images[0]]
    
        
    outputdict = {'-pix_fmt': 'yuv420p', '-crf': '23',
                  '-vf': f'setpts=1.*PTS'}
                  
    if fps is not None:
        outputdict.update({'-r': str(fps)})

    writer = skvideo.io.FFmpegWriter(path,
                                     outputdict=outputdict)
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()



def log_videos(pred_images, gt_images, condition_images, logger, global_step, data_dict):
    bs = pred_images.shape[0]
    for b in range(bs):
        pred_video_frames = []
        gt_video_frames = []
        for v in range(pred_images.shape[1]):
            frame = pred_images[b, v]
            pred_video_frames.append(frame)

        for v in range(gt_images.shape[1]):
            frame = gt_images[b, v]
            gt_video_frames.append(frame)
        
        pred_video_array = np.stack(pred_video_frames)
        gt_video_array = np.stack(gt_video_frames)
        pred_video_array = np.transpose(pred_video_array, (0, 3, 1, 2))
        gt_video_array = np.transpose(gt_video_array, (0, 3, 1, 2))

        condition_views = []

        # Add condition images visualization
        condition_row = np.concatenate([
            condition_images[b, j] for j in range(condition_images.shape[1])
        ], axis=1)

        # Log condition views
        condition_views.append(wandb.Image(
            condition_row,
            # caption=f"Condition Views"
        ))

        logger.experiment.log({
            f"reconstruction_video/{data_dict['data_id'][b]}": wandb.Video(pred_video_array, fps=16, format="mp4"),
            f"gt_video/{data_dict['data_id'][b]}": wandb.Video(gt_video_array, fps=16, format="mp4"),
            f"condition_views/{data_dict['data_id'][b]}": condition_views,
        }, step=global_step)



def visualize_policy(images_all, policy, save_path=None, log=False):
    """Visualize selected patches according to the policy, making other areas white.
    
    Args:
        images_all: Images of shape [7, H, W, C]
        policy: Tensor of shape [16, 2007] containing indices for 16 different views
        save_path: Path to save the visualization (optional)
        log: If True, return list of images instead of saving
    Returns:
        If log=True, returns list of visualizations for each view
    """
    # Constants
    patch_size = 8
    tokens_per_image = 32 * 32  # 1024 tokens per image (32x32 patches)
    policy_visualizations = [] if log else None
    
    # Process each view
    for view_idx in range(policy.shape[0]):
        # Create a white canvas for all 7 images
        H, W = images_all.shape[1:3]
        combined_mask = np.zeros((H, W * 4, 3), dtype=np.uint8)
        combined_mask[:, :, 0] = 255  # Set red channel to 255
        
        for img_idx in range(4):
            img = images_all[img_idx]
            
            # Calculate which patches belong to this image
            start_idx = img_idx * tokens_per_image
            end_idx = (img_idx + 1) * tokens_per_image
            
            # Find selected patches for this image
            image_patches = policy[view_idx][
                (policy[view_idx] >= start_idx) & 
                (policy[view_idx] < end_idx)
            ]
            
            # Convert global indices to local patch coordinates
            local_indices = image_patches - start_idx
            for idx in local_indices:
                # Convert token index to x,y coordinates
                y = (idx // 32) * patch_size
                x = (idx % 32) * patch_size
                # Copy the selected patch from original image to mask
                combined_mask[y:y+patch_size, x + img_idx*W:x + img_idx*W + patch_size] = \
                    img[y:y+patch_size, x:x+patch_size]
        
        if log:
            policy_visualizations.append(combined_mask)
        else:
            # Save the combined visualization for this view
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray(combined_mask).save(f"{save_path}_view{view_idx}.png")
    
    return policy_visualizations if log else None

def vis_policy_per_scene(
        images_all, 
        policy, 
        num_context_views,
        patch_size=8, 
        save_path=None, 
        log=False
    ):
    # Get actual image dimensions
    H, W = images_all.shape[1:3]
    
    # Calculate tokens per image based on actual dimensions
    tokens_per_image = (H * W) // (patch_size**2)
    
    # Calculate number of patches in each dimension
    n_patch_h = H // patch_size
    n_patch_w = W // patch_size
    
    # Create a canvas for all context views
    combined_mask = np.zeros((H, W * num_context_views, 3), dtype=np.uint8)
    combined_mask[:, :, 0] = 255  # Set red channel to 255
    
    for img_idx in range(num_context_views):
        img = images_all[img_idx]
        
        # Calculate which patches belong to this image
        start_idx = img_idx * tokens_per_image
        end_idx = (img_idx + 1) * tokens_per_image
        
        # Find selected patches for this image
        image_patches = policy[
            (policy >= start_idx) & 
            (policy < end_idx)
        ]
        
        # Convert global indices to local patch coordinates
        local_indices = image_patches - start_idx
        for idx in local_indices:
            # Convert linear token index to 2D patch coordinates
            # Assuming row-major order: idx = row * width + col
            # Where row ranges from 0 to n_patch_h-1 and col ranges from 0 to n_patch_w-1
            patch_idx_y = idx // n_patch_w
            patch_idx_x = idx % n_patch_w
            
            # Convert patch indices to pixel coordinates
            y = patch_idx_y * patch_size
            x = patch_idx_x * patch_size
            
            # Ensure we don't go out of bounds
            if y < H and x < W:
                # Copy the selected patch from original image to mask
                combined_mask[y:y+patch_size, x + img_idx*W:x + img_idx*W + patch_size] = \
                    img[y:y+patch_size, x:x+patch_size]
    
    if log:
        policy_visualizations = combined_mask
    else:
        # Save the combined visualization for this view
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(combined_mask).save(f"{save_path}.png")
    
    return policy_visualizations if log else None