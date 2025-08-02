import torch
from torch import Tensor
from jaxtyping import Float
from einops import reduce, rearrange
import functools
import warnings
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn as nn
from typing import Dict, Tuple
from skimage.metrics import structural_similarity



# Suppress warnings for LPIPS loss loading
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum.*")


class MetricComputer(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        # self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)


    def forward(
            self, 
            predicted: Float[Tensor, "batch channel height width"], 
            ground_truth: Float[Tensor, "batch channel height width"]
        ) -> Tuple[Dict[str, Float[Tensor, "batch"]], Float[Tensor, "batch"]]:
        metric_dict = {}

        # metric_dict['ssim'] = self.ssim(predicted, ground_truth)
        metric_dict['ssim'] = compute_ssim(predicted, ground_truth).mean()
        psnr_raw = compute_psnr(predicted, ground_truth)
        metric_dict['psnr'] = psnr_raw.mean()
        metric_dict['lpips'] = compute_lpips(predicted, ground_truth).mean()

        return metric_dict, psnr_raw


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, "batch"]:
    """
    Compute Peak Signal-to-Noise Ratio between ground truth and predicted images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width], values in [0, 1]
        predicted: Images with shape [batch, channel, height, width], values in [0, 1]
        
    Returns:
        PSNR values for each image in the batch
    """
    ground_truth = torch.clamp(ground_truth, 0, 1)
    predicted = torch.clamp(predicted, 0, 1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * torch.log10(mse) 



@functools.lru_cache(maxsize=None)
def get_lpips_model(net_type="vgg", device="cuda"):
    from lpips import LPIPS
    return LPIPS(net=net_type).to(device)

@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
    normalize: bool = True,
) -> Float[Tensor, "batch"]:
    """
    Compute Learned Perceptual Image Patch Similarity between images.
    
    Args:
        ground_truth: Images with shape [batch, channel, height, width]
        predicted: Images with shape [batch, channel, height, width]
        The value range is [0, 1] when we have set the normalize flag to True.
        It will be [-1, 1] when the normalize flag is set to False.
    Returns:
        LPIPS values for each image in the batch (lower is better)
    """

    _lpips_fn = get_lpips_model(device=predicted.device)
    batch_size = 10  # Process in batches to save memory
    values = [
        _lpips_fn(
            ground_truth[i : i + batch_size],
            predicted[i : i + batch_size],
            normalize=normalize,
        )
        for i in range(0, ground_truth.shape[0], batch_size)
    ]
    return torch.cat(values, dim=0).squeeze()


def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)