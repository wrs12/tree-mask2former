import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from torch import nn
from torch.nn import functional as F


def resize_padding_mask(mask, size):
    # A feature cell touching padding is excluded from geometry supervision.
    return F.adaptive_max_pool2d(mask[:, None].float(), size)[:, 0].bool()


def mask_centroids(probabilities, valid=None):
    """Normalized (x, y) centers for [N, H, W] masks, in float32."""
    probabilities = probabilities.float()
    if valid is not None:
        probabilities = probabilities * valid.to(probabilities)
    height, width = probabilities.shape[-2:]
    x = (torch.arange(width, device=probabilities.device).float() + 0.5) / width
    y = (torch.arange(height, device=probabilities.device).float() + 0.5) / height
    mass = probabilities.sum((-2, -1)).clamp_min(1e-6)
    cx = (probabilities.sum(-2) * x).sum(-1) / mass
    cy = (probabilities.sum(-1) * y).sum(-1) / mass
    return torch.stack((cx, cy), dim=-1)


class TreeGeometryEncoder(nn.Module):
    def __init__(self, mask_dim):
        super().__init__()
        width = max(8, mask_dim // 4)

        def branch():
            return nn.Sequential(
                nn.Conv2d(mask_dim, width, 3, padding=1),
                nn.GroupNorm(1, width),
                nn.ReLU(),
                nn.Conv2d(width, 1, 1),
            )

        self.center = branch()
        self.boundary = branch()
        self.distance = branch()
        self.fusion = nn.Conv2d(3, mask_dim, 3, padding=1)
        nn.init.normal_(self.fusion.weight, std=1e-3)
        nn.init.zeros_(self.fusion.bias)
        nn.init.constant_(self.center[-1].bias, -2.19)

    def forward(self, features):
        predictions = {
            'center_logits': self.center(features),
            'boundary_logits': self.boundary(features),
            'distance_logits': self.distance(features),
        }
        priors = torch.cat([value.sigmoid() for value in predictions.values()], 1)
        return features + self.fusion(priors), predictions


class TreeAwareQueryGenerator(nn.Module):
    def __init__(self, mask_dim, hidden_dim, num_queries, num_guided, nms_kernel):
        super().__init__()
        if num_guided < 0:
            num_guided = max(1, num_queries // 2)
        if not 1 <= num_guided <= num_queries:
            raise ValueError('NUM_GUIDED_QUERIES must be in [1, NUM_OBJECT_QUERIES]')
        if nms_kernel < 1 or nms_kernel % 2 == 0:
            raise ValueError('NMS_KERNEL must be a positive odd integer')
        self.num_guided = num_guided
        self.nms_kernel = nms_kernel
        self.projection = nn.Linear(mask_dim, hidden_dim, bias=False)
        # Geometry heads receive explicit losses even at zero query residual.
        nn.init.zeros_(self.projection.weight)

    def forward(self, features, center_logits, queries, padding_mask=None):
        batch, channels, height, width = features.shape
        count = min(self.num_guided, height * width)
        scores = center_logits[:, 0].float().sigmoid()
        valid = torch.ones_like(scores, dtype=torch.bool)
        if padding_mask is not None:
            valid = ~resize_padding_mask(padding_mask, (height, width))
        scores_for_nms = scores.masked_fill(~valid, -1.0)
        pooled = F.max_pool2d(
            scores_for_nms[:, None], self.nms_kernel, 1, self.nms_kernel // 2
        )[:, 0]
        peaks = (scores_for_nms == pooled) & valid
        ranked = scores.masked_fill(~peaks, -1.0).flatten(1)
        indices = ranked.topk(count, dim=1).indices
        confidence = scores.flatten(1).gather(1, indices)
        confidence = confidence * peaks.flatten(1).gather(1, indices)
        sampled = features.flatten(2).transpose(1, 2).gather(
            1, indices[:, :, None].expand(batch, count, channels)
        )
        residual = self.projection(sampled)
        residual = residual * confidence.to(residual.dtype)[:, :, None] * 0.1
        guided = queries[:count] + residual.transpose(0, 1).to(queries.dtype)
        return torch.cat((guided, queries[count:]), dim=0)


@torch.no_grad()
def build_tree_targets(masks, padding_mask, output_size, sigma):
    """Build per-instance geometry before merging; never use the mask union EDT.

    Distance is an internal EDT normalized independently for each instance.
    Heatmaps use its maximum; matching uses mask centroids instead.
    """
    height, width = output_size
    full_height, full_width = masks.shape[-2:]
    valid = ~resize_padding_mask(padding_mask[None], output_size)[0]
    center = np.zeros((height, width), np.float32)
    boundary = np.zeros((full_height, full_width), np.float32)
    distance = torch.zeros(output_size, device=masks.device, dtype=torch.float32)
    foreground = torch.zeros_like(distance)
    yy, xx = np.mgrid[:height, :width]
    for binary in masks.detach().bool().cpu().numpy():
        if not binary.any():
            continue
        edt = distance_transform_edt(np.pad(binary, 1))[1:-1, 1:-1]
        cy, cx = np.unravel_index(edt.argmax(), edt.shape)
        cy = int(np.clip(np.floor((cy + 0.5) * height / full_height), 0, height - 1))
        cx = int(np.clip(np.floor((cx + 0.5) * width / full_width), 0, width - 1))
        gaussian = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        center = np.maximum(center, gaussian.astype(np.float32))
        # Explicit border_value=0 gives a defined boundary for cropped masks.
        structure = np.ones((3, 3), dtype=bool)
        ring = binary_dilation(binary, structure=structure) & ~binary_erosion(
            binary, structure=structure, border_value=0
        )
        boundary = np.maximum(boundary, ring.astype(np.float32))
        normalized = torch.from_numpy((edt / edt.max()).astype(np.float32)).to(masks.device)
        resized = F.interpolate(
            normalized[None, None], output_size, mode='bilinear', align_corners=False
        )[0, 0]
        distance = torch.maximum(distance, resized)
        binary_tensor = torch.from_numpy(binary.copy()).to(masks.device).float()
        foreground = torch.maximum(
            foreground, F.adaptive_max_pool2d(binary_tensor[None, None], output_size)[0, 0]
        )
    boundary = torch.from_numpy(boundary).to(masks.device)
    boundary = F.adaptive_max_pool2d(boundary[None, None], output_size)[0, 0]
    return {
        'center': torch.from_numpy(center).to(masks.device) * valid,
        'boundary': boundary * valid,
        'distance': distance * valid,
        'foreground': foreground * valid,
        'valid': valid,
    }
