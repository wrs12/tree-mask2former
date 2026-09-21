# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


def calculate_uncertainty(logits):
    """
    We estimate uncertainty as L1 distance between 0.0 and the logit prediction in
    'logits' for the foreground class in `classes`.

    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all
            images. The values are logits.

    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty
            scores with the most uncertain locations having the highest score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


@torch.no_grad()
def sample_scale_aware_unique_boundary_point_coords(
    target_masks,
    num_points=512,
    small_boundary_weight=0.0,
    medium_boundary_weight=0.1,
    large_boundary_weight=0.2,
    reference_size=1024,
    boundary_kernel_size=3,
    chunk_size=8,
):
    """
    Sample unique GT boundary points for V3B auxiliary boundary loss.

    Args:
        target_masks:
            [N, 1, H, W]

    Returns:
        boundary_coords:
            [N, num_points, 2]

        valid_boundary:
            [N, num_points]

        boundary_weights:
            [N]

    Scale-aware weights:
        small:  equivalent side < 32       -> 0.0
        medium: 32 <= equivalent side < 96 -> 0.1
        large:  equivalent side >= 96      -> 0.2

    Boundary definition:
        dilation - erosion

    Boundary sampling:
        unique sampling without replacement

    Important:
        Scale estimation is performed inside the same chunk loop used for
        boundary extraction. No second boundary pass is introduced.
    """

    assert target_masks.dim() == 4
    assert target_masks.shape[1] == 1
    assert num_points > 0
    assert boundary_kernel_size >= 3
    assert boundary_kernel_size % 2 == 1
    assert chunk_size >= 1

    num_masks = target_masks.shape[0]
    height = target_masks.shape[-2]
    width = target_masks.shape[-1]
    device = target_masks.device

    assert num_points <= height * width

    if num_masks == 0:
        return (
            torch.empty(
                (0, num_points, 2),
                device=device,
                dtype=torch.float32,
            ),
            torch.empty(
                (0, num_points),
                device=device,
                dtype=torch.bool,
            ),
            torch.empty(
                (0,),
                device=device,
                dtype=torch.float32,
            ),
        )

    padding = boundary_kernel_size // 2

    image_area = float(height * width)
    reference_area = float(reference_size * reference_size)

    small_area_threshold = float(32 * 32)
    medium_area_threshold = float(96 * 96)

    boundary_coord_chunks = []
    valid_boundary_chunks = []
    boundary_weight_chunks = []

    for start in range(0, num_masks, chunk_size):
        end = min(start + chunk_size, num_masks)

        masks = (target_masks[start:end] > 0.5).float()

        # Scale estimation is intentionally done here so that target masks
        # are not traversed in a separate scale-analysis function.
        mask_areas = masks[:, 0].sum(dim=(-2, -1))
        area_ratio = mask_areas / image_area
        equivalent_area = area_ratio * reference_area

        boundary_weights = torch.full(
            (end - start,),
            large_boundary_weight,
            device=device,
            dtype=torch.float32,
        )

        small_mask = equivalent_area < small_area_threshold
        medium_mask = (
            (equivalent_area >= small_area_threshold)
            & (equivalent_area < medium_area_threshold)
        )

        boundary_weights[small_mask] = small_boundary_weight
        boundary_weights[medium_mask] = medium_boundary_weight

        dilated = F.max_pool2d(
            masks,
            kernel_size=boundary_kernel_size,
            stride=1,
            padding=padding,
        )

        eroded = 1.0 - F.max_pool2d(
            1.0 - masks,
            kernel_size=boundary_kernel_size,
            stride=1,
            padding=padding,
        )

        boundary = (dilated - eroded) > 0.5
        boundary_flat = boundary[:, 0].flatten(1)

        random_scores = torch.rand(
            boundary_flat.shape,
            device=device,
            dtype=torch.float32,
        )

        random_scores.masked_fill_(~boundary_flat, -1.0)

        selected_scores, selected_linear_indices = torch.topk(
            random_scores,
            k=num_points,
            dim=1,
            largest=True,
            sorted=False,
        )

        valid_boundary = selected_scores >= 0.0

        y = torch.div(
            selected_linear_indices,
            width,
            rounding_mode="floor",
        )

        x = selected_linear_indices % width

        x = (x.float() + 0.5) / float(width)
        y = (y.float() + 0.5) / float(height)

        boundary_coords = torch.stack([x, y], dim=-1)

        boundary_coord_chunks.append(boundary_coords)
        valid_boundary_chunks.append(valid_boundary)
        boundary_weight_chunks.append(boundary_weights)

    boundary_coords = torch.cat(boundary_coord_chunks, dim=0)
    valid_boundary = torch.cat(valid_boundary_chunks, dim=0)
    boundary_weights = torch.cat(boundary_weight_chunks, dim=0)

    assert boundary_coords.shape == (
        num_masks,
        num_points,
        2,
    )

    assert valid_boundary.shape == (
        num_masks,
        num_points,
    )

    assert boundary_weights.shape == (num_masks,)

    return boundary_coords, valid_boundary, boundary_weights


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the
           outputs of the model
        2) we supervise each pair of matched ground-truth / prediction
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef,
        losses,
        num_points,
        oversample_ratio,
        importance_sample_ratio,
    ):
        """Create the criterion.

        Parameters:
            num_classes: number of object categories, omitting the special
                no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as
                values their relative weight
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied
        """
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # Original Mask2Former pointwise mask loss parameters.
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # V3B scale-aware auxiliary boundary loss parameters.
        self.boundary_num_points = 512
        self.boundary_kernel_size = 3
        self.boundary_chunk_size = 8
        self.boundary_reference_size = 1024

        self.small_boundary_weight = 0.0
        self.medium_boundary_weight = 0.1
        self.large_boundary_weight = 0.2

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)."""
        assert "pred_logits" in outputs

        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)

        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),
            target_classes,
            self.empty_weight,
        )

        return {"loss_ce": loss_ce}

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the original Mask2Former point mask and dice losses."""
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        masks = [t["masks"] for t in targets]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()

        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # No need to upsample predictions as we are using normalized coordinates.
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )

            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(
                point_logits,
                point_labels,
                num_masks,
            ),
            "loss_dice": dice_loss_jit(
                point_logits,
                point_labels,
                num_masks,
            ),
        }

        del src_masks
        del target_masks

        return losses

    def loss_boundary(self, outputs, targets, indices, num_masks):
        """
        Compute V3B scale-aware auxiliary GT boundary BCE loss.

        Important:
            - final decoder only
            - does not replace original Mask2Former points
            - 512 unique boundary points per matched instance
            - small weight  = 0.0
            - medium weight = 0.1
            - large weight  = 0.2
        """

        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]

        if src_masks.shape[0] == 0:
            return {"loss_boundary": src_masks.sum() * 0.0}

        matched_target_masks = []

        for target, (_, target_indices) in zip(targets, indices):
            target_indices = target_indices.to(target["masks"].device)

            matched_target_masks.append(
                target["masks"].index_select(
                    0,
                    target_indices,
                )
            )

        target_masks = torch.cat(matched_target_masks, dim=0)
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            (
                boundary_coords,
                valid_boundary,
                boundary_weights,
            ) = sample_scale_aware_unique_boundary_point_coords(
                target_masks,
                num_points=self.boundary_num_points,
                small_boundary_weight=self.small_boundary_weight,
                medium_boundary_weight=self.medium_boundary_weight,
                large_boundary_weight=self.large_boundary_weight,
                reference_size=self.boundary_reference_size,
                boundary_kernel_size=self.boundary_kernel_size,
                chunk_size=self.boundary_chunk_size,
            )

            boundary_labels = point_sample(
                target_masks,
                boundary_coords,
                align_corners=False,
            ).squeeze(1)

        boundary_logits = point_sample(
            src_masks,
            boundary_coords,
            align_corners=False,
        ).squeeze(1)

        boundary_bce = F.binary_cross_entropy_with_logits(
            boundary_logits,
            boundary_labels,
            reduction="none",
        )

        valid_boundary = valid_boundary.to(boundary_bce.dtype)

        valid_counts = valid_boundary.sum(dim=1)

        instance_boundary_loss = (
            (boundary_bce * valid_boundary).sum(dim=1)
            / valid_counts.clamp(min=1.0)
        )

        has_boundary = (valid_counts > 0).to(instance_boundary_loss.dtype)

        instance_boundary_loss = (
            instance_boundary_loss
            * has_boundary
            * boundary_weights.to(instance_boundary_loss.dtype)
        )

        loss_boundary = instance_boundary_loss.sum() / num_masks

        del src_masks
        del target_masks

        return {"loss_boundary": loss_boundary}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [
                torch.full_like(src, i)
                for i, (src, _) in enumerate(indices)
            ]
        )

        src_idx = torch.cat(
            [src for (src, _) in indices]
        )

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat(
            [
                torch.full_like(tgt, i)
                for i, (_, tgt) in enumerate(indices)
            ]
        )

        tgt_idx = torch.cat(
            [tgt for (_, tgt) in indices]
        )

        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks,
        }

        assert loss in loss_map, f"do you really want to compute {loss} loss?"

        return loss_map[loss](
            outputs,
            targets,
            indices,
            num_masks,
        )

    def forward(self, outputs, targets):
        """This performs the loss computation."""

        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs"
        }

        # Final-layer Hungarian matching.
        indices = self.matcher(
            outputs_without_aux,
            targets,
        )

        num_masks = sum(
            len(t["labels"])
            for t in targets
        )

        num_masks = torch.as_tensor(
            [num_masks],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)

        num_masks = torch.clamp(
            num_masks / get_world_size(),
            min=1,
        ).item()

        # Original Mask2Former final losses.
        losses = {}

        for loss in self.losses:
            losses.update(
                self.get_loss(
                    loss,
                    outputs,
                    targets,
                    indices,
                    num_masks,
                )
            )

        # V3B auxiliary boundary loss.
        # IMPORTANT: final decoder only.
        losses.update(
            self.loss_boundary(
                outputs_without_aux,
                targets,
                indices,
                num_masks,
            )
        )

        # Original Mask2Former auxiliary losses.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(
                    aux_outputs,
                    targets,
                )

                for loss in self.losses:
                    l_dict = self.get_loss(
                        loss,
                        aux_outputs,
                        targets,
                        indices,
                        num_masks,
                    )

                    l_dict = {
                        k + f"_{i}": v
                        for k, v in l_dict.items()
                    }

                    losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__

        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(
                self.importance_sample_ratio
            ),
        ]

        _repr_indent = 4

        lines = [
            head
        ] + [
            " " * _repr_indent + line
            for line in body
        ]

        return "\n".join(lines)