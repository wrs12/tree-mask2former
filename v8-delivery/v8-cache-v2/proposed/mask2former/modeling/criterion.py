# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from detectron2.utils.comm import get_world_size

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Compute the DICE loss, similar to generalized IOU for masks.
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)

    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)

    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(dice_loss)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Compute point-wise sigmoid BCE loss.
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs,
        targets,
        reduction="none",
    )

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(sigmoid_ce_loss)


def calculate_uncertainty(logits):
    """
    Estimate uncertainty as the negative absolute logit magnitude.
    """
    assert logits.shape[1] == 1

    gt_class_logits = logits.clone()

    return -(torch.abs(gt_class_logits))


def morphological_boundary(masks, kernel_size=3):
    """
    Compute a differentiable morphological boundary map.

    Args:
        masks:
            [N, 1, H, W], values in [0, 1]

    Returns:
        boundaries:
            [N, 1, H, W]

    Boundary:
        dilation(mask) - erosion(mask)
    """

    assert masks.dim() == 4
    assert masks.shape[1] == 1
    assert kernel_size >= 3
    assert kernel_size % 2 == 1

    padding = kernel_size // 2

    dilated = F.max_pool2d(
        masks,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    eroded = 1.0 - F.max_pool2d(
        1.0 - masks,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    boundaries = (dilated - eroded).clamp(min=0.0, max=1.0)

    return boundaries


class SetCriterion(nn.Module):
    """This class computes the loss for DETR."""

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
        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # Original Mask2Former point sampling.
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

        # Final-only Boundary Dice.
        self.boundary_kernel_size = 3

    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss."""

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
        """
        Compute the original Mask2Former point-based mask BCE and Dice losses.
        """

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
        Compute final-only Boundary Dice loss.

        The predicted mask probability map is converted to a differentiable
        morphological boundary map.

        The matched GT mask is resized to the prediction-mask resolution and
        converted to the same morphological boundary representation.

        No random sampling, top-k, or extra point sampling is used.
        """

        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        src_masks = outputs["pred_masks"][src_idx]

        if src_masks.shape[0] == 0:
            return {"loss_boundary": src_masks.sum() * 0.0}

        pred_height = src_masks.shape[-2]
        pred_width = src_masks.shape[-1]

        matched_target_masks = []

        for target, (_, target_indices) in zip(targets, indices):
            if target_indices.numel() == 0:
                continue

            target_indices = target_indices.to(target["masks"].device)

            target_masks = target["masks"].index_select(
                0,
                target_indices,
            )

            target_masks = target_masks[:, None].to(src_masks)

            target_masks = F.interpolate(
                target_masks,
                size=(pred_height, pred_width),
                mode="nearest",
            )

            matched_target_masks.append(target_masks)

        if len(matched_target_masks) == 0:
            return {"loss_boundary": src_masks.sum() * 0.0}

        target_masks = torch.cat(matched_target_masks, dim=0)

        assert target_masks.shape[0] == src_masks.shape[0]
        assert target_masks.shape[-2:] == src_masks.shape[-2:]

        src_masks = src_masks[:, None]

        pred_probs = src_masks.sigmoid()

        pred_boundaries = morphological_boundary(
            pred_probs,
            kernel_size=self.boundary_kernel_size,
        )

        with torch.no_grad():
            target_masks = (target_masks > 0.5).to(pred_probs.dtype)

            target_boundaries = morphological_boundary(
                target_masks,
                kernel_size=self.boundary_kernel_size,
            )

        pred_boundaries = pred_boundaries.flatten(1)
        target_boundaries = target_boundaries.flatten(1)

        numerator = 2.0 * (
            pred_boundaries * target_boundaries
        ).sum(dim=1)

        denominator = (
            pred_boundaries.sum(dim=1)
            + target_boundaries.sum(dim=1)
        )

        instance_boundary_loss = 1.0 - (
            (numerator + 1.0)
            / (denominator + 1.0)
        )

        valid_boundary = (
            target_boundaries.sum(dim=1) > 0
        ).to(instance_boundary_loss.dtype)

        instance_boundary_loss = (
            instance_boundary_loss * valid_boundary
        )

        loss_boundary = instance_boundary_loss.sum() / num_masks

        del src_masks
        del target_masks
        del pred_probs
        del pred_boundaries
        del target_boundaries

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
        """
        Perform loss computation.
        """

        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != "aux_outputs"
        }

        # Final decoder matching.
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

        losses = {}

        # Original final-layer Mask2Former losses.
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

        # Boundary Dice:
        # final decoder only.
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