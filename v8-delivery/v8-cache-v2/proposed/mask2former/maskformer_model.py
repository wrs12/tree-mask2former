# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import BitMasks, Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .modeling.tree_criterion import TreeAwareSetCriterion
from .modeling.tree_geometry import build_tree_targets, mask_centroids


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        tree_build_boundary: bool = False,
        tree_build_distance: bool = False,
        tree_cache_enabled: bool = False,
        tree_cache_stride: int = 4,
    ):
        super().__init__()

        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.tree_enabled = getattr(getattr(sem_seg_head, "predictor", None), "tree_enabled", False)
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata

        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility

        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = (
            sem_seg_postprocess_before_inference
        )

        self.register_buffer(
            "pixel_mean",
            torch.Tensor(pixel_mean).view(-1, 1, 1),
            False,
        )

        self.register_buffer(
            "pixel_std",
            torch.Tensor(pixel_std).view(-1, 1, 1),
            False,
        )

        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        self.tree_build_boundary = tree_build_boundary
        self.tree_build_distance = tree_build_distance
        self.tree_cache_enabled = tree_cache_enabled
        self.tree_cache_stride = tree_cache_stride

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        sem_seg_head = build_sem_seg_head(
            cfg,
            backbone.output_shape(),
        )

        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        tree = cfg.MODEL.TREE_AWARE
        if tree.ENABLED and not getattr(getattr(sem_seg_head, 'predictor', None), 'tree_enabled', False):
            raise ValueError('V8 requires MultiScaleMaskedTransformerDecoder')

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            cost_geometry=tree.GEOMETRY_MATCH_COST if tree.ENABLED else 0.0,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}

            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {
                        k + f"_{i}": v
                        for k, v in weight_dict.items()
                    }
                )

            weight_dict.update(aux_weight_dict)

        # Geometry, separation, and matched boundary losses are final-only.
        if tree.MASK_BOUNDARY_WEIGHT > 0:
            weight_dict['loss_boundary'] = tree.MASK_BOUNDARY_WEIGHT
        if tree.ENABLED:
            weight_dict.update({
                'loss_tree_center': tree.CENTER_WEIGHT,
                'loss_tree_boundary': tree.BOUNDARY_WEIGHT,
                'loss_tree_distance': tree.DISTANCE_WEIGHT,
                'loss_query_separation': tree.SEPARATION_WEIGHT,
            })

        losses = [
            "labels",
            "masks",
        ]

        criterion_class = TreeAwareSetCriterion if tree.ENABLED else SetCriterion
        tree_arguments = {}
        if tree.ENABLED:
            if tree.CENTER_WEIGHT <= 0:
                raise ValueError('V8 hard Top-K queries require CENTER_WEIGHT > 0')
            tree_arguments = {
                'center_sigma': tree.CENTER_SIGMA,
                'separation_margin': tree.SEPARATION_MARGIN,
                'separation_radius': tree.SEPARATION_RADIUS,
                'positive_weight': tree.SEPARATION_POS_WEIGHT,
                'negative_weight': tree.SEPARATION_NEG_WEIGHT,
            }
        criterion = criterion_class(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            **tree_arguments,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": (
                cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD
            ),
            "overlap_threshold": (
                cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD
            ),
            "metadata": MetadataCatalog.get(
                cfg.DATASETS.TRAIN[0]
            ),
            "size_divisibility": (
                cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY
            ),
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "tree_build_boundary": tree.BOUNDARY_WEIGHT > 0,
            "tree_build_distance": tree.DISTANCE_WEIGHT > 0,
            "tree_cache_enabled": tree.CACHE_ENABLED,
            "tree_cache_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = [
            x["image"].to(self.device)
            for x in batched_inputs
        ]

        images = [
            (x - self.pixel_mean) / self.pixel_std
            for x in images
        ]

        images = ImageList.from_tensors(
            images,
            self.size_divisibility,
        )

        features = self.backbone(
            images.tensor
        )

        padding_mask = None
        if self.tree_enabled:
            padding_mask = torch.ones(
                images.tensor.shape[0], *images.tensor.shape[-2:],
                dtype=torch.bool, device=images.tensor.device,
            )
            for batch, (height, width) in enumerate(images.image_sizes):
                padding_mask[batch, :height, :width] = False
        outputs = self.sem_seg_head(features, padding_mask)

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [
                    x["instances"].to(self.device)
                    for x in batched_inputs
                ]

                targets = self.prepare_targets(
                    gt_instances,
                    images,
                    batched_inputs=batched_inputs,
                    geometry_size=(
                        outputs['tree_geometry']['center_logits'].shape[-2:]
                        if self.tree_enabled else None
                    ),
                )

            else:
                targets = None

            losses = self.criterion(
                outputs,
                targets,
            )

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]

                else:
                    losses.pop(k)

            return losses

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]

        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(
                images.tensor.shape[-2],
                images.tensor.shape[-1],
            ),
            mode="bilinear",
            align_corners=False,
        )

        del outputs

        processed_results = []

        for (
            mask_cls_result,
            mask_pred_result,
            input_per_image,
            image_size,
        ) in zip(
            mask_cls_results,
            mask_pred_results,
            batched_inputs,
            images.image_sizes,
        ):
            height = input_per_image.get(
                "height",
                image_size[0],
            )

            width = input_per_image.get(
                "width",
                image_size[1],
            )

            processed_results.append({})

            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(
                    sem_seg_postprocess
                )(
                    mask_pred_result,
                    image_size,
                    height,
                    width,
                )

                mask_cls_result = mask_cls_result.to(
                    mask_pred_result
                )

            if self.semantic_on:
                r = retry_if_cuda_oom(
                    self.semantic_inference
                )(
                    mask_cls_result,
                    mask_pred_result,
                )

                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(
                        sem_seg_postprocess
                    )(
                        r,
                        image_size,
                        height,
                        width,
                    )

                processed_results[-1]["sem_seg"] = r

            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(
                    self.panoptic_inference
                )(
                    mask_cls_result,
                    mask_pred_result,
                )

                processed_results[-1][
                    "panoptic_seg"
                ] = panoptic_r

            if self.instance_on:
                instance_r = retry_if_cuda_oom(
                    self.instance_inference
                )(
                    mask_cls_result,
                    mask_pred_result,
                )

                processed_results[-1][
                    "instances"
                ] = instance_r

        return processed_results

    def prepare_targets(self, targets, images, geometry_size=None, batched_inputs=None):
        h_pad, w_pad = images.tensor.shape[-2:]

        new_targets = []

        for batch, targets_per_image in enumerate(targets):
            gt_masks = targets_per_image.gt_masks
            if isinstance(gt_masks, BitMasks):
                gt_masks = gt_masks.tensor

            padded_masks = torch.zeros(
                (
                    gt_masks.shape[0],
                    h_pad,
                    w_pad,
                ),
                dtype=gt_masks.dtype,
                device=gt_masks.device,
            )

            padded_masks[
                :,
                : gt_masks.shape[1],
                : gt_masks.shape[2],
            ] = gt_masks

            target = {
                'labels': targets_per_image.gt_classes,
                'masks': padded_masks,
            }
            if geometry_size is not None:
                height, width = images.image_sizes[batch]
                padding = torch.ones((h_pad, w_pad), device=gt_masks.device, dtype=torch.bool)
                padding[:height, :width] = False
                target['padding_mask'] = padding
                if self.tree_cache_enabled:
                    if batched_inputs is None or 'tree_geometry_cache' not in batched_inputs[batch]:
                        raise ValueError('CACHE_ENABLED requires the updated ITD mapper and offline cache')
                    cached = batched_inputs[batch]['tree_geometry_cache']
                    meta = cached['meta']
                    if tuple(meta['image_size']) != (height, width):
                        raise ValueError('Cached geometry image size differs from augmented input')
                    stride = self.tree_cache_stride
                    if (h_pad, w_pad) != (geometry_size[0] * stride, geometry_size[1] * stride):
                        raise ValueError('Geometry output stride differs from offline cache')
                    if meta['stride'] != stride or meta['divisor'] != self.size_divisibility:
                        raise ValueError('Cached geometry stride or padding divisor changed')
                    if meta['sigma'] != self.criterion.center_sigma:
                        raise ValueError('Cached center sigma changed')
                    gh, gw = meta['feature_size']
                    if gh > geometry_size[0] or gw > geometry_size[1]:
                        raise ValueError('Cached geometry is larger than the current batch')
                    fields = ['center', 'valid']
                    if self.tree_build_boundary:
                        fields.append('boundary')
                    if self.tree_build_distance:
                        fields.extend(('distance', 'foreground'))
                    geometry = {}
                    for field in fields:
                        tensor = cached['maps'][field]
                        if tuple(tensor.shape) != (gh, gw):
                            raise ValueError('Cached geometry map shape mismatch')
                        dtype = torch.bool if field == 'valid' else torch.float32
                        tensor = tensor.to(device=gt_masks.device, dtype=dtype)
                        geometry[field] = F.pad(tensor, (0, geometry_size[1] - gw,
                                                       0, geometry_size[0] - gh))
                    centroids = cached['centroid_pixels'].to(device=gt_masks.device, dtype=torch.float32)
                    if centroids.shape != (gt_masks.shape[0], 2):
                        raise ValueError('Cached instance ordering/count changed')
                    target['tree_centroids'] = centroids / centroids.new_tensor([w_pad, h_pad])
                    target['tree_geometry'] = geometry
                else:
                    target['tree_centroids'] = mask_centroids(padded_masks.float(), ~padding)
                    target['tree_geometry'] = build_tree_targets(
                        padded_masks, padding, geometry_size, self.criterion.center_sigma,
                        build_boundary=self.tree_build_boundary,
                        build_distance=self.tree_build_distance,
                    )
            new_targets.append(target)

        return new_targets

    def semantic_inference(
        self,
        mask_cls,
        mask_pred,
    ):
        mask_cls = F.softmax(
            mask_cls,
            dim=-1,
        )[..., :-1]

        mask_pred = mask_pred.sigmoid()

        semseg = torch.einsum(
            "qc,qhw->chw",
            mask_cls,
            mask_pred,
        )

        return semseg

    def panoptic_inference(
        self,
        mask_cls,
        mask_pred,
    ):
        scores, labels = F.softmax(
            mask_cls,
            dim=-1,
        ).max(-1)

        mask_pred = mask_pred.sigmoid()

        keep = (
            labels.ne(self.sem_seg_head.num_classes)
            & (scores > self.object_mask_threshold)
        )

        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = (
            cur_scores.view(-1, 1, 1)
            * cur_masks
        )

        h, w = cur_masks.shape[-2:]

        panoptic_seg = torch.zeros(
            (h, w),
            dtype=torch.int32,
            device=cur_masks.device,
        )

        segments_info = []
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            return panoptic_seg, segments_info

        cur_mask_ids = cur_prob_masks.argmax(0)

        stuff_memory_list = {}

        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()

            isthing = (
                pred_class
                in self.metadata.thing_dataset_id_to_contiguous_id.values()
            )

            mask_area = (
                cur_mask_ids == k
            ).sum().item()

            original_area = (
                cur_masks[k] >= 0.5
            ).sum().item()

            mask = (
                (cur_mask_ids == k)
                & (cur_masks[k] >= 0.5)
            )

            if (
                mask_area > 0
                and original_area > 0
                and mask.sum().item() > 0
            ):
                if (
                    mask_area / original_area
                    < self.overlap_threshold
                ):
                    continue

                if not isthing:
                    if (
                        int(pred_class)
                        in stuff_memory_list.keys()
                    ):
                        panoptic_seg[
                            mask
                        ] = stuff_memory_list[
                            int(pred_class)
                        ]

                        continue

                    stuff_memory_list[
                        int(pred_class)
                    ] = (
                        current_segment_id + 1
                    )

                current_segment_id += 1

                panoptic_seg[
                    mask
                ] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(isthing),
                        "category_id": int(pred_class),
                    }
                )

        return panoptic_seg, segments_info

    def instance_inference(
        self,
        mask_cls,
        mask_pred,
    ):
        image_size = mask_pred.shape[-2:]

        scores = F.softmax(
            mask_cls,
            dim=-1,
        )[:, :-1]

        labels = (
            torch.arange(
                self.sem_seg_head.num_classes,
                device=self.device,
            )
            .unsqueeze(0)
            .repeat(
                self.num_queries,
                1,
            )
            .flatten(
                0,
                1,
            )
        )

        scores_per_image, topk_indices = (
            scores.flatten(
                0,
                1,
            ).topk(
                self.test_topk_per_image,
                sorted=False,
            )
        )

        labels_per_image = labels[
            topk_indices
        ]

        topk_indices = (
            topk_indices
            // self.sem_seg_head.num_classes
        )

        mask_pred = mask_pred[
            topk_indices
        ]

        if self.panoptic_on:
            keep = torch.zeros_like(
                scores_per_image
            ).bool()

            for i, lab in enumerate(
                labels_per_image
            ):
                keep[i] = (
                    lab
                    in self.metadata.thing_dataset_id_to_contiguous_id.values()
                )

            scores_per_image = (
                scores_per_image[keep]
            )

            labels_per_image = (
                labels_per_image[keep]
            )

            mask_pred = mask_pred[
                keep
            ]

        result = Instances(
            image_size
        )

        result.pred_masks = (
            mask_pred > 0
        ).float()

        result.pred_boxes = Boxes(
            torch.zeros(
                mask_pred.size(0),
                4,
            )
        )

        mask_scores_per_image = (
            mask_pred.sigmoid().flatten(1)
            * result.pred_masks.flatten(1)
        ).sum(1) / (
            result.pred_masks.flatten(1).sum(1)
            + 1e-6
        )

        result.scores = (
            scores_per_image
            * mask_scores_per_image
        )

        result.pred_classes = (
            labels_per_image
        )

        return result