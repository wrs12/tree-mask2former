import copy
import itertools
import os
from typing import Any, Dict, List, Set

import numpy as np
import torch

from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    detection_utils as utils,
    transforms as T,
)
from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.events import CommonMetricPrinter, JSONWriter

class ITDCOCOEvaluator(COCOEvaluator):

    def _eval_predictions(self, predictions, img_ids=None):
        """
        COCO evaluation with larger maxDets for high-density instance segmentation.
        """

        super()._eval_predictions(
            predictions,
            img_ids
        )

        if hasattr(self, "_coco_eval"):

            for task, coco_eval in self._coco_eval.items():

                coco_eval.params.maxDets = [
                    1,
                    10,
                    700
                ]

                print(
                    f"[ITDCOCOEvaluator] "
                    f"{task} maxDets = "
                    f"{coco_eval.params.maxDets}"
                )

class ITDMask2FormerMapper:
    """
    ITD 数据集 Mapper。

    训练：
    - RGB PNG 读取
    - 多尺度 resize
    - 水平/垂直翻转
    - 亮度/对比度增强
    - COCO polygon/RLE 转实例 mask
    - gt_masks 转为 Mask2Former 需要的 Tensor

    验证：
    - RGB PNG 读取
    - 确定性 resize
    """

    def __init__(self, cfg, is_train=True):

        self.is_train = is_train
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.input_format = cfg.INPUT.FORMAT
        self.tree_cache = None
        tree = getattr(cfg.MODEL, 'TREE_AWARE', None)
        if is_train and tree is not None and tree.ENABLED and tree.CACHE_ENABLED:
            from mask2former.modeling.tree_geometry_cache import TreeGeometryCache, cache_profile
            self.tree_cache = TreeGeometryCache(
                tree.CACHE_DIR, cfg.DATASETS.TRAIN, cache_profile(cfg),
                tree.BOUNDARY_WEIGHT > 0, tree.DISTANCE_WEIGHT > 0,
                tree.CACHE_MEMORY_ITEMS,
            )

        if self.input_format != "RGB":
            raise ValueError(
                "本 Mapper 按 RGB 读取 PNG，"
                f"cfg.INPUT.FORMAT 应为 RGB，实际为 {self.input_format}"
            )

        if is_train:

            self.augmentations = T.AugmentationList([
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TRAIN,
                    max_size=cfg.INPUT.MAX_SIZE_TRAIN,
                    sample_style="choice",
                ),

                T.RandomFlip(
                    prob=0.5,
                    horizontal=True,
                    vertical=False,
                ),

                T.RandomFlip(
                    prob=0.5,
                    horizontal=False,
                    vertical=True,
                ),

                T.RandomBrightness(
                    intensity_min=0.8,
                    intensity_max=1.2,
                ),

                T.RandomContrast(
                    intensity_min=0.8,
                    intensity_max=1.2,
                ),
            ])

        else:

            test_short_edge = cfg.INPUT.MIN_SIZE_TEST

            if isinstance(test_short_edge, int):
                test_short_edge = (test_short_edge,)

            self.augmentations = T.AugmentationList([
                T.ResizeShortestEdge(
                    short_edge_length=test_short_edge,
                    max_size=cfg.INPUT.MAX_SIZE_TEST,
                    sample_style="choice",
                )
            ])


    @staticmethod
    def _validate_image(image, file_name):

        image = np.asarray(image)

        if image.ndim == 2:
            image = np.repeat(
                image[:, :, None],
                3,
                axis=2,
            )

        if image.ndim != 3:
            raise ValueError(
                f"影像维度错误：{image.shape}，文件：{file_name}"
            )

        if image.shape[2] == 1:
            image = np.repeat(
                image,
                3,
                axis=2,
            )

        elif image.shape[2] > 3:
            image = image[:, :, :3]

        if image.shape[2] != 3:
            raise ValueError(
                f"影像必须为三通道：{image.shape}，文件：{file_name}"
            )

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        image = np.clip(
            image,
            0,
            255,
        ).astype(np.uint8)

        return np.ascontiguousarray(image)


    def __call__(self, dataset_dict):

        cache_record = dataset_dict
        dataset_dict = copy.deepcopy(
            dataset_dict
        )

        file_name = dataset_dict["file_name"]

        image = utils.read_image(
            file_name,
            format=self.input_format,
        )

        image = self._validate_image(
            image,
            file_name,
        )

        original_h, original_w = image.shape[:2]

        dataset_dict.setdefault(
            "height",
            original_h,
        )

        dataset_dict.setdefault(
            "width",
            original_w,
        )

        aug_input = T.AugInput(image)

        transforms = self.augmentations(
            aug_input
        )

        image = aug_input.image

        image_shape = image.shape[:2]

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(
                image.transpose(2, 0, 1)
            )
        )

        annotations = dataset_dict.pop(
            "annotations",
            None,
        )

        if self.is_train:

            if annotations is None:
                raise ValueError(
                    f"训练样本没有 annotations：{file_name}"
                )

            annos = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                )
                for obj in annotations
                if obj.get("iscrowd", 0) == 0
            ]

            instances = utils.annotations_to_instances(
                annos,
                image_shape,
                mask_format=self.mask_format,
            )

            instances = utils.filter_empty_instances(
                instances
            )

            if (
                hasattr(instances, "gt_masks")
                and hasattr(
                    instances.gt_masks,
                    "tensor",
                )
            ):
                instances.gt_masks = (
                    instances.gt_masks.tensor
                )

            if not instances.has("gt_masks"):
                # annotations_to_instances omits this field for an empty annotation list.
                instances.gt_masks = torch.zeros((0, *image_shape), dtype=torch.bool)
            dataset_dict["instances"] = instances
            if self.tree_cache is not None:
                horizontal = vertical = False
                for transform in transforms.transforms:
                    if isinstance(transform, T.HFlipTransform):
                        horizontal = not horizontal
                    elif isinstance(transform, T.VFlipTransform):
                        vertical = not vertical
                    elif not isinstance(transform, (T.ResizeTransform, T.NoOpTransform, T.BlendTransform)):
                        raise ValueError(f'Unsupported cached geometry transform: {type(transform).__name__}')
                maps, centroids, meta = self.tree_cache.get(
                    cache_record, image_shape, horizontal, vertical
                )
                if meta['num_instances'] != len(instances):
                    raise ValueError('Augmented GT instance count changed; rebuild cache')
                # Readers never mutate the LRU arrays; the model creates padded device tensors.
                dataset_dict['tree_geometry_cache'] = {
                    'maps': {key: torch.from_numpy(value) for key, value in maps.items()},
                    'centroid_pixels': torch.from_numpy(centroids),
                    'meta': meta,
                }

        return dataset_dict


class ITDMask2FormerTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):

        return build_detection_train_loader(
            cfg,
            mapper=ITDMask2FormerMapper(
                cfg,
                is_train=True,
            ),
        )


    @classmethod
    def build_test_loader(
        cls,
        cfg,
        dataset_name,
    ):

        return build_detection_test_loader(
            cfg,
            dataset_name,
            mapper=ITDMask2FormerMapper(
                cfg,
                is_train=False,
            ),
        )


    @classmethod
    def build_evaluator(
        cls,
        cfg,
        dataset_name,
        output_folder=None,
    ):

        if output_folder is None:
    
            output_folder = os.path.join(
                cfg.OUTPUT_DIR,
                "inference",
                dataset_name,
            )
    
        os.makedirs(
            output_folder,
            exist_ok=True,
        )
    
    
        evaluator_kwargs = dict(
            dataset_name=dataset_name,
            tasks=("segm",),
            distributed=True,
            output_dir=output_folder,
        )
    
    
        evaluator = ITDCOCOEvaluator(
            **evaluator_kwargs
        )
    
    
        print(
            f"[ITDCOCOEvaluator] "
            f"Dataset={dataset_name}, "
            f"maxDets=700"
        )
    
    
        return evaluator


    @classmethod
    def build_lr_scheduler(
        cls,
        cfg,
        optimizer,
    ):

        return build_lr_scheduler(
            cfg,
            optimizer,
        )


    @classmethod
    def build_optimizer(
        cls,
        cfg,
        model,
    ):

        weight_decay_norm = (
            cfg.SOLVER.WEIGHT_DECAY_NORM
        )

        weight_decay_embed = (
            cfg.SOLVER.WEIGHT_DECAY_EMBED
        )

        defaults = {
            "lr": cfg.SOLVER.BASE_LR,
            "weight_decay": (
                cfg.SOLVER.WEIGHT_DECAY
            ),
        }

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []

        memo: Set[
            torch.nn.Parameter
        ] = set()

        for module_name, module in model.named_modules():

            for module_param_name, value in (
                module.named_parameters(
                    recurse=False
                )
            ):

                if not value.requires_grad:
                    continue

                if value in memo:
                    continue

                memo.add(value)

                hyperparams = copy.copy(
                    defaults
                )

                if "backbone" in module_name:

                    hyperparams["lr"] *= (
                        cfg.SOLVER.BACKBONE_MULTIPLIER
                    )

                if (
                    "relative_position_bias_table"
                    in module_param_name
                    or "absolute_pos_embed"
                    in module_param_name
                ):
                    hyperparams[
                        "weight_decay"
                    ] = 0.0

                if isinstance(
                    module,
                    norm_module_types,
                ):
                    hyperparams[
                        "weight_decay"
                    ] = weight_decay_norm

                if isinstance(
                    module,
                    torch.nn.Embedding,
                ):
                    hyperparams[
                        "weight_decay"
                    ] = weight_decay_embed

                params.append({
                    "params": [value],
                    **hyperparams,
                })


        def maybe_add_full_model_gradient_clipping(
            optimizer_class,
        ):

            clip_norm_value = (
                cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            )

            enabled = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE
                == "full_model"
                and clip_norm_value > 0.0
            )

            if not enabled:
                return optimizer_class

            class FullModelGradientClippingOptimizer(
                optimizer_class
            ):

                def step(
                    self,
                    closure=None,
                ):

                    all_params = itertools.chain(
                        *[
                            group["params"]
                            for group
                            in self.param_groups
                        ]
                    )

                    torch.nn.utils.clip_grad_norm_(
                        all_params,
                        clip_norm_value,
                    )

                    return super().step(
                        closure=closure
                    )

            return FullModelGradientClippingOptimizer


        optimizer_type = (
            cfg.SOLVER.OPTIMIZER.upper()
        )

        if optimizer_type == "SGD":

            optimizer_class = (
                maybe_add_full_model_gradient_clipping(
                    torch.optim.SGD
                )
            )

            optimizer = optimizer_class(
                params,
                lr=cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
            )

        elif optimizer_type == "ADAMW":

            optimizer_class = (
                maybe_add_full_model_gradient_clipping(
                    torch.optim.AdamW
                )
            )

            optimizer = optimizer_class(
                params,
                lr=cfg.SOLVER.BASE_LR,
            )

        else:

            raise NotImplementedError(
                f"不支持 optimizer：{optimizer_type}"
            )

        if (
            cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE
            != "full_model"
        ):

            optimizer = maybe_add_gradient_clipping(
                cfg,
                optimizer,
            )

        return optimizer

    def build_hooks(self):
        """Build trainer hooks with a configurable metric writer period.

        Set ``cfg.WRITER_PERIOD`` in the training script, e.g. 100.
        If it is not provided, Detectron2's usual 20-iteration cadence is used.
        """

        hooks_list = super().build_hooks()

        writer_period = int(
            getattr(
                self.cfg,
                "WRITER_PERIOD",
                20,
            )
        )

        if writer_period <= 0:
            raise ValueError(
                f"WRITER_PERIOD must be a positive integer, got {writer_period}"
            )

        print(
            f"[ITD Trainer] WRITER_PERIOD = {writer_period}"
        )

        replaced = False

        for index, hook in enumerate(hooks_list):
            if isinstance(hook, hooks.PeriodicWriter):
                hooks_list[index] = hooks.PeriodicWriter(
                    [
                        CommonMetricPrinter(self.max_iter),
                        JSONWriter(
                            os.path.join(
                                self.cfg.OUTPUT_DIR,
                                "metrics.json",
                            )
                        ),
                    ],
                    period=writer_period,
                )
                replaced = True

        print(
            f"[ITD Trainer] PeriodicWriter replaced = {replaced}, "
            f"period = {writer_period}"
        )

        return hooks_list
