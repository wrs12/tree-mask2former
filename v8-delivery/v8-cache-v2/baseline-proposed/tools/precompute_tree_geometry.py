#!/usr/bin/env python3
"""Precompute exact ITD resize/flip targets on CPU; no model or CUDA required."""
import argparse
import importlib.util
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, PROJECT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load_file('_offline_geometry', 'mask2former/modeling/tree_geometry.py')
cache = load_file('_offline_cache', 'mask2former/modeling/tree_geometry_cache.py')


def precompute_record(record, cfg, profile, folder, mapper_class, transforms,
                      build_boundary, build_distance, flips=True):
    key = cache.sample_key(record)
    entry = {'annotation_signature': cache.annotation_signature(record), 'variants': {}}
    for size in profile['short_edges']:
        for horizontal, vertical in ((False, False), (True, False), (False, True), (True, True)) if flips else ((False, False),):
            variant_cfg = cfg.clone()
            variant_cfg.defrost()
            variant_cfg.MODEL.TREE_AWARE.CACHE_ENABLED = False
            mapper = mapper_class(variant_cfg, is_train=True)
            # Apply annotations through the original mapper, including rasterization and filtering.
            augmentations = [transforms.ResizeShortestEdge((size,), profile['max_size'], 'choice')]
            if horizontal:
                augmentations.append(transforms.RandomFlip(prob=1.0, horizontal=True, vertical=False))
            if vertical:
                augmentations.append(transforms.RandomFlip(prob=1.0, horizontal=False, vertical=True))
            mapper.augmentations = transforms.AugmentationList(augmentations)
            mapped = mapper(record)
            height, width = mapped['image'].shape[-2:]
            variant = cache.variant_key((height, width), horizontal, vertical)
            if variant in entry['variants']:
                continue  # Different short edges can hit the same MAX_SIZE-limited dimensions.
            masks = mapped['instances'].gt_masks
            if hasattr(masks, 'tensor'):
                masks = masks.tensor
            if masks.device.type != 'cpu':
                raise ValueError('Precomputation requires CPU masks')
            divisor, stride = profile['divisor'], profile['stride']
            hp = (height + divisor - 1) // divisor * divisor
            wp = (width + divisor - 1) // divisor * divisor
            padded = torch.zeros((len(masks), hp, wp), dtype=torch.bool)
            padded[:, :height, :width] = masks.bool()
            padding = torch.ones((hp, wp), dtype=torch.bool)
            padding[:height, :width] = False
            output_size = (hp // stride, wp // stride)
            maps = geometry.build_tree_targets(
                padded, padding, output_size, profile['sigma'],
                build_boundary=build_boundary, build_distance=build_distance,
            )
            fields = ['center', 'valid']
            if build_boundary:
                fields.append('boundary')
            if build_distance:
                fields.extend(('distance', 'foreground'))
            maps = {field: maps[field] for field in fields}
            centroids = geometry.mask_centroids(padded.float(), ~padding)
            centroids = (centroids * centroids.new_tensor([wp, hp])).numpy()
            meta = {
                'image_size': [height, width], 'padded_size': [hp, wp],
                'feature_size': list(output_size), 'stride': stride, 'divisor': divisor,
                'sigma': profile['sigma'], 'horizontal': horizontal, 'vertical': vertical,
                'num_instances': len(masks),
            }
            entry['variants'][variant] = cache.write_variant(folder, key, variant, maps, centroids, meta)
    return key, entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-file', required=True, help='Export cfg.dump() AFTER notebook overrides')
    parser.add_argument('--data-root', required=True, help='Root containing training_set and validation_split/validation_set')
    parser.add_argument('--shared-tools-root', required=True, help='Directory containing your mytools package')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--workers', type=int, default=1, help='CPU images in parallel; start with 1 or 2 to limit host RAM')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    sys.path.insert(0, str(Path(args.shared_tools_root).resolve()))
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from detectron2.data import DatasetCatalog, transforms as T
    from mytools.dataset_utils import find_validation_root, build_dataset_specs, register_all_itd_datasets
    from mytools.itd_trainer import ITDMask2FormerMapper
    cfg = get_cfg()
    add_deeplab_config(cfg)
    load_file('_offline_config', 'mask2former/config.py').add_maskformer2_config(cfg)
    cfg.WRITER_PERIOD = 20  # User notebook's existing custom field.
    cfg.merge_from_file(args.config_file)
    root = Path(args.data_root).resolve()
    specs = build_dataset_specs(root / 'training_set', find_validation_root(root))
    train_names = [name for name, spec in specs.items() if spec['split'] == 'train']
    val_names = [name for name, spec in specs.items() if spec['split'] == 'val']
    register_all_itd_datasets(specs, train_names, val_names)
    datasets = args.datasets or list(dict.fromkeys((*cfg.DATASETS.TRAIN, *cfg.DATASETS.TEST)))
    output = Path(args.output_dir or cfg.MODEL.TREE_AWARE.CACHE_DIR).resolve()
    boundary = cfg.MODEL.TREE_AWARE.BOUNDARY_WEIGHT > 0
    distance = cfg.MODEL.TREE_AWARE.DISTANCE_WEIGHT > 0
    # Avoid nested CPU thread pools; these settings affect this offline process only.
    torch.set_num_threads(1)
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial
    for dataset in datasets:
        if dataset not in specs:
            raise KeyError(f'{dataset} was not registered by mytools.dataset_utils')
        train = specs[dataset]['split'] == 'train'
        profile = cache.cache_profile(cfg, is_train=train)
        folder = output / dataset
        records = DatasetCatalog.get(dataset)
        manifest = {
            'schema': cache.SCHEMA, 'dataset': dataset, 'profile': profile,
            'build_boundary': boundary, 'build_distance': distance, 'entries': {},
        }
        operation = partial(precompute_record, cfg=cfg, profile=profile, folder=folder,
                            mapper_class=ITDMask2FormerMapper, transforms=T,
                            build_boundary=boundary, build_distance=distance, flips=train)
        # map preserves order, so registration/manifest ordering stays deterministic.
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, (key, entry) in enumerate(executor.map(operation, records), 1):
                if key in manifest['entries']:
                    raise ValueError(f'Duplicate cache identity in {dataset}')
                manifest['entries'][key] = entry
                if index % 20 == 0 or index == len(records):
                    print(f'{dataset}: {index}/{len(records)} images', flush=True)
        cache.write_manifest(folder, manifest)
        print(f'Saved {folder / "manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
