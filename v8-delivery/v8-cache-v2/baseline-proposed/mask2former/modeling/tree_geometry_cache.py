"""CPU cache utilities. No EDT or model execution occurs while reading."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA = 1


def sample_key(record):
    identity = [str(record['image_id']), str(Path(record['file_name']).resolve())]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def annotation_signature(record):
    def encode(value):
        if isinstance(value, bytes):
            return value.decode('ascii')
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    text = json.dumps(record.get('annotations', []), sort_keys=True,
                      separators=(',', ':'), default=encode)
    return hashlib.sha256(text.encode()).hexdigest()


def variant_key(image_size, horizontal=False, vertical=False):
    height, width = map(int, image_size)
    return f'{height}x{width}_h{int(horizontal)}v{int(vertical)}'


def cache_profile(cfg, is_train=True):
    sizes = cfg.INPUT.MIN_SIZE_TRAIN if is_train else cfg.INPUT.MIN_SIZE_TEST
    if isinstance(sizes, int):
        sizes = (sizes,)
    stride = int(cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE)
    divisor = int(cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY)
    if stride <= 0 or divisor <= 0 or divisor % stride:
        raise ValueError('Cache requires positive SIZE_DIVISIBILITY divisible by COMMON_STRIDE')
    return {
        'short_edges': sorted(set(map(int, sizes))),
        'max_size': int(cfg.INPUT.MAX_SIZE_TRAIN if is_train else cfg.INPUT.MAX_SIZE_TEST),
        'stride': stride, 'divisor': divisor,
        'sigma': float(cfg.MODEL.TREE_AWARE.CENTER_SIGMA),
        'format': str(cfg.INPUT.FORMAT), 'mask_format': str(cfg.INPUT.MASK_FORMAT),
    }


def write_variant(folder, key, variant, geometry, centroid_pixels, meta):
    """Atomic NPZ, float32 center/distance; never store pickle objects."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = f'{key}_{variant}.npz'
    values = {}
    for field, value in geometry.items():
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        dtype = np.bool_ if field == 'valid' else (
            np.uint8 if field in ('boundary', 'foreground') else np.float32
        )
        values[field] = np.asarray(value, dtype=dtype)
    values['centroid_pixels'] = np.asarray(centroid_pixels, dtype=np.float32)
    values['meta'] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    fd, temporary = tempfile.mkstemp(prefix='.geometry-', suffix='.npz', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **values)
        os.replace(temporary, folder / name)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return name


def write_manifest(folder, manifest):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.manifest-', dir=folder)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(manifest, stream, sort_keys=True)
        os.replace(temporary, folder / 'manifest.json')
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class TreeGeometryCache:
    """Bounded host-memory LRU; loaded in DataLoader workers by the ITD mapper."""
    def __init__(self, root, datasets, profile, build_boundary, build_distance, max_items=8):
        self.root = Path(root).resolve()
        self.profile = profile
        self.build_boundary = bool(build_boundary)
        self.build_distance = bool(build_distance)
        self.max_items = int(max_items)
        if self.max_items < 0:
            raise ValueError('CACHE_MEMORY_ITEMS must be nonnegative')
        self.indices = []
        self.lru = OrderedDict()
        for dataset in datasets:
            folder = self.root / dataset
            with (folder / 'manifest.json').open() as stream:
                manifest = json.load(stream)
            if manifest['schema'] != SCHEMA or manifest['dataset'] != dataset:
                raise ValueError(f'Invalid geometry manifest for {dataset}')
            if manifest['profile'] != profile:
                raise ValueError(f'Cache resize/padding/sigma profile changed for {dataset}; precompute again')
            if self.build_boundary and not manifest['build_boundary']:
                raise ValueError(f'{dataset} cache has no Boundary; precompute with its weight > 0')
            if self.build_distance and not manifest['build_distance']:
                raise ValueError(f'{dataset} cache has no Distance; precompute with its weight > 0')
            self.indices.append((folder, manifest['entries']))

    def get(self, record, image_size, horizontal=False, vertical=False, signature=None):
        key = sample_key(record)
        variant = variant_key(image_size, horizontal, vertical)
        matches = [(folder, entries[key]) for folder, entries in self.indices if key in entries]
        if len(matches) != 1:
            raise KeyError(f'Expected one cache record for image_id={record["image_id"]}; found {len(matches)}')
        folder, entry = matches[0]
        signature = annotation_signature(record) if signature is None else signature
        if entry['annotation_signature'] != signature:
            raise ValueError('Annotations changed; precompute geometry again')
        if variant not in entry['variants']:
            raise KeyError(f'Missing geometry variant {variant}; no online EDT fallback')
        relative = entry['variants'][variant]
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError('Geometry cache path escapes its dataset folder')
        fields = ['center', 'valid']
        if self.build_boundary:
            fields.append('boundary')
        if self.build_distance:
            fields.extend(('distance', 'foreground'))
        cache_identity = (str(path), tuple(fields))
        if cache_identity in self.lru:
            result = self.lru.pop(cache_identity)
            self.lru[cache_identity] = result
            return result
        with np.load(path, allow_pickle=False) as archive:
            geometry = {field: np.ascontiguousarray(archive[field]) for field in fields}
            centroids = np.ascontiguousarray(archive['centroid_pixels'])
            meta = json.loads(archive['meta'].tobytes().decode())
        expected = tuple(meta['feature_size'])
        if tuple(meta['image_size']) != tuple(image_size):
            raise ValueError('Cache image size mismatch')
        if any(array.shape != expected for array in geometry.values()):
            raise ValueError('Cache feature map shape mismatch')
        if meta['stride'] != self.profile['stride'] or meta['divisor'] != self.profile['divisor']:
            raise ValueError('Cache feature stride/padding divisor mismatch')
        if meta['sigma'] != self.profile['sigma']:
            raise ValueError('Cache center sigma mismatch')
        if bool(meta['horizontal']) != bool(horizontal) or bool(meta['vertical']) != bool(vertical):
            raise ValueError('Cache flip mismatch')
        if centroids.shape != (meta['num_instances'], 2):
            raise ValueError('Cache centroid count mismatch')
        result = (geometry, centroids, meta)
        if self.max_items:
            self.lru[cache_identity] = result
            while len(self.lru) > self.max_items:
                self.lru.popitem(last=False)
        return result
