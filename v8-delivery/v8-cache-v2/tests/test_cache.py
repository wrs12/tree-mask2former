import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import os

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import transforms as T
from detectron2.structures import BitMasks, ImageList

ROOT = Path(__file__).resolve().parents[1]
PROPOSED = ROOT / os.environ.get('V8_SOURCE_DIR', 'proposed')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


geo = load('revision_geo', PROPOSED / 'mask2former/modeling/tree_geometry.py')
old_geo = load('old_geo', ROOT / 'tests/fixtures/previous_tree_geometry.py')
cache = load('revision_cache', PROPOSED / 'mask2former/modeling/tree_geometry_cache.py')
precompute = load('precompute', PROPOSED / 'tools/precompute_tree_geometry.py')
mapper_module = load('revision_mapper', ROOT / 'shared/mytools/itd_trainer.py')
# Real mapper imports only this utility; bypass package init/CUDA ops in CPU tests.
for name in ('mask2former', 'mask2former.modeling'):
    package = ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package
sys.modules['mask2former.modeling.tree_geometry_cache'] = cache


def cfg_for_test():
    cfg = get_cfg()
    add_deeplab_config(cfg)
    load('revision_config', PROPOSED / 'mask2former/config.py').add_maskformer2_config(cfg)
    cfg.INPUT.FORMAT = 'RGB'
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.INPUT.MIN_SIZE_TRAIN = (32, 48)
    cfg.INPUT.MAX_SIZE_TRAIN = 64
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32
    cfg.MODEL.TREE_AWARE.ENABLED = True
    cfg.DATASETS.TRAIN = ('synthetic_train',)
    return cfg


def prepare_method():
    tree = ast.parse((PROPOSED / 'mask2former/maskformer_model.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare_targets')
    namespace = dict(torch=torch, F=F, BitMasks=BitMasks, mask_centroids=geo.mask_centroids,
                     build_tree_targets=lambda *a, **k: (_ for _ in ()).throw(AssertionError('online EDT called')))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(PROPOSED), 'exec'), namespace)
    return namespace['prepare_targets']


@pytest.mark.parametrize('count', [0, 1, 5])
def test_builder_equivalence(count):
    torch.manual_seed(10)
    masks = torch.rand(count, 32, 64) > .7
    padding = torch.ones(32, 64, dtype=torch.bool)
    padding[:29, :55] = False
    masks &= ~padding
    original = old_geo.build_tree_targets(masks, padding, (8, 16), 2)
    revised = geo.build_tree_targets(masks, padding, (8, 16), 2)
    for field in original:
        torch.testing.assert_close(original[field], revised[field], rtol=0, atol=0)
    center_only = geo.build_tree_targets(masks, padding, (8, 16), 2, False, False)
    torch.testing.assert_close(center_only['center'], original['center'], rtol=0, atol=0)
    assert not center_only['boundary'].any()
    assert not center_only['distance'].any()
    assert not center_only['foreground'].any()


def create_record(folder):
    path = folder / 'image.png'
    Image.fromarray(np.arange(43 * 61 * 3, dtype=np.uint8).reshape(43, 61, 3)).save(path)
    # Polygon touching image boundary, nonsymmetric instance, and a second tiny polygon.
    return dict(image_id=7, file_name=str(path), height=43, width=61, annotations=[
        dict(category_id=0, bbox=[0, 0, 28, 23], bbox_mode=1,
             segmentation=[[0, 0, 28, 0, 25, 23, 3, 19]], iscrowd=0),
        dict(category_id=0, bbox=[40, 29, 9, 10], bbox_mode=1,
             segmentation=[[40, 29, 49, 29, 49, 39, 40, 39]], iscrowd=0),
    ])


def create_cache(tmp_path, boundary=True, distance=True):
    cfg = cfg_for_test()
    record = create_record(tmp_path)
    profile = cache.cache_profile(cfg)
    folder = tmp_path / 'cache' / 'synthetic_train'
    key, entry = precompute.precompute_record(record, cfg, profile, folder,
        mapper_module.ITDMask2FormerMapper, T, boundary, distance)
    cache.write_manifest(folder, dict(schema=cache.SCHEMA, dataset='synthetic_train', profile=profile,
        build_boundary=boundary, build_distance=distance, entries={key: entry}))
    return cfg, record, profile, folder


@pytest.mark.parametrize('boundary,distance', [(False, False), (True, True)])
def test_real_mapper_variants_and_batch_padding(tmp_path, boundary, distance):
    cfg, record, profile, folder = create_cache(tmp_path, boundary, distance)
    cfg.MODEL.TREE_AWARE.CACHE_DIR = str(folder.parent)
    cfg.MODEL.TREE_AWARE.CACHE_ENABLED = True
    cfg.MODEL.TREE_AWARE.BOUNDARY_WEIGHT = float(boundary)
    cfg.MODEL.TREE_AWARE.DISTANCE_WEIGHT = float(distance)
    mapper = mapper_module.ITDMask2FormerMapper(cfg)
    prepare = prepare_method()
    model = SimpleNamespace(tree_cache_enabled=True, tree_cache_stride=4, size_divisibility=32,
        tree_build_boundary=boundary, tree_build_distance=distance, criterion=SimpleNamespace(center_sigma=2))
    for size in (32, 48):
        for hf, vf in ((False, False), (True, False), (False, True), (True, True)):
            augs = [T.ResizeShortestEdge((size,), 64, 'choice')]
            if hf:
                augs.append(T.RandomFlip(1, horizontal=True, vertical=False))
            if vf:
                augs.append(T.RandomFlip(1, horizontal=False, vertical=True))
            # Photometric augmentation remains supported and leaves supervision unchanged.
            augs += [T.RandomBrightness(.8, 1.2), T.RandomContrast(.8, 1.2)]
            mapper.augmentations = T.AugmentationList(augs)
            mapped = mapper(record)
            for extra_pad in (0, 32):
                images = ImageList.from_tensors([mapped['image'], torch.zeros(3, 64 + extra_pad, 96 + extra_pad)], 32)
                size_out = tuple(d // 4 for d in images.tensor.shape[-2:])
                result = prepare(model, [mapped['instances']], images, size_out, [mapped])[0]
                hp, wp = images.tensor.shape[-2:]
                h, w = mapped['image'].shape[-2:]
                padding = torch.ones(hp, wp, dtype=torch.bool)
                padding[:h, :w] = False
                expected = geo.build_tree_targets(result['masks'], padding, size_out, 2, boundary, distance)
                for field, value in result['tree_geometry'].items():
                    torch.testing.assert_close(value, expected[field], rtol=0, atol=0)
                torch.testing.assert_close(result['tree_centroids'], geo.mask_centroids(result['masks'], ~padding), rtol=1e-6, atol=1e-7)


def test_fail_closed_and_lru(tmp_path):
    cfg, record, profile, folder = create_cache(tmp_path)
    reader = cache.TreeGeometryCache(folder.parent, ['synthetic_train'], profile, True, True, max_items=1)
    manifest = json.loads((folder / 'manifest.json').read_text())
    variants = list(next(iter(manifest['entries'].values()))['variants'])
    for key in variants:
        shape, flips = key.split('_')
        h, w = map(int, shape.split('x'))
        reader.get(record, (h, w), flips[1] == '1', flips[3] == '1')
        assert len(reader.lru) == 1
    with pytest.raises(KeyError, match='no online EDT'):
        reader.get(record, (1, 1))
    changed = dict(record, annotations=[])
    with pytest.raises(ValueError, match='Annotations changed'):
        reader.get(changed, (32, 45))
    with pytest.raises(ValueError, match='profile changed'):
        cache.TreeGeometryCache(folder.parent, ['synthetic_train'], dict(profile, sigma=3), True, True)
    # Full cache can serve A; A-only cache cannot silently serve B/C.
    only_center = cache.TreeGeometryCache(folder.parent, ['synthetic_train'], profile, False, False)
    key = variants[0]
    shape, flips = key.split('_')
    values = only_center.get(record, tuple(map(int, shape.split('x'))), flips[1] == '1', flips[3] == '1')[0]
    assert set(values) == {'center', 'valid'}
    manifest['build_distance'] = False
    cache.write_manifest(folder, manifest)
    with pytest.raises(ValueError, match='no Distance'):
        cache.TreeGeometryCache(folder.parent, ['synthetic_train'], profile, True, True)


def test_empty_instances_cache(tmp_path):
    cfg = cfg_for_test()
    record = create_record(tmp_path)
    record['annotations'] = []
    profile = cache.cache_profile(cfg)
    key, entry = precompute.precompute_record(record, cfg, profile, tmp_path / 'cache',
        mapper_module.ITDMask2FormerMapper, T, False, False)
    for filename in entry['variants'].values():
        with np.load(tmp_path / 'cache' / filename, allow_pickle=False) as a:
            assert a['centroid_pixels'].shape == (0, 2)
            assert not a['center'].any()


def test_encoder_and_query_unchanged():
    def classes(path):
        return {node.name: ast.dump(node) for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)}
    assert classes(PROPOSED / 'mask2former/modeling/tree_geometry.py') == classes(ROOT / 'tests/fixtures/previous_tree_geometry.py')


@pytest.mark.parametrize('count', [0, 1, 4])
def test_separation_value_and_gradient(count):
    def extract(path):
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'loss_separation')
        namespace = dict(torch=torch, F=F)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace['loss_separation']
    old = extract(ROOT / 'tests/fixtures/previous_tree_criterion.py')
    new = extract(PROPOSED / 'mask2former/modeling/tree_criterion.py')
    torch.manual_seed(42)
    states = [torch.randn(1, 5, 8) for _ in range(3)]
    assignments = [(torch.arange(count), torch.arange(count).flip(0)),
                   (torch.arange(count), torch.arange(count)),
                   (torch.arange(count).flip(0), torch.arange(count))]
    targets = [{'tree_centroids': torch.rand(count, 2) * .1}]
    settings = SimpleNamespace(separation_radius=.15, separation_margin=.2, positive_weight=1, negative_weight=1)
    def run(fn):
        features = [s.clone().requires_grad_() for s in states]
        predictions = [dict(query_features=f) for f in features]
        loss = fn(settings, predictions[-1], [assignments[-1]], predictions[:-1],
                  [[a] for a in assignments[:-1]], targets)['loss_query_separation']
        loss.backward()
        return loss, features
    a, af = run(old)
    b, bf = run(new)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for left, right in zip(af, bf):
        if left.grad is None:
            assert right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_cli_custom_dataset_registration(tmp_path):
    import subprocess
    record = create_record(tmp_path)
    # Match the real helper's directory contract, including the other validation splits.
    root = tmp_path / 'data'
    for relative, dataset in [('training_set/dataset_4', 'Dataset_4_train'),
                              ('validation_split/Dataset_3_val', 'Dataset_3_val'),
                              ('validation_split/Dataset_4_val', 'Dataset_4_val'),
                              ('validation_split/Dataset_5_val', 'Dataset_5_val')]:
        folder = root / relative
        (folder / 'images').mkdir(parents=True)
        (folder / 'annotation').mkdir()
        Image.open(record['file_name']).save(folder / 'images/image.png')
        annotations = []
        for index, anno in enumerate(record['annotations'], 1):
            annotations.append({key: value for key, value in anno.items() if key != 'bbox_mode'} |
                dict(id=index, image_id=7, category_id=1, area=anno['bbox'][2] * anno['bbox'][3]))
        data = dict(images=[dict(id=7, file_name='image.png', height=43, width=61)],
                    annotations=annotations, categories=[dict(id=1, name='tree')])
        (folder / f'annotation/{dataset}.json').write_text(json.dumps(data))
    cfg = cfg_for_test()
    cfg.DATASETS.TRAIN = ('Dataset_4_train',)
    cfg.DATASETS.TEST = ('Dataset_4_val',)
    cfg.MODEL.TREE_AWARE.BOUNDARY_WEIGHT = 0.0
    cfg.MODEL.TREE_AWARE.DISTANCE_WEIGHT = 0.0
    cfg.WRITER_PERIOD = 1
    config = tmp_path / 'effective.yaml'
    config.write_text(cfg.dump())
    output = tmp_path / 'cli_cache'
    subprocess.run([sys.executable, str(PROPOSED / 'tools/precompute_tree_geometry.py'),
        '--config-file', str(config), '--data-root', str(root), '--shared-tools-root', str(ROOT / 'shared'),
        '--output-dir', str(output), '--workers', '2'], check=True, capture_output=True, text=True)
    for dataset, variants in [('Dataset_4_train', 8), ('Dataset_4_val', 1)]:
        manifest = json.loads((output / dataset / 'manifest.json').read_text())
        assert len(manifest['entries']) == 1
        assert len(next(iter(manifest['entries'].values()))['variants']) == variants
        assert not manifest['build_boundary']
        assert not manifest['build_distance']


@pytest.mark.parametrize('cost_geometry', [0.0, 0.2])
def test_matcher_broadcast_equivalence(cost_geometry):
    sys.modules['mask2former.modeling.tree_geometry'] = geo
    old = load('mask2former.modeling.previous_matcher', ROOT / 'tests/fixtures/previous_matcher.py')
    new = load('mask2former.modeling.revised_matcher', PROPOSED / 'mask2former/modeling/matcher.py')
    torch.manual_seed(4)
    masks = torch.rand(5, 32, 64) > .5
    outputs = dict(pred_logits=torch.randn(1, 11, 2), pred_masks=torch.randn(1, 11, 8, 16))
    targets = [dict(labels=torch.zeros(5, dtype=torch.int64), masks=masks,
                    padding_mask=torch.zeros(32, 64, dtype=torch.bool), tree_centroids=geo.mask_centroids(masks))]
    costs = []
    def solve(matrix):
        from scipy.optimize import linear_sum_assignment
        costs.append(matrix.clone())
        return linear_sum_assignment(matrix)
    old.linear_sum_assignment = new.linear_sum_assignment = solve
    torch.manual_seed(42)
    a = old.HungarianMatcher(num_points=31, cost_geometry=cost_geometry)(outputs, targets)
    torch.manual_seed(42)
    b = new.HungarianMatcher(num_points=31, cost_geometry=cost_geometry)(outputs, targets)
    torch.testing.assert_close(costs[0], costs[1], rtol=0, atol=0)
    for (asrc, atgt), (bsrc, btgt) in zip(a, b):
        assert torch.equal(asrc, bsrc) and torch.equal(atgt, btgt)


def test_mask_loss_selection_and_gradient_equivalence():
    utils = ModuleType('mask2former.utils')
    utils.__path__ = []
    sys.modules['mask2former.utils'] = utils
    load('mask2former.utils.misc', ROOT / 'tests/fixtures/misc.py')
    old = load('mask2former.modeling.previous_criterion', ROOT / 'tests/fixtures/previous_criterion.py')
    new = load('mask2former.modeling.revised_criterion', PROPOSED / 'mask2former/modeling/criterion.py')
    torch.manual_seed(42)
    logits = torch.randn(2, 7, 8, 16)
    targets = [dict(masks=torch.rand(count, 32, 64) > .5) for count in (5, 3)]
    indices = [(torch.tensor([1, 5]), torch.tensor([4, 1])), (torch.tensor([2]), torch.tensor([2]))]
    kwargs = dict(num_classes=1, matcher=None, weight_dict={}, eos_coef=.1,
                  losses=['masks'], num_points=31, oversample_ratio=3, importance_sample_ratio=.75)
    def run(module):
        source = logits.clone().requires_grad_()
        torch.manual_seed(4)
        losses = module.SetCriterion(**kwargs).loss_masks(dict(pred_masks=source), targets, indices, 3)
        sum(losses.values()).backward()
        return losses, source.grad
    a, ag = run(old)
    b, bg = run(new)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    torch.testing.assert_close(ag, bg, rtol=0, atol=0)
