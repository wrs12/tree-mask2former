from pathlib import Path
import re

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


def _dataset_number(path):
    m = re.search(r"(\d+)", path.name)
    return int(m.group(1)) if m else 10**9


def _find_json(annotation_dir, expected_name):
    expected = annotation_dir / expected_name

    if expected.exists():
        return expected

    json_files = sorted(annotation_dir.glob("*.json"))

    if len(json_files) == 1:
        print(
            f"[提示] 未找到 {expected_name}，"
            f"自动使用 {json_files[0].name}"
        )
        return json_files[0]

    raise FileNotFoundError(
        f"无法唯一确定标注 JSON：{annotation_dir}\n"
        f"期望：{expected_name}\n"
        f"实际：{[p.name for p in json_files]}"
    )


def find_validation_root(ROOT):
    validation_root_candidates = (
        ROOT / "validation_split",
        ROOT / "validation_set",
    )

    for root in validation_root_candidates:

        if not root.exists():
            continue

        candidates = (
            root / "Dataset_3_val",
            root / "dataset_3_val",
            root / "dataset_3",
        )

        if any(p.exists() for p in candidates):
            return root

    raise FileNotFoundError(
        "没有找到拆分后的验证集。\n"
        "已检查：\n"
        f"{ROOT / 'validation_split'}\n"
        f"{ROOT / 'validation_set'}"
    )


def find_val_dir(VAL_ROOT, dataset_id):
    candidates = (
        VAL_ROOT / f"Dataset_{dataset_id}_val",
        VAL_ROOT / f"dataset_{dataset_id}_val",
        VAL_ROOT / f"dataset_{dataset_id}",
    )

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"找不到 Dataset_{dataset_id}_val 目录：\n"
        + "\n".join(str(p) for p in candidates)
    )


def build_dataset_specs(TRAIN_ROOT, VAL_ROOT):

    if not TRAIN_ROOT.exists():
        raise FileNotFoundError(
            f"训练集目录不存在：{TRAIN_ROOT}"
        )

    train_dirs = [
        p for p in TRAIN_ROOT.iterdir()
        if p.is_dir()
        and re.fullmatch(
            r"dataset_\d+",
            p.name,
            flags=re.I,
        )
    ]

    train_dirs = sorted(
        train_dirs,
        key=_dataset_number,
    )

    if not train_dirs:
        raise RuntimeError(
            f"{TRAIN_ROOT} 下没有 dataset_1、dataset_2 ..."
        )

    specs = {}

    # =========================
    # 训练集
    # =========================
    for p in train_dirs:

        idx = _dataset_number(p)

        image_root = p / "images"
        annotation_dir = p / "annotation"

        ann_file = _find_json(
            annotation_dir,
            f"Dataset_{idx}_train.json",
        )

        if not image_root.exists():
            raise FileNotFoundError(
                f"图片目录不存在：{image_root}"
            )

        name = f"Dataset_{idx}_train"

        specs[name] = {
            "split": "train",
            "dataset_id": idx,
            "ann_file": ann_file,
            "image_root": image_root,
        }

    # =========================
    # 验证集
    # =========================
    for idx in (3, 4, 5):

        p = find_val_dir(
            VAL_ROOT,
            idx,
        )

        image_root = p / "images"
        annotation_dir = p / "annotation"

        ann_file = _find_json(
            annotation_dir,
            f"Dataset_{idx}_val.json",
        )

        if not image_root.exists():
            raise FileNotFoundError(
                f"图片目录不存在：{image_root}"
            )

        name = f"Dataset_{idx}_val"

        specs[name] = {
            "split": "val",
            "dataset_id": idx,
            "ann_file": ann_file,
            "image_root": image_root,
        }

    return specs


def register_all_itd_datasets(
    DATASET_SPECS,
    TRAIN_NAMES,
    VAL_NAMES,
):

    for name, spec in DATASET_SPECS.items():

        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)

        try:
            if name in MetadataCatalog.list():
                MetadataCatalog.remove(name)
        except Exception:
            pass

        register_coco_instances(
            name,
            {},
            str(spec["ann_file"]),
            str(spec["image_root"]),
        )

        MetadataCatalog.get(name).set(
            evaluator_type="coco"
        )

    print("已注册训练集：", TRAIN_NAMES)
    print("已注册验证集：", VAL_NAMES)