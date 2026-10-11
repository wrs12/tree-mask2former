# 文件与修改位置

推荐 baseline-proposed；proposed 是 V7 兼容路径。以下是完整代码，不是单个示例。

| 文件 | 关键位置 | 作用 |
|---|---|---|
| [baseline-proposed/configs/tree-aware/baseline.yaml](baseline-proposed/configs/tree-aware/baseline.yaml) | 配置全文 | V8-A/B/C或缓存配置 |
| [baseline-proposed/configs/tree-aware/tree_aware_v8.yaml](baseline-proposed/configs/tree-aware/tree_aware_v8.yaml) | 配置全文 | V8-A/B/C或缓存配置 |
| [baseline-proposed/configs/tree-aware/v8_a.yaml](baseline-proposed/configs/tree-aware/v8_a.yaml) | 配置全文 | V8-A/B/C或缓存配置 |
| [baseline-proposed/configs/tree-aware/v8_a_cache.yaml](baseline-proposed/configs/tree-aware/v8_a_cache.yaml) | 配置全文 | V8-A/B/C或缓存配置 |
| [baseline-proposed/configs/tree-aware/v8_b.yaml](baseline-proposed/configs/tree-aware/v8_b.yaml) | 配置全文 | V8-A/B/C或缓存配置 |
| [baseline-proposed/configs/tree-aware/v8_b_cache.yaml](baseline-proposed/configs/tree-aware/v8_b_cache.yaml) | 配置全文 | V8-A/B/C或缓存配置 |
| [baseline-proposed/configs/tree-aware/v8_c.yaml](baseline-proposed/configs/tree-aware/v8_c.yaml) | 配置全文 | V8-A/B/C或缓存配置 |
| [baseline-proposed/configs/tree-aware/v8_c_cache.yaml](baseline-proposed/configs/tree-aware/v8_c_cache.yaml) | 配置全文 | V8-A/B/C或缓存配置 |
| [baseline-proposed/mask2former/config.py](baseline-proposed/mask2former/config.py) | add_maskformer2_config:6 | 新增 V8 和缓存配置；基础接口保留 |
| [baseline-proposed/mask2former/maskformer_model.py](baseline-proposed/mask2former/maskformer_model.py) | MaskFormer:23, __init__:29, from_config:97, forward:225, prepare_targets:335 | from_config→构造器→prepare_targets 接入缓存，forward 原始输出保留 |
| [baseline-proposed/mask2former/modeling/criterion.py](baseline-proposed/mask2former/modeling/criterion.py) | dice_loss:21, sigmoid_ce_loss:48, calculate_uncertainty:73, morphological_boundary:90, SetCriterion:133, __init__:140, forward:255 | baseline 的 SetCriterion 类完全一致；仅增加 V8 boundary helper |
| [baseline-proposed/mask2former/modeling/matcher.py](baseline-proposed/mask2former/modeling/matcher.py) | batch_dice_loss:17, batch_sigmoid_ce_loss:40, HungarianMatcher:72, __init__:80, forward:175 | 仅增加 V8 geometry cost/空 GT 处理，原采样及基本 cost 保留 |
| [baseline-proposed/mask2former/modeling/pixel_decoder/msdeformattn.py](baseline-proposed/mask2former/modeling/pixel_decoder/msdeformattn.py) | MSDeformAttnTransformerEncoderOnly:23, __init__:24, forward:61, MSDeformAttnTransformerEncoderLayer:92, __init__:93, forward:122, MSDeformAttnTransformerEncoder:134, __init__:135, forward:155, MSDeformAttnPixelDecoder:165, __init__:167, from_config:295 | baseline版保留原始文件；V7兼容版旧模块开关 |
| [baseline-proposed/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py](baseline-proposed/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py) | SelfAttentionLayer:18, __init__:20, forward:65, CrossAttentionLayer:76, __init__:78, forward:127, FFNLayer:139, __init__:141, forward:176, _get_activation_fn:182, MLP:193, __init__:196, forward:202, MultiScaleMaskedTransformerDecoder:209, __init__:237, from_config:350, forward:382 | V8接入；separation关闭时跳过 Query states；baseline没有V7 adapter |
| [baseline-proposed/mask2former/modeling/tree_criterion.py](baseline-proposed/mask2former/modeling/tree_criterion.py) | TreeAwareSetCriterion:11, __init__:12, loss_separation:95, forward:135 | V8 几何/分离 loss；负对计数移除 host 同步 |
| [baseline-proposed/mask2former/modeling/tree_geometry.py](baseline-proposed/mask2former/modeling/tree_geometry.py) | resize_padding_mask:8, mask_centroids:13, TreeGeometryEncoder:27, __init__:28, forward:48, TreeAwareQueryGenerator:58, __init__:59, forward:73, build_tree_targets:99 | 新 Encoder/Query 与条件 GT builder；两个类和原 V8 完全一致 |
| [baseline-proposed/mask2former/modeling/tree_geometry_cache.py](baseline-proposed/mask2former/modeling/tree_geometry_cache.py) | sample_key:14, annotation_signature:19, variant_key:33, cache_profile:38, write_variant:55, write_manifest:81, TreeGeometryCache:94, __init__:96 | 分片读取、校验、LRU 与原子写入 |
| [baseline-proposed/tools/precompute_tree_geometry.py](baseline-proposed/tools/precompute_tree_geometry.py) | load_file:14, precompute_record:25, main:82 | 离线调用真实注册流程和mapper，生成增强变体 |
| [shared/mytools/dataset_utils.py](shared/mytools/dataset_utils.py) | _dataset_number:8, _find_json:13, find_validation_root:35, find_val_dir:63, build_dataset_specs:80, register_all_itd_datasets:173 | 用户上传原文，未修改，供注册脚本使用/参考 |
| [shared/mytools/itd_trainer.py](shared/mytools/itd_trainer.py) | ITDCOCOEvaluator:21, ITDMask2FormerMapper:49, __init__:66, __call__:183, ITDMask2FormerTrainer:301 | 真实mapper按尺寸/翻转读取cache，空GT字段处理；其他Trainer代码保留 |

## 补丁统计与应用验证

已分别在对应提交的临时快照验证正向应用（index）与反向应用；未应用到仓库的模型源文件。

```text
 configs/tree-aware/baseline.yaml                   |   8 +
 configs/tree-aware/tree_aware_v8.yaml              |  21 ++
 configs/tree-aware/v8_a.yaml                       |  10 +
 configs/tree-aware/v8_a_cache.yaml                 |   7 +
 configs/tree-aware/v8_b.yaml                       |   6 +
 configs/tree-aware/v8_b_cache.yaml                 |   7 +
 configs/tree-aware/v8_c.yaml                       |   2 +
 configs/tree-aware/v8_c_cache.yaml                 |   7 +
 mask2former/config.py                              |  26 ++
 mask2former/maskformer_model.py                    | 408 ++++++++++++++-------
 mask2former/modeling/criterion.py                  |  43 +++
 mask2former/modeling/matcher.py                    |  19 +-
 .../mask2former_transformer_decoder.py             |  50 ++-
 mask2former/modeling/tree_criterion.py             | 161 ++++++++
 mask2former/modeling/tree_geometry.py              | 153 ++++++++
 mask2former/modeling/tree_geometry_cache.py        | 168 +++++++++
 tools/precompute_tree_geometry.py                  | 144 ++++++++
 17 files changed, 1100 insertions(+), 140 deletions(-)

 configs/tree-aware/baseline.yaml                   |   8 +
 configs/tree-aware/tree_aware_v8.yaml              |  21 +++
 configs/tree-aware/v8_a.yaml                       |  10 ++
 configs/tree-aware/v8_a_cache.yaml                 |   7 +
 configs/tree-aware/v8_b.yaml                       |   6 +
 configs/tree-aware/v8_b_cache.yaml                 |   7 +
 configs/tree-aware/v8_c.yaml                       |   2 +
 configs/tree-aware/v8_c_cache.yaml                 |   7 +
 mask2former/config.py                              |  26 ++++
 mask2former/maskformer_model.py                    | 135 ++++++++++++++---
 mask2former/modeling/matcher.py                    |  19 ++-
 mask2former/modeling/pixel_decoder/msdeformattn.py | 106 +++++--------
 .../mask2former_transformer_decoder.py             |  80 +++++++---
 mask2former/modeling/tree_criterion.py             | 161 ++++++++++++++++++++
 mask2former/modeling/tree_geometry.py              | 153 +++++++++++++++++++
 mask2former/modeling/tree_geometry_cache.py        | 168 +++++++++++++++++++++
 tools/precompute_tree_geometry.py                  | 144 ++++++++++++++++++
 17 files changed, 955 insertions(+), 105 deletions(-)
```
