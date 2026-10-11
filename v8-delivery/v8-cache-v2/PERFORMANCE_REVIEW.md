# V8 性能审核、baseline 起点与缓存修改说明

## 1. 建议从哪个版本开始

**最新约束：不改原始 Mask2Former 的基础采样、匹配和 loss 逻辑。已撤回 matcher repeat→expand、mask loss 调整转换顺序两项尝试。**

建议从你仓库的 `c2aa3786db69edd4a5fa3abb8d31efeaa990a4b1` 开始。

已通过 Git 历史和官方 GitHub HEAD 核对：官方 facebookresearch/Mask2Former 的 HEAD 是 `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a`；你的 c2aa378 与它只有 matcher 中一行 AMP 兼容性修复：`autocast(enabled=False)` 改为 `torch.amp.autocast("cuda", enabled=False)`。模型结构仍是官方 baseline。

交付两条独立路径：

- **推荐：baseline-v8-cache.patch + baseline-proposed/**，针对 c2aa378。Pixel Decoder 保留 baseline 原文，Decoder 中直接没有 V7 adapter；基础 criterion 不包含 V7 Boundary Dice，只有 V8 subclass 使用的形态学辅助函数。原始 inference 方法和 forward 推理后处理保留 baseline 文本。
- **兼容：v7-v8-cache.patch + proposed/**，针对原始 V7 的 d209982。保留旧模块开关以兼容 V7；V8 YAML 将两个 LEGACY 开关关闭。基础 criterion 保留原始 V7 文本；它的旧边界计算问题仅作为审核发现，不在本次改动。推荐 baseline 路径本来没有该旧计算。

两份都是完整补丁，不能连续叠加；也不要叠加旧版 tree-aware-v8.patch。已有自行修改的 V8 时，先保存当前代码，再按完整文件合并。你本地 pasted 版本已经增加 build_boundary/build_distance，这次完整文件包含这些参数。

V7 保留为实验对照，避免覆盖。baseline、V7、V8-A/B/C 使用同一数据拆分、训练增强、700 Queries、初始化来源和训练计划。若 V8 用 V7 训练后的 checkpoint 初始化，而 baseline 用官方 checkpoint，就引入了另一项实验变量。

## 2. 审核范围与证据边界

阅读了模型 forward、Pixel Decoder、Transformer Decoder、Hungarian matcher、原始 criterion、新增 geometry/query/criterion，以及你上传的 notebook、dataset_utils.py、itd_trainer.py、mask2former_predictor.py。

这是训练调用链的代码审核和 CPU 一致性验证。未在云端运行单木训练，未生成实际 Dataset_4 缓存，未执行 4090 性能测试；不声称修改后一定达到某个秒数。Encoder 0.025s、Query 0.027s、EDT 5s 是你提供的测量，CUDA 异步计时的方式尚未核实。

## 3. 当前哪些计算最值得处理

令 N 为一张图片 GT 实例数，Q 为 Queries 数，H×W 为输入大小，h×w 为 mask feature 大小，P 为采样点数，L 为预测阶段数。

| 路径 | 成本/风险 | 本次处理 |
|---|---|---|
| 在线逐实例 EDT | 每步 O(NHW)，CPU 工作，可能有 GPU→CPU 同步 | 缓存模式完全移出训练循环 |
| GT GPU→CPU→GPU | masks 已搬到 GPU 后又转 NumPy，再搬回 dense maps | mapper worker 直接读取 CPU cache，只上传小监督图 |
| GT centroid | 在线把全部 padded GT 转 float，再做全分辨率 reduction | 离线保存 N×2 像素坐标，在线只按 batch 尺寸归一化 |
| 不启用的 Boundary/Distance GT | 权重为 0 仍做形态学、resize、foreground pooling | config 驱动生成开关；A 不做这些计算 |
| foreground per-instance pool | N 次转换与 pooling | 逐实例 union 后只做一次 pooling，数学等价 |
| 基础 criterion Boundary Dice | V7 即使最终丢弃也先计算 | 只记录问题；不改基础 criterion；推荐 baseline 没有这条旧调用 |
| matcher 采样坐标复制 | 每层对 GT/Query 分别 repeat 同一坐标 | 按你的最新要求保留原写法，不修改基础采样 |
| mask loss GT float 转换 | 先转换所有 GT，之后才选 matched GT | 按你的最新要求保留原顺序，不修改基础 loss |
| Query states | A/B separation=0 仍逐层额外 normalize 和保存 | 只在 separation>0 时保存，参数及 checkpoint 键不变 |
| separation 负对计数 | 每张图 pairs.sum().item() 导致 host 同步 | tensor 累加、clamp_min(1) 归一化，公式不变 |
| auxiliary matching | 继承 Mask2Former，每阶段独立匹配与点损失 | 保留；共享最终匹配会改变监督方式 |
| 700 Queries + 高分辨率 | attention、mask projection、匹配都可能很重 | 不擅自减 Q/点数/层数；在相同设置下比较 |
| 日志/评估/checkpoint | notebook 两步就评估、保存，writer 每步 | 提供测量建议，未替你改实验计划 |

“复杂”不等于应该删除。Transformer、逐层 matching、point loss 是模型算法的一部分；优先移除重复目标构建与无效工作，之后根据 4090 profiler 判断下一步。

## 4. 4090 的内存成本为什么也要考虑

4090 的 24GB 显存不能掩盖 CPU EDT 开销。还要避免巨大的临时 tensor。

以 N=700、1024×1024 为例：bool GT 约 700MiB；相同 GT 转 float32 约 2.73GiB。matcher 和 point loss 可能在多个阶段重新转换。这次缓存 centroid 避免了 V8 新增的全分辨率 centroid 转换；原始 matcher 和 mask loss 的转换顺序不变，仍可能产生较大分配。按你的要求，这次不优化它们。

采样坐标以 Q=700、P=12544、float32、2 个坐标分量计算，每份显式 repeat 约 67MiB。这是审核发现，最终代码仍使用 repeat，没有改为 expand。point_sample 内部还会分配转换后的 grid。

1152 相对 1024，像素数为 1.266 倍；768 相对 1024 为 0.5625 倍。不同尺寸的 iteration 本来就不能直接比较。相同 checkpoint、同一数据增强策略下先看稳定均值，再决定是否做 1024 固定尺寸的新实验。

本次未默认使用 torch.compile、梯度 checkpointing、减少 Queries/点数、共享 matcher、近似 EDT、float16 cache。它们分别涉及编译兼容性、额外重算、实验变量或数值变化，应在确定具体瓶颈后单独评估。

## 5. 为什么没有直接缓存原图 geometry 后再 resize/flip

你提供的 mapper 实际执行：ResizeShortestEdge、多尺度 choice、水平随机翻转、垂直随机翻转、Brightness、Contrast。它没有读取 `cfg.INPUT.RANDOM_FLIP`，所以只在 config 中关闭 RandomFlip 并不能关闭这里的翻转。

EDT 极大值受 resize 后的栅格化影响；有多个等大极值时，np.argmax 的首次位置也受翻转影响。Gaussian 的 sigma 定义在 feature pixels；boundary 的宽度与 resize/pooling 有关。原图 cache 简单 resize、或者直接翻转带右下 padding 的低分辨率图，都可能改变监督。

因此离线处理 **每个实际 resize 尺寸 × 每种翻转组合**，重新经过你原来的 mapper 转 annotation、生成 mask、过滤空实例，再计算 geometry。训练时根据真正执行的 transforms 读取对应变体。Brightness/Contrast 不影响 GT，不生成额外 cache。

四个训练短边 `(768,896,1024,1152)` 最多产生 16 个训练变体/图片；MAX_SIZE 限制可能让若干 resize 产生相同实际 H/W，会自动去重。验证集按测试 resize，不生成翻转变体。正常验证只做推理，不需要 GT geometry cache，生成 val cache 是为了完整交付和离线检查。

可以继续保留你的多尺度和两种翻转。若你决定固定 `(1024,)`，应对所有实验一起改、重新导出 config 并生成 cache；这时最多 4 个训练变体。Crop、旋转等新几何增强目前不受支持，会报错，而非读取错误地图。

## 6. 缓存格式为什么是分片 NPZ

不是一个巨大 pkl 一次性塞入主进程。缓存目录结构：

```text
tree_cache/
  Dataset_4_train/
    manifest.json
    <sample_sha256>_1024x1024_h0v0.npz
    <sample_sha256>_1024x1024_h1v0.npz
    ...
  Dataset_4_val/
    manifest.json
    ...
```

逻辑上仍是 `image_id → variant → geometry`。键同时包含 image_id 和完整文件路径，避免不同图片复用相同 ID；移动数据路径后需要重建缓存。

- center/distance 为 float32，避免量化误差。
- boundary/foreground 为 uint8，valid 为 bool；这些目标本来只有 0/1。
- centroid_pixels 为 float32 的 N×2。
- meta 存储 image_size、padded_size、feature_size、stride、divisor、sigma、翻转和实例数。
- manifest 保存 dataset、预处理 profile、标注签名、可用分支、变体文件索引。
- NPZ 使用 allow_pickle=False 读取；原子替换文件和 manifest，避免把半写入文件作为完成结果。
- 每个 DataLoader worker 最多保留 CACHE_MEMORY_ITEMS 个解压结果，默认 8，可设 0 关闭 LRU。
- 全分支 cache 可服务 A，只读取 center/valid；A-only cache 不能服务 B/C，会要求重建。

在 stride=4、1024 输入下，一个全分支变体原始数组约 704KiB + centroid/meta；16 个同等尺寸变体约 11MiB/图片，压缩后大小取决于数据。磁盘容量需要按样本数考虑，优先使用本地 SSD。并发 precompute 会增大主机 RAM 占用，默认 workers=1，通常从 1 或 2 开始。

不要在训练过程中覆盖正在读取的 cache。标注、sigma、resize、padding、mapper 栅格化/过滤规则改变后都应重新生成。修改 worker 数/LRU 大小不改变 cache 数值。

## 7. cache reader 检查什么

初始化检查 schema、dataset、resize profile、sigma、stride、divisor、图像格式、mask 格式、所需分支。每次读取检查样本键、标注签名、实际尺寸、翻转组合、每张图 shape 和 centroid 数量。

缺文件、缺变体、A-only cache 用于 B/C、标注改变、输出 stride 改变，都直接报错。**CACHE_ENABLED=True 时没有在线 EDT fallback。** CACHE_ENABLED=False 保留明确的旧在线接口用于兼容；它不是性能优化模式。

缓存只提供监督目标；前向使用的是模型自己预测的 geometry。测试/预测时 mapper 不读取 GT cache，模型不依赖标注，不引入信息泄漏。

## 8. 模型如何接入，以及 batch padding 对齐

新流程：

```text
DatasetCatalog record
  → ITD mapper 的实际 Resize/Flip/photometric transforms
  → 按尺寸/翻转读取 CPU cache
  → batched_inputs['tree_geometry_cache']
  → 模型正常 backbone / Pixel Decoder / V8 / Transformer Decoder
  → prepare_targets 上传 cache maps，补齐 batch feature padding
  → criterion
```

原始 masks、labels 和分类/分割输出尺寸保持不变。cache 是新增 input dictionary 字段。

离线 maps 在“单张图按 divisor 补齐”后生成。实际 batch 可能被另一张较大图撑大；只在 feature 图右侧/下侧补 0，valid 补 False。由于 padded H/W 与 stride 对齐，有效区域内的 feature 像素坐标与 Gaussian sigma 不变，不能把 cache 整张再 resize。

centroid 保存的是像素坐标，在线用 `[batch_padded_width, batch_padded_height]` 相除，得到 matcher/separation 原本使用的归一化坐标。不能直接复用离线单图归一化结果，否则多图 batch 的匹配和邻域半径会错。

模型检查实际输出与 COMMON_STRIDE 的关系，要求正的 SIZE_DIVISIBILITY，且能被 COMMON_STRIDE 整除。当前脚本不猜测 backbone 的负值自动 padding 配置；若你的配置 SIZE_DIVISIBILITY=-1，要先明确实际 divisor。

## 9. V8-A/B/C 的目标和开关

| 实验 | Center target | Boundary target | Distance + foreground target | 几何匹配 | Query separation |
|---|---|---|---|---|---|
| baseline | 不构建 | 不构建 | 不构建 | 关闭 | 关闭 |
| V8-A | 构建 | 关闭 | 关闭 | 关闭 | 关闭 |
| V8-B | 构建 | 构建 | 构建 | 关闭 | 关闭 |
| V8-C | 构建 | 构建 | 构建 | 开启 | 开启 |

A 的 BOUNDARY_WEIGHT 和 DISTANCE_WEIGHT 是 0.0；B/C 为 1.0。使用浮点 YAML 数值，避免 YACS 导入 config dump 时出现 int/float 类型错误。

中心仍来自每个实例 EDT 的极大值。所以 **A 虽然关闭 Distance loss，离线仍需要 EDT 来定位 Center**；关掉 Distance 并不能消除在线 Center 的 EDT。性能模式通过 cache 把这部分全部提前。

`TreeGeometryEncoder` 的三个预测分支和融合结构完全保留；A 只是关闭两项监督，并关闭 mask feature fusion。按照原 V8 设计，Query Generator 仍从 geometry feature 采样。若将 A 改成“只运行 center 分支”，就属于另一个消融定义，这次没有做。

保留原 point mask BCE/Dice、逐阶段 auxiliary matching、中心定位规则、几何匹配代价、Query separation 正负对的定义和权重。

## 10. cfg 作用域修复

`prepare_targets()` 不访问 cfg。配置只在 from_config 读取：

```python
"tree_build_boundary": tree.BOUNDARY_WEIGHT > 0,
"tree_build_distance": tree.DISTANCE_WEIGHT > 0,
"tree_cache_enabled": tree.CACHE_ENABLED,
"tree_cache_stride": cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
```

通过 __init__ 保存到 self，prepare_targets 使用 self。新增 bool 参数带默认 False，保留直接调用构造器时的兼容性；新字段不改变模型参数形状或 checkpoint 键。非缓存模式的 build_tree_targets 明确接收 build_boundary/build_distance。

## 11. foreground pooling 为什么可以合并

对于二值 mask，union 是逐像素 max；adaptive max pool 也是区域 max。因此

`max_i pool(mask_i) = pool(max_i mask_i)`。

只需 union 一次，再 pool 一次。boundary 不能用 union 后形态学代替逐实例形态学，否则相邻树的内部边界消失；Distance 更不能用 union EDT 代替逐实例归一化 EDT。这两条原理保留。

## 12. separation 修改为什么数学等价

原负对计数是 `pairs.sum().item()`，每张图把一个 GPU 标量拉回 Python。新代码让 count 保留为 tensor，最后 `negative_total / negative_count.clamp_min(1)`。

计数和归一化数值相同，不对计数求导。仍保留同图邻近不同 GT 的排斥、相邻预测状态同 GT 的吸引。没有改变 margin/radius、正负对权重或独立匹配。

A/B separation=0 时不保存 query_features；V8-C 继续保存所有需要的预测状态。新增属性不是 Parameter，不改变 checkpoint 键。

## 13. notebook 怎么改（先导出最终 cfg）

在 `cfg = setup_cfg()` 之后、创建 Trainer 之前添加：

```python
from pathlib import Path

# 必须先应用新版模型代码及新版 mytools/itd_trainer.py。
cfg.MODEL.TREE_AWARE.CACHE_ENABLED = True
cfg.MODEL.TREE_AWARE.CACHE_DIR = "/home/user/code/tree_cache"
cfg.MODEL.TREE_AWARE.CACHE_MEMORY_ITEMS = 8

# A: 0.0 / 0.0；B/C: 1.0 / 1.0（你的实验 YAML 已有这些设置）。
effective_cfg = Path("/home/user/code/v8_effective.yaml")
effective_cfg.write_text(cfg.dump(), encoding="utf-8")
print(effective_cfg)
```

不要直接用仓库中的 COCO 示例 YAML 生成 Dataset_4 cache。notebook 后续覆盖了 DATASETS、resize、类别、Query 数等参数；必须导出覆盖后的 cfg。

首次训练前在终端运行，项目路径按你的实际目录：

```bash
cd /home/user/code/mask2former/Mask2Former_treeaware_v8
python tools/precompute_tree_geometry.py \
  --config-file /home/user/code/v8_effective.yaml \
  --data-root /home/user/code \
  --shared-tools-root /home/user/code/mask2former \
  --output-dir /home/user/code/tree_cache \
  --datasets Dataset_4_train Dataset_4_val \
  --workers 1
```

脚本使用你真实的注册接口，不是另写标准 COCO 注册：

```python
specs = build_dataset_specs(root / 'training_set', find_validation_root(root))
register_all_itd_datasets(specs, train_names, val_names)
```

这个接口需要三个参数。find_validation_root/build_dataset_specs 仍按你提供的规则寻找 Dataset_3/4/5_val；数据目录不足时原 helper 会报错，这是其现有行为。

脚本直接加载 geometry/cache/config 文件，不导入整套模型，也不构建模型；EDT 由 CPU 执行。可以在训练环境使用同一 Python；不要求 4090参与预计算。导出 config 后仍需等 precompute 完成，再执行 Trainer 的训练 cell。

若要用一个 full cache 跑 A/B/C：用 B 或 C 的 `BOUNDARY_WEIGHT=1.0, DISTANCE_WEIGHT=1.0` 配置预计算一次，保证其他 profile 相同；A 的 reader 自动只解压所需目标。若用 A 配置预计算，只包含 center/valid，不能用于 B/C。

## 14. 2 iteration 的 notebook 不能代表稳定速度

你上传的 notebook 当前设置 MAX_ITER=2、EVAL_PERIOD=2、CHECKPOINT_PERIOD=2、WRITER_PERIOD=1；还会在最后单独 test。这样初始化、worker 启动、首次 CUDA kernel、训练结束 hook、COCO 评估和模型文件写入都可能占短跑的大部分时间。

这不能单凭代码认定是日志中的 18s/iter 的原因；必须区分 trainer 的 iteration timer 与 notebook 总 wall time。EDT 已有明确测量，应先解决。建议另外复制一份 benchmark cfg，先跑几十步，剔除 warmup，测 train iteration；正式训练保持你最终确认的实验计划。

benchmark 时 baseline/V8 必须同尺寸策略；每步日志写入可改为 10/20 步，评估和 checkpoint 设在测量区间之外。不要把验证集 AP 时间算作训练计算成本，也不要直接减少正式实验的评估次数而未记录。

DataLoader 用原来的 4 workers 开始观察 data_time；提高 worker 数并不一定更快，可能增加 PNG decode、polygon rasterization 的竞争及主机内存。优先使用本地 SSD cache，并检查 worker 不发生重启/频繁缓存缺失。

## 15. CUDA 计时和后续定位

CPU `time.time()` 包住 GPU forward 可能只量到 kernel 提交时间。精确 GPU 时间可用 CUDA events，在读取结果时同步；CPU EDT、加载、Hungarian SciPy 则要看 CPU wall time。不要只用 CPU timer 断定 encoder/query 没有 GPU 开销。

建议在独立短测中记录：data_time、整个 forward、matcher/point loss、backward、optimizer step、CPU EDT 调用次数、max_memory_allocated。torch.profiler 同时看 CPU/CUDA，查 aten::to、copy_、grid_sampler、einsum、cudaMemcpy、SciPy/Hungarian附近 host 等待。

cache 模式里 EDT 调用次数应为 0。若仍出现 EDT，先确认实际 import 的 tree_geometry.py、itd_trainer.py 路径以及 CACHE_ENABLED；项目父目录和多个 V8 文件夹容易加载到旧源码。

新 matcher 仍独立计算每层 geometry cost（C），其 soft centroid 是预测 mask 的全分辨率 reduction；不能复用 GT cache 代替预测中心，也不能随意只对 final 层增加 geometry cost，否则匹配规则变化。若它成为瓶颈，需要单独设计、验证新的实验。

## 16. evaluator 和 predictor 的独立问题

你的 ITDCOCOEvaluator 是先调用 `super()._eval_predictions()`，然后才尝试改 `_coco_eval.params.maxDets`。这不足以保证当次评估按 700 detections 计算；某些 Detectron2 版本甚至没有该属性。应使用相应版本 COCOEvaluator 的初始化参数 max_dets_per_image 等正式接口，在评估前设置。此问题影响 AP 解释，不是 EDT 训练瓶颈；本次未顺带改 evaluator，避免把评估语义变化混入性能补丁。

mask2former_predictor.py 构建模型一次，predict_record 用 no_grad，之后 `.to('cpu')`、轮廓提取、绘图、NPZ 写入是离线预测/可视化成本。它不在 Trainer 的训练 forward 链路中，本次未修改。大批可视化应与训练计时分开；没有理由为了训练速度改它的输出格式。

## 17. 验证了什么

分别对 V7 兼容版和 baseline 版运行 CPU 针对性测试，各 15 项通过：

- 原/新 geometry builder：空 GT、单实例、多实例，所有五种目标逐值一致。
- Center-only 开关：Center 一致，关闭的目标为 0，cache 不存其 dense maps。
- 使用上传的真实 ITD mapper：2 尺寸 × 4 翻转 × 2 batch padding，分别覆盖 A 和 full targets；缓存与在线目标逐值一致。
- Brightness/Contrast 保留，CPU cache 到 prepare_targets 接入，测试中的在线 EDT 被替换为抛异常，确保没有调用。
- centroid 重新归一化正确；float32舍入允许绝对 1e-7、相对 1e-6。
- 标注/尺寸/profile/所需分支错误 fail closed，LRU 数量有界，full cache 服务 A。
- 空 annotation list 的 mapper 补齐空 gt_masks。
- Encoder/Query 类 AST 与旧交付完全一致。
- separation loss 空/单/多实例的数值、梯度逐值一致。
- matcher 基础采样写法未变；cost matrix、匹配结果一致（geometry cost 0 和 0.2）。
- point mask BCE/Dice 原函数未变；数值和梯度一致。
- CLI 用实际 mytools.dataset_utils 注册合成 train/val 数据，workers=2，输出完整 manifests/variants。

测试环境为 Python 3.10、torch 2.1.2+cpu、NumPy 1.23.5、SciPy 1.15.3、Detectron2。云端安装的较旧 Detectron2 用到 np.bool，NumPy 1.26 导致 polygon 转 mask 失败，所以仅在测试虚拟环境调整 NumPy 版本；没有往产品代码加全局 np.bool monkey patch。测试有第三方弃用警告。

尚未验证真实 Dataset_4、大规模缓存磁盘量、完整模型 CUDA forward/backward、AP、不同行为的其他 mapper、4090 实际速度。这些需要真实数据与 GPU；CPU 单元测试不能替代整机验收。

## 18. 在自己机器上的验收顺序

1. 在保留 V7 的前提下，建立 baseline c2aa378 的新实验目录/分支，应用推荐补丁和 shared-tools.patch。
2. 确认模型/mapper 实际加载路径；使用同样初始化与超参数。
3. 导出最终 cfg，预计算缓存；A-only 与 full 目录不要误混。
4. 先用少量数据确认 cache 无缺失、loss 有限、关闭分支没有相应 loss；cache 模式 EDT 次数为 0。
5. 相同配置下对比 baseline、在线 V8 与 cached V8 的稳定 step 时间、data_time、显存峰值。
6. 完整训练后才比较 AP；训练速度目标是接近 V7，需要实测，不能从删掉一段 EDT 代码直接保证。

最后的文件与具体修改位置见 FILES.md。完整架构与公式还可参考上一级《V8完整技术详解》，其中旧版在线 target 章节已由本文的 cache 流程取代。
