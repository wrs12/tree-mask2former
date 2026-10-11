# V8 完整代码与性能优化交付（v2）

**推荐从 baseline `c2aa378` 开始。这里包含全部修改后的文件、完整 Git diff、外部 mapper 和详细文档，不是单个代码片段。**

- [下载全部代码和文档](V8-cache-complete.zip)
- [性能审核、原理、配置与使用步骤](PERFORMANCE_REVIEW.md)
- [逐文件修改位置](FILES.md)
- [baseline 完整 Git diff（推荐）](baseline-v8-cache.patch)
- [V7 完整 Git diff（另一条兼容路径）](v7-v8-cache.patch)
- [外部 mytools mapper 补丁（两条路径均需要）](shared-tools.patch)

在 GitHub 打开 ZIP 文件，使用 Download raw file 下载。也可下载本分支的仓库 ZIP，完整代码文件可直接浏览：

| 路径 | 内容 |
|---|---|
| [baseline-proposed/](baseline-proposed/) | 推荐：以 c2aa378 为起点的全部修改后模型文件、预计算工具、A/B/C 与缓存 YAML |
| [proposed/](proposed/) | 可选：以 d209982 V7 为起点的兼容版本；不要与 baseline 版叠加 |
| [shared/mytools/](shared/mytools/) | 用户工程父目录的 itd_trainer.py、未修改的 dataset_utils.py |
| [tests/](tests/) | CPU 一致性测试及旧版参考 fixtures；不属于运行时模型 |

## 应用顺序

先保留现有 V7/V8 自行改动。以下命令是给你在本地操作的步骤，云端未将补丁应用到仓库的模型源码。

推荐路径：在你的模型仓库从 `c2aa3786db69edd4a5fa3abb8d31efeaa990a4b1` 建立独立分支，进入该分支后：

```bash
git apply --check /下载路径/baseline-v8-cache.patch
git apply /下载路径/baseline-v8-cache.patch
```

兼容 V7 则从 `d209982ebe463dbb54881117fa7beefc3a064b8e` 应用 v7-v8-cache.patch，二选一，不要再叠加旧版 tree-aware-v8.patch。

进入含有 `mytools/` 的项目父目录，再应用外部补丁：

```bash
cd /home/user/code/mask2former
git apply --check /下载路径/shared-tools.patch
git apply /下载路径/shared-tools.patch
```

若你后续改过 mapper，按完整文件合并；不要强行应用不匹配的 patch。dataset_utils.py 是你上传的原文，无需替换。

然后按 PERFORMANCE_REVIEW.md 第 13 节导出 notebook **最终 cfg**，预计算 cache，完成后再训练。示例 YAML 的 COCO 父配置需要你原有 notebook 的 Dataset_4 等覆盖，不可直接当成最终单木配置。

## 保留的基础逻辑与验证范围

按你的最新要求，原始 Mask2Former 的 sampling、matcher 基础 cost、逐层独立匹配和 point mask loss 保留原写法。baseline SetCriterion 整个类通过 AST 与 c2aa378 比对完全一致；只添加 V8 subclass 使用的 boundary helper。两项基础数据复制优化已经撤回。

V8 新增 geometry/query 数学保持原设计；性能改动在离线监督缓存、条件 target 构建、缓存对齐、V8 Query states 和 separation 计数中。

- baseline 与 V7 兼容版本的 CPU 一致性测试：各 15 项通过。
- 三份 patch 均验证正向与反向应用，所有 Python 文件通过语法解析。
- 没有运行真实 Dataset_4 训练，没有实际数据 cache，没有 4090 CUDA 性能/AP 测试。
- CACHE_ENABLED 默认 False，性能模式必须明确启用；启用后 cache 缺失直接报错，不回退在线 EDT。

如需在安装好项目依赖的环境复核测试，在此交付目录运行：

```bash
V8_SOURCE_DIR=baseline-proposed python -m pytest -q tests/test_cache.py
python -m pytest -q tests/test_cache.py
```

完整包只包含本次相关源文件和文档，运行仍依赖原 Mask2Former 仓库、Detectron2 及你的数据目录。
