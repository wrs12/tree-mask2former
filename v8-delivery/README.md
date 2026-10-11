# Tree-aware Mask2Former V8 交付

## 最新版：缓存优化与 baseline 起点

请优先查看 [v8-cache-v2 完整交付目录](v8-cache-v2/README.md)。

- [下载完整代码包](v8-cache-v2/V8-cache-complete.zip)
- [性能审核与详细修改说明](v8-cache-v2/PERFORMANCE_REVIEW.md)
- [逐文件修改位置](v8-cache-v2/FILES.md)
- [基于 baseline 的完整 Git diff（推荐）](v8-cache-v2/baseline-v8-cache.patch)
- [外部 mytools mapper 补丁](v8-cache-v2/shared-tools.patch)
- [完整模型设计与公式说明](V8完整技术详解.md)：其中旧版在线 target 生成章节已由新版缓存流程取代。

所有完整源文件也放在新版目录里，可以直接浏览。最新版本保留原始 Mask2Former 的基础算法，不包含此前讨论后撤回的两项数据复制优化。模型源码未直接应用这些补丁；此分支保存的是交付材料。

## 旧版存档

以下是最初针对 V7 d209982 的 11 文件交付，包含在线 EDT。保留供比对，**不要与最新版叠加应用**。

- [旧版完整代码包](tree-aware-v8-complete.zip)
- [旧版 Git diff](tree-aware-v8.patch)
- [旧版修改说明](V8修改说明.md)

GitHub 上打开 ZIP，点击 Download raw file 下载。
