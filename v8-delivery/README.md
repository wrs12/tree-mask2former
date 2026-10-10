# Tree-aware Mask2Former V8 delivery

This directory contains the complete proposed V8 changes based on commit `d209982ebe463dbb54881117fa7beefc3a064b8e`.

- [Complete code package](tree-aware-v8-complete.zip): patch, documentation, and all 11 proposed files.
- [Complete Git diff](tree-aware-v8.patch)
- [Implementation guide](V8修改说明.md)

Open a file on GitHub and use its download button or Raw view to save it.

The patch has not been applied to the repository source. Only Python syntax parsing and `git apply --check` were performed; the model and training were not run. The example configurations inherit COCO settings and require your existing tree dataset configuration for tree experiments.
