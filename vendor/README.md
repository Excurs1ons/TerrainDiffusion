# Vendored terrain_diffusion

此目录包含从上游仓库精简拷贝的推理管线代码（`world_pipeline` 依赖闭包）。

- **上游 / Upstream**: https://github.com/xandergos/terrain-diffusion
- **作者 / Author**: Alexander Goslin (xandergos)
- **许可证 / License**: MIT（见项目根目录 `LICENSE`）

## 与上游的差异 / Deviations from upstream

仅一处功能性修改，其余代码与上游逐字节一致：

| 文件 | 行 | 改动 | 原因 |
|---|---|---|---|
| `inference/world_pipeline.py` | 1356 | `grid_sample(..., mode='bilinear')` → `mode='bicubic'` | 气候通道（t_season / precip / p_cv / beta）直接双线性上采样会在 256px 粗单元边界留下可见网格纹；双三次插值一阶导数连续，消除网格。代价是极值处轻微 overshoot |

Code in this directory is copied from the upstream inference pipeline.
Copyright Alexander Goslin, licensed under MIT. See `LICENSE` at the
repository root. The only deviation from upstream is the single-line
change documented above.
