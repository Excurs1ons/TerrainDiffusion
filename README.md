# TerrainDiffusion 纹理导出工具

基于 [terrain-diffusion](https://huggingface.co/xandergos/terrain-diffusion-30m) 扩散模型的地形纹理生成 CLI。输入一块世界坐标区域，输出 6 通道 32-bit float TIFF（高程 + 5 通道气候），可直接作为 GPU 纹理（R32Float）在 ComputeShader 中采样。

## 环境

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) 包管理
- CUDA GPU（推荐，CPU 可跑但慢 ~10×）

```bash
uv sync
```

## 快速开始

```bash
# 生成 256x256 高程+气候纹理 (seed=1)
uv run python terrain_export.py --i1 0 --j1 0 --i2 256 --j2 256 --seed 1 -o smoke.tif

# 生成 1024x1024, fp16 加速, 附带归一化 PNG 预览
uv run python terrain_export.py --i1 0 --j1 0 --i2 1024 --j2 1024 --seed 42 --dtype fp16 --normalize -o big.tif

# 仅高程, 不要气候
uv run python terrain_export.py --i1 0 --j1 0 --i2 512 --j2 512 --no-climate -o elev_only.tif
```

输出落到 `output/<文件名主干>/`：

```
output/smoke/
├── smoke_elev.tif       # 高程 (米), 海平面=0, 可正可负
├── smoke_temp.tif       # 实际温度 (°C) = 基准 + 递减率 × max(elev,0)
├── smoke_t_season.tif   # 温度季节性 (WorldClim bio4 去趋势)
├── smoke_precip.tif     # 年降水量 (mm/yr, WorldClim bio12)
├── smoke_p_cv.tif       # 降水季节性 (CV%, WorldClim bio15)
└── smoke_beta.tif       # 温度递减率 (°C/m)
```

`--normalize` 时每个通道额外生成一张 0-255 灰度 PNG 预览。

## 坐标系统

- 全局像素坐标，原点 `(0,0)` 在左上角，`i` 轴向南、`j` 轴向东
- 分辨率 **30 m/像素**（模型固定）
- 例：`(0,0)→(256,256)` = 7.68 km × 7.68 km
- 任意大小可一次生成（内部自动 tile + 重叠混合），无需手动分块拼接
- 相同 `--seed` + 相同坐标 ⇒ 完全相同的地形

## 首次运行

1. **模型权重**：从 HuggingFace 拉取（默认走 `HF_ENDPOINT=https://hf-mirror.com` 镜像，缓存到 `HF_HOME=F:\.hf_cache`，均可在 `terrain_export.py` 顶部改）
2. **WorldClim 数据**：`synthetic_map.py` 自动从 UC Davis 下载 4 个气候 tif 到 `data/global/`（约 22 MB，一次性）；`etopo_10m.tif` 已随仓库提供
3. **统计缓存**：`data/global/synthetic_map_stats.json` 首次自动生成

## 项目结构

```
├── terrain_export.py        # 主 CLI
├── analyze.py               # 输出 TIFF 值域分析
├── vendor/
│   └── terrain_diffusion/   # 内嵌的推理库 (world_pipeline 及其依赖闭包)
├── data/global/             # WorldClim 数据 + 统计缓存 (运行时填充)
└── output/                  # 导出产物 (默认 output/output_*.tif; -o big.tif → output/big/)
```

`vendor/terrain_diffusion` 是从上游仓库精简拷贝的推理闭包（coarse → latent → decoder 三级扩散管线），包名与内部 import 保持原样。

## 已知特性

- 4 个气候通道（t_season / precip / p_cv / beta）是从 ~7.7 km 分辨率的 coarse map 直接双线性上采样的，**没有**扩散模型补细节——这是设计行为，它们代表大尺度气候先验。`elev` 和 `temp` 因经过完整扩散管线 / 被 elev 高频调制，无网格痕迹。

## 致谢 / Credit

本项目的推理核心来自 **Alexander Goslin (xandergos)** 的
[terrain-diffusion](https://github.com/xandergos/terrain-diffusion)
（[论文](https://arxiv.org/abs/2512.08309)），
代码以 MIT 许可证收录于 [`vendor/terrain_diffusion/`](vendor/terrain_diffusion/)。
预训练模型来自 HuggingFace
[xandergos/terrain-diffusion-30m](https://huggingface.co/xandergos/terrain-diffusion-30m)。

气候数据来自 [WorldClim v2.1](https://www.worldclim.org/)，
地形数据来自 ETOPO。详见 [NOTICE.md](NOTICE.md)。

## 许可证 / License

[MIT](LICENSE) — 与上游保持一致。
