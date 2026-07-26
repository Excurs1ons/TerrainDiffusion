# TerrainDiffusion 纹理导出工具

把 [terrain-diffusion](https://huggingface.co/xandergos/terrain-diffusion-30m) 扩散模型封装成一条命令：给定世界坐标区域，输出 6 通道 32-bit float TIFF（高程 + 气候），可直接作为 GPU 纹理（R32Float）在 ComputeShader 中采样。

本项目聚焦**部署、使用、工程化**——不改模型、不做研究，上游推理管线原样内嵌于 `vendor/`。

## 环境

- Python ≥ 3.12，[uv](https://docs.astral.sh/uv/)，CUDA GPU（CPU 可跑但慢 ~10×）

```bash
uv sync
```

## 使用

```bash
# 256x256, seed=1
uv run python terrain_export.py --i1 0 --j1 0 --i2 256 --j2 256 --seed 1 -o out.tif

# 1024x1024, fp16 加速, 附带 PNG 预览
uv run python terrain_export.py --i1 0 --j1 0 --i2 1024 --j2 1024 --seed 42 --dtype fp16 --normalize -o big.tif

# 仅高程
uv run python terrain_export.py --i1 0 --j1 0 --i2 512 --j2 512 --no-climate -o elev.tif
```

`-o out.tif` → `output/out/out_elev.tif` 等（`-o output.tif` 则落到 `output/` 根）。完整参数见 `--help`。

## 输出通道

| 文件 | 含义 | 单位/范围 |
|---|---|---|
| `_elev.tif` | 高程 | 米，海平面=0，约 -500~5000 |
| `_temp.tif` | 实际温度 = 基准 + 递减率×max(elev,0) | °C，约 -30~30 |
| `_t_season.tif` | 温度季节性（WorldClim bio4 去趋势） | 大陆性高 / 海洋性低 |
| `_precip.tif` | 年降水量（WorldClim bio12） | mm/yr，0~10000 |
| `_p_cv.tif` | 降水季节性（WorldClim bio15, CV%） | 季风高 / 均匀低 |
| `_beta.tif` | 温度递减率 | °C/m，约 -0.004~-0.01 |

## 坐标系统

- 全局像素坐标，原点 `(0,0)` 左上，`i` 向南、`j` 向东，**30 m/像素**
- `(0,0)→(256,256)` = 7.68 km × 7.68 km；任意大小一次生成，无需手动分块
- 相同 `--seed` + 相同坐标 ⇒ 完全相同的地形

## 首次运行

1. 模型权重自动从 HuggingFace 拉取（默认 `hf-mirror` 镜像，缓存 `F:\.hf_cache`，见 `terrain_export.py` 顶部）
2. WorldClim 气候数据自动下载到 `data/global/`（约 22 MB，一次性）
3. `data/global/synthetic_map_stats.json` 自动生成

## 工程说明

- `vendor/terrain_diffusion/`：上游推理闭包原样拷贝（coarse→latent→decoder 三级扩散），包名与 import 未改，便于追踪上游
- 气候 4 通道（t_season/precip/p_cv/beta）是 ~7.7 km 分辨率 coarse map 的双线性上采样，无扩散细化——设计如此，代表大尺度气候先验；`elev`/`temp` 经完整扩散管线，无网格痕迹

## 致谢与许可

推理核心来自 **Alexander Goslin (xandergos)** 的 [terrain-diffusion](https://github.com/xandergos/terrain-diffusion)（[论文](https://arxiv.org/abs/2512.08309)，[模型](https://huggingface.co/xandergos/terrain-diffusion-30m)），MIT 许可收录于 `vendor/`。气候数据 [WorldClim v2.1](https://www.worldclim.org/)，地形数据 ETOPO。详见 [NOTICE.md](NOTICE.md)。

[MIT](LICENSE)
