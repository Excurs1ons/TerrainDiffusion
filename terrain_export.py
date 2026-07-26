"""
Terrain Diffusion — 推理导出工具
===================================
运行 terrain-diffusion 推理，每个通道输出为单独的 32-bit float TIFF，
可直接用图像查看器预览，也可零转换上传 GPU 用于 ComputeShader。

=== 数据管线概览 ===

  WorldClim 真实数据 (bio1/4/12/15)       随机种子
        │                                      │
        ▼                                      ▼
  synthetic_map.py ──→ coarse map (5ch) ──→ 扩散模型 (UNet)
  (低分辨率全球先验)      │                      │
                     双线性上采样              高分辨率细节
                        │                      │
                        ▼                      ▼
                  _compute_climate() ←── elev (扩散输出)
                        │
                        ▼
                  climate (5ch) + elev (1ch) = 6 个 F32 TIFF

=== 输出文件 (均为 F32 单通道 TIFF, 256x256) ===

  ┌───────────────────┬──────────────────────────────────────────────────────────┐
  │ 文件后缀          │ 含义                                                     │
  ├───────────────────┼──────────────────────────────────────────────────────────┤
  │  _elev.tif        │ 高程 (elevation)                                         │
  │                   │   单位: 米 (m)，海平面=0，可正(陆地)可负(海底)           │
  │                   │   扩散模型直接输出的高分辨率高程图                         │
  │                   │   经 sign(z)·z² 逆变换还原为真实海拔                     │
  │                   │   典型范围: -500m ~ 5000m                                │
  │                   │   GPU 纹理: R32Float，ComputeShader 中直接采样高度       │
  ├───────────────────┼──────────────────────────────────────────────────────────┤
  │  _temp.tif        │ 实际温度 (realistic temperature)                         │
  │                   │   单位: °C                                               │
  │                   │   = temp_baseline + beta × max(elev, 0)                  │
  │                   │   temp_baseline: coarse map 上采样的海平面基准温度       │
  │                   │   beta: 温度递减率 (每升高1m降温多少)                    │
  │                   │   基于 WorldClim bio1 校准，叠加高程修正                 │
  │                   │   典型范围: -30°C (高山/极地) ~ +30°C (热带海平面)       │
  ├───────────────────┼──────────────────────────────────────────────────────────┤
  │  _t_season.tif    │ 温度季节性 (temperature seasonality)                     │
  │                   │   对应 WorldClim bio4（月度温度标准差 × 10）             │
  │                   │   已去趋势: detrended = raw - (a×temp + b)               │
  │                   │   去除了温度对季节性的线性影响，保留纯季节性信号         │
  │                   │   值越大 = 冬夏温差越大（大陆性气候，如西伯利亚）       │
  │                   │   值越小 = 全年温差小（海洋性/热带气候，如新加坡）       │
  │                   │   GPU 用途: 决定植被类型、雪线动态                       │
  ├───────────────────┼──────────────────────────────────────────────────────────┤
  │  _precip.tif      │ 年降水量 (annual precipitation)                          │
  │                   │   单位: mm/yr                                            │
  │                   │   对应 WorldClim bio12                                   │
  │                   │   基于 Perlin 噪声校准的全球降水分布                     │
  │                   │   典型值: 0 (沙漠) ~ 10000mm (热带雨林)                  │
  │                   │   GPU 用途: 河流生成、植被密度、侵蚀模拟                 │
  ├───────────────────┼──────────────────────────────────────────────────────────┤
  │  _p_cv.tif        │ 降水季节性 (precipitation seasonality, CV%)              │
  │                   │   对应 WorldClim bio15（降水变异系数）                   │
  │                   │   = (月降水标准差 / 月降水均值) × 100%                   │
  │                   │   值越大 = 旱雨季差异越大（季风气候，如印度）           │
  │                   │   值越小 = 全年降水均匀（热带雨林 / 西欧）               │
  │                   │   GPU 用途: 洪水风险、季节性河流、生态系统类型           │
  ├───────────────────┼──────────────────────────────────────────────────────────┤
  │  _beta.tif        │ 温度递减率 (temperature lapse rate)                      │
  │                   │   单位: °C/m                                             │
  │                   │   每升高 1 米温度下降多少度                               │
  │                   │   公式: (-6.5 + 0.0015 × precip).clip(-9.8, -4.0) / 1000│
  │                   │   受降水量调制: 湿润地区递减率更大（水汽多，降温快）     │
  │                   │   典型值: -0.004 ~ -0.01 °C/m（即 -4 ~ -10 °C/km）      │
  │                   │   国际标准大气: -6.5 °C/km = -0.0065 °C/m                │
  │                   │   GPU 用途: 实时高程→温度修正，动态气候系统             │
  └───────────────────┴──────────────────────────────────────────────────────────┘

=== 数据来源 ===

  WorldClim v2.1 (wc2.1_10m_bio_1/4/12/15.tif)
  10 arc-minute 分辨率全球气候数据，用于:
    1. synthetic_map.py 中生成 coarse map（低分辨率先验）
    2. 气候统计校准（温度-季节性回归、递减率计算）
    3. 作为扩散模型的条件输入，确保生成结果符合真实气候分布

用法:
  python terrain_export.py --i1 0 --j1 0 --i2 256 --j2 256 -o output.tif
    # → output/output_elev.tif, output/output_temp.tif, ...

  python terrain_export.py --i1 0 --j1 0 --i2 512 --j2 512 -o big.tif
    # → output/big/big_elev.tif, ...

  python terrain_export.py --help  # 查看所有参数
"""

import argparse
import os
import sys
import time
import warnings

# 过滤 huggingface_hub 的弃用警告
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")

# 使用 hf-mirror 镜像 HuggingFace，避免 SSL / 网络问题
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 模型缓存目录
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.hf_cache"))

import numpy as np
import torch


def import_terrain_diffusion() -> None:
    """把项目内 vendor/ 加入 sys.path，加载自带的 terrain_diffusion 包。"""
    vendor_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
    if not os.path.isdir(os.path.join(vendor_dir, "terrain_diffusion")):
        raise RuntimeError(f"Cannot find vendored package at {vendor_dir}\\terrain_diffusion")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    import terrain_diffusion  # noqa: F401


import_terrain_diffusion()

from terrain_diffusion.inference.world_pipeline import WorldPipeline  # noqa: E402


from PIL import Image


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_pipeline(**kwargs) -> WorldPipeline:
    """
    构建 WorldPipeline。

    内部工作流程:
        1. 从 HuggingFace 下载预训练扩散模型权重
        2. 加载 coarse map（低分辨率全球气候/高程先验，约 0.5° 网格）
           coarse map 由 synthetic_map.py 从 WorldClim 数据生成，包含 5 通道:
             [0] elev        — 粗高程（编码域: sign(z)·sqrt(|z|)）
             [1] temp        — 基准温度 (°C)（WorldClim bio1）
             [2] temp_std    — 温度标准差（WorldClim bio4 去趋势后）
             [3] precip      — 年降水量 (mm/yr)（WorldClim bio12）
             [4] precip_std  — 降水标准差（WorldClim bio15）
        3. 扩散模型在 coarse map 条件引导下生成高分辨率细节
        4. pipeline.bind() 固定随机种子，确保后续 get() 调用可复现

    参数:
        model:      HuggingFace 模型 ID
                    默认 "xandergos/terrain-diffusion-30m"
                    30m = 30 米分辨率（约 1 arc-second）
                    模型包含 UNet 扩散权重 + coarse map 先验数据
        device:     推理设备 "cuda" / "cpu"
                    None = 自动检测，优先 CUDA
                    CUDA 推理速度约 10x CPU
        seed:       随机种子 (int64)
                    控制扩散模型的初始噪声，决定地形形态
                    相同种子 + 相同坐标 = 完全相同的地形
                    None = 使用当前时间戳作为随机种子
        dtype:      推理精度
                    "fp32" — 32-bit 浮点，最精确（默认）
                    "fp16" — 16-bit 半精度，速度快 2x，显存减半，可能有精度损失
                    "bf16" — Brain Float 16，动态范围同 fp32 但精度低，Ampere+ GPU
        cache_size: 扩散模型 latent 缓存上限
                    缓存已生成的 tile，避免重复推理
                    支持后缀: K/M/G（如 "200M", "1G"）
                    越大缓存越多 tile，但占用更多内存/显存
        batch_size: 扩散模型单次推理 batch size
                    越大越快（并行处理多个 tile），但显存占用线性增长
                    4 = 默认，适合 8GB+ 显存
                    如遇 OOM 可降至 2 或 1
    """
    device = kwargs.pop("device", None) or select_device()
    model = kwargs.pop("model", "xandergos/terrain-diffusion-30m")
    seed = kwargs.pop("seed", None)
    dtype_str = kwargs.pop("dtype", "fp32")
    cache_size = kwargs.pop("cache_size", "200M")
    batch_size = kwargs.pop("batch_size", 4)

    dtype_map = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}

    print(f"Loading {model} on {device} ({dtype_str})...")
    t0 = time.perf_counter()

    pipeline = WorldPipeline.from_pretrained(
        model,
        seed=seed,
        latents_batch_size=batch_size,
        log_mode="info",
        dtype=dtype_map.get(dtype_str),
        caching_strategy="direct",
        cache_limit=_parse_cache(cache_size),
    )
    pipeline.to(device)
    pipeline.bind()
    print(f"Loaded in {time.perf_counter()-t0:.1f}s. Seed: {pipeline.seed}")
    return pipeline


def _parse_cache(val: str) -> int:
    val = val.strip().upper()
    if val.endswith("G"):
        return int(float(val[:-1]) * 1024**3)
    elif val.endswith("M"):
        return int(float(val[:-1]) * 1024**2)
    elif val.endswith("K"):
        return int(float(val[:-1]) * 1024)
    return int(val)


def _resolve_output_stem(output_path: str) -> str:
    """
    将用户给的 -o 路径解析为输出文件主干，统一放到 output/<stem>/ 目录下。

    特殊规则: 主干为 "output" 时直接落到 output/ 根目录，避免 output/output/ 套娃。

    例:
        "output.tif"       → "output/output"      (即 output/output_elev.tif ...)
        "big.tif"          → "output/big/big"     (即 output/big/big_elev.tif ...)
        "foo/bar.tif"      → "output/bar/bar"     (忽略中间目录, 只取文件名主干)
    """
    stem = os.path.splitext(os.path.basename(output_path))[0]
    out_dir = "output" if stem == "output" else os.path.join("output", stem)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, stem)


def _write_tiff_f32(path: str, arr: np.ndarray) -> None:
    """写入单通道 32-bit float TIFF。"""
    img = Image.fromarray(arr.astype(np.float32), mode='F')
    img.save(path)


def _write_exr_f32(path: str, arr: np.ndarray) -> None:
    """写入单通道 32-bit float OpenEXR (通道名 'R')。

    注意: openexr 3.4.x / 3.2.10 在 Windows+Py3.12 上 import 即 segfault,
    本工具固定使用 3.2.3 (见 pyproject.toml)。
    """
    import Imath
    import OpenEXR

    a = np.ascontiguousarray(arr.astype(np.float32))
    h, w = a.shape
    header = OpenEXR.Header(w, h)
    header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    out = OpenEXR.OutputFile(path, header)
    try:
        out.writePixels({'R': a.tobytes()})
    finally:
        out.close()


# 格式 → (写入函数, 扩展名)
_FORMAT_WRITERS = {
    "tif": (_write_tiff_f32, ".tif"),
    "exr": (_write_exr_f32, ".exr"),
}


def _write_normalized_png(path: str, arr: np.ndarray) -> None:
    """
    将 float 数组归一化到 0-255 并保存为 8-bit 灰度 PNG。
    归一化: (arr - min) / (max - min) * 255
    用于可视化预览，普通图像查看器可直接打开。
    """
    a = arr.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-10:
        normed = np.zeros_like(a, dtype=np.uint8)
    else:
        normed = ((a - lo) / (hi - lo) * 255).astype(np.uint8)
    img = Image.fromarray(normed, mode='L')
    img.save(path)


def export_terrain(
    pipeline: WorldPipeline,
    i1: int, j1: int, i2: int, j2: int,
    output_path: str,
    with_climate: bool = True,
    vertical_scale: float = 1.0,
    normalize: bool = False,
    fmt: str = "tif",
) -> dict:
    """
    执行推理并输出 32-bit float 纹理，每个通道一个单独文件。

    内部流程:
        1. pipeline.get(i1,j1,i2,j2) 调用:
           a. _compute_elev(): 从 coarse map 裁出目标区域，扩散模型生成高分辨率高程
              - coarse map 分辨率 = 32×latent_compression 像素/block
              - 扩散模型在 coarse 条件引导下填充细节
              - 输出编码域高程: sign(z)·sqrt(|z|)，内部逆变换为真实海拔
           b. _compute_climate(): 从 coarse map 上采样气候特征
              - temp_baseline, beta 上采样到目标分辨率
              - temp_realistic = temp_baseline + beta × max(elev, 0)
              - 组装 5 通道: [temp, t_season, precip, p_cv, beta]

    参数:
        pipeline:       已加载的 WorldPipeline 实例
        i1, j1, i2, j2: 生成区域的坐标边界 (左上角 i1,j1 到右下角 i2,j2)
                        坐标系: 原点在左上角 (0,0)，i 轴向南，j 轴向东
                        单位: 像素（30m 分辨率，不可更改）
                        例如 (0,0,256,256) = 256×256 像素 = 7.68km × 7.68km
                        可超出全球范围，超出部分自动填充
        output_path:    输出路径文件名 (只用文件名主干, 忽略扩展名和中间目录)
                        主干为 "output" 时落到 output/ 根: output/output_elev.<fmt> 等
                        其他主干落到 output/<stem>/: 例如 "big.tif" → output/big/big_elev.<fmt> ...
        with_climate:   是否计算并导出气候数据（5 通道）
                        False = 仅输出 _elev
        vertical_scale: 垂直缩放因子
                        高程值 × 此系数后写入文件
                        默认 1.0 = 1 像素 = 1 米
                        用于 Rust 引擎单位转换（如需要厘米可设 100.0）
        normalize:      是否额外输出归一化 PNG 预览图
                        True = 每个通道额外生成 _xxx.png（灰度 0-255）
                        归一化: (val - min) / (max - min) × 255
                        可用普通图像查看器直接打开
                        False = 仅输出主格式（GPU 纹理用途）
        fmt:            主输出格式 "tif" / "exr"（均为 32-bit float 单通道）
                        tif = F32 TIFF (默认, 通用)
                        exr = F32 OpenEXR (通道名 'R', 影视/HDR 管线和部分 DCC 更友好)

    Returns:
        dict with keys:
            width:  输出宽度 (像素)
            height: 输出高度 (像素)
            paths:  生成的文件路径列表
    """
    if fmt not in _FORMAT_WRITERS:
        raise ValueError(f"Unsupported format {fmt!r}, choose from {sorted(_FORMAT_WRITERS)}")
    write_f32, ext = _FORMAT_WRITERS[fmt]

    print(f"Inferring terrain [{i1}:{i2}, {j1}:{j2}] (format={fmt})...")
    t0 = time.perf_counter()

    # pipeline.get() 返回:
    #   elev:    (H, W) float32 — 高程，单位米（已完成 sign(z)·z² 逆变换）
    #   climate: (5, H, W) float32 — 5 通道气候数据（见文件头注释）
    with torch.no_grad():
        result = pipeline.get(i1, j1, i2, j2, with_climate=with_climate)

    elev = result["elev"]  # (H, W) float32, meters
    h, w = elev.shape
    elapsed = time.perf_counter() - t0
    print(f"  Elevation: {h}x{w}, {elapsed*1000:.1f}ms")

    # ── 高程纹理 (F32) ──────────────────────────────────────────────
    # 扩散模型输出编码域高程: encoded = sign(z) · sqrt(|z|)
    # pipeline._compute_elev() 内部已完成逆变换:
    #   elev = sign(encoded) · encoded²
    # 最终值为真实海拔（米），海平面 = 0，可正（陆地）可负（海底）
    # 典型范围: -500m ~ 5000m
    # GPU 纹理用途: 作为 R32Float 纹理采样，在 ComputeShader 中还原地形高度
    elev_np = elev.cpu().numpy().astype(np.float32)
    stem = _resolve_output_stem(output_path)
    paths = []

    elev_path = f"{stem}_elev{ext}"
    write_f32(elev_path, elev_np)
    print(f"  Written: {elev_path} ({os.path.getsize(elev_path)/1024:.1f}KB) [elev]")
    paths.append(elev_path)
    if normalize:
        png_path = f"{stem}_elev.png"
        _write_normalized_png(png_path, elev_np)
        print(f"  Written: {png_path} ({os.path.getsize(png_path)/1024:.1f}KB) [elev PNG]")
        paths.append(png_path)

    # ── 气候纹理 (每通道单独 F32) ────────────────────────────────────
    # climate 张量由 pipeline._compute_climate() 计算，形状 (5, H, W)
    #
    # 计算流程:
    #   1. 从 coarse map（低分辨率全球先验）双线性上采样到目标分辨率
    #   2. 用高程修正温度: temp_realistic = temp_baseline + beta * max(elev, 0)
    #   3. 组装 5 通道输出
    #
    # 各通道详细说明:
    #
    #   [0] temp (temp_realistic) — 实际温度
    #       公式: temp_baseline + beta × max(elevation, 0)
    #       temp_baseline: 从 coarse map 上采样的海平面基准温度 (°C)
    #       beta: 温度递减率，每升高 1 米降温多少度
    #       物理含义: 考虑海拔后的真实年均温
    #       典型值: -30°C (高山/极地) ~ +30°C (热带海平面)
    #       数据来源: WorldClim bio1 校准
    #
    #   [1] t_season (temperature seasonality) — 温度季节性
    #       对应 WorldClim bio4，原始为月度温度标准差 × 10
    #       synthetic_map.py 中已做去趋势处理:
    #         temp_std_detrended = temp_std - (a × temp + b)
    #       去除了温度对季节性的线性影响，保留纯季节性信号
    #       物理含义: 冬夏温差幅度
    #       值大 (~300+) = 大陆性气候（如西伯利亚，冬夏极端温差）
    #       值小 (~50)  = 海洋性/热带气候（如新加坡，全年温差小）
    #
    #   [2] precip (annual precipitation) — 年降水量
    #       对应 WorldClim bio12
    #       从 coarse map 双线性上采样，基于 Perlin 噪声校准
    #       物理含义: 一年总降水量
    #       典型值: 0 (沙漠) ~ 10000mm (热带雨林，如乞拉朋齐)
    #       用途: 决定植被类型、河流分布、雪线高度
    #
    #   [3] p_cv (precipitation seasonality / CV%) — 降水季节性
    #       对应 WorldClim bio15，降水变异系数
    #       = (月降水量标准差 / 月降水量均值) × 100%
    #       物理含义: 降水的季节分配均匀程度
    #       值大 (~100%+) = 季风气候（旱季几乎无雨，雨季集中）
    #       值小 (~25%)  = 全年均匀（如西欧、热带雨林）
    #       用途: 影响洪水风险、农业规划、生态系统类型
    #
    #   [4] beta (temperature lapse rate) — 温度递减率
    #       公式: (-6.5 + 0.0015 × precip).clip(-9.8, -4.0) / 1000
    #       单位: °C/m（每升高 1 米温度下降多少度）
    #       受降水量调制: 湿润地区空气含水汽多，递减率更大（降温更快）
    #       典型值: -0.004 ~ -0.01 °C/m（即 -4 ~ -10 °C/km）
    #       国际标准大气: -6.5 °C/km = -0.0065 °C/m
    #       用途: 在 [0] temp 计算中作为高程→温度修正系数
    climate = result.get("climate")
    has_climate = with_climate and climate is not None

    if has_climate:
        climate_np = climate.cpu().numpy().astype(np.float32)  # (5, H, W)
        climate_names = ["temp", "t_season", "precip", "p_cv", "beta"]
        for idx, name in enumerate(climate_names):
            p = f"{stem}_{name}{ext}"
            write_f32(p, climate_np[idx])
            print(f"  Written: {p} ({os.path.getsize(p)/1024:.1f}KB) [{name}]")
            paths.append(p)
            if normalize:
                png_p = f"{stem}_{name}.png"
                _write_normalized_png(png_p, climate_np[idx])
                print(f"  Written: {png_p} ({os.path.getsize(png_p)/1024:.1f}KB) [{name} PNG]")
                paths.append(png_p)

    return {
        "width": w,
        "height": h,
        "paths": paths,
    }


def main():
    # ── 命令行参数说明 ──────────────────────────────────────────────
    # 坐标系统: 全局像素坐标，原点在左上角 (0, 0)
    #   i 轴 = 纵向（向南），j 轴 = 横向（向东）
    #   分辨率: 30m/像素（模型固定），不可更改
    #   例: (0,0)→(256,256) = 256×256 像素 = 7.68km × 7.68km 区域
    #   坐标可超出 [0, 全球尺寸] 范围，超出部分自动填充
    parser = argparse.ArgumentParser(
        description="Export terrain-diffusion output to F32 TIFF textures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成 256x256 高程+气候纹理
  python terrain_export.py --i1 0 --j1 0 --i2 256 --j2 256 -o output.tif

  # 指定种子和精度
  python terrain_export.py --i1 0 --j1 0 --i2 512 --j2 512 --seed 42 --dtype fp16 -o big.tif

  # 仅高程，不要气候
  python terrain_export.py --i1 0 --j1 0 --i2 256 --j2 256 --no-climate -o elev_only.tif

  # 输出 F32 TIFF + 归一化 PNG 预览（可用图片查看器打开）
  python terrain_export.py --i1 0 --j1 0 --i2 256 --j2 256 --normalize -o output.tif

  # 输出 OpenEXR (F32, 通道名 'R')
  python terrain_export.py --i1 0 --j1 0 --i2 256 --j2 256 --format exr -o output.tif
""")

    # 模型与推理参数
    parser.add_argument("--model", default="xandergos/terrain-diffusion-30m",
                        help="HF 模型 ID (默认: xandergos/terrain-diffusion-30m, 30m 分辨率)")
    parser.add_argument("--device", default=None,
                        help="推理设备: cuda / cpu (默认自动检测, 优先 CUDA)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子, 控制地形形态 (None=随机, 相同种子=相同地形)")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32",
                        help="推理精度: fp32=精确(默认), fp16/bf16=快速但可能有损失")
    parser.add_argument("--cache-size", default="200M",
                        help="latent 缓存上限, 支持 K/M/G 后缀 (默认: 200M)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="扩散模型 batch size, 越大越快但显存占用线性增长 (默认: 4)")

    # 区域坐标参数
    parser.add_argument("--i1", type=int, required=True,
                        help="区域左上角 i 坐标 (像素, i 轴向南)")
    parser.add_argument("--j1", type=int, required=True,
                        help="区域左上角 j 坐标 (像素, j 轴向东)")
    parser.add_argument("--i2", type=int, required=True,
                        help="区域右下角 i 坐标 (像素, 不含)")
    parser.add_argument("--j2", type=int, required=True,
                        help="区域右下角 j 坐标 (像素, 不含)")

    # 输出控制参数
    parser.add_argument("--vertical-scale", type=float, default=1.0,
                        help="垂直缩放因子, 高程值 × 此系数 (默认 1.0, 即 1 像素=1 米)")
    parser.add_argument("--no-climate", action="store_true",
                        help="仅输出高程, 不计算/导出气候数据 (5 通道)")
    parser.add_argument("--normalize", action="store_true",
                        help="额外输出归一化 PNG 预览图 (灰度 0-255, 可直接查看)")
    parser.add_argument("--format", choices=["tif", "exr"], default="tif",
                        help="主输出格式: tif=F32 TIFF (默认), exr=F32 OpenEXR (通道 'R')")
    parser.add_argument("-o", "--output", default="output.tif",
                        help="输出文件名 (只用主干; 'output' → output/ 根, 其他 → output/<stem>/<stem>_*)")
    args = parser.parse_args()

    pipeline = build_pipeline(
        model=args.model,
        device=args.device,
        seed=args.seed,
        dtype=args.dtype,
        cache_size=args.cache_size,
        batch_size=args.batch_size,
    )

    export_terrain(
        pipeline,
        i1=args.i1, j1=args.j1, i2=args.i2, j2=args.j2,
        output_path=args.output,
        with_climate=not args.no_climate,
        vertical_scale=args.vertical_scale,
        normalize=args.normalize,
        fmt=args.format,
    )


if __name__ == "__main__":
    main()
