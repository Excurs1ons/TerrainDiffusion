# 第三方声明 / Third-Party Notices

本项目包含来自以下上游项目的代码，并按其许可证使用。

## terrain-diffusion

- **来源 / Source**: https://github.com/xandergos/terrain-diffusion
- **作者 / Author**: Alexander Goslin (xandergos)
- **许可证 / License**: MIT License (见仓库根目录 [LICENSE](LICENSE))
- **论文 / Paper**: https://arxiv.org/abs/2512.08309
- **项目主页 / Website**: https://xandergos.github.io/terrain-diffusion/
- **模型 / Model**: https://huggingface.co/xandergos/terrain-diffusion-30m

`vendor/terrain_diffusion/` 目录下的代码摘自上游仓库的推理管线
（`world_pipeline` 及其依赖闭包），除拷贝外未作修改，
版权归 Alexander Goslin 所有，依 MIT 许可证授权使用。

The code under `vendor/terrain_diffusion/` is copied from the upstream
inference pipeline (the `world_pipeline` dependency closure) without
modification. Copyright Alexander Goslin, used under the MIT License.

## 数据 / Data

- **WorldClim v2.1** (https://www.worldclim.org/): 气候栅格数据
  (`wc2.1_10m_bio_*.tif`)，首次运行时自动下载。WorldClim 数据可免费用于
  学术与商业用途，需注明来源。
- **ETOPO** (`etopo_10m.tif`): 全球地形数据。

---

本项目自身的代码（`terrain_export.py`、`analyze.py` 等）同样以 MIT 许可证发布。
