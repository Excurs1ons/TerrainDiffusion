"""分析输出 TIFF 的值域分布 (默认读取 output/output_*.tif)"""
import numpy as np
from PIL import Image

base = "F:/repos/PyCharm/TerrainDiffusion/output"
names = ["elev", "temp", "t_season", "precip", "p_cv", "beta"]

for n in names:
    path = f"{base}/output_{n}.tif"
    img = Image.open(path)
    arr = np.array(img, dtype=np.float32)
    print(f"{n:12s}  shape={arr.shape}  "
          f"min={arr.min():12.4f}  max={arr.max():12.4f}  "
          f"mean={arr.mean():12.4f}  std={arr.std():12.4f}  "
          f"p1={np.percentile(arr,1):12.4f}  p99={np.percentile(arr,99):12.4f}")
