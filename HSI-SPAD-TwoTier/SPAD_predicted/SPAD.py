import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from pathlib import Path

# 设置基础目录
BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'output' / 'models'
FIGURE_DIR = BASE_DIR / 'output' / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 参数 ----------
file_name ="CK_HS-116_maize_pixel_spectra_spec_smoothed_predictions"
EXCEL_PATH   = MODEL_DIR / f"{file_name}.xlsx"
IMG_SIZE     = (480, 480)
CMAP_OPENCV  = cv2.COLORMAP_TURBO
CMAP_SEABORN = 'jet'  # 使用彩虹色
OUT_PNG      = FIGURE_DIR / f"{file_name}_SPAD_pseudo.png"
OUT_SEABORN  = FIGURE_DIR / f"{file_name}_SPAD_seaborn.png"

# ---------- 读取excel中的值 ----------
df = pd.read_excel(EXCEL_PATH)

# ---------- 创建空画布 ----------
H, W = IMG_SIZE
canvas = np.full((H, W), np.nan, dtype=float)

# ---------- 填充数值 ----------
for _, row in df.iterrows():
    y, x, val = int(row['y']), int(row['x']), row['Predicted_SPAD']
    if 0 <= y < H and 0 <= x < W:
        canvas[y, x] = val

# ---------- 创建掩模 (标记植株区域) ----------
mask = ~np.isnan(canvas)

# ---------- 归一化 (仅对植株区域) ----------
if np.any(mask):
    vmin, vmax = np.nanmin(canvas), np.nanmax(canvas)
    canvas_normalized = np.zeros_like(canvas)
    canvas_normalized[mask] = 255 * (canvas[mask] - vmin) / (vmax - vmin)
else:
    canvas_normalized = np.zeros_like(canvas)

# 转换为 uint8
canvas_uint8 = np.clip(canvas_normalized, 0, 255).astype(np.uint8)

# ---------- OpenCV 伪彩色图 ----------
# 应用伪彩色
pseudo = cv2.applyColorMap(canvas_uint8, CMAP_OPENCV)

# 修改2: 将非植株区域设置为黑色
pseudo[~mask] = [0, 0, 0]  # BGR 黑色 (已经是黑色，不需要修改)

# 保存图像
cv2.imwrite(str(OUT_PNG), pseudo)
print(f'已保存 OpenCV 伪彩色图: {OUT_PNG}')

# ---------- Seaborn 预览 ----------
plt.figure(figsize=(5, 5))

# 修改3: 创建背景为黑色的画布
# 使用全0数组代替全NaN数组
background = np.zeros(canvas.shape)  # 全0数组
norm = Normalize(vmin=0, vmax=40)  # 设置颜色范围0-40
# 然后在heatmap中添加参数: norm=norm
# 绘制黑色背景
# 注意: 使用vmin=0, vmax=0 使所有值映射为黑色
sns.heatmap(background, cmap='gray', cbar=False, square=True,
            vmin=0, vmax=0, alpha=1.0)  # alpha=1.0 确保不透明

# 修改4: 叠加植株区域的热力图，使用彩虹色
# 注意: 设置mask=~mask 只在植株区域显示
ax = sns.heatmap(canvas,
            cmap=CMAP_SEABORN,  # 使用彩虹色
            # cbar_kws={'label': 'Predicted SPAD'}, #调整颜色条
            cbar_kws={
                            # 'label': 'Predicted SPAD',
                            'fraction': 0.045,  # 调整颜色条高度为图的5%
                            'aspect': 20,      # 调整颜色条的宽高比
                            'pad': 0.05        # 调整颜色条与图的间距
                        },
            square=True,
            linewidths=0,
            alpha=1.0,  # 不透明
            mask=~mask,
            norm=norm,
            xticklabels = False,  # 不显示x轴刻度标签
            yticklabels = False)  # 不显示y轴刻度标签 # 只在植株区域显示




plt.tight_layout()
plt.savefig(str(OUT_SEABORN), dpi=600, transparent=False)
plt.show()
print(f'已保存 Seaborn 预览图: {OUT_SEABORN}')
