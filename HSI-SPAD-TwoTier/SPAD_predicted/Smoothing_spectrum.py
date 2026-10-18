import pandas as pd
import numpy as np
from pathlib import Path
from Preprocessing.Preprocessing import MA

# 设置基础目录
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'output' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    # 读取新光谱数据
    file_name="DS_HS-58_maize_pixel_spectra"
    data_df = pd.read_excel(DATA_DIR / f"{file_name}.xlsx",
                            header=None)

    # 提取坐标
    y_coords = data_df.iloc[1:, 0].values
    x_coords = data_df.iloc[1:, 1].values

    # 提取光谱数据
    new_spectra = data_df.iloc[1:, 2:].values.astype(float)

    # 对光谱数据进行平滑处理
    smoothed_spectra = MA(new_spectra)

    # 创建包含坐标和平滑后光谱数据的结果DataFrame
    # 首先创建列名列表
    band_names = data_df.iloc[0, 2:].values
    columns = ['y', 'x'] + band_names.tolist()

    # 组合数据
    results_data = np.column_stack((y_coords, x_coords, smoothed_spectra))

    # 创建DataFrame
    results = pd.DataFrame(results_data, columns=columns)

    # 保存结果
    output_path = MODEL_DIR / f"{file_name}_spec_smoothed.xlsx"
    results.to_excel(output_path, index=False)
    print(f"平滑后的光谱数据已保存到: {output_path}")
