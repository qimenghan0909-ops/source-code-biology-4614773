# predict_new.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from Preprocessing.Preprocessing import MSC  # 确保使用修改后的函数

# 设置基础目录
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'output' / 'models'


def predict_spectral(new_spectra, model_dir=str(MODEL_DIR)):
    """
    对新光谱进行预测
    new_spectra : (n_samples, n_wavelength)  原始光谱
    return      : (n_samples,) 的预测值
    """
    # 1. 加载所有预处理参数和模型
    mean_spectrum = joblib.load(f'{model_dir}/mean_spectrum.pkl')
    selected_waves = joblib.load(f'{model_dir}/cars_waves.pkl')
    svr_model = joblib.load(f'{model_dir}/svr_model.pkl')

    # 2. 使用训练集的参数进行预处理
    new_msc = MSC(new_spectra, mean_spectrum)  # 关键：使用训练集的均值

    # 3. 应用训练时的波长选择
    new_feature = new_msc[:, selected_waves]
    print(new_feature)
    # 4. 使用保存的模型进行预测
    y_pred = svr_model.predict(new_feature)

    return y_pred.flatten()  # 确保输出为一维数组



if __name__ == '__main__':
    file_name="DS_HS-58_maize_pixel_spectra_spec_smoothed"
    # 读取新光谱数据（格式应与训练数据一致）
    data_df = pd.read_excel(MODEL_DIR / f"{file_name}.xlsx",
                           header=None)
    new_spectra = data_df.iloc[1:, 2:].values.astype(float)

    # 进行预测
    y_hat = predict_spectral(new_spectra)

    # 保存选择的波长索引和对应的实际波长值
    selected_waves = joblib.load(f'{MODEL_DIR}/cars_waves.pkl')
    
    # 读取训练数据的波长信息（第一行）
    train_data_df = pd.read_excel(DATA_DIR / "all_Spec.xlsx",
                                   sheet_name='Sheet1', header=None)
    wavelengths = train_data_df.iloc[0, 2:].values.astype(float)
    
    # 提取对应的实际波长值
    selected_wavelengths = wavelengths[selected_waves]
    
    # 保存为Excel文件
    waves_df = pd.DataFrame({
        '序号': range(1, len(selected_waves) + 1),
        '波长索引': selected_waves,
        '实际波长值': selected_wavelengths
    })
    waves_df.to_excel(MODEL_DIR / f"{file_name}_selected_waves.xlsx", index=False)
    print(f"波长索引已保存：{len(selected_waves)} 个波长 -> {MODEL_DIR / f'{file_name}_selected_waves.xlsx'}")

    # 保存结果
    results = pd.DataFrame({
        'y': data_df.iloc[1:, 0].values,
        'x': data_df.iloc[1:, 1].values,
        'Predicted_SPAD': y_hat
    })
    results.to_excel(MODEL_DIR / f"{file_name}_predictions.xlsx", index=False)
    print(f"成功预测 {len(y_hat)} 个样本，结果已保存：{MODEL_DIR / f'{file_name}_predictions.xlsx'}")
