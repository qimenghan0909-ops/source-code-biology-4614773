# train_save.py
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

from Preprocessing.Preprocessing import MSC
from WaveSelect.Cars import CARS_Cloud
from DataLoad.DataLoad import spxy
from Regression.ClassicRgs import Svregression

# 设置基础目录
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'output' / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

save_dir = str(MODEL_DIR)

def train_and_save_model(data, label, save_dir=save_dir):
    """
    训练 MSC → CARS → SPXY 划分 -> SVR 模型并保存
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # 1. MSC 预处理
    # data_msc = MSC(data)
    mean_spectrum = np.mean(data, axis=0)
    data_msc = MSC(data, mean_spectrum)  # 使用计算好的均值

    # 2. CARS 选择波长
    selected_waves = CARS_Cloud(data_msc, label)
    X_feature = data_msc[:, selected_waves]

    # 3. SPXY 划分
    X_train, X_test, y_train, y_test = spxy(X_feature, label)

    # 4. SVR 训练
    rmse, r2, mae, y_pred, y_train_pred, model = Svregression(X_train, X_test, y_train, y_test)

    # 5. 保存
    joblib.dump(X_train, f'{save_dir}/X_train.pkl')
    joblib.dump(y_train, f'{save_dir}/y_train.pkl')
    joblib.dump(mean_spectrum, f'{save_dir}/mean_spectrum.pkl')
    joblib.dump(selected_waves, f'{save_dir}/cars_waves.pkl')
    joblib.dump(model, f'{save_dir}/svr_model.pkl')


    print(f"模型已保存到 {save_dir}/")
    return r2,rmse,  mae, y_test,y_train,y_pred, y_train_pred

if __name__ == '__main__':
    # 读取光谱和标签
    data_df = pd.read_excel(DATA_DIR / "all_Spec.xlsx",
                            sheet_name='Sheet1', header=None)
    data = data_df.iloc[1:, :].values.astype(float)

    label_df = pd.read_excel(DATA_DIR / "all_SPAD.xlsx")
    label = label_df.iloc[:, 1].values

    data_name="0717"
    Rmse_r2_mae = []
    # 训练并保存
    R2,RMSE, MAE,y_test,y_train,y_pred, y_train_pred = train_and_save_model(data, label)
    print(f"训练完成 ->  R2={R2:.4f},RMSE={RMSE:.4f}, MAE={MAE:.4f}")

    temp = [R2,RMSE, MAE]
    Rmse_r2_mae.append(temp)
    print(R2,RMSE, MAE)
    print(Rmse_r2_mae)
    # 创建结果DataFrame并添加列名
    results_df = pd.DataFrame(
        Rmse_r2_mae,
        columns=['R2','RMSE', 'MAE']
    )

    # 保存结果到Excel（带列名）
    results_df.to_excel(MODEL_DIR / f'all_RMSE_R2_MAE_all_methods_{data_name}.xlsx', index=False)

    # 保存预测值和真实值（可选）
    pd.DataFrame(y_train_pred).to_excel(MODEL_DIR / f'all_y_train_pre_{data_name}.xlsx', index=False)
    pd.DataFrame(y_train).to_excel(MODEL_DIR / f'all_y_train_{data_name}.xlsx', index=False)
    pd.DataFrame(y_pred).to_excel(MODEL_DIR / f'all_y_test_pre_{data_name}.xlsx', index=False)
    pd.DataFrame(y_test).to_excel(MODEL_DIR / f'all_y_test_{data_name}.xlsx', index=False)
