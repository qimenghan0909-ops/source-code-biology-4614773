import numpy as np
from DataLoad.DataLoad import SetSplit
from Preprocessing.Preprocessing import Preprocessing
from WaveSelect.WaveSelcet import SpctrumFeatureSelcet
from Regression.Rgs import QuantitativeAnalysis
import pandas as pd
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed
import os
from pathlib import Path

# 设置基础目录
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings('ignore')


def SpectralQuantitativeAnalysis(data, label, ProcessMethods, FslecetedMethods, SetSplitMethods, model, windows, plsr,
                                 feature_num, random_seed):
    """
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法
    :param SetSplitMethods : string, 划分数据集的方法
    :param model : string, 定量分析模型
    :param random_seed: int, 随机种子
    :return: Rmse: float, Rmse回归误差评估指标
             R2: float, 回归拟合,
             Mae: float, Mae回归误差评估指标
             y_pred: float, 预测值
             y_test: float, 参考值
             y_train_pred: float, 训练数据预测值
             y_train_: float, 训练数据参考值
    """
    ProcesedData = Preprocessing(ProcessMethods, data, windows)  # 预处理
    FeatrueData, labels = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label, feature_num)  # 光谱特征选择
    # 数据集划分
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.2,
                                                randomseed=random_seed)
    Rmse, R2, Mae, y_pred, y_train_pred = QuantitativeAnalysis(model, X_train, X_test, y_train, y_test, plsr)  # 定量分析
    return Rmse, R2, Mae, y_pred, y_test, y_train_pred, y_train


def run_experiment(reg_model, pre_method, key_band_method, split_method, data, label, windows, plsr, feature_num,
                   random_seed):
    """运行单个实验并返回结果"""
    RMSE, R2, MAE, y_pred, y_test, y_train_pred, y_train = SpectralQuantitativeAnalysis(
        data, label, pre_method, key_band_method, split_method, reg_model, windows, plsr, feature_num, random_seed
    )
    return [RMSE, R2, MAE, pre_method, key_band_method, split_method, reg_model, random_seed]


def run_full_experiment(data, label, regression_model, preprocessing_methods,
                        split_data_methods, WaveSelect_methods, feature_num,
                        windows, plsr, file_name, data_name, run_id, seeds):
    """运行完整的实验流程（一次运行）"""
    print(f"\n{'=' * 50}")
    print(f"开始运行 #{run_id + 1}，随机种子: {seeds[run_id]}")
    print(f"{'=' * 50}")

    # 生成所有参数组合
    param_combinations = [
        (reg_model, pre_method, key_band_method, split_method)
        for reg_model in regression_model
        for pre_method in preprocessing_methods
        for split_method in split_data_methods
        for key_band_method in WaveSelect_methods
    ]

    # 使用并行计算运行所有实验
    results = Parallel(n_jobs=10, verbose=10)(
        delayed(run_experiment)(
            reg_model, pre_method, key_band_method, split_method,
            data, label, windows, plsr, feature_num, seeds[run_id]
        )
        for reg_model, pre_method, key_band_method, split_method in tqdm(
            param_combinations,
            total=len(param_combinations),
            desc=f"运行 #{run_id + 1} 进度"
        )
    )

    # 创建结果DataFrame并添加列名
    results_df = pd.DataFrame(
        results,
        columns=['RMSE', 'R2', 'MAE', 'Preprocessing', 'FeatureSelection', 'SplitMethod', 'RegressionModel',
                 'RandomSeed']
    )

    # 添加运行ID列
    results_df['RunID'] = run_id + 1

    # 创建输出目录
    os.makedirs('data/output', exist_ok=True)

    # 保存结果到Excel
    output_path = f'data/output/{file_name}_RMSE_R2_MAE_all_methods_{data_name}_run{run_id + 1}.xlsx'
    results_df.to_excel(output_path, index=False)
    print(f"运行 #{run_id + 1} 完成! 结果已保存到: {output_path}")

    return results_df


if __name__ == '__main__':
    file_names = ["all"]
    data_name = "0719"
    num_runs = 10  # 运行次数
    seeds = [123]*10  # 每次运行的随机种子

    # 确保有足够的随机种子
    # if len(seeds) < num_runs:
    #     seeds = seeds + [seeds[-1] * (i + 1) for i in range(num_runs - len(seeds))]

    # 存储所有运行的结果
    all_results = []

    for file_name in file_names:
        # 加载数据
        data_path = DATA_DIR / f"{file_name}_Spec.xlsx"
        data_df = pd.read_excel(data_path, sheet_name='Sheet1', header=None)
        data = data_df.iloc[1:, :].values
        data = np.array(data)

        # 加载SPAD信息
        df = pd.read_excel(DATA_DIR / f"{file_name}_SPAD.xlsx")
        column_data = df.iloc[:, 1].values
        label = np.array(column_data)

        regression_model = ["Pls", "SVR", "ELM", "MLR"]
        preprocessing_methods = ["None", "MA", "SG", "MSC", "D1", "D2", "DT", "WAVE"]
        split_data_methods = ["random", "spxy", "ks"]
        WaveSelect_methods = ["None", "Cars", "Lars", "Uve", "Spa", "GA", "Pca"]
        ##设置窗口window范围为5到19，feature_num设置为【10，99】，plsr的范围设置为【8，当前feature_num】
        feature_num = 30
        windows = 11
        plsr = None

        # 运行多次实验
        for run_id in range(num_runs):
            results_df = run_full_experiment(
                data, label, regression_model, preprocessing_methods,
                split_data_methods, WaveSelect_methods, feature_num,
                windows, plsr, file_name, data_name, run_id, seeds
            )
            all_results.append(results_df)

        # 合并所有运行的结果
        combined_results = pd.concat(all_results, ignore_index=True)

        # 保存合并后的结果
        combined_output_path = f'data/output/{file_name}_RMSE_R2_MAE_all_methods_{data_name}_combined_{num_runs}runs.xlsx'
        combined_results.to_excel(combined_output_path, index=False)
        print(f"\n所有{num_runs}次运行完成! 合并结果已保存到: {combined_output_path}")