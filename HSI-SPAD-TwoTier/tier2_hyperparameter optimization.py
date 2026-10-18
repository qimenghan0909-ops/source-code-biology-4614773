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
                                 feature_num, random_seed, N=20, f=30, cv=5, C=2, gamma=1e-07, kernel='linear'):
    """
    :param data: shape (n_samples, n_features), 光谱数据
    :param label: shape (n_samples, ), 光谱数据对应的标签(理化性质)
    :param ProcessMethods: string, 预处理的方法, 具体可以看预处理模块
    :param FslecetedMethods: string, 光谱波长筛选的方法
    :param SetSplitMethods : string, 划分数据集的方法
    :param model : string, 定量分析模型
    :param random_seed: int, 随机种子
    :param N: CARS参数 - 迭代次数
    :param f: CARS参数 - 选择的特征比例因子
    :param cv: CARS参数 - 交叉验证折数
    :param C: SVR参数 - 惩罚系数
    :param gamma: SVR参数 - 核函数系数
    :param kernel: SVR参数 - 核函数类型
    :return: Rmse: float, Rmse回归误差评估指标
             R2: float, 回归拟合,
             Mae: float, Mae回归误差评估指标
             y_pred: float, 预测值
             y_test: float, 参考值
             y_train_pred: float, 训练数据预测值
             y_train_: float, 训练数据参考值
    """
    ProcesedData = Preprocessing(ProcessMethods, data, windows)  # 预处理
    FeatrueData, labels = SpctrumFeatureSelcet(FslecetedMethods, ProcesedData, label, feature_num, N, f, cv)  # 光谱特征选择
    # 数据集划分
    X_train, X_test, y_train, y_test = SetSplit(SetSplitMethods, FeatrueData, labels, test_size=0.2,
                                                randomseed=random_seed)
    Rmse, R2, Mae, y_pred, y_train_pred = QuantitativeAnalysis(model, X_train, X_test, y_train, y_test,
                                                               C=C, gamma=gamma, kernel=kernel, num=plsr)  # 定量分析
    return Rmse, R2, Mae, y_pred, y_test, y_train_pred, y_train


def run_experiment(reg_model, pre_method, key_band_method, split_method, data, label, windows, plsr, feature_num,
                   random_seed, N, f, cv, C, gamma, kernel):
    """运行单个实验并返回结果"""
    RMSE, R2, MAE, y_pred, y_test, y_train_pred, y_train = SpectralQuantitativeAnalysis(
        data, label, pre_method, key_band_method, split_method, reg_model, windows, plsr, feature_num, random_seed,
        N=N, f=f, cv=cv, C=C, gamma=gamma, kernel=kernel
    )
    return [RMSE, R2, MAE, pre_method, key_band_method, split_method, reg_model, random_seed, N, f, cv, C, gamma,
            kernel]


def run_full_experiment(data, label, regression_model, preprocessing_methods,
                        split_data_methods, WaveSelect_methods,
                        file_name, data_name, run_id, seeds):
    """运行完整的实验流程（一次运行）"""
    print(f"\n{'=' * 50}")
    print(f"开始运行 #{run_id + 1}，随机种子: {seeds[run_id]}")
    print(f"{'=' * 50}")

    # 定义参数搜索空间
    cars_param_space = {
        'N': list(range(2,51)),  # CARS迭代次数
        'f': [30],  # CARS选择的特征比例因子
        'cv': [5, 10]  # CARS交叉验证折数
    }

    svr_param_space = {
        'C': list(range(1,101)),  # SVR惩罚系数
        'gamma': [1e-07],  # SVR核函数系数
        'kernel': ['linear', 'rbf']  # SVR核函数类型
    }

    # 固定其他参数
    feature_num = 30
    windows = 11

    # 存储所有参数组合
    param_combinations = []

    # 生成所有参数组合
    for reg_model in regression_model:
        for pre_method in preprocessing_methods:
            for split_method in split_data_methods:
                for key_band_method in WaveSelect_methods:
                    # 处理CARS参数
                    if key_band_method == "Cars":
                        cars_params = [(n, f, cv) for n in cars_param_space['N']
                                       for f in cars_param_space['f']
                                       for cv in cars_param_space['cv']]
                    else:
                        cars_params = [(20, 30, 5)]  # 非CARS方法使用默认值

                    # 处理SVR参数
                    if reg_model == "SVR":
                        svr_params = [(c, g, k) for c in svr_param_space['C']
                                      for g in svr_param_space['gamma']
                                      for k in svr_param_space['kernel']]
                    else:
                        svr_params = [(2, 1e-07, 'linear')]  # 非SVR方法使用默认值

                    # 处理PLSR成分数
                    if reg_model == "Pls":
                        max_components = min(feature_num, 30)
                        plsr_range = range(1, max_components + 1)
                    else:
                        plsr_range = [None]

                    # 生成最终参数组合
                    for (N, f, cv) in cars_params:
                        for (C, gamma, kernel) in svr_params:
                            for plsr_comp in plsr_range:
                                param_combinations.append((
                                    reg_model, pre_method, key_band_method, split_method,
                                    windows, feature_num, plsr_comp, N, f, cv, C, gamma, kernel
                                ))

    print(f"总实验数量: {len(param_combinations)}")

    # 使用并行计算运行所有实验
    results = Parallel(n_jobs=8, verbose=10)(
        delayed(run_experiment)(
            reg_model, pre_method, key_band_method, split_method,
            data, label, windows, plsr_comp, feature_num, seeds[run_id],
            N, f, cv, C, gamma, kernel
        )
        for reg_model, pre_method, key_band_method, split_method,
        windows, feature_num, plsr_comp, N, f, cv, C, gamma, kernel in tqdm(
            param_combinations,
            total=len(param_combinations),
            desc=f"运行 #{run_id + 1} 进度"
        )
    )

    # 创建结果DataFrame
    results_data = []
    for (reg_model, pre_method, key_band_method, split_method,
         windows, feature_num, plsr_comp, N, f, cv, C, gamma, kernel), res in zip(param_combinations, results):
        RMSE, R2, MAE, *_ = res
        results_data.append({
            'RMSE': RMSE,
            'R2': R2,
            'MAE': MAE,
            'Preprocessing': pre_method,
            'FeatureSelection': key_band_method,
            'SplitMethod': split_method,
            'RegressionModel': reg_model,
            'WindowSize': windows,
            'FeatureNum': feature_num,
            'PLS_Components': plsr_comp if reg_model == "Pls" else "N/A",
            'CARS_N': N if key_band_method == "Cars" else "N/A",
            'CARS_f': f if key_band_method == "Cars" else "N/A",
            'CARS_cv': cv if key_band_method == "Cars" else "N/A",
            'SVR_C': C if reg_model == "SVR" else "N/A",
            'SVR_gamma': gamma if reg_model == "SVR" else "N/A",
            'SVR_kernel': kernel if reg_model == "SVR" else "N/A",
            'RandomSeed': seeds[run_id],
            'RunID': run_id + 1
        })

    results_df = pd.DataFrame(results_data)

    # 创建输出目录
    os.makedirs('data/output', exist_ok=True)

    # 保存结果
    output_path = f'data/output/{file_name}_optimized_params_{data_name}_run{run_id + 1}.xlsx'
    results_df.to_excel(output_path, index=False)
    print(f"运行 #{run_id + 1} 完成! 结果已保存到: {output_path}")

    return results_df


if __name__ == '__main__':
    file_names = ["all"]
    data_name = "0707"
    num_runs = 10
    seeds = [123]*10  # 每次运行的随机种子

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

        # 使用需要优化的方法
        regression_model = ["SVR"]
        preprocessing_methods = ["MSC"]
        split_data_methods = ["spxy"]
        WaveSelect_methods = ["Cars"]

        # 运行多次实验
        for run_id in range(num_runs):
            results_df = run_full_experiment(
                data, label, regression_model, preprocessing_methods,
                split_data_methods, WaveSelect_methods,
                file_name, data_name, run_id, seeds
            )
            all_results.append(results_df)

        # 合并所有运行的结果
        combined_results = pd.concat(all_results, ignore_index=True)

        # 保存合并后的结果
        combined_output_path = f'data/output/{file_name}_optimized_params_{data_name}_combined_{num_runs}runs.xlsx'
        combined_results.to_excel(combined_output_path, index=False)
        print(f"\n所有{num_runs}次运行完成! 合并结果已保存到: {combined_output_path}")