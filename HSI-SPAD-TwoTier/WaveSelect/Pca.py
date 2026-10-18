from sklearn.decomposition import PCA

def Pca(X, nums=30):
    """
       使用主成分分析(PCA)对光谱数据进行降维

        :param X: 原始光谱数据，形状为 (n_samples, n_features)
        :param nums: 保留的主成分数量，默认为30
        :return: X_reduction：降维后的光谱数据
        """
    # 1. 创建PCA对象并设置参数
    # =================================================
    # 初始化PCA模型，设置要保留的主成分数量
    pca = PCA(n_components=nums)  # 保留的特征维度数

    # 2. 训练PCA模型
    # =================================================
    # 使用原始数据拟合PCA模型
    # 这一步会计算数据的主成分方向（特征向量）和方差解释量（特征值）
    pca.fit(X)

    # 3. 执行数据转换
    # =================================================
    # 将原始数据投影到主成分空间
    # 结果是一个降维后的数据矩阵，形状为 (n_samples, nums)
    X_reduction = pca.transform(X)

    # 4. 返回降维后的数据
    return X_reduction
