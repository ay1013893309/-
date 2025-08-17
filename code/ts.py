import math
import numpy as np
import pandas as pd
import os
from numpy.linalg import norm
from sklearn.utils import check_array
from scipy.spatial import distance_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.impute import SimpleImputer

__author__ = 'taoll'


class MC_NCLWO(object):
    def __init__(self,
                 max_weight_percentile=95,
                 clear_ratio=None,
                 alpha=0.5,
                 p_norm=2,
                 verbose=False):

        self.max_weight_percentile = max_weight_percentile
        self.clear_ratio = clear_ratio
        self.alpha = alpha
        self.p_norm = p_norm
        self.verbose = verbose

    def fit(self, X, y):
        self.X = check_array(X)
        self.y = np.array(y)

        classes = np.unique(y)

        sizes = np.array([sum(y == c) for c in classes])
        indices = np.argsort(sizes)[::-1]
        self.unique_classes_ = classes[indices]

        self.observation_dict = {c: X[y == c] for c in classes}
        self.maj_class_ = self.unique_classes_[0]
        self.sort_sizes = sizes[indices]

        self.n_max = max(sizes)

        self.n = self.n_max - self.sort_sizes
        if self.verbose:
            print(
                'Majority class is %s and total number of classes is %s'
                % (self.maj_class_, max(sizes)))

    def fit_sample(self, X, y):
        self.fit(X, y)

        for i in range(1, len(self.observation_dict)):
            current_class = self.unique_classes_[i]

            reshape_points, reshape_labels = self.reshape_observations_dict()
            oversampled_points, oversampled_labels = self.generate_samples(reshape_points, reshape_labels,
                                                                           current_class, self.n[i])
            self.observation_dict = {cls: oversampled_points[oversampled_labels == cls] for cls in self.unique_classes_}

        reshape_points, reshape_labels = self.reshape_observations_dict()

        return reshape_points, reshape_labels

    def clear_feature_noise(self, majority_points, minority_points):

        translations = np.zeros(majority_points.shape)

        kept_indices = np.full(len(majority_points), True)

        if self.clear_ratio is None:
            self.clear_ratio = int(((len(minority_points) / len(majority_points))) * 100)
        heat_threshold = np.percentile(self.gi, q=self.clear_ratio)
        for i in range(len(minority_points)):
            num_maj_in_radius = 0
            asc_index = np.argsort(self.min_to_maj_distances[i])

            if self.gi[i] > heat_threshold:
                for j in range(1, len(asc_index)):
                    remain_heat = self.gi[i] * math.exp(-self.alpha * (1 / self.radii[i]) * (
                            self.min_to_maj_distances[i, asc_index[j]] - self.radii[i]))
                    num_maj_in_radius += 1

                    if remain_heat <= heat_threshold:
                        self.radii[i] = self.min_to_maj_distances[i, asc_index[j]]
                        break

            if num_maj_in_radius > 0:
                for j in range(num_maj_in_radius):
                    majority_point = majority_points[asc_index[j]]

                    d = np.sum(np.abs(majority_point - minority_points[i]) ** self.p_norm) ** (1 / self.p_norm)
                    translation = (self.radii[i] - d) / d * (majority_point - minority_points[i])
                    translations[asc_index[j]] = translations[asc_index[j]] + translation

        majority_points = majority_points.astype(np.float64)
        majority_points += translations
        return majority_points

    def generate_samples(self, X, y, minority_class, generate_num):
        minority_points = X[y == minority_class].copy()
        majority_points = X[y != minority_class].copy()
        minority_labels = y[y == minority_class].copy()
        majority_labels = y[y != minority_class].copy()

        self.compute_weight(majority_points, minority_points, generate_num)

        majority_points = self.clear_feature_noise(majority_points, minority_points)

        appended = []
        for i in range(len(minority_points)):
            minority_point = minority_points[i]
            min_to_min_dis = self.min_to_min_distances[i]
            asc_min_index = np.argsort(min_to_min_dis)

            for _ in range(int(self.gi[i])):
                random_vector = np.random.uniform(-1, 1, len(minority_point))
                direction_vector = minority_point + (2 * np.random.rand() - 1) * (
                        random_vector - minority_point)

                direction_unit_vector = direction_vector / norm(direction_vector)
                new_data = minority_point + direction_unit_vector * np.random.rand() * self.radii[i]
                appended.append(new_data)

        if len(appended) > 0:
            points = np.concatenate([majority_points, minority_points, appended])
            labels = np.concatenate([majority_labels, minority_labels, np.tile([minority_class], len(appended))])
        else:
            points = np.concatenate([majority_points, minority_points])
            labels = np.concatenate([majority_labels, minority_labels])
        return points, labels

    def compute_weight(self, majority_points, minority_points, generate_num):
        self.min_to_maj_distances = distance_matrix(minority_points, majority_points, self.p_norm)
        self.min_to_min_distances = distance_matrix(minority_points, minority_points, self.p_norm)

        self.radii = np.zeros(len(minority_points))
        for i in range(len(minority_points)):
            asc_index = np.argsort(self.min_to_maj_distances[i])
            radius = self.min_to_maj_distances[i, asc_index[0]]
            self.radii[i] = radius

        min_weight = np.zeros(len(minority_points))

        regoin_count = np.zeros(len(minority_points))

        min_to_kmaj_dis = np.zeros(len(minority_points))

        for i in range(len(minority_points)):
            for j in range(len(self.min_to_min_distances[i])):
                if self.min_to_min_distances[i][j] < self.radii[i] and i != j:
                    regoin_count[i] = regoin_count[i] + 1

            asc_max_index = np.argsort(self.min_to_maj_distances[i])
            min_to_kmaj_dis[i] = self.min_to_maj_distances[asc_max_index[0]]
            min_weight[i] = 1.0 / (min_to_kmaj_dis[i] * (regoin_count[i] + 1.0) / self.radii[i])

        max_weight = np.percentile(min_weight, q=self.max_weight_percentile)
        for i in range(len(minority_points)):
            if min_weight[i] >= max_weight:
                min_weight[i] = max_weight

        weight_sum = np.sum(min_weight)
        for i in range(len(minority_points)):
            min_weight[i] = min_weight[i] / weight_sum
        self.min_weight = min_weight

        self.gi = np.rint(min_weight * generate_num).astype(np.int32)

    def reshape_observations_dict(self):
        reshape_points = []
        reshape_labels = []

        for cls in self.observation_dict.keys():
            if len(self.observation_dict[cls]) > 0:
                reshape_points.append(self.observation_dict[cls])
                reshape_labels.append(np.tile([cls], len(self.observation_dict[cls])))

        reshape_points = np.concatenate(reshape_points)
        reshape_labels = np.concatenate(reshape_labels)

        return reshape_points, reshape_labels


def preprocess_data(X, y):
    """数据预处理"""
    # 处理缺失值 - 使用中位数填充
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 处理偏态分布
    skewed_features = X_imputed.apply(lambda x: abs(x.skew()) > 0.75)

    for feat in skewed_features[skewed_features].index:
        # 动态设置分位数数量
        n_samples = len(X_imputed)
        n_quantiles = min(1000, max(10, n_samples // 2))

        qt = QuantileTransformer(n_quantiles=n_quantiles,
                                 output_distribution='normal')
        X_imputed[feat] = qt.fit_transform(X_imputed[[feat]])

    # 标准化
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

    return X_scaled, y


def evaluate_with_decision_tree(X, y, test_size=0.2, random_state=42):
    """
    使用决策树分类器评估数据集

    参数:
    X: 特征矩阵
    y: 目标向量
    test_size: 测试集比例
    random_state: 随机种子

    返回:
    评估指标字典
    """
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 创建并训练决策树模型
    model = DecisionTreeClassifier(
        max_depth=5,  # 限制树深度防止过拟合
        min_samples_split=10,  # 增加分裂最小样本数
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }

    return metrics


def evaluate_mc_nclwo_with_decision_tree(X, y, n_runs=5):
    """
    使用决策树评估MC_NCLWO过采样方法

    参数:
    X: 原始特征矩阵
    y: 原始目标向量
    n_runs: 重复实验次数

    返回:
    平均评估指标字典
    """
    # 存储每次运行的指标
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }

    print(f"Evaluating MC_NCLWO with Decision Tree over {n_runs} runs...")

    for run in range(n_runs):
        print(f"\n=== Run {run + 1}/{n_runs} ===")

        # 创建MC_NCLWO过采样器
        sampler = MC_NCLWO(
            max_weight_percentile=95,
            clear_ratio=None,
            alpha=0.5,
            p_norm=2,
            verbose=False
        )

        # 应用过采样
        try:
            X_resampled, y_resampled = sampler.fit_sample(X, y)
            print(f"Resampled dataset size: {len(X_resampled)} (Original: {len(X)})")
        except Exception as e:
            print(f"Error during resampling: {str(e)}")
            continue

        # 使用决策树评估
        metrics = evaluate_with_decision_tree(X_resampled, y_resampled, random_state=42 + run)

        # 保存指标
        for key in all_metrics:
            if key in metrics and metrics[key] is not None:
                all_metrics[key].append(metrics[key])

        # 打印本次运行结果
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # 计算平均指标
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:  # 确保列表不为空
            avg_metrics[key] = np.mean(values)

    # 打印平均结果
    print("\nAverage metrics:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")

    return avg_metrics


def compare_without_resampling(X, y, n_runs=5):
    """
    比较不使用过采样的原始数据性能

    参数:
    X: 原始特征矩阵
    y: 原始目标向量
    n_runs: 重复实验次数

    返回:
    平均评估指标字典
    """
    # 存储每次运行的指标
    all_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }

    print(f"Evaluating without resampling over {n_runs} runs...")

    for run in range(n_runs):
        print(f"\n=== Run {run + 1}/{n_runs} ===")

        # 使用决策树评估原始数据
        metrics = evaluate_with_decision_tree(X, y, random_state=42 + run)

        # 保存指标
        for key in all_metrics:
            if key in metrics and metrics[key] is not None:
                all_metrics[key].append(metrics[key])

        # 打印本次运行结果
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    # 计算平均指标
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:  # 确保列表不为空
            avg_metrics[key] = np.mean(values)

    # 打印平均结果
    print("\nAverage metrics:")
    for metric, value in avg_metrics.items():
        print(f"  {metric}: {value:.4f}")

    return avg_metrics


def load_data(file_path, target_column='bug'):
    """
    加载数据集（使用您提供的格式）

    参数:
    file_path: CSV文件路径
    target_column: 目标列名称

    返回:
    X: 特征矩阵
    y: 目标向量
    """
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return None, None

    data = pd.read_csv(file_path)
    print(f"Dataset shape: {data.shape}")

    # 删除前三列标识信息列
    if data.shape[1] > 3:
        feature_columns = data.columns[3:-1]
        target_column = data.columns[-1]
        X = data[feature_columns]
        y = data[target_column]
    else:
        raise ValueError("CSV文件列数不足，请检查数据格式")

    # 将目标变量转换为二分类：大于0表示有缺陷（1），等于0表示无缺陷（0）
    y = y.apply(lambda x: 1 if x > 0 else 0)

    # 输出类分布
    class_counts = y.value_counts()
    print(f"Original class distribution:\n{class_counts}")

    return X, y


def main():
    # 使用您提供的加载数据集方法
    data_path = r"G:\pycharm\lutrs\de分层欠采样\datacunshu\arc.csv"  # 请替换为您的实际路径

    # 加载数据
    X, y = load_data(data_path)
    if X is None or y is None:
        return

    # 数据预处理
    print("\nPreprocessing data...")
    X = X.apply(pd.to_numeric, errors='coerce')
    X_preprocessed, y = preprocess_data(X, y)

    # 评估不使用过采样的原始数据
    print("\n=== Evaluating without resampling ===")
    without_resampling_metrics = compare_without_resampling(X_preprocessed, y, n_runs=3)

    # 评估使用MC_NCLWO过采样的数据
    print("\n=== Evaluating with MC_NCLWO resampling ===")
    with_resampling_metrics = evaluate_mc_nclwo_with_decision_tree(X_preprocessed, y, n_runs=3)

    # 比较结果
    print("\n=== Comparison Results ===")
    print("Without resampling:")
    for metric, value in without_resampling_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\nWith MC_NCLWO resampling:")
    for metric, value in with_resampling_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # 计算改进百分比
    print("\nImprovement (%):")
    for metric in without_resampling_metrics:
        if metric in with_resampling_metrics:
            improvement = ((with_resampling_metrics[metric] - without_resampling_metrics[metric]) /
                           without_resampling_metrics[metric]) * 100
            print(f"  {metric}: {improvement:.2f}%")


if __name__ == "__main__":
    main()