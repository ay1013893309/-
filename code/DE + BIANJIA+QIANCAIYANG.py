import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import differential_evolution
from sklearn.metrics import (matthews_corrcoef, f1_score, precision_score,
                             recall_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

# 忽略警告
warnings.filterwarnings('ignore')


class EnhancedFWRSUS:
    """
    增强版特征加权风险分级欠采样
    针对性能问题进行了多项优化
    """

    def __init__(self, k=5, lambda_comp=0.3, max_ratio=2.0, gmean_threshold=0.7,
                 de_popsize=10, de_maxiter=5, cost_matrix=None, min_maj_ratio=0.5,
                 min_min_samples=5, risk_threshold=0.3):
        """
        参数初始化
        :param min_min_samples: 最少保留的少数类样本数(默认5)
        :param risk_threshold: 高风险样本判定阈值(默认0.3)
        """
        self.k = k
        self.lambda_comp = lambda_comp
        self.max_ratio = max_ratio
        self.gmean_threshold = gmean_threshold
        self.de_popsize = de_popsize
        self.de_maxiter = de_maxiter
        self.cost_matrix = cost_matrix if cost_matrix else [5, 1]
        self.min_maj_ratio = min_maj_ratio
        self.min_min_samples = min_min_samples
        self.risk_threshold = risk_threshold
        self.alpha = None

    def _gmean_score(self, y_true, y_pred):
        """计算G-Mean评估指标"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall_min = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall_maj = tn / (tn + fp) if (tn + fp) > 0 else 0
        return np.sqrt(recall_min * recall_maj)

    def _de_objective(self, alpha, X_train, y_train):
        """
        差分进化优化目标函数 - 改进版
        加入成本敏感指标和F1分数
        """
        weighted_X = X_train * alpha
        knn = NearestNeighbors(n_neighbors=5).fit(weighted_X)
        _, indices = knn.kneighbors(weighted_X)
        y_pred = np.round(np.mean(y_train[indices], axis=1))

        # 计算MCC和F1
        mcc = matthews_corrcoef(y_train, y_pred)
        f1 = f1_score(y_train, y_pred, average='binary')

        # 组合目标函数
        return -(0.7 * mcc + 0.3 * f1)  # 加权优化

    def _de_optimization(self, X_train, y_train):
        d = X_train.shape[1]
        bounds = [(-1, 1)] * d
        result = differential_evolution(
            func=lambda alpha: self._de_objective(alpha, X_train, y_train),
            bounds=bounds,
            popsize=self.de_popsize * d,
            maxiter=self.de_maxiter,
            tol=0.01
        )
        return result.x

    def _weighted_distance(self, X1, X2):
        if self.alpha is None:
            return np.sqrt(np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2))
        weighted_X1 = X1 * self.alpha
        weighted_X2 = X2 * self.alpha
        return np.sqrt(np.sum((weighted_X1[:, np.newaxis] - weighted_X2) ** 2, axis=2))

    def _risk_stratification(self, X_maj, X_min):
        """增强的风险分级逻辑"""
        # 保护机制1: 确保有足够的少数类样本
        if len(X_min) < self.min_min_samples:
            return {
                "safe": list(range(len(X_maj))),
                "borderline": [],
                "noise": [],
                "high_risk": []
            }

        # 计算所有样本间加权距离
        all_samples = np.vstack([X_maj, X_min])
        dist_matrix = self._weighted_distance(X_maj, all_samples)

        # 计算τ(少数类样本间距中位数)
        min_dists = self._weighted_distance(X_min, X_min)
        np.fill_diagonal(min_dists, np.inf)
        min_distances = np.min(min_dists, axis=1)
        tau = np.median(min_distances) if len(min_distances) > 0 else 0

        risk_categories = {
            "safe": [], "borderline": [], "noise": [], "high_risk": []
        }

        # 计算少数类聚类中心
        if len(X_min) > 10:
            kmeans = KMeans(n_clusters=min(5, len(X_min)), random_state=42)
            kmeans.fit(X_min)
            min_centers = kmeans.cluster_centers_
        else:
            min_centers = X_min

        # 对每个多数类样本进行风险分级
        for i in range(len(X_maj)):
            distances = dist_matrix[i, :]
            nearest_indices = np.argsort(distances)[1:self.k + 1]
            min_count = np.sum(nearest_indices >= len(X_maj))

            # 计算到最近少数类中心的距离
            min_center_dist = np.min(self._weighted_distance(X_maj[i].reshape(1, -1), min_centers))

            # 分级逻辑 - 更保守的策略
            if min_count == 0:
                risk_categories["safe"].append(i)
            elif min_count == self.k and min_center_dist < tau * 0.5:
                risk_categories["noise"].append(i)
            else:
                min_dist = np.min(distances[len(X_maj):])
                risk_score = min_dist / tau

                if risk_score < self.risk_threshold and len(risk_categories["high_risk"]) < len(X_maj) * 0.5:
                    risk_categories["high_risk"].append(i)
                else:
                    risk_categories["borderline"].append(i)

        return risk_categories

    def _generate_compensation_samples(self, X_min, high_risk_samples):
        """改进的合成样本生成"""
        synthetic_samples = []

        # 使用KMeans找到少数类聚类中心
        if len(X_min) > 10:
            kmeans = KMeans(n_clusters=min(5, len(X_min)), random_state=42)
            kmeans.fit(X_min)
            min_centers = kmeans.cluster_centers_
        else:
            min_centers = X_min

        for sample in high_risk_samples:
            # 选择最近的少数类中心
            dists = pairwise_distances(sample.reshape(1, -1), min_centers)
            center_idx = np.argmin(dists)
            min_center = min_centers[center_idx]

            # 生成合成样本 - 更靠近少数类区域
            synthetic = min_center + self.lambda_comp * (sample - min_center)
            synthetic_samples.append(synthetic)

        return np.array(synthetic_samples)

    def fit_resample(self, X, y):
        X_min = X[y == 1]
        X_maj = X[y == 0]
        original_maj_count = len(X_maj)
        original_min_count = len(X_min)

        # 保护机制: 确保有足够的少数类样本
        if original_min_count < self.min_min_samples:
            print(f"警告: 少数类样本不足({original_min_count}<{self.min_min_samples}), 跳过采样")
            return X, y

        # 优化特征权重
        self.alpha = self._de_optimization(X, y)

        X_resampled = [X_min.copy()]
        y_resampled = [np.ones(len(X_min))]

        max_iterations = 10  # 最大迭代次数
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            risk_cats = self._risk_stratification(X_maj, X_min)

            # 限制移除样本数量
            to_remove = risk_cats["noise"] + risk_cats["high_risk"]
            min_maj_keep = max(int(len(X_min) * self.min_maj_ratio), 1)  # 至少保留1个
            max_remove = min(
                int(original_maj_count * (1 - self.min_maj_ratio)),
                len(X_maj) - min_maj_keep  # 确保多数类保留数 >= 最小比例
            )
            if len(to_remove) > max_remove:
                to_remove = to_remove[:max_remove]

            # 部分移除边界类(30%)
            borderline = risk_cats["borderline"]
            num_border_remove = min(
                int(len(borderline) * 0.3),
                max_remove - len(to_remove)
            )
            if num_border_remove > 0:
                to_remove += list(np.random.choice(borderline, num_border_remove, replace=False))

            keep_indices = list(set(range(len(X_maj))) - set(to_remove))
            X_maj_keep = X_maj[keep_indices]

            # 生成合成样本（限制合成数量，避免过度生成）
            high_risk_samples = X_maj[risk_cats["high_risk"]]
            if len(high_risk_samples) > 0 and len(X_min) >= self.min_min_samples:
                # 控制合成样本数量，避免过多
                max_synthetic = min(len(high_risk_samples), int(len(X_maj_keep) * 0.3))
                synthetic = self._generate_compensation_samples(X_min, high_risk_samples[:max_synthetic])
                X_resampled.append(synthetic)
                y_resampled.append(np.ones(len(synthetic)))

            X_resampled.append(X_maj_keep)
            y_resampled.append(np.zeros(len(X_maj_keep)))

            # 合并当前结果
            current_X = np.vstack(X_resampled)
            current_y = np.hstack(y_resampled)

            # 终止条件检查（使用当前少数类数量而非初始值）
            current_min_count = len(current_X[current_y == 1])
            current_maj_count = len(current_X[current_y == 0])

            if current_min_count > 0:
                ratio = current_maj_count / current_min_count
                if ratio <= self.max_ratio:
                    break

            # 保底机制
            if (current_maj_count <= original_maj_count * self.min_maj_ratio or
                    len(np.unique(current_y)) < 2):
                break

            X_maj = X_maj_keep
            X_resampled = [X_min.copy()]  # 重置，只保留原始少数类
            y_resampled = [np.ones(len(X_min))]

        # 最终合并
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        # 确保至少有两个类别
        if len(np.unique(y_resampled)) < 2:
            print("警告: 采样后只有一个类别，添加安全样本")
            safe_samples = X_maj[:min(5, len(X_maj))]
            X_resampled = np.vstack([X_resampled, safe_samples])
            y_resampled = np.hstack([y_resampled, np.zeros(len(safe_samples))])

        return X_resampled, y_resampled


def evaluate_model(model, X_train, y_train, X_test, y_test, cost_matrix=None):
    """评估模型性能，增加成本敏感指标"""
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"模型训练错误: {str(e)}")
        # 返回默认指标
        return {
            'f1': 0,
            'precision': 0,
            'recall': 0,
            'mcc': 0,
            'auc': 0,
            'total_cost': 0,
            'cost_f1': 0
        }
    y_proba = model.predict_proba(X_test)[:, 1]  # 取正类概率

    # 基础指标
    metrics = {
        'f1': f1_score(y_test, y_pred, average='binary'),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }

    # 成本敏感指标
    if cost_matrix:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        fn_cost, fp_cost = cost_matrix
        total_cost = fn * fn_cost + fp * fp_cost

        cost_precision = tp / (tp + fp + 1e-7)
        cost_recall = tp / (tp + fn + 1e-7)
        cost_f1 = 2 * (cost_precision * cost_recall) / (cost_precision + cost_recall + 1e-7)

        metrics.update({
            'total_cost': total_cost,
            'cost_f1': cost_f1
        })

    return metrics


def plot_metrics(metrics, title):
    """绘制性能指标对比图"""
    fig, ax = plt.subplots(figsize=(14, 7))

    methods = list(metrics.keys())
    metric_names = ['f1', 'precision', 'recall', 'mcc', 'auc', 'total_cost', 'cost_f1']

    bar_width = 0.15
    index = np.arange(len(metric_names))

    for i, method in enumerate(methods):
        scores = [metrics[method].get(metric, 0) for metric in metric_names]
        ax.bar(index + i * bar_width, scores, bar_width, label=method)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(index + bar_width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metric_names)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()


def plot_feature_importance(alpha, feature_names):
    """绘制特征重要性图"""
    importance = pd.DataFrame({
        'feature': feature_names,
        'weight': alpha
    }).sort_values('weight', key=abs, ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='weight', y='feature', data=importance)
    plt.title('Feature Importance (DE-Optimized Weights)')
    plt.xlabel('Weight')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('Feature_Importance.png')
    plt.show()


def main():
    # 加载数据集
    data_path = "/code/data/ant-1.3.csv"
    df = pd.read_csv(data_path)

    # 数据预处理：将bug > 0视为有缺陷
    df['label'] = (df['bug'] > 0).astype(int)

    # 删除前三列（name, version, name）
    df = df.drop(df.columns[[0, 1, 2]], axis=1)

    # 检查数据特性
    print(f"数据集: {data_path}")
    print(f"实例数量: {len(df)}")
    print(f"缺陷比例: {df['label'].mean():.2%}")

    # 特征/标签分离
    feature_names = df.drop(columns=['bug', 'label']).columns.tolist()
    X = df.drop(columns=['bug', 'label']).values
    y = df['label'].values

    # 应用min-max归一化
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # 定义分类器
    classifiers = {
        "KNN": KNeighborsClassifier(),
        "RF": RandomForestClassifier(random_state=42),
        "LR": LogisticRegression(max_iter=1000, random_state=42),
        "NB": GaussianNB()
    }

    # 成本敏感矩阵 [FN_cost, FP_cost]
    cost_matrix = [5, 1]  # 误判缺陷的代价是误判非缺陷的5倍

    # 定义采样方法
    sampling_methods = {
        "原始数据": None,
        "FWRSUS": EnhancedFWRSUS(k=5, lambda_comp=0.3, cost_matrix=cost_matrix, min_maj_ratio=0.5),
        "随机欠采样": "RUS"
    }

    # 存储结果
    all_results = {clf_name: {method: {} for method in sampling_methods}
                   for clf_name in classifiers}

    # 单次划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    print(f"训练集缺陷比例: {y_train.mean():.2%}, 测试集缺陷比例: {y_test.mean():.2%}")

    for clf_name, clf in classifiers.items():
        for method_name, sampler in sampling_methods.items():
            # 处理不同采样方法
            if method_name == "原始数据":
                X_res, y_res = X_train, y_train
            elif method_name == "随机欠采样":
                from imblearn.under_sampling import RandomUnderSampler
                rus = RandomUnderSampler(random_state=42)
                X_res, y_res = rus.fit_resample(X_train, y_train)
            else:
                X_res, y_res = sampler.fit_resample(X_train, y_train)

            # 确保有多个类别
            if len(np.unique(y_res)) < 2:
                print(f"警告: {clf_name} + {method_name} 采样后只有一个类别")
                # 添加一个虚拟样本以创建多个类别
                X_res = np.vstack([X_res, X_res[0]])
                y_res = np.append(y_res, 1 - y_res[0])

            # 评估模型性能
            metrics = evaluate_model(clf, X_res, y_res, X_test, y_test, cost_matrix)
            all_results[clf_name][method_name] = metrics

            print(f"  {clf_name} + {method_name}: F1={metrics['f1']:.4f}, Cost={metrics.get('total_cost', 0)}")

    # 打印结果
    print("\n最终性能评估结果:")
    for clf_name, methods in all_results.items():
        print(f"\n分类器: {clf_name}")
        print("方法\t\tF1\t\tPrecision\tRecall\t\tMCC\t\tAUC\t\tCost")
        for method_name, metrics in methods.items():
            print(f"{method_name[:10]}\t{metrics['f1']:.4f}\t{metrics['precision']:.4f}\t\t"
                  f"{metrics['recall']:.4f}\t\t{metrics['mcc']:.4f}\t\t{metrics['auc']:.4f}\t\t{metrics.get('total_cost', 0)}")

    # 可视化结果
    for clf_name in classifiers:
        plot_metrics(all_results[clf_name], f"性能指标对比 - {clf_name}")

    # 绘制特征重要性图
    if 'FWRSUS' in sampling_methods and hasattr(sampling_methods['FWRSUS'], 'alpha'):
        plot_feature_importance(sampling_methods['FWRSUS'].alpha, feature_names)


if __name__ == "__main__":
    main()