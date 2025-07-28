import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import time
import csv
import warnings
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from scipy.optimize import differential_evolution
from sklearn.neighbors import NearestNeighbors


class LTROS:
    """Learning-To-Rank OverSampling (LTROS)实现"""

    def __init__(self, k=5, termination='ratio', random_state=None):
        """
        参数:
        k: 生成样本时的近邻数
        termination: 终止条件类型 ('ratio' 或 'thres')
        random_state: 随机种子
        """
        self.k = k
        self.termination = termination
        self.random_state = random_state
        self.weights_ = None
        self.theta_ = None
        self.ratio_ = None
        np.random.seed(random_state)

    def _rank_score(self, X_min, weights):
        """计算少数类样本的排序分数"""
        return np.dot(X_min, weights)

    def _sigmoid(self, x):
        """Sigmoid函数转换"""
        return 1 / (1 + np.exp(-x))

    def _generate_samples(self, X_min, weights, selected_indices, n_gen):
        """基于高分样本生成新样本"""
        if len(selected_indices) == 0:
            return np.empty((0, X_min.shape[1]))

        # 使用k近邻算法
        nn = NearestNeighbors(n_neighbors=self.k + 1).fit(X_min)
        indices = selected_indices.copy()
        np.random.shuffle(indices)

        # 确定每个高分样本应生成的样本数
        samples_per_point = np.full(len(indices), n_gen // len(indices))
        samples_per_point[:n_gen % len(indices)] += 1

        synthetic = []
        for idx, count in zip(indices, samples_per_point):
            # 基于权重分数计算生成强度
            base_multiplier = self._sigmoid(weights[np.argmax(weights)])
            for _ in range(count):
                # 获取k近邻（排除自身）
                neighbors = nn.kneighbors([X_min[idx]], return_distance=False)[0][1:]
                j = np.random.choice(neighbors)
                # 动态λ: 基于权重的高斯扰动
                λ = np.clip(np.random.normal(0.5, 0.2 * base_multiplier), 0.1, 0.9)
                new_sample = X_min[idx] + λ * (X_min[j] - X_min[idx])
                synthetic.append(new_sample)

        return np.vstack(synthetic) if synthetic else np.empty((0, X_min.shape[1]))

    def fit_resample(self, X_min, weights, y_min=None):
        """
        执行过采样
        X_min: 少数类样本的特征
        weights: 优化后的权重向量
        y_min: 少数类样本的标签（可选）
        """
        if y_min is None:
            y_min = np.ones(len(X_min))

        # 计算样本排序分数
        scores = self._rank_score(X_min, weights)

        # 确定需要生成的样本数量
        if self.termination == 'thres':
            selected = scores > self.theta_
            n_gen = np.sum(selected)  # 每个高分样本生成1个样本
        else:
            n_gen = int(len(X_min) * self.ratio_)
            sorted_idx = np.argsort(-scores)
            top_k = min(len(scores), max(1, int(len(scores) * self.ratio_)))
            selected = np.zeros_like(scores, dtype=bool)
            selected[sorted_idx[:top_k]] = True

        # 生成新样本
        synthetic = self._generate_samples(X_min, weights, np.where(selected)[0], n_gen)

        # 合并结果
        synthetic_y = np.ones(len(synthetic)) * (1 if len(y_min) == 0 else y_min[0])
        return synthetic, synthetic_y


# 修改后的评估函数，使用LTROS过采样
def fit(bound, *param):
    """评估函数：使用LTROS过采样后评估模型性能"""
    de_dataset = []
    for de_i in param:
        de_dataset.append(de_i)

    de_dataset = np.array(de_dataset)
    de_x = de_dataset
    de_y = de_dataset[:, -1]
    de_total_mcc = 0

    # 创建LTROS对象
    ltros = LTROS(k=5, termination='ratio', random_state=42)

    # 设置优化后的参数（在后续代码中会被设置）
    ltros.ratio_ = 0.5  # 临时值，实际在优化过程中设置
    ltros.weights_ = bound  # 使用边界参数作为权重向量

    for train, test in skf.split(de_x, de_y):
        de_train_x = de_x[train]
        de_train_y = de_y[train]

        for sub_train, sub_test in skf.split(de_train_x, de_train_y):
            de_sub_test_x = de_train_x[sub_test]
            de_sub_test_y = de_train_y[sub_test]
            de_sub_test_x = de_sub_test_x[:, 0:-1]

            de_sub_train_x = de_train_x[sub_train]

            # 分离少数类和多数类
            defect_indices = np.where(de_sub_train_x[:, -1] > 0)[0]
            defect_x = de_sub_train_x[defect_indices]
            clean_x = de_sub_train_x[de_sub_train_x[:, -1] == 0]

            # 对少数类样本应用LTROS过采样
            synthetic_x, synthetic_y = ltros.fit_resample(
                defect_x[:, 0:-1],
                bound,  # 使用当前权重向量
                y_min=defect_x[:, -1]
            )

            # 构建平衡的训练集
            synthetic_set = np.c_[synthetic_x, synthetic_y]
            de_sub_train_x = np.vstack([clean_x, defect_x, synthetic_set])

            # 准备训练数据和标签
            de_sub_train_y = de_sub_train_x[:, -1]
            de_sub_train_x = de_sub_train_x[:, 0:-1]

            # 训练分类器
            de_clf = classifier_for_selection[classifier]
            de_clf.fit(de_sub_train_x, de_sub_train_y)

            # 评估性能
            de_predict_result = de_clf.predict(de_sub_test_x)
            de_mcc = matthews_corrcoef(de_sub_test_y, de_predict_result)
            de_total_mcc = de_total_mcc + de_mcc

    de_average_mcc = de_total_mcc / 25
    return -de_average_mcc  # 最小化负MCC


# 主程序执行
if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    warnings.filterwarnings('ignore')
    classifier = "knn"  # 可以改为其他分类器
    skf = StratifiedKFold(n_splits=5)

    # 创建输出目录
    output_dir = './results/ltros/'
    os.makedirs(output_dir, exist_ok=True)

    # 初始化分类器字典
    classifier_for_selection = {
        "knn": neighbors.KNeighborsClassifier(),
        "svm": svm.SVC(),
        "rf": RandomForestClassifier(random_state=0),
        "dt": tree.DecisionTreeClassifier(random_state=0),
        "lr": LogisticRegression(random_state=0),
        "nb": GaussianNB()
    }

    # 创建结果文件
    single = open(os.path.join(output_dir, f'{classifier}_ltros_results.csv'), 'w', newline='')
    single_writer = csv.writer(single)
    single_writer.writerow(["inputfile", "mcc", "auc", "balance", "fmeasure", "precision", "pd", "pf"])

    # 数据集处理和分析循环
    for iteration in range(1):
        for inputfile in os.listdir("./data"):
            if not inputfile.endswith('.csv'):
                continue
            if inputfile in ["PC1.csv", "PC2.csv", "PC3.csv", "synapse-1.0.csv",
                             "jedit-4.3.csv", "synapse-1.1.csv", "LICENSE"]:
                continue

            print(f"Processing {inputfile}...")
            start_time = time.time()

            # 加载数据集
            dataset = pd.read_csv(os.path.join("./data", inputfile))

            # 清理数据
            dataset = dataset.drop(columns=["name", "version", "name.1"], errors="ignore")
            total_number = len(dataset)

            # 处理目标变量
            defect_ratio = len(dataset[dataset["bug"] > 0]) / total_number
            if defect_ratio > 0.45:
                print(f"{inputfile} defect ratio larger than 0.45")
                continue

            # 将bug转换为二进制标签
            dataset["bug"] = dataset["bug"].apply(lambda x: 1 if x > 0 else 0)

            # 特征标准化
            for col in dataset.columns[:-1]:  # 排除最后一列bug列
                col_min = dataset[col].min()
                col_max = dataset[col].max()
                # 避免除以0
                if col_max != col_min:
                    dataset[col] = (dataset[col] - col_min) / (col_max - col_min)

            # 转换为NumPy数组
            data_arr = dataset.to_numpy()
            y = data_arr[:, -1]
            X = data_arr

            # 设置优化边界
            n_features = X.shape[1] - 1
            bounds = [(-1.0, 1.0)] * n_features

            # 优化权重向量
            print("Optimizing weights...")
            optimal_result = differential_evolution(
                fit,
                bounds,
                args=data_arr,
                popsize=10,
                maxiter=10,
                mutation=0.3,
                recombination=0.9,
                disp=True,
                seed=42
            )
            optimal_weight = optimal_result.x

            # 初始化评估指标
            total_metrics = {
                "auc": 0, "balance": 0, "fmeasure": 0,
                "precision": 0, "recall": 0, "pf": 0, "mcc": 0
            }

            # 使用优化的权重评估模型性能
            ltros = LTROS(k=5, termination='ratio', random_state=42)
            ltros.weights_ = optimal_weight
            ltros.ratio_ = 1.0  # 可根据需要调整比例

            print("Evaluating model performance...")
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_test = X_test[:, -1]
                X_test = X_test[:, :-1]

                # 分离训练集中的少数类和多数类
                defect_indices = np.where(X_train[:, -1] > 0)[0]
                defect_x = X_train[defect_indices]
                clean_x = X_train[X_train[:, -1] == 0]

                # 应用LTROS过采样
                synthetic_x, synthetic_y = ltros.fit_resample(
                    defect_x[:, :-1],  # 特征
                    optimal_weight,  # 优化后的权重
                    y_min=defect_x[:, -1]  # 标签
                )

                # 构建平衡的训练集
                synthetic_set = np.c_[synthetic_x, synthetic_y]
                balanced_train = np.vstack([clean_x, defect_x, synthetic_set])
                X_train_final = balanced_train[:, :-1]
                y_train_final = balanced_train[:, -1]

                # 训练分类器
                clf = classifier_for_selection[classifier]
                clf.fit(X_train_final, y_train_final)

                # 预测和评估
                y_pred = clf.predict(X_test)

                # 计算评估指标
                true_neg, false_pos, false_neg, true_pos = confusion_matrix(y_test, y_pred).ravel()

                # 收集各项指标
                total_metrics["auc"] += roc_auc_score(y_test, y_pred)
                total_metrics["mcc"] += matthews_corrcoef(y_test, y_pred)
                total_metrics["fmeasure"] += f1_score(y_test, y_pred)
                total_metrics["recall"] += recall_score(y_test, y_pred)
                total_metrics["pf"] += false_pos / (true_neg + false_pos)
                total_metrics["precision"] += precision_score(y_test, y_pred)
                total_metrics["balance"] += 1 - (
                            ((0 - total_metrics["pf"] / 5) ** 2 + (1 - total_metrics["recall"] / 5) ** 2) / 2) ** 0.5

            # 计算平均指标
            avg_metrics = {k: v / 5 for k, v in total_metrics.items()}

            # 写入结果
            single_writer.writerow([
                inputfile,
                avg_metrics["mcc"],
                avg_metrics["auc"],
                avg_metrics["balance"],
                avg_metrics["fmeasure"],
                avg_metrics["precision"],
                avg_metrics["recall"],  # pd
                avg_metrics["pf"]
            ])

            print(f"Completed {inputfile} in {time.time() - start_time:.2f} seconds")
            print("--------------------------------------")

    # 关闭文件
    single.close()
    print("All done!")
