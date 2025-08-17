import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score, \
    confusion_matrix, matthews_corrcoef
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="UndefinedMetricWarning")

# 设置全局字体
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})


class RASUSampler:
    def __init__(self, features, target, risk_features=None):
        """
        风险感知分层欠采样器（改进分箱版本）

        参数:
        features: 特征DataFrame
        target: 目标Series
        risk_features: 风险特征列表
        """
        self.data = pd.concat([features, target], axis=1)
        self.target_name = target.name
        self.risk_features = risk_features if risk_features else features.columns.tolist()
        self.majority_class = 0
        self.minority_class = 1
        self.feature_importance = None
        self.risk_weights = None
        self.debug_info = {}

    def precompute_risk_weights(self):
        """预计算特征风险权重（改进分箱版本）"""
        print("\nPrecomputing risk weights with robust binning...")
        X = self.data.drop(self.target_name, axis=1)
        y = self.data[self.target_name]

        # 计算特征重要性
        self.feature_importance = mutual_info_classif(X, y, random_state=42)
        self.feature_importance = dict(zip(X.columns, self.feature_importance))

        # 归一化特征重要性
        total = sum(self.feature_importance.values())
        if total > 0:
            for feature in self.feature_importance:
                self.feature_importance[feature] /= total
        else:
            # 如果所有重要性为零，则平均分配
            n_features = len(self.risk_features)
            for feature in self.risk_features:
                self.feature_importance[feature] = 1 / n_features

        print("Feature importance:")
        for feature, imp in self.feature_importance.items():
            print(f"  {feature}: {imp:.4f}")

        # 计算每个特征的风险权重
        self.risk_weights = {}
        for feature in self.risk_features:
            try:
                # 确保特征存在
                if feature not in self.data.columns:
                    print(f"  Warning: Feature '{feature}' not found. Skipping.")
                    self.risk_weights[feature] = {}
                    continue

                # 检查特征是否可分箱
                feature_data = self.data[feature]
                n_unique = feature_data.nunique()

                if n_unique <= 1:
                    print(f"  Warning: Feature '{feature}' has only one unique value. Skipping binning.")
                    self.risk_weights[feature] = {None: self.feature_importance[feature]}
                    continue

                # 动态确定分箱数量
                n_bins = min(5, max(2, n_unique))  # 最少2个，最多5个分箱

                # 尝试分位数分箱
                try:
                    self.data[f'{feature}_bin'], bins = pd.qcut(
                        feature_data,
                        q=n_bins,
                        duplicates='drop',
                        retbins=True
                    )
                    bin_method = "qcut"
                except Exception as e:
                    print(f"  Warning: qcut failed for '{feature}': {str(e)}. Using kmeans binning.")
                    try:
                        # 使用KMeans分箱
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=n_bins, random_state=42)
                        clusters = kmeans.fit_predict(feature_data.values.reshape(-1, 1))
                        self.data[f'{feature}_bin'] = pd.Series(clusters).astype(str)
                        bin_method = "kmeans"
                        bins = kmeans.cluster_centers_.flatten()
                    except Exception as e2:
                        print(
                            f"  Error: kmeans binning also failed for '{feature}': {str(e2)}. Using equal-width binning.")
                        try:
                            self.data[f'{feature}_bin'], bins = pd.cut(
                                feature_data,
                                bins=n_bins,
                                duplicates='drop',
                                retbins=True
                            )
                            bin_method = "cut"
                        except Exception as e3:
                            print(f"  Critical: All binning methods failed for '{feature}'. Using single bin.")
                            self.data[f'{feature}_bin'] = pd.Series(["all"] * len(self.data), index=self.data.index)
                            bin_method = "single"
                            bins = [feature_data.min(), feature_data.max()]

                # 存储分箱信息用于调试
                self.debug_info[feature] = {
                    'method': bin_method,
                    'bins': bins.tolist() if hasattr(bins, 'tolist') else bins,
                    'n_bins': n_bins
                }

                # 计算每个箱的缺陷率
                bin_stats = self.data.groupby(f'{feature}_bin')[self.target_name].agg(['mean', 'count'])

                # 使用Sigmoid函数优化权重计算
                bin_stats['defect_ratio'] = bin_stats['mean']
                bin_stats['weight'] = 1 / (1 + np.exp(-20 * (bin_stats['defect_ratio'] - 0.3))) * \
                                      self.feature_importance[feature]

                self.risk_weights[feature] = bin_stats['weight'].to_dict()

                print(f"  Feature '{feature}' binning ({bin_method}):")
                print(bin_stats[['count', 'defect_ratio', 'weight']])

            except Exception as e:
                print(f"  Error processing feature '{feature}': {str(e)}")
                self.risk_weights[feature] = {}

    def adaptive_sampling(self):
        """执行风险感知分层欠采样（改进分箱版本）"""
        self.precompute_risk_weights()

        # 计算每个样本的综合风险分数
        self.data['risk_score'] = 0.0

        print("\nCalculating risk scores...")
        for feature in self.risk_features:
            bin_col = f'{feature}_bin'

            # 检查分箱列是否存在
            if bin_col not in self.data.columns:
                print(f"  Warning: Bin column '{bin_col}' not found. Skipping feature '{feature}'.")
                continue

            # 检查特征是否有风险权重
            if feature not in self.risk_weights or not self.risk_weights[feature]:
                print(f"  Warning: No risk weights for feature '{feature}'. Skipping.")
                continue

            try:
                # 映射权重
                weights = self.data[bin_col].map(self.risk_weights[feature])

                # 处理缺失值
                if weights.isna().any():
                    avg_weight = np.mean(list(self.risk_weights[feature].values()))
                    weights = weights.fillna(avg_weight)
                    print(
                        f"  Filled {weights.isna().sum()} missing weights for '{feature}' with average {avg_weight:.4f}")

                weights = weights.astype(float)
                self.data['risk_score'] += weights
                print(f"  Added weights for feature '{feature}'")
            except Exception as e:
                print(f"  Error mapping weights for feature '{feature}': {str(e)}")

        # 标准化风险分数
        min_score = self.data['risk_score'].min()
        max_score = self.data['risk_score'].max()
        range_score = max_score - min_score

        if range_score > 0:
            self.data['risk_score'] = (self.data['risk_score'] - min_score) / range_score
        else:
            # 如果所有风险分数相同，则设置为0.5
            self.data['risk_score'] = 0.5
            print("Warning: All risk scores are identical. Setting to 0.5")

        print(f"Risk score range: {min_score:.4f} to {max_score:.4f}")

        # 按风险分数分箱
        try:
            # 使用分位数分箱
            self.data['risk_bin'] = pd.qcut(self.data['risk_score'], q=5, duplicates='drop')
            bin_method = "qcut"
        except Exception as e:
            print(f"Warning: qcut failed for risk score: {str(e)}. Using kmeans binning.")
            try:
                # 使用KMeans分箱
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=5, random_state=42)
                clusters = kmeans.fit_predict(self.data['risk_score'].values.reshape(-1, 1))
                self.data['risk_bin'] = pd.Series(clusters).astype(str)
                bin_method = "kmeans"
            except Exception as e2:
                print(f"Error: kmeans binning also failed: {str(e2)}. Using equal-width binning.")
                try:
                    self.data['risk_bin'] = pd.cut(self.data['risk_score'], bins=5, duplicates='drop')
                    bin_method = "cut"
                except Exception as e3:
                    print(f"Critical: All binning methods failed. Using single bin.")
                    self.data['risk_bin'] = pd.Series(["all"] * len(self.data), index=self.data.index)
                    bin_method = "single"

        # 存储风险分箱信息
        self.debug_info['risk_score'] = {'method': bin_method}

        # 执行分层抽样
        sampled_dfs = []
        majority_data = self.data[self.data[self.target_name] == self.majority_class]
        bin_risk_levels = sorted(majority_data['risk_bin'].unique())

        minority_data = self.data[self.data[self.target_name] == self.minority_class]
        minority_count = len(minority_data)

        print("\nRisk-based sampling strategy:")
        for i, bin_name in enumerate(bin_risk_levels):
            bin_group = majority_data[majority_data['risk_bin'] == bin_name]
            bin_size = len(bin_group)

            if bin_size == 0:
                print(f"  Bin {i + 1}: Empty bin. Skipping.")
                continue

            # 使用S型曲线分配采样率（高风险区保留更多样本）
            preserve_rate = 0.25 + 0.50 / (1 + np.exp(-3 * (i - len(bin_risk_levels) / 2)))
            n_samples = max(int(bin_size * preserve_rate), 15)  # 保证最小样本量

            # 确保抽样数量不超过分箱大小
            if n_samples > bin_size:
                n_samples = bin_size
                print(f"  Bin {i + 1}: Adjusted sampling size to {n_samples} (max available)")
            else:
                print(f"  Bin {i + 1}: {bin_size} samples → preserving {preserve_rate:.1%} ({n_samples} samples)")

            sampled = bin_group.sample(n=n_samples, random_state=42)
            sampled_dfs.append(sampled)

        sampled_majority = pd.concat(sampled_dfs)
        minority_data = self.data[self.data[self.target_name] == self.minority_class]

        # 合并多数类样本和所有少数类样本
        balanced_data = pd.concat([sampled_majority, minority_data])

        # 报告最终采样比例
        final_majority = len(sampled_majority)
        final_minority = len(minority_data)
        final_ratio = final_majority / max(1, final_minority)
        print(f"\nFinal sampling result: Majority {final_majority}, Minority {final_minority}")
        print(f"Final majority:minority ratio: {final_ratio:.2f}:1")

        return balanced_data

    def save_debug_info(self, file_path="rasu_debug_info.json"):
        """保存调试信息到文件"""
        import json
        with open(file_path, 'w') as f:
            json.dump(self.debug_info, f, indent=4)
        print(f"Debug info saved to {file_path}")


def preprocess_data(X, y):
    """数据预处理"""
    # 处理缺失值 - 使用中位数填充
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 处理偏态分布
    skewed_features = X_imputed.apply(lambda x: abs(x.skew()) > 0.75)

    for feat in skewed_features[skewed_features].index:
        n_samples = len(X_imputed)
        n_quantiles = min(1000, max(10, n_samples // 2))
        qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')
        X_imputed[feat] = qt.fit_transform(X_imputed[[feat]])

    # 标准化
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

    return X_scaled, y


def evaluate_model(model, X_test, y_test):
    """评估模型性能并返回指标字典"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba) if y_proba is not None else 0,
        'AUC-PR': average_precision_score(y_test, y_proba) if y_proba is not None else 0,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics


def plot_class_distribution(y, title_suffix="", save_path=None):
    """可视化类别分布"""
    class_counts = y.value_counts()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index,
                     palette='viridis', legend=False)

    total = len(y)
    for i, v in enumerate(class_counts.values):
        ax.text(i, v + 0.02 * total, f'{v}\n({v / total:.1%})',
                ha='center', fontsize=12, fontweight='bold')

    plt.title(f'Class Distribution {title_suffix}')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.ylim(0, total * 1.15)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_confusion_matrix(cm, title_suffix="", save_path=None):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Defect'],
                yticklabels=['Normal', 'Defect'],
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    plt.title(f'Confusion Matrix {title_suffix}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_metrics_comparison(metrics, save_path=None):
    """可视化不同指标的对比"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # 准备数据
    metrics_data = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    keys = list(metrics_data.keys())
    values = list(metrics_data.values())

    # 创建颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(keys)))

    # 绘制条形图
    bars = ax.bar(keys, values, color=colors)

    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                f'{values[i]:.4f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 设置图表属性
    ax.set_title('Model Performance Metrics')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.grid(True, axis='y', alpha=0.3)

    # 添加描述性文本
    plt.figtext(0.5, 0.01,
                f"Overall Performance: MCC = {metrics['MCC']:.4f}, AUC-PR = {metrics['AUC-PR']:.4f}",
                ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.3, "pad": 5})

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def rasu_only_evaluation(file_path, target_column='bug', n_runs=5):
    """
    仅应用风险感知分层欠采样(RASU)的评估

    参数:
    file_path: CSV文件路径
    target_column: 目标列名称
    n_runs: 重复实验次数
    """
    # 1. 从CSV文件加载数据
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return None

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
    plot_class_distribution(y, "(Original)", "rasu_class_distribution_original.png")

    # 2. 数据预处理
    print("\nPreprocessing data...")
    X = X.apply(pd.to_numeric, errors='coerce')

    # 删除所有值都相同的列
    constant_cols = []
    for col in X.columns:
        if X[col].nunique() == 1:
            constant_cols.append(col)

    if constant_cols:
        print(f"Removing constant columns: {constant_cols}")
        X = X.drop(columns=constant_cols)

    X_preprocessed, y = preprocess_data(X, y)

    # 存储结果
    all_metrics = {
        'Precision': [],
        'Recall': [],
        'F1 Score': [],
        'MCC': [],
        'AUC': [],
        'AUC-PR': []
    }

    # 存储采样信息
    sampling_ratios = []

    print(f"\nStarting RASU Sampling evaluation with {n_runs} runs...")
    for run in range(n_runs):
        print(f"\n=== Run {run + 1}/{n_runs} ===")

        # 使用不同的随机种子确保结果可重复
        random_state = 42 + run

        # 分割数据集 (80% 训练, 20% 测试)
        X_train, X_test, y_train, y_test = train_test_split(
            X_preprocessed, y,
            test_size=0.2,
            stratify=y,
            random_state=random_state
        )

        # 检查目标列是否有足够的类别
        if len(np.unique(y_train)) < 2:
            print("Warning: Only one class present in training data. Skipping run.")
            continue

        # 3. 应用风险感知分层欠采样(RASU)
        print("Applying Risk-Aware Stratified Undersampling (RASU)...")
        rasu = RASUSampler(pd.DataFrame(X_train, columns=X_preprocessed.columns),
                           pd.Series(y_train, name=target_column))
        balanced_data = rasu.adaptive_sampling()

        # 移除辅助列
        X_bal = balanced_data.drop([col for col in balanced_data.columns
                                    if col.endswith('_bin') or col in ['risk_score', 'risk_bin', target_column]],
                                   axis=1, errors='ignore')
        y_bal = balanced_data[target_column]

        # 记录采样比例
        majority_count = sum(y_bal == 0)
        minority_count = sum(y_bal == 1)
        sampling_ratio = majority_count / minority_count if minority_count > 0 else 0
        sampling_ratios.append(sampling_ratio)

        # 可视化采样后的类分布 (仅第一次运行时)
        if run == 0:
            plot_class_distribution(y_bal, "(After RASU Sampling)", "rasu_class_distribution_balanced.png")
            #
            # # 可视化特征分布变化 (前3个特征)
            # for i, feature in enumerate(X_bal.columns[:min(3, len(X_bal.columns))]):
            #     rasu.plot_distribution_comparison(
            #         pd.DataFrame(X_train, columns=X_preprocessed.columns),
            #         X_bal,
            #         feature,
            #         f"rasu_distribution_{feature}.png"
            #     )

        # 4. 在采样数据上训练模型
        print("Training model on RASU-sampled data...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_bal, y_bal)

        # 5. 评估模型
        metrics = evaluate_model(model, X_test, y_test)

        # 保存指标
        for key in all_metrics:
            if key in metrics:
                all_metrics[key].append(metrics[key])

        # 6. 可视化结果 (仅第一次运行时)
        if run == 0:
            # 绘制混淆矩阵
            plot_confusion_matrix(metrics['confusion_matrix'], "(RASU Sampling)", "rasu_confusion_matrix.png")

            # 绘制指标对比
            plot_metrics_comparison(metrics, "rasu_metrics.png")

            # 计算特征重要性
            feature_importances = pd.DataFrame({
                'Feature': X_bal.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            # 可视化前15个重要特征
            plt.figure(figsize=(12, 8))
            sns.barplot(
                x='Importance',
                y='Feature',
                data=feature_importances.head(15),
                hue='Feature',  # 添加hue参数避免警告
                palette='viridis',
                legend=False
            )
            plt.title('Feature Importance (RASU Sampling)')
            plt.tight_layout()
            plt.savefig('rasu_feature_importance.png', dpi=300)
            plt.show()

    # 计算平均指标
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    # 创建结果数据框
    results_df = pd.DataFrame({
        'Metric': list(all_metrics.keys()),
        'Mean': [np.mean(all_metrics[key]) for key in all_metrics],
        'StdDev': [np.std(all_metrics[key]) for key in all_metrics]
    })

    # 打印结果
    print("\n" + "=" * 80)
    print("RASU Sampling Evaluation Results")
    print("=" * 80)
    print(f"Dataset: {file_path}")
    print(f"Total runs: {n_runs}")

    # 性能指标摘要
    print("\nPerformance Metrics Summary:")
    print(f"Precision: {avg_metrics['Precision']:.4f}")
    print(f"Recall: {avg_metrics['Recall']:.4f}")
    print(f"F1 Score: {avg_metrics['F1 Score']:.4f}")
    print(f"MCC: {avg_metrics['MCC']:.4f}")
    print(f"AUC: {avg_metrics['AUC']:.4f}")
    print(f"AUC-PR: {avg_metrics['AUC-PR']:.4f}")

    # 采样分析
    print("\nSampling Analysis:")
    print(f"Average sampling ratio (majority:minority): {np.mean(sampling_ratios):.2f}:1")
    print(f"Min sampling ratio: {min(sampling_ratios):.2f}:1")
    print(f"Max sampling ratio: {max(sampling_ratios):.2f}:1")

    # 可视化多轮实验结果
    plt.figure(figsize=(14, 10))
    metrics_to_plot = ['Recall', 'MCC', 'AUC', 'AUC-PR']

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i + 1)
        plt.plot(range(1, n_runs + 1), all_metrics[metric], 'o-', linewidth=2)
        plt.title(f'{metric} over Runs')
        plt.xlabel('Run')
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.grid(True)

        # 标注平均线
        avg = np.mean(all_metrics[metric])
        plt.axhline(avg, color='r', linestyle='--')
        plt.text(0.5, avg + 0.02, f'Avg: {avg:.4f}', color='r')

    plt.tight_layout()
    plt.savefig('rasu_performance_over_runs.png', dpi=300)
    plt.show()

    return {
        'metrics': avg_metrics,
        'sampling_ratios': sampling_ratios
    }


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 从CSV文件加载真实数据
    data_path = r"G:\pycharm\lutrs\code\datacunshu\ant-1.7.csv"  # 请替换为你的实际路径

    # 运行仅RASU采样评估
    results = rasu_only_evaluation(
        file_path=data_path,
        n_runs=5
    )

    if results:
        print("\nRASU Sampling evaluation completed successfully!")
        print(f"Average recall: {results['metrics']['Recall']:.4f}")
    else:
        print("\nRASU Sampling evaluation encountered an error.")