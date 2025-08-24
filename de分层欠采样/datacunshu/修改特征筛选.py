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
from sklearn.svm import SVC
import shap
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

        # 计算特征重要性（互信息
        # self.feature_importance = mutual_info_classif(X, y, random_state=42)
        # self.feature_importance = dict(zip(X.columns, self.feature_importance))

        # X = self.data.drop(self.target_name, axis=1)
        # y = self.data[self.target_name]
        # # 使用KNN计算特征重要性

        # model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        # model.fit(X, y)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # 计算平均|SHAP|值作为重要性
        if isinstance(shap_values, list):
            # 对于分类问题，取正类的SHAP值
            shap_abs = shap_values[1]
        else:
            shap_abs = shap_values[0]

        importance = np.abs(shap_abs).mean(axis=1)
        # # 使用随机森林计算特征重要性（修改点）
        # model = SVC(kernel='rbf', probability=True, random_state=42)
        # model.fit(X, y)
        # 获取特征重要性
        # importances = model.feature_importances_
        self.feature_importance = dict(zip(X.columns, importance))
        # 基于模型性能估计特征重要性
        # self.feature_importance = {}
        # cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # baseline_score = np.mean(cross_val_score(model, X, y, cv=cv, scoring='recall'))
        # baseline_score = np.mean(cross_val_score(model, X, y, cv=cv, scoring='matthews_corrcoef'))
        #
        # for feature in self.risk_features:
        #     X_reduced = X.drop(feature, axis=1)
        #     reduced_score = np.mean(cross_val_score(model, X_reduced, y, cv=cv, scoring='matthews_corrcoef'))
        #     importance = max(0, baseline_score - reduced_score)  # 特征删除导致性能下降的程度
        #     self.feature_importance[feature] = importance
        # 归一化特征重要性
        self.risk_features = [
            feature for feature, imp in self.feature_importance.items()
            if imp >= 0
        ]
        #
        # 确保至少保留10个特征（防止过度筛选）
        if len(self.risk_features) < 10:
            # 按重要性排序并取前10个
            sorted_features = sorted(self.feature_importance.items(),
                                     key=lambda x: x[1], reverse=True)
            self.risk_features = [feature for feature, _ in sorted_features[:10]]

        print(f"Selected {len(self.risk_features)} features (from {len(X.columns)} total)")

        # 更新数据，只保留筛选后的特征
        self.data = self.data[self.risk_features + [self.target_name]]
        # total = sum(self.feature_importance.values())
        total = sum(self.feature_importance[feature] for feature in self.risk_features)

        if total > 0:
            for feature in self.risk_features:
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

                self.data[f'{feature}_bin'], bins = pd.qcut(
                        feature_data,
                        q=n_bins,
                        duplicates='drop',
                        retbins=True
                    )
                bin_method = "qcut"


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

        # # 按风险分数分箱
        # try:
        #     # 使用分位数分箱
        self.data['risk_bin'] = pd.qcut(self.data['risk_score'], q=5, duplicates='drop')
        bin_method = "qcut"
        # except Exception as e:
        #     print(f"Warning: qcut failed for risk score: {str(e)}. Using kmeans binning.")
        #     try:
        #         # 使用KMeans分箱
        #         from sklearn.cluster import KMeans
        #         kmeans = KMeans(n_clusters=5, random_state=42)
        #         clusters = kmeans.fit_predict(self.data['risk_score'].values.reshape(-1, 1))
        #         self.data['risk_bin'] = pd.Series(clusters).astype(str)
        #         bin_method = "kmeans"
        #     except Exception as e2:
        #         print(f"Error: kmeans binning also failed: {str(e2)}. Using equal-width binning.")
        #         try:
        #             self.data['risk_bin'] = pd.cut(self.data['risk_score'], bins=5, duplicates='drop')
        #             bin_method = "cut"
        #         except Exception as e3:
        #             print(f"Critical: All binning methods failed. Using single bin.")
        #             self.data['risk_bin'] = pd.Series(["all"] * len(self.data), index=self.data.index)
        #             bin_method = "single"

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


def preprocess_data(X, y, visualize=True, save_dir="skewness_plots"):
    """数据预处理"""
    # 创建保存目录
    if visualize and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 1. 检查缺失比例
    max_missing = 0.3
    missing_ratio = X.isnull().mean()

    # 2. 移除高缺失特征
    high_missing = missing_ratio[missing_ratio > max_missing].index.tolist()
    if high_missing:
        print(f"Removing features with high missing ratio (> {max_missing:.0%}):")
        for feature in high_missing:
            ratio = missing_ratio[feature]
            print(f"  {feature}: {ratio:.1%} missing")
        X = X.drop(columns=high_missing)
    # 处理缺失值 - 使用中位数填充
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    #
    # # 处理偏态分布
    # skewed_features = X_imputed.apply(lambda x: abs(x.skew()) > 0.75)
    # skewed_features = skewed_features[skewed_features].index.tolist()
    #
    # print(f"\nFound {len(skewed_features)} skewed features:")
    # print(skewed_features)
    #
    # # 可视化偏态特征处理前后的分布
    # if visualize and skewed_features:
    #     # 创建子图布局
    #     n_cols = min(3, len(skewed_features))
    #     n_rows = (len(skewed_features) + n_cols - 1) // n_cols
    #
    #     # 绘制处理前的分布
    #     plt.figure(figsize=(15, 5 * n_rows))
    #     plt.suptitle("Feature Distributions Before Transformation", fontsize=16)
    #
    #     for i, feat in enumerate(skewed_features):
    #         plt.subplot(n_rows, n_cols, i + 1)
    #         sns.histplot(X_imputed[feat], kde=True)
    #         plt.title(f"{feat} (Skew: {X_imputed[feat].skew():.2f})")
    #         plt.xlabel("Value")
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(save_dir, "before_transformation.png"), dpi=300)
    #     plt.show()
    #
    # # 应用QuantileTransformer处理偏态特征
    # transformed_features = []
    # for feat in skewed_features:
    #     try:
    #         n_samples = len(X_imputed)
    #         n_quantiles = min(1000, max(10, n_samples // 2))
    #         qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')
    #         X_imputed[feat] = qt.fit_transform(X_imputed[[feat]])
    #         transformed_features.append(feat)
    #     except Exception as e:
    #         print(f"Error transforming feature {feat}: {str(e)}")
    #
    # print(f"\nSuccessfully transformed {len(transformed_features)} features")
    #
    # # 可视化处理后的分布
    # if visualize and transformed_features:
    #     # 创建子图布局
    #     n_cols = min(3, len(transformed_features))
    #     n_rows = (len(transformed_features) + n_cols - 1) // n_cols
    #
    #     # 绘制处理后的分布
    #     plt.figure(figsize=(15, 5 * n_rows))
    #     plt.suptitle("Feature Distributions After Transformation", fontsize=16)
    #
    #     for i, feat in enumerate(transformed_features):
    #         plt.subplot(n_rows, n_cols, i + 1)
    #         sns.histplot(X_imputed[feat], kde=True)
    #         plt.title(f"{feat} (Skew: {X_imputed[feat].skew():.2f})")
    #         plt.xlabel("Value")
    #
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(save_dir, "after_transformation.png"), dpi=300)
    #     plt.show()

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


def light_feature_selection(X, feature_importance, threshold=0.0):
    """
    轻度特征筛选：移除重要性低于阈值的特征

    参数:
    X: 特征DataFrame
    feature_importance: 特征重要性字典
    threshold: 重要性阈值（默认0.01）

    返回:
    筛选后的特征DataFrame
    保留的特征列表
    """
    # 筛选重要性高于阈值的特征
    selected_features = [feature for feature, imp in feature_importance.items() if imp >= threshold]

    # 确保至少保留5个特征
    if len(selected_features) < 5:
        # 按重要性排序并取前5个
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feature for feature, _ in sorted_features[:5]]
        print(f"Warning: Few features selected. Keeping top 5 features.")

    print(f"\nSelected {len(selected_features)} features after light feature selection:")
    print(selected_features)

    return X[selected_features], selected_features


def rasu_only_evaluation(file_path, target_column='bug', n_folds=5):
    """
    仅应用风险感知分层欠采样(RASU)的评估
    添加轻度特征筛选：移除重要性<0.01的特征

    参数:
    file_path: CSV文件路径
    target_column: 目标列名称
    n_folds: 交叉验证折数
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
        feature_columns = data.columns[:-1]
        target_column = data.columns[-1]
        X = data[feature_columns]
        y = data[target_column]

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

    X_preprocessed, y = preprocess_data(
        X,
        y,
        visualize=True,
        save_dir="skewness_plots"
    )

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

    # 存储特征选择信息
    selected_features_list = []

    print(f"\nStarting RASU Sampling evaluation with {n_folds}-fold cross-validation...")

    # 创建分层五折交叉验证器
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_preprocessed, y)):
        print(f"\n=== Fold {fold_idx + 1}/{n_folds} ===")

        # 分割数据集
        X_train, X_test = X_preprocessed.iloc[train_index], X_preprocessed.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 检查目标列是否有足够的类别
        if len(np.unique(y_train)) < 2:
            print("Warning: Only one class present in training data. Skipping fold.")
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

        # 4. 轻度特征筛选
        print("Performing light feature selection...")
        # 获取特征重要性
        feature_importance = rasu.feature_importance

        # 进行轻度特征筛选
        X_bal_selected, selected_features = light_feature_selection(X_bal, feature_importance, threshold =0.01)

        # 存储特征选择信息（用于第一次运行的可视化）
        if fold_idx == 0:
            selected_features_list = selected_features

        # 可视化采样后的类分布 (仅第一次运行时)
        if fold_idx == 0:
            plot_class_distribution(y_bal, "(After RASU Sampling)", "rasu_class_distribution_balanced.png")

        # 5. 在采样和筛选后的数据上训练模型
        print(f"Training model on RASU-sampled and feature-selected data ({len(selected_features)} features)...")
        # model = RandomForestClassifier(
        #     n_estimators=100,
        #     random_state=42,
        #     n_jobs=-1
        # )
        model = KNeighborsClassifier(
            n_neighbors=5,  # 选择5个最近邻（默认值）
            # weights='uniform',  # 所有邻居权重相等
            # algorithm='auto',  # 自动选择最优算法（KDTree/BallTree/Brute）
            # leaf_size=30,  # 树结构叶子节点大小
            # p=2,  # 欧氏距离（p=2）
            # metric='minkowski',  # 闵可夫斯基距离（p=2时等价欧氏距离）
            n_jobs=-1  # 使用所有CPU核心并行计算
        )
        model.fit(X_bal_selected, y_bal)

        # 6. 评估模型（使用原始测试集特征，但应用相同的特征筛选）
        X_test_selected = X_test[selected_features]
        metrics = evaluate_model(model, X_test_selected, y_test)

        # 保存指标
        for key in all_metrics:
            if key in metrics:
                all_metrics[key].append(metrics[key])

        # 7. 可视化结果 (仅第一次运行时)
        if fold_idx == 0:
            # 绘制混淆矩阵
            plot_confusion_matrix(metrics['confusion_matrix'], "(RASU Sampling)", "rasu_confusion_matrix.png")

            # 绘制指标对比
            plot_metrics_comparison(metrics, "rasu_metrics.png")

            # 计算特征重要性
            feature_importances = pd.DataFrame({
                'Feature': selected_features,
                'Importance': model.weights
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
            plt.title('Feature Importance (After Selection)')
            plt.tight_layout()
            plt.savefig('rasu_feature_importance.png', dpi=300)
            plt.show()

            # 可视化特征筛选结果
            plt.figure(figsize=(14, 8))

            # 获取所有特征的重要性
            all_features = list(feature_importance.keys())
            all_importances = [feature_importance[feature] for feature in all_features]

            # 排序特征
            sorted_idx = np.argsort(all_importances)[::-1]
            sorted_features = [all_features[i] for i in sorted_idx]
            sorted_importances = [all_importances[i] for i in sorted_idx]

            # 创建颜色映射：选中的特征为绿色，未选中的为灰色
            colors = ['green' if feature in selected_features else 'gray' for feature in sorted_features]

            # 绘制条形图
            plt.bar(range(len(sorted_features)), sorted_importances, color=colors)
            plt.axhline(y=0.01, color='r', linestyle='--', label='Threshold (0.01)')
            plt.title('Feature Importance and Selection')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.xticks(range(len(sorted_features)), sorted_features, rotation=90)
            plt.legend()

            # 添加计数标签
            selected_count = len(selected_features)
            total_count = len(all_features)
            plt.text(0.05, 0.95, f"Selected: {selected_count}/{total_count} features",
                     transform=plt.gca().transAxes, fontsize=12)

            plt.tight_layout()
            plt.savefig('feature_selection_result.png', dpi=300)
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
    print(f"Total folds: {n_folds}")

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
        plt.plot(range(1, n_folds + 1), all_metrics[metric], 'o-', linewidth=2)
        plt.title(f'{metric} over Folds')
        plt.xlabel('Fold')
        plt.ylabel(metric)
        plt.ylim(0, 1)
        plt.grid(True)

        # 标注平均线
        avg = np.mean(all_metrics[metric])
        plt.axhline(avg, color='r', linestyle='--')
        plt.text(0.5, avg + 0.02, f'Avg: {avg:.4f}', color='r')

    plt.tight_layout()
    plt.savefig('rasu_performance_over_folds.png', dpi=300)
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
    data_path = r"D:\-\stability-of-smote-main\AEEM\converted_PDE.csv"  # 请替换为你的实际路径

    # 运行仅RASU采样评估
    results = rasu_only_evaluation(
        file_path=data_path,
        n_folds=5
    )

    if results:
        print("\nRASU Sampling evaluation completed successfully!")
        print(f"Average recall: {results['metrics']['Recall']:.4f}")
    else:
        print("\nRASU Sampling evaluation encountered an error.")