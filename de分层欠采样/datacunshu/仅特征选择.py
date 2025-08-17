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
from sklearn.feature_selection import mutual_info_classif
import warnings

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


class DEFeatureSelector:
    def __init__(self, X, y, n_features, pop_size=20, max_iter=30, F=0.8, CR=0.9):
        """
        差分进化特征选择器

        参数:
        X: 特征矩阵 (n_samples, n_features)
        y: 目标向量 (n_samples,)
        n_features: 原始特征数量
        pop_size: 种群大小
        max_iter: 最大迭代次数
        F: 差分权重
        CR: 交叉概率
        """
        self.X = X
        self.y = y
        self.n_features = n_features
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        self.best_fitness_history = []
        self.best_feature_set_history = []

    def enhanced_objective(self, feature_mask):
        """多目标适应度函数"""
        selected_idx = np.where(feature_mask > 0.5)[0]
        if len(selected_idx) == 0:
            return 1000  # 惩罚空特征集

        X_sub = self.X[:, selected_idx]
        n_selected = len(selected_idx)

        # 1. 性能评估 (AUC-PR)
        try:
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )

            # 根据目标类别数量选择评分指标
            if len(np.unique(self.y)) == 2:
                scoring = 'average_precision'
            else:
                scoring = 'roc_auc_ovr'

            # 动态设置交叉验证折数
            min_class_size = min(np.bincount(self.y))
            n_splits = min(5, min_class_size) if min_class_size > 0 else 3
            n_splits = max(2, n_splits)

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_sub, self.y, scoring="matthews_corrcoef", cv=cv, n_jobs=-1)
            performance = np.mean(scores)
        except Exception as e:
            print(f"Warning: Enhanced objective function error: {str(e)}")
            performance = 0.01

        # 2. 特征相关性评估
        try:
            mi_scores = mutual_info_classif(X_sub, self.y, random_state=42)
            relevance = np.mean(mi_scores)
        except:
            relevance = 0

        # 3. 特征冗余惩罚
        try:
            # 检查并移除常数值特征
            non_constant_mask = np.std(X_sub, axis=0) > 1e-6
            X_sub_non_constant = X_sub[:, non_constant_mask]

            if X_sub_non_constant.shape[1] > 1:
                corr_matrix = np.corrcoef(X_sub_non_constant, rowvar=False)
                np.fill_diagonal(corr_matrix, 0)
                redundancy = np.mean(np.abs(corr_matrix))
            else:
                redundancy = 0
        except:
            redundancy = 0.5

        # 4. 复杂度惩罚
        complexity_penalty = 0.5 * (n_selected / self.n_features)

        # 5. 召回率评估
        # try:
        #     recall_scores = cross_val_score(
        #         model, X_sub, self.y,
        #         scoring='recall',
        #         cv=cv, n_jobs=-1
        #     )
        #     recall = np.mean(recall_scores)
        # except:
        #     recall = 0.4
        #
        # # 最终适应度 (最小化问题)
        # fitness = (0.6 * recall + 0.4 * performance + 0.3 * relevance - 0.2 * redundancy - 0.3 * complexity_penalty)
        fitness = performance

        return -fitness

    def evolve_generation(self, population):
        """执行完整的一代差分进化"""
        mutant_population = np.zeros_like(population)
        trial_population = np.zeros_like(population)

        # 1. 变异阶段
        for i in range(self.pop_size):
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant_population[i] = a + self.F * (b - c)
            mutant_population[i] = np.clip(mutant_population[i], 0, 1)

        # 2. 交叉阶段
        for i in range(self.pop_size):
            cross_points = np.random.random(self.n_features) < self.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.n_features)] = True
            trial_population[i] = np.where(cross_points, mutant_population[i], population[i])

        # 3. 选择阶段
        new_population = population.copy()
        for i in range(self.pop_size):
            current_fitness = self.enhanced_objective(population[i])
            trial_fitness = self.enhanced_objective(trial_population[i])
            if trial_fitness < current_fitness:
                new_population[i] = trial_population[i]
        return new_population

    def optimize(self):
        """执行完整的差分进化优化"""
        # 初始化种群 (连续向量)
        population = np.random.uniform(0, 1, (self.pop_size, self.n_features))

        # 记录初始最佳
        best_idx = np.argmin([self.enhanced_objective(ind) for ind in population])
        best_solution = population[best_idx].copy()
        best_fitness = self.enhanced_objective(best_solution)

        # 进化迭代
        for gen in range(self.max_iter):
            population = self.evolve_generation(population)
            current_fitnesses = [self.enhanced_objective(ind) for ind in population]
            gen_best_idx = np.argmin(current_fitnesses)
            gen_best_fitness = current_fitnesses[gen_best_idx]

            if gen_best_fitness < best_fitness:
                best_solution = population[gen_best_idx].copy()
                best_fitness = gen_best_fitness

            # 记录历史
            self.best_fitness_history.append(-best_fitness)
            self.best_feature_set_history.append(np.where(best_solution > 0.5)[0])

            # 进度输出
            if (gen + 1) % 5 == 0:
                n_features = len(np.where(best_solution > 0.5)[0])
                print(f"Generation {gen + 1}/{self.max_iter} - Fitness: {-best_fitness:.4f}, Features: {n_features}")

        # 获取最优特征子集
        optimal_mask = best_solution > 0.5
        return np.arange(self.n_features)[optimal_mask]

    def plot_evolution(self, save_path=None):
        """可视化进化过程"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, 'b-o', linewidth=2)
        plt.title('Differential Evolution Optimization Progress')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Performance + Relevance - Redundancy)')
        plt.grid(True)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


def preprocess_data(X, y):
    """数据预处理"""
    # 处理缺失值 - 使用中位数填充
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # # 处理偏态分布
    # skewed_features = X_imputed.apply(lambda x: abs(x.skew()) > 0.75)
    #
    # for feat in skewed_features[skewed_features].index:
    #     n_samples = len(X_imputed)
    #     n_quantiles = min(1000, max(10, n_samples // 2))
    #     qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')
    #     X_imputed[feat] = qt.fit_transform(X_imputed[[feat]])

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


def plot_feature_importance(feature_importances, title="", save_path=None):
    """可视化特征重要性"""
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importances,
                palette='viridis', hue='Feature', legend=False)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def feature_selection_only_evaluation(file_path, target_column='bug', n_runs=1,
                                      de_pop_size=20, de_max_iter=30):
    """
    仅应用差分进化特征选择(DE)的评估

    参数:
    file_path: CSV文件路径
    target_column: 目标列名称
    n_runs: 重复实验次数
    de_pop_size: DE种群大小
    de_max_iter: DE最大迭代次数
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
    plot_class_distribution(y, "(Original)", "de_class_distribution.png")

    # 2. 数据预处理
    print("\nPreprocessing data...")
    X = X.apply(pd.to_numeric, errors='coerce')
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

    # 存储特征选择信息
    selected_features_history = []
    feature_counts = []

    print(f"\nStarting DE Feature Selection evaluation with {n_runs} runs...")
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

        # 3. 应用差分进化特征选择
        print("Applying Differential Evolution Feature Selection...")
        selector = DEFeatureSelector(
            X_train.values,
            y_train.values,
            n_features=X_train.shape[1],
            pop_size=de_pop_size,
            max_iter=de_max_iter
        )

        # 执行特征选择
        optimal_features_idx = selector.optimize()
        optimal_features = X_train.columns[optimal_features_idx]

        # 记录特征选择结果
        selected_features_history.append(optimal_features.tolist())
        feature_counts.append(len(optimal_features))

        print(f"Selected {len(optimal_features)} features: {optimal_features.tolist()}")

        # 可视化进化过程 (仅第一次运行时)
        if run == 0:
            selector.plot_evolution("de_evolution.png")

        # 4. 在选定特征上训练模型
        print("Training model on selected features...")
        X_train_selected = X_train.iloc[:, optimal_features_idx]
        X_test_selected = X_test.iloc[:, optimal_features_idx]

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train_selected, y_train)

        # 5. 评估模型
        metrics = evaluate_model(model, X_test_selected, y_test)

        # 保存指标
        for key in all_metrics:
            if key in metrics:
                all_metrics[key].append(metrics[key])

        # 6. 可视化结果 (仅第一次运行时)
        if run == 0:
            # 绘制混淆矩阵
            plot_confusion_matrix(metrics['confusion_matrix'], "(DE Feature Selection)", "de_confusion_matrix.png")

            # 计算特征重要性
            feature_importances = pd.DataFrame({
                'Feature': optimal_features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            # 可视化特征重要性
            plot_feature_importance(feature_importances.head(15),
                                    "Top Features Selected by DE",
                                    "de_feature_importance.png")

    # 计算平均指标
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}

    # 创建结果数据框
    results_df = pd.DataFrame({
        'Metric': list(all_metrics.keys()),
        'Mean': [np.mean(all_metrics[key]) for key in all_metrics],
        'StdDev': [np.std(all_metrics[key]) for key in all_metrics]
    })

    # 特征选择稳定性分析
    feature_stability = {}
    for feature in X_preprocessed.columns:
        selected_count = sum(1 for features in selected_features_history if feature in features)
        feature_stability[feature] = selected_count / n_runs

    # 按稳定性排序
    feature_stability_df = pd.DataFrame({
        'Feature': list(feature_stability.keys()),
        'Selection Rate': list(feature_stability.values())
    }).sort_values('Selection Rate', ascending=False)

    # 打印结果
    print("\n" + "=" * 80)
    print("DE Feature Selection Evaluation Results")
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

    # 特征选择分析
    print("\nFeature Selection Analysis:")
    print(f"Average number of features selected: {np.mean(feature_counts):.1f}")
    print(f"Min features selected: {min(feature_counts)}")
    print(f"Max features selected: {max(feature_counts)}")

    print("\nTop 10 most frequently selected features:")
    print(feature_stability_df.head(10))

    # 可视化特征稳定性
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Selection Rate', y='Feature',
                data=feature_stability_df.head(20),
                palette='viridis', hue='Feature', legend=False)
    plt.title('Top 20 Most Stable Features (DE Selection)')
    plt.xlabel('Selection Rate')
    plt.tight_layout()
    plt.savefig('de_feature_stability.png', dpi=300)
    plt.show()

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
    plt.savefig('de_performance_over_runs.png', dpi=300)
    plt.show()

    return {
        'metrics': avg_metrics,
        'feature_stability': feature_stability_df,
        'feature_counts': feature_counts
    }


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 从CSV文件加载真实数据
    data_path = r"G:\pycharm\lutrs\de分层欠采样\datacunshu\Ivy-2.0.csv"  # 请替换为你的实际路径

    # 运行仅特征选择评估
    results = feature_selection_only_evaluation(
        file_path=data_path,
        n_runs=1,
        de_pop_size=20,
        de_max_iter=20
    )

    if results:
        print("\nDE Feature Selection evaluation completed successfully!")
        print(f"Average recall: {results['metrics']['Recall']:.4f}")
    else:
        print("\nDE Feature Selection evaluation encountered an error.")