import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score, \
    confusion_matrix, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone


class DEFeatureSelector:
    def __init__(self, X, y, n_features, pop_size=30, max_iter=50, F=0.8, CR=0.9, random_state=42):
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
        random_state: 随机种子
        """
        self.X = X
        self.y = y
        self.n_features = n_features
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F
        self.CR = CR
        self.random_state = random_state
        np.random.seed(random_state)
        self.best_fitness_history = []
        self.best_feature_set_history = []
        self.best_solution_history = []
        self.feature_weights = np.ones(n_features)  # 初始化所有特征权重为1

    def enhanced_objective(self, feature_mask):
        """多目标适应度函数"""
        selected_idx = np.where(feature_mask > 0.5)[0]
        if len(selected_idx) == 0:
            selected_idx = np.arange(self.n_features)  # 保留所有特征

        X_sub = self.X[:, selected_idx]
        n_selected = len(selected_idx)

        # 1. 性能评估 (AUC-PR)
        try:
            auc_pr = np.mean(cross_val_score(
                RandomForestClassifier(random_state=self.random_state),
                X_sub, self.y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='average_precision'
            ))
        except Exception:
            auc_pr = 0

        # 2. 特征相关性评估
        try:
            relevance = np.mean(mutual_info_classif(X_sub, self.y, random_state=self.random_state))
        except:
            relevance = 0

        # 3. 特征冗余惩罚
        try:
            redundancy = np.mean(np.corrcoef(X_sub, rowvar=False))
        except:
            redundancy = 0

        # 4. 复杂度惩罚
        complexity_penalty = 0.5 * (n_selected / self.n_features)

        # 5. 召回率评估
        try:
            recall = np.mean(cross_val_score(
                RandomForestClassifier(random_state=self.random_state),
                X_sub, self.y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='recall'
            ))
        except:
            recall = 0

        # 计算适应度
        fitness = (0.6 * recall + 0.4 * auc_pr + 0.3 * relevance - 0.2 * redundancy - 0.3 * complexity_penalty)

        # 更新特征权重
        for idx in selected_idx:
            self.feature_weights[idx] += fitness  # 根据适应度更新权重

        return -fitness  # 最小化问题

    def evolve_generation(self, population):
        """Perform mutation, crossover, and selection to evolve the population."""
        new_population = np.zeros_like(population)
        for i in range(self.pop_size):
            # Mutation: Select three random individuals (different from i)
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]

            # Generate mutant vector
            mutant = a + self.F * (b - c)
            mutant = np.clip(mutant, 0, 1)  # Ensure values are within [0, 1]

            # Crossover: Create trial vector
            crossover_mask = np.random.rand(self.n_features) < self.CR
            if not np.any(crossover_mask):
                crossover_mask[np.random.randint(0, self.n_features)] = True
            trial = np.where(crossover_mask, mutant, population[i])

            # Selection: Evaluate fitness and select the better individual
            if self.enhanced_objective(trial) < self.enhanced_objective(population[i]):
                new_population[i] = trial
            else:
                new_population[i] = population[i]

        return new_population
    def optimize(self):
        """执行完整的差分进化优化"""
        population = np.random.uniform(0, 1, (self.pop_size, self.n_features))
        best_idx = np.argmin([self.enhanced_objective(ind) for ind in population])
        best_solution = population[best_idx].copy()
        best_fitness = self.enhanced_objective(best_solution)

        no_improve = 0
        for gen in range(self.max_iter):
            new_population = self.evolve_generation(population)
            new_best_idx = np.argmin([self.enhanced_objective(ind) for ind in new_population])
            new_best_solution = new_population[new_best_idx].copy()
            new_best_fitness = self.enhanced_objective(new_best_solution)

            if new_best_fitness < best_fitness:
                best_solution = new_best_solution
                best_fitness = new_best_fitness
                no_improve = 0
            else:
                no_improve += 1

            self.best_fitness_history.append(-best_fitness)
            self.best_solution_history.append(best_solution)

            if no_improve >= 10:  # 提前终止条件
                break

        return self.feature_weights  # 返回所有特征的权重

    def plot_evolution(self):
        """可视化进化过程"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, 'b-o', linewidth=2)
        plt.title('Differential Evolution Optimization Progress')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Performance + Relevance - Redundancy)')
        plt.grid(True)
        plt.savefig('de_evolution.png')
        plt.show()


class RASUSampler:
    def __init__(self, features, target, feature_weights, risk_features=None, random_state=42):
        """
        风险感知分层欠采样器

        参数:
        features: 特征DataFrame
        target: 目标Series
        feature_weights: 特征权重向量
        risk_features: 风险特征列表
        random_state: 随机种子
        """
        self.data = pd.concat([features, target], axis=1)
        self.target_name = target.name
        self.feature_weights = feature_weights
        self.risk_features = risk_features if risk_features else features.columns.tolist()
        self.random_state = random_state
        np.random.seed(random_state)
        self.majority_class = 0
        self.minority_class = 1
        self.feature_importance = {}  # 初始化为空字典
        self.risk_weights = None

    def precompute_risk_weights(self):
        """使用传入的特征权重计算风险权重"""
        # 确保 feature_importance 被初始化
        if not hasattr(self, 'feature_importance') or self.feature_importance is None:
            self.feature_importance = {}

        # 归一化特征重要性
        total_weight = np.sum(np.abs(self.feature_weights))
        for i, feature in enumerate(self.risk_features):
            if total_weight > 0:
                self.feature_importance[feature] = self.feature_weights[i] / total_weight
            else:
                self.feature_importance[feature] = 0

        # 计算每个特征的风险权重
        self.risk_weights = {}
        for feature in self.risk_features:
            try:
                # 检查特征是否有足够的值进行分箱
                unique_values = self.data[feature].nunique()
                if unique_values < 2:
                    print(f"Warning: Feature {feature} has only {unique_values} unique values. Skipping binning.")
                    self.risk_weights[feature] = {None: self.feature_importance[feature]}
                    continue

                # 尝试分箱 - 使用更健壮的方法
                try:
                    # 尝试分5箱
                    self.data[f'{feature}_bin'], bins = pd.qcut(
                        self.data[feature],
                        q=5,
                        duplicates='drop',
                        retbins=True
                    )
                except ValueError:
                    # 如果5箱失败，尝试更少的分箱
                    n_bins = min(5, unique_values)
                    if n_bins < 2:
                        n_bins = 2
                    self.data[f'{feature}_bin'], bins = pd.qcut(
                        self.data[feature],
                        q=n_bins,
                        duplicates='drop',
                        retbins=True
                    )

                # 计算每个箱的缺陷率
                bin_stats = self.data.groupby(f'{feature}_bin')[self.target_name].agg(['mean', 'count'])
                bin_stats['defect_ratio'] = bin_stats['mean']

                # 使用Sigmoid函数计算权重
                bin_stats['weight'] = 1 / (1 + np.exp(-20 * (bin_stats['defect_ratio'] - 0.3))) * \
                                      self.feature_importance[feature]
                self.risk_weights[feature] = bin_stats['weight'].to_dict()
            except Exception as e:
                print(f"Warning: Failed to compute risk weights for feature {feature}: {str(e)}")
                # 如果失败，使用特征重要性作为默认权重
                self.risk_weights[feature] = {None: self.feature_importance[feature]}

    def adaptive_sampling(self):
        """执行自适应分层抽样"""
        if not hasattr(self, 'risk_weights'):
            self.precompute_risk_weights()

        # 确保 feature_importance 被初始化
        if not hasattr(self, 'feature_importance') or self.feature_importance is None:
            print("Warning: feature_importance not initialized. Initializing now.")
            self.precompute_risk_weights()

        # 计算每个样本的综合风险分数
        self.data['risk_score'] = 0.0
        for feature in self.risk_features:
            bin_col = f'{feature}_bin'
            if bin_col in self.data.columns:
                # 确保 risk_weights 存在
                if hasattr(self, 'risk_weights') and feature in self.risk_weights:
                    weights = self.data[bin_col].map(self.risk_weights[feature]).fillna(0).astype(float)
                else:
                    weights = pd.Series(0, index=self.data.index)
                    print(f"Warning: Risk weights not found for feature {feature}")
            else:
                # 如果分箱列不存在，使用特征重要性作为权重
                if hasattr(self, 'feature_importance') and feature in self.feature_importance:
                    weight_value = self.feature_importance[feature]
                else:
                    weight_value = 0
                weights = pd.Series(weight_value, index=self.data.index)
                print(f"Warning: Using feature importance for {feature} as bin column {bin_col} not found")

            self.data['risk_score'] += weights

        # 标准化风险分数
        min_score = self.data['risk_score'].min()
        max_score = self.data['risk_score'].max()
        self.data['risk_score'] = (self.data['risk_score'] - min_score) / (max_score - min_score + 1e-10)

        # 按风险分数分箱
        try:
            self.data['risk_bin'] = pd.qcut(self.data['risk_score'], q=5, duplicates='drop')
        except ValueError:
            n_bins = min(5, len(self.data['risk_score'].unique()))
            if n_bins < 2:
                self.data['risk_bin'] = 0
            else:
                self.data['risk_bin'] = pd.qcut(self.data['risk_score'], q=n_bins, duplicates='drop')

        # 分层抽样
        majority_data = self.data[self.data[self.target_name] == self.majority_class]
        minority_data = self.data[self.data[self.target_name] == self.minority_class]

        # 检查是否有多数类样本
        if len(majority_data) == 0:
            print("Warning: No majority class samples found. Returning minority data only.")
            return minority_data

        # 检查是否有风险分箱列
        if 'risk_bin' not in majority_data.columns:
            print("Warning: 'risk_bin' column not found. Using random undersampling.")
            n_samples = min(len(majority_data), int(1.5 * len(minority_data)))
            sampled_majority = majority_data.sample(n=n_samples, random_state=self.random_state)
            return pd.concat([sampled_majority, minority_data])

        # 获取风险分箱
        bin_risk_levels = sorted(majority_data['risk_bin'].unique())

        # 检查是否有分箱
        if len(bin_risk_levels) == 0:
            print("Warning: No risk bins available. Using random undersampling.")
            n_samples = min(len(majority_data), int(1.5 * len(minority_data)))
            sampled_majority = majority_data.sample(n=n_samples, random_state=self.random_state)
            return pd.concat([sampled_majority, minority_data])

        imbalance_ratio = len(majority_data) / max(1, len(minority_data))
        base_preserve = 0.2 + 0.3 * np.exp(-imbalance_ratio / 10)

        print("\nRisk-based sampling strategy:")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1, Base preserve rate: {base_preserve:.1%}")

        sampled_dfs = []
        allocated_samples = 0
        target_majority_size = int(1.5 * len(minority_data))

        # 确保至少有一个分箱被处理
        at_least_one_bin_processed = False

        for i, bin_name in enumerate(bin_risk_levels):
            bin_group = majority_data[majority_data['risk_bin'] == bin_name]
            bin_size = len(bin_group)

            if bin_size == 0:
                print(f"  - Bin {i + 1}: Skipped (0 samples)")
                continue

            # 标记至少有一个分箱被处理
            at_least_one_bin_processed = True

            # 风险等级计算 (0=最低风险, 1=最高风险)
            risk_level = i / max(1, len(bin_risk_levels) - 1)

            # 动态调整保留率：高风险区域保留更多样本
            risk_factor = 0.6 + 0.4 * risk_level
            preserve_rate = min(1.0, base_preserve * risk_factor)

            # 计算目标样本量
            n_samples = max(1, int(bin_size * preserve_rate))  # 至少保留1个样本

            # 按剩余需要分配样本量
            remaining_needed = max(0, target_majority_size - allocated_samples)
            bin_allocation = min(bin_size, remaining_needed, int(0.3 * target_majority_size))

            if allocated_samples < target_majority_size:
                n_samples = min(bin_size, max(n_samples, bin_allocation))
            else:
                n_samples = min(bin_size, max(1, int(0.5 * n_samples)))

            if n_samples > bin_size:
                n_samples = bin_size
                print(f"  - Bin {i + 1}: Adjusted sampling size to {n_samples} (max available)")

            print(f"  - Bin {i + 1}: {bin_size} samples → preserving {n_samples} samples")

            sampled = bin_group.sample(n=n_samples, random_state=self.random_state)
            sampled_dfs.append(sampled)
            allocated_samples += n_samples

        # 检查是否有分箱被处理
        if not at_least_one_bin_processed:
            print("Warning: No bins processed. Using random undersampling as fallback.")
            n_samples = min(len(majority_data), target_majority_size)
            sampled_majority = majority_data.sample(n=n_samples, random_state=self.random_state)
            return pd.concat([sampled_majority, minority_data])

        # 检查 sampled_dfs 是否为空
        if len(sampled_dfs) == 0:
            print("Warning: No bins were sampled. Using random undersampling as fallback.")
            n_samples = min(len(majority_data), target_majority_size)
            sampled_majority = majority_data.sample(n=n_samples, random_state=self.random_state)
        else:
            sampled_majority = pd.concat(sampled_dfs)

        return pd.concat([sampled_majority, minority_data])

    def plot_distribution_comparison(self, original_data, sampled_data, feature):
        """可视化采样前后特征分布对比"""
        plt.figure(figsize=(10, 6))
        sns.kdeplot(original_data[feature], label='Original', fill=True)
        sns.kdeplot(sampled_data[feature], label='RASU Sampled', fill=True)
        plt.title(f'Distribution Comparison: {feature}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(f'rasu_distribution_{feature}.png')
        plt.show()


def preprocess_data(X, y):
    """数据预处理"""
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    skewed_features = X_imputed.apply(lambda x: abs(x.skew()) > 0.75)
    for feat in skewed_features[skewed_features].index:
        n_samples = len(X_imputed)
        n_quantiles = min(1000, max(10, n_samples // 2))
        qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')
        X_imputed[feat] = qt.fit_transform(X_imputed[[feat]])

    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)
    return X_scaled, y


def DE_RASU_pipeline(file_path, target_column='bug', n_iter=30, pop_size=20, random_state=42):
    """
    DE-RASU完整工作流程

    参数:
    file_path: CSV文件路径
    target_column: 目标列名称
    n_iter: 差分进化迭代次数
    pop_size: 种群大小
    random_state: 随机种子
    """
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

    # 将目标变量转换为二分类
    y = y.apply(lambda x: 1 if x > 0 else 0)

    # 输出类分布
    class_counts = y.value_counts()
    print(f"Class distribution:\n{class_counts}")
    if class_counts.max() > 0:
        defect_rate = class_counts[1] / class_counts.sum()
        print(f"Defect rate: {defect_rate:.2%}")

    # 数据预处理
    print("\nPreprocessing data...")
    X = X.apply(pd.to_numeric, errors='coerce')
    X_preprocessed, y = preprocess_data(X, y)

    # 差分进化特征选择
    print("\nStarting Differential Evolution Feature Selection...")
    selector = DEFeatureSelector(
        X_preprocessed.values,
        y.values,
        n_features=X_preprocessed.shape[1],
        max_iter=n_iter,
        pop_size=pop_size,
        random_state=random_state
    )

    # 获取特征权重向量
    feature_weights = selector.optimize()

    # 可视化特征重要性
    feature_importance = pd.DataFrame({
        'Feature': X_preprocessed.columns,
        'Weight': feature_weights
    }).sort_values('Weight', ascending=False)

    print("\nTop 10 Feature Weights:")
    print(feature_importance.head(10))

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Weight', y='Feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Weights')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

    # 可视化进化过程
    selector.plot_evolution()

    # 风险感知分层欠采样
    print("\nPerforming Risk-Aware Stratified Undersampling...")
    rasu = RASUSampler(
        X_preprocessed,
        y,
        feature_weights=feature_weights,
        random_state=random_state
    )
    balanced_data = rasu.adaptive_sampling()

    # 可视化分布变化
    for feature in X_preprocessed.columns[:min(3, len(X_preprocessed.columns))]:
        rasu.plot_distribution_comparison(X_preprocessed, balanced_data, feature)

    # 准备训练数据
    X_bal = balanced_data.drop(columns=[rasu.target_name, 'risk_score', 'risk_bin'], errors='ignore')
    for col in balanced_data.columns:
        if col.endswith('_bin'):
            X_bal = X_bal.drop(columns=col, errors='ignore')

    y_bal = balanced_data[rasu.target_name]

    # 输出采样后的类分布
    print("\nBalanced dataset class distribution:")
    print(y_bal.value_counts())

    if len(np.unique(y_bal)) < 2:
        print("Warning: Sampled dataset has only one class")
        return None

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=random_state
    )

    # 训练最终模型（使用特征权重）
    print("\nTraining Final Model with Feature Weighting...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )

    # 使用特征权重调整模型（可选）
    # 这里我们使用特征权重作为样本权重
    sample_weights = np.zeros(len(X_train))
    for i, (_, row) in enumerate(X_train.iterrows()):
        weight_sum = 0
        for j, feature in enumerate(X_train.columns):
            weight_sum += feature_weights[j] * row[feature]
        sample_weights[i] = weight_sum

    # 归一化样本权重
    sample_weights = (sample_weights - sample_weights.min()) / (sample_weights.max() - sample_weights.min() + 1e-10)
    sample_weights = sample_weights * 0.5 + 0.5  # 调整到0.5-1.0范围

    model.fit(X_train, y_train, sample_weight=sample_weights)

    # 评估模型
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'auc_pr': average_precision_score(y_test, y_proba),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    print("\nModel Evaluation:")
    print(f"AUC-PR: {metrics['auc_pr']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Defect'], yticklabels=['Normal', 'Defect'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # 决策边界可视化（如果特征数>=2）
    if len(X_bal.columns) >= 2:
        print("\nVisualizing decision boundary for top 2 features...")
        top_features = feature_importance.head(2)['Feature'].values
        feat1, feat2 = top_features

        plt.figure(figsize=(10, 8))

        # 创建网格点
        x_min, x_max = X_bal[feat1].min() - 1, X_bal[feat1].max() + 1
        y_min, y_max = X_bal[feat2].min() - 1, X_bal[feat2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        # 创建网格数据
        grid_data = pd.DataFrame({
            feat1: xx.ravel(),
            feat2: yy.ravel()
        })

        # 为其他特征添加中位数
        for feat in X_bal.columns:
            if feat not in [feat1, feat2]:
                grid_data[feat] = X_bal[feat].median()

        # 预测概率
        Z = model.predict_proba(grid_data[X_bal.columns])[:, 1]
        Z = Z.reshape(xx.shape)

        # 绘制等高线
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')

        # 绘制散点图
        plt.scatter(X_bal[feat1], X_bal[feat2], c=y_bal, cmap='bwr', s=20, edgecolors='k')
        plt.title('Decision Boundary Visualization')
        plt.xlabel(feat1)
        plt.ylabel(feat2)
        plt.savefig('decision_boundary.png')
        plt.show()

    # 保存结果
    result = {
        'model': model,
        'feature_weights': feature_weights,
        'metrics': metrics,
        'sampled_data': balanced_data,
        'feature_importance': feature_importance
    }

    # 保存特征选择历史
    feature_history = pd.DataFrame({
        'generation': range(len(selector.best_solution_history)),
        'fitness': selector.best_fitness_history,
        'num_features': [len(np.where(sol > 0.5)[0]) for sol in selector.best_solution_history]
    })
    feature_history.to_csv('de_feature_history.csv', index=False)

    return result


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 从CSV文件加载真实数据
    data_path = r"G:\pycharm\lutrs\de分层欠采样\datacunshu\ant-1.4.csv"
    random_seed = 42

    # 运行DE-RASU管道
    results = DE_RASU_pipeline(
        file_path=data_path,
        n_iter=20,
        pop_size=15,
        random_state=random_seed
    )

    if results:
        print("\nPipeline completed successfully!")
        print(f"Final recall: {results['metrics']['recall']:.4f}")
        print(f"Feature weights saved to 'feature_importance.csv'")

        # 保存特征重要性
        results['feature_importance'].to_csv('feature_importance.csv', index=False)
    else:
        print("\nPipeline encountered an error.")