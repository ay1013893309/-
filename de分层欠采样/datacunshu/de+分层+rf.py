import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score, \
    confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
# 在文件开头添加导入
from sklearn.metrics import matthews_corrcoef
from sklearn.tree import DecisionTreeClassifier


class DEFeatureSelector:
    def __init__(self, X, y, n_features, pop_size=30, max_iter=50, F=0.8, CR=0.9):
        """
        完整版差分进化特征选择器

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
            # 修改为使用随机森林分类器
            model = RandomForestClassifier(
                n_estimators=50,  # 使用50棵树以提高效率
                max_depth=5,  # 控制过拟合
                random_state=42,
                n_jobs=-1  # 使用所有CPU核心
            )

            # 使用分层交叉验证
            # 检查是否是二分类问题
            if len(np.unique(self.y)) == 2:
                scoring = 'average_precision'
            else:
                scoring = 'roc_auc_ovr'  # 多分类使用ROC-AUC

            # 根据样本量动态设置交叉验证折数
            min_class_size = min(np.bincount(self.y))
            n_splits = min(5, min_class_size) if min_class_size > 0 else 3
            n_splits = max(2, n_splits)  # 至少2折

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            scores = cross_val_score(model, X_sub, self.y,
                                     scoring=scoring,
                                     cv=cv, n_jobs=-1)
            performance = np.mean(scores)
        except Exception as e:
            print(f"Warning: Enhanced objective function error: {str(e)}")
            performance = 0.01

        # 2. 特征相关性评估
        try:
            # 使用互信息
            mi_scores = mutual_info_classif(X_sub, self.y, random_state=42)
            relevance = np.mean(mi_scores)
        except:
            relevance = 0

        # 3. 特征冗余惩罚
        try:
            # 检查并移除常数值特征
            non_constant_mask = np.std(X_sub, axis=0) > 1e-6
            X_sub_non_constant = X_sub[:, non_constant_mask]

            if X_sub_non_constant.shape[1] > 1:  # 至少需要两个特征才能计算相关性
                corr_matrix = np.corrcoef(X_sub_non_constant, rowvar=False)
                np.fill_diagonal(corr_matrix, 0)
                redundancy = np.mean(np.abs(corr_matrix))
            else:
                redundancy = 0  # 无法计算相关性时设为0
        except:
            redundancy = 0.5

        # 4. 复杂度惩罚
        complexity_penalty = 0.5 * (n_selected / self.n_features)
        # 在函数顶部计算召回率（新增）
        try:
            recall_scores = cross_val_score(
                model, X_sub, self.y,
                scoring='recall',
                cv=cv, n_jobs=-1
            )
            recall = np.mean(recall_scores)
        except:
            recall = 0.4

        # # 最终适应度 (最大化问题)
        # fitness = (0.5 * performance + 0.3 * relevance -
        #            0.4 * redundancy - 0.3 * complexity_penalty)
        fitness = (0.4 * performance + 0.3 * relevance - 0.2 * redundancy)

        return -fitness  # 最小化问题

    def evolve_generation(self, population):
        """执行完整的一代差分进化"""
        mutant_population = np.zeros_like(population)
        trial_population = np.zeros_like(population)

        # 1. 变异阶段
        for i in range(self.pop_size):
            # 随机选择三个不同的个体
            indices = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant_population[i] = a + self.F * (b - c)

            # 确保变异后仍在[0,1]范围内
            mutant_population[i] = np.clip(mutant_population[i], 0, 1)

        # 2. 交叉阶段
        for i in range(self.pop_size):
            cross_points = np.random.random(self.n_features) < self.CR
            # 确保至少有一个维度交叉
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.n_features)] = True

            trial_population[i] = np.where(cross_points,
                                           mutant_population[i],
                                           population[i])

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

            # 更新最佳个体
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
    def __init__(self, features, target, risk_features=None):
        """
        风险感知分层欠采样器

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

    def precompute_risk_weights(self):
        """预计算特征风险权重"""
        # 训练简单模型获取特征重要性
        X = self.data.drop(self.target_name, axis=1)
        y = self.data[self.target_name]
        # 使用KNN计算特征重要性
        model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        model.fit(X, y)
        # # 使用随机森林计算特征重要性（修改点）
        # model = RandomForestClassifier(
        #     n_estimators=100,
        #     random_state=42,
        #     n_jobs=-1
        # )
        # model.fit(X, y)
        # # 获取特征重要性
        # importances = model.feature_importances_
        # self.feature_importance = dict(zip(X.columns, importances))
        # 基于模型性能估计特征重要性
        self.feature_importance = {}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        # baseline_score = np.mean(cross_val_score(model, X, y, cv=cv, scoring='recall'))
        baseline_score = np.mean(cross_val_score(model, X, y, cv=cv, scoring='matthews_corrcoef'))

        for feature in self.risk_features:
            X_reduced = X.drop(feature, axis=1)
            reduced_score = np.mean(cross_val_score(model, X_reduced, y, cv=cv, scoring='matthews_corrcoef'))
            importance = max(0, baseline_score - reduced_score)  # 特征删除导致性能下降的程度
            self.feature_importance[feature] = importance

        # 归一化特征重要性
        total = sum(self.feature_importance.values())
        for feature in self.feature_importance:
            if total > 0:
                self.feature_importance[feature] /= total

        # 计算每个特征的风险权重
        self.risk_weights = {}
        for feature in self.risk_features:
            # 按特征分箱
            self.data[f'{feature}_bin'], bins = pd.qcut(
                self.data[feature],
                q=5,
                duplicates='drop',
                retbins=True
            )

            # 计算每个箱的缺陷率
            bin_stats = self.data.groupby(f'{feature}_bin')[self.target_name].agg(['mean', 'count'])

            # 使用Sigmoid函数优化权重计算
            bin_stats['defect_ratio'] = bin_stats['mean']
            bin_stats['weight'] = 1 / (1 + np.exp(-20 * (bin_stats['defect_ratio'] - 0.3))) * self.feature_importance[
                feature]

            self.risk_weights[feature] = bin_stats['weight'].to_dict()

    def adaptive_sampling(self):
        """Perform adaptive stratified sampling."""
        if not hasattr(self, 'risk_weights'):
            self.precompute_risk_weights()

        # Calculate the overall risk score for each sample
        self.data['risk_score'] = 0.0  # Initialize as float

        for feature in self.risk_features:
            bin_col = f'{feature}_bin'
            weights = self.data[bin_col].map(self.risk_weights[feature]).astype(float)
            self.data['risk_score'] += weights

        # Normalize risk scores
        min_score = self.data['risk_score'].min()
        max_score = self.data['risk_score'].max()
        self.data['risk_score'] = (self.data['risk_score'] - min_score) / (max_score - min_score + 1e-10)

        # Use `duplicates='drop'` to handle duplicate bin edges
        self.data['risk_bin'] = pd.qcut(self.data['risk_score'], q=5, duplicates='drop')

        # Perform stratified sampling
        sampled_dfs = []
        majority_data = self.data[self.data[self.target_name] == self.majority_class]
        bin_risk_levels = sorted(majority_data['risk_bin'].unique())

        minority_data = self.data[self.data[self.target_name] == self.minority_class]

        minority_count = len(minority_data)
        count = minority_count / 5
        print("\nRisk-based sampling strategy:")
        for i, bin_name in enumerate(bin_risk_levels):
            bin_group = majority_data[majority_data['risk_bin'] == bin_name]
            bin_size = len(bin_group)

            preserve_rate = 0.25 + 0.50 / (1 + np.exp(-3 * (i - len(bin_risk_levels) / 2)))
            n_samples = max(int(bin_size * preserve_rate), 15)

            if n_samples > bin_size:
                n_samples = bin_size
                print(f"  - Bin {i + 1}: Adjusted sampling size to {n_samples} (max available)")

            print(f"  - Bin {i + 1}: {bin_size} samples → preserving {preserve_rate:.1%} ({n_samples} samples)")

            sampled = bin_group.sample(n=n_samples)
            sampled_dfs.append(sampled)

        sampled_majority = pd.concat(sampled_dfs)
        minority_data = self.data[self.data[self.target_name] == self.minority_class]

        return pd.concat([sampled_majority, minority_data])

    # def adaptive_sampling(self):
    #     """执行风险感知分层自适应欠采样"""
    #     if not hasattr(self, 'risk_weights'):
    #         self.precompute_risk_weights()
    #
    #     # 计算每个样本的综合风险分数
    #     self.data['risk_score'] = 0.0  # 初始化为浮点数
    #
    #     # 计算每个样本的总风险分数
    #     for feature in self.risk_features:
    #         bin_col = f'{feature}_bin'
    #         weights = self.data[bin_col].map(self.risk_weights[feature])
    #         self.data['risk_score'] += weights.astype(float)
    #
    #     # 标准化风险分数 (0-1范围)
    #     min_score = self.data['risk_score'].min()
    #     max_score = self.data['risk_score'].max()
    #     range_score = max_score - min_score + 1e-10  # 避免除零错误
    #     self.data['risk_score'] = (self.data['risk_score'] - min_score) / range_score
    #
    #     # 按风险分数分箱处理重复边缘情况
    #     try:
    #         self.data['risk_bin'] = pd.qcut(self.data['risk_score'], q=5, duplicates='drop')
    #     except ValueError:
    #         # 处理特殊情况下无法分箱的情况
    #         n_bins = min(5, len(self.data['risk_score'].unique()))
    #         if n_bins < 2:
    #             self.data['risk_bin'] = 0
    #         else:
    #             self.data['risk_bin'] = pd.qcut(self.data['risk_score'], q=n_bins, duplicates='drop')
    #
    #     # 分离多数类和少数类
    #     majority_data = self.data[self.data[self.target_name] == self.majority_class]
    #     minority_data = self.data[self.data[self.target_name] == self.minority_class]
    #
    #     # 计算不平衡比率和动态基础保留率
    #     imbalance_ratio = len(majority_data) / max(1, len(minority_data))  # 避免除零
    #     base_preserve = 0.2 + 0.3 * np.exp(-imbalance_ratio / 10)  # 不平衡度越高，保留比例越低
    #
    #     # 获取风险分箱并确保有序
    #     bin_risk_levels = sorted(majority_data['risk_bin'].unique())
    #
    #     print("\n动态风险采样策略:")
    #     print(f"不平衡比率: {imbalance_ratio:.2f}:1, 基准保留率: {base_preserve:.1%}")
    #
    #     sampled_dfs = []
    #     total_minority = max(1, len(minority_data))  # 确保分母不为零
    #     target_majority_size = int(1.5 * total_minority)  # 目标多数类样本量为少数类的1.5倍
    #     allocated_samples = 0
    #
    #     for i, bin_name in enumerate(bin_risk_levels):
    #         bin_group = majority_data[majority_data['risk_bin'] == bin_name]
    #         bin_size = len(bin_group)
    #
    #         if bin_size == 0:
    #             continue
    #
    #         # 风险等级计算 (0=最低风险, 1=最高风险)
    #         risk_level = i / max(1, len(bin_risk_levels) - 1)
    #
    #         # 动态调整保留率：高风险区域保留更多样本
    #         risk_factor = 0.6 + 0.4 * risk_level  # 高风险区域权重更高
    #         preserve_rate = min(1.0, base_preserve * risk_factor)
    #
    #         # 计算目标样本量
    #         n_samples = max(10, int(bin_size * preserve_rate))  # 至少保留10个样本
    #
    #         # 按剩余需要分配样本量
    #         remaining_needed = max(0, target_majority_size - allocated_samples)
    #         bin_allocation = min(bin_size, remaining_needed, int(0.3 * target_majority_size))
    #
    #         if allocated_samples < target_majority_size:
    #             # 如果尚未达到目标样本量，按分配量调整
    #             n_samples = min(bin_size, max(n_samples, bin_allocation))
    #         else:
    #             # 如果已超过目标量，减少取样量
    #             n_samples = min(bin_size, max(10, int(0.5 * n_samples)))
    #
    #         # 确保样本量合理
    #         if n_samples > bin_size:
    #             n_samples = bin_size
    #             print(f"  - 分箱 {i + 1}: 调整样本量至最大值 {n_samples}")
    #
    #         # 更新已分配样本计数
    #         allocated_samples += n_samples
    #
    #         print(f"  - 分箱 {i + 1}: 风险等级 {risk_level:.2f}, {bin_size} 样本")
    #         print(f"      基准保留率: {preserve_rate:.1%}, 目标样本: {n_samples}")
    #
    #         # 执行抽样（当样本量小于分箱大小时）
    #         if n_samples < bin_size:
    #             sampled = bin_group.sample(n=n_samples, random_state=42)
    #         else:
    #             sampled = bin_group
    #
    #         sampled_dfs.append(sampled)
    #
    #     # 合并多数类样本并添加所有少数类样本
    #     sampled_majority = pd.concat(sampled_dfs)
    #
    #     # 最终采样比例报告
    #     final_majority = len(sampled_majority)
    #     final_minority = len(minority_data)
    #     final_ratio = final_majority / max(1, final_minority)
    #
    #     print(f"\n最终采样结果: 多数类 {final_majority}, 少数类 {final_minority}")
    #     print(f"最终多数:少数比例: {final_ratio:.2f}:1")
    #
    #     return pd.concat([sampled_majority, minority_data])

    # def adaptive_sampling(self):
    #     """执行自适应分层抽样"""
    #     if not hasattr(self, 'risk_weights'):
    #         self.precompute_risk_weights()
    #
    #     # 计算每个样本的综合风险分数
    #     self.data['risk_score'] = 0.0  # 初始化为浮点数
    #
    #     for feature in self.risk_features:
    #         bin_col = f'{feature}_bin'
    #         # 将权重转换为浮点数
    #         weights = self.data[bin_col].map(self.risk_weights[feature]).astype(float)
    #         self.data['risk_score'] += weights
    #
    #     # 标准化风险分数
    #     min_score = self.data['risk_score'].min()
    #     max_score = self.data['risk_score'].max()
    #     self.data['risk_score'] = (self.data['risk_score'] - min_score) / (max_score - min_score + 1e-10)
    #
    #     # 按风险分数分箱
    #     self.data['risk_bin'] = pd.qcut(self.data['risk_score'], q=5)
    #
    #     # 分层抽样
    #     sampled_dfs = []
    #     majority_data = self.data[self.data[self.target_name] == self.majority_class]
    #     bin_risk_levels = sorted(majority_data['risk_bin'].unique())
    #
    #     print("\nRisk-based sampling strategy:")
    #     for i, bin_name in enumerate(bin_risk_levels):
    #         bin_group = majority_data[majority_data['risk_bin'] == bin_name]
    #         bin_size = len(bin_group)
    #
    #         # 使用S型曲线分配采样率（低风险区保留更多样本）
    #         preserve_rate = 0.25 + 0.50 / (1 + np.exp(-3 * (i - len(bin_risk_levels) / 2)))
    #         n_samples = max(int(bin_size * preserve_rate), 15)  # 保证最小样本量
    #
    #         # 确保抽样数量不超过分箱大小
    #         if n_samples > bin_size:
    #             n_samples = bin_size
    #             print(f"  - Bin {i + 1}: Adjusted sampling size to {n_samples} (max available)")
    #
    #         print(f"  - Bin {i + 1}: {bin_size} samples → preserving {preserve_rate:.1%} ({n_samples} samples)")
    #
    #         sampled = bin_group.sample(n=n_samples, random_state=42)
    #         sampled_dfs.append(sampled)
    #
    #     # 合并多数类样本并添加少数类样本
    #     sampled_majority = pd.concat(sampled_dfs)
    #     minority_data = self.data[self.data[self.target_name] == self.minority_class]
    #
    #     return pd.concat([sampled_majority, minority_data])

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


def DE_RASU_pipeline(file_path, target_column='bug', n_iter=30, pop_size=20):
    """
    DE-RASU完整工作流程

    参数:
    file_path: CSV文件路径
    target_column: 目标列名称
    n_iter: 差分进化迭代次数
    pop_size: 种群大小
    """
    # 1. 从CSV文件加载数据
    print(f"Loading data from {file_path}...")
    if not os.path.exists(file_path):
        print(f"Error: File not found at path: {file_path}")
        return None

    data = pd.read_csv(file_path)
    print(f"Dataset shape: {data.shape}")

    # 删除前三列标识信息列
    if data.shape[1] > 3:  # 确保有足够的列
        # 保留特征列（从第3列开始到倒数第二列）和目标列（最后一列）
        feature_columns = data.columns[0:-1]
        target_column = data.columns[-1]

        X = data[feature_columns]
        y = data[target_column]
    # else:
    #     raise ValueError("CSV文件列数不足，请检查数据格式")

    # 将目标变量转换为二分类：大于0表示有缺陷（1），等于0表示无缺陷（0）
    y = y.apply(lambda x: 1 if x > 0 else 0)

    # 输出类分布
    class_counts = y.value_counts()
    print(f"Class distribution:\n{class_counts}")
    if class_counts.max() > 0:
        defect_rate = class_counts[1] / class_counts.sum()
        print(f"Defect rate: {defect_rate:.2%}")

    # 2. 数据预处理
    print("\nPreprocessing data...")
    # 确保所有特征是数值型
    X = X.apply(pd.to_numeric, errors='coerce')
    X_preprocessed, y = preprocess_data(X, y)

    # 3. 差分进化特征选择
    print("\nStarting Differential Evolution Feature Selection...")
    selector = DEFeatureSelector(X_preprocessed.values, y.values,
                                 n_features=X_preprocessed.shape[1],
                                 max_iter=n_iter, pop_size=pop_size)
    optimal_features_idx = selector.optimize()
    optimal_features = X_preprocessed.columns[optimal_features_idx]
    print(f"\nSelected {len(optimal_features)} features: {optimal_features.tolist()}")

    # 可视化进化过程
    selector.plot_evolution()

    # 4. 风险感知分层欠采样
    print("\nPerforming Risk-Aware Stratified Undersampling...")
    rasu = RASUSampler(X_preprocessed[optimal_features], y)
    balanced_data = rasu.adaptive_sampling()

    # 5. 可视化分布变化
    for feature in optimal_features[:min(3, len(optimal_features))]:  # 可视化前3个特征
        rasu.plot_distribution_comparison(X_preprocessed, balanced_data, feature)

    # 6. 训练最终模型
    print("\nTraining Final Model...")
    X_bal = balanced_data.drop(rasu.target_name, axis=1)
    y_bal = balanced_data[rasu.target_name]

    # 移除在采样过程中添加的辅助列（分箱列和风险列）
    bin_columns = [col for col in X_bal.columns if col.endswith('_bin')]
    risk_columns = ['risk_score', 'risk_bin', 'temp_weight']  # 可能存在的列
    columns_to_drop = bin_columns + [col for col in risk_columns if col in X_bal.columns]
    X_bal = X_bal.drop(columns=columns_to_drop, errors='ignore')

    # 输出采样后的类分布
    print("Balanced dataset class distribution:")
    print(y_bal.value_counts())

    # 检查目标列是否有足够的类别
    if len(np.unique(y_bal)) < 2:
        print("警告：采样后的数据集只有一个类别，无法训练分类模型")
        return None

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
    )

    # 6. 训练最终模型
    print("\nTraining Final Model with Random Forest...")
    # 使用更强大的随机森林分类器
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    # model =DecisionTreeClassifier();
    # 训练模型
    model.fit(X_train, y_train)

    # 7. 评估模型
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # 计算MCC指标
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics = {
        'auc_pr': average_precision_score(y_test, y_proba) if y_proba is not None else 0,
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 0,
        'f1': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'mcc': mcc,  # 添加MCC指标
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

    print("\nModel Evaluation:")
    print(f"AUC-PR: {metrics['auc_pr']:.4f}")
    # print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")  # 打印MCC值
    print(f"AUC: {metrics['roc_auc']:.4f}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Defect'], yticklabels=['Normal', 'Defect'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # 在 DE_RASU_pipeline 函数中找到决策边界可视化部分，修改为：

    # 8. KNN没有特征重要性，但我们可以绘制决策边界
    # 选择一个特征对进行可视化
    if len(optimal_features) >= 2:
        print("\nVisualizing decision boundary for top 2 features...")
        plt.figure(figsize=(10, 8))

        # 使用前两个特征
        selected_feat1, selected_feat2 = optimal_features[:2]

        # 创建网格点
        x_min, x_max = X_bal[selected_feat1].min() - 1, X_bal[selected_feat1].max() + 1
        y_min, y_max = X_bal[selected_feat2].min() - 1, X_bal[selected_feat2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

        # 创建一个只包含这两个特征的数据集
        # 其他特征使用中位数填充
        grid_data = pd.DataFrame({
            selected_feat1: xx.ravel(),
            selected_feat2: yy.ravel()
        })

        # 为其他特征添加中位数
        for feat in optimal_features[2:]:
            grid_data[feat] = X_bal[feat].median()

        # 确保特征顺序与训练数据一致
        grid_data = grid_data[optimal_features]

        # 预测网格点
        Z = model.predict_proba(grid_data)[:, 1]
        Z = Z.reshape(xx.shape)

        # 绘制等高线
        plt.contourf(xx, yy, Z, alpha=0.4)

        # 绘制散点图
        plt.scatter(X_bal[selected_feat1], X_bal[selected_feat2],
                    c=y_bal, cmap=plt.cm.bwr, s=20, edgecolors='k')

        plt.title('Decision Boundary Visualization')
        plt.xlabel(selected_feat1)
        plt.ylabel(selected_feat2)
        plt.savefig('knn_decision_boundary.png')
        plt.show()

    # 9. 保存结果
    result = {
        'model': model,
        'selected_features': optimal_features.tolist(),
        'metrics': metrics,
        'sampled_data': balanced_data
    }

    # 保存特征选择历史
    feature_history = pd.DataFrame({
        'generation': range(len(selector.best_feature_set_history)),
        'fitness': selector.best_fitness_history,
        'num_features': [len(feats) for feats in selector.best_feature_set_history]
    })
    feature_history.to_csv('de_feature_history.csv', index=False)

    return result


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 从CSV文件加载真实数据
    data_path = r"D:\-\de分层欠采样\datacunshu\Log4j-1.1.csv"  # 使用原始字符串

    # 运行DE-RASU管道
    results = DE_RASU_pipeline(file_path=data_path,
                               n_iter=30,
                               pop_size=15)

    if results:
        print("\nPipeline completed successfully!")
        print(f"Final recall: {results['metrics']['recall']:.4f}")
    else:
        print("\nPipeline encountered an error. See above output for details.")