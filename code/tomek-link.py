# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.neighbors import NearestNeighbors
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (precision_score, recall_score, f1_score,
#                              matthews_corrcoef, roc_auc_score,
#                              precision_recall_curve, auc)
# from collections import Counter
# import matplotlib.pyplot as plt
# import joblib
# import time
# import os
#
#
# # 优化点1：添加缓存机制减少重复计算
# def load_data(cache_path="data_cache.pkl"):
#     # """加载数据并添加缓存机制"""
#     # if os.path.exists(cache_path):
#     #     return joblib.load(cache_path)
#
#     df = pd.read_csv(r"G:\pycharm\lutrs\code\datacunshu\ant-1.7.csv")
#     y_raw = df['bug'].values
#     y = (y_raw > 0).astype(int)
#     df = df.drop(columns=['name', 'version', 'name1', 'bug'], errors='ignore')
#     X = df.select_dtypes(include=['int64', 'float64']).values
#
#     # 添加数据清洗：处理缺失值
#     if np.isnan(X).any():
#         print(f"发现 {np.isnan(X).sum()} 个缺失值，使用中位数填充")
#         from sklearn.impute import SimpleImputer
#         imputer = SimpleImputer(strategy='median')
#         X = imputer.fit_transform(X)
#
#     joblib.dump((X, y), cache_path)
#     return X, y
#
#
# # 优化点2：增加降采样方法选择
# def reduce_majority_samples(X, y, method='knn', majority_label=0, k=5, keep_ratio=0.7):
#     """
#     降低多数类样本数量
#     支持两种方法：
#     - knn: 基于KNN密度估计 (默认)
#     - tomenk: 使用Tomek Links去除边界样本
#     """
#     if method == 'knn':
#         return knn_reduction(X, y, majority_label, k, keep_ratio)
#     elif method == 'tomek':
#         return tomek_reduction(X, y, majority_label)
#     else:
#         raise ValueError(f"未知的降采样方法: {method}")
#
#
# def knn_reduction(X, y, majority_label=0, k=5, keep_ratio=0.7):
#     """基于KNN的多数类降采样"""
#     X_major = X[y == majority_label]
#
#     # 优化点3：使用KDTree加速近邻搜索
#     neigh = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(X_major)
#     dists, _ = neigh.kneighbors(X_major)
#
#     # 排除自身距离
#     avg_dists = np.mean(dists[:, 1:], axis=1)
#     threshold = np.percentile(avg_dists, keep_ratio * 100)
#     keep_mask = avg_dists <= threshold
#
#     # 多数类去冗余，少数类全保留
#     X_cleaned = np.concatenate([X_major[keep_mask], X[y != majority_label]])
#     y_cleaned = np.concatenate([y[y == majority_label][keep_mask], y[y != majority_label]])
#
#     print(f"[KNN降采样] 原始多数类样本数: {len(X_major)}，保留: {sum(keep_mask)}")
#     return X_cleaned, y_cleaned
#
#
# def tomek_reduction(X, y, majority_label=0):
#     X_major = X[y == majority_label]
#     X_minor = X[y != majority_label]
#     # 记录多数类样本在原始X中的全局索引
#     major_global_indices = np.where(y == majority_label)[0]  # 关键：保存全局索引
#
#     neigh = NearestNeighbors(n_neighbors=1).fit(X)
#     distances, indices = neigh.kneighbors(X_major)  # indices是全局索引
#
#     tomek_links = []
#     for i in range(len(X_major)):
#         neighbor_global_idx = indices[i][0]  # 多数类样本i的最近邻的全局索引
#         if y[neighbor_global_idx] != majority_label:  # 最近邻是少数类
#             # 检查少数类样本的最近邻是否是当前多数类样本（全局索引比较）
#             minor_neigh_idx = neigh.kneighbors([X[neighbor_global_idx]], n_neighbors=1)[1][0][0]
#             if minor_neigh_idx == major_global_indices[i]:  # 用全局索引比较
#                 tomek_links.append(i)  # i是X_major中的局部索引，可用于筛选
#     # 后续逻辑不变...
#
#     # 移除Tomek Links中的多数类样本
#     keep_mask = np.ones(len(X_major), dtype=bool)
#     keep_mask[tomek_links] = False
#
#     # 构建新数据集
#     X_cleaned = np.concatenate([X_major[keep_mask], X_minor])
#     y_cleaned = np.concatenate([y[y == majority_label][keep_mask], y[y != majority_label]])
#
#     print(f"[Tomek降采样] 找到 {len(tomek_links)} 个Tomek links，移除对应多数类样本")
#     return X_cleaned, y_cleaned
#
#
# # 优化点4：添加交叉验证评估
# def cross_validate_model_with_reduction(X, y, model, reduction_method='tomek', n_splits=5):
#     skf = StratifiedKFold(n_splits=n_splits)
#     metrics = {'precision': [], 'recall': [], 'f1': [], 'mcc': [], 'auc': []}
#
#     for train_idx, val_idx in skf.split(X, y):
#         X_tr, X_val = X[train_idx], X[val_idx]
#         y_tr, y_val = y[train_idx], y[val_idx]
#
#         # 仅对训练fold降采样（关键：避免数据泄露）
#         X_tr_clean, y_tr_clean = reduce_majority_samples(X_tr, y_tr, method=reduction_method)
#
#         model.fit(X_tr_clean, y_tr_clean)
#         probs = model.predict_proba(X_val)[:, 1]
#
#         # 使用最佳阈值而非固定0.5
#         precision, recall, thresholds = precision_recall_curve(y_val, probs)
#         f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
#         best_threshold_idx = np.argmax(f1_scores)
#         best_threshold = thresholds[best_threshold_idx]
#
#         y_pred = (probs >= best_threshold).astype(int)
#
#         metrics['precision'].append(precision_score(y_val, y_pred, zero_division=0))
#         metrics['recall'].append(recall_score(y_val, y_pred, zero_division=0))
#         metrics['f1'].append(f1_score(y_val, y_pred, zero_division=0))
#         metrics['mcc'].append(matthews_corrcoef(y_val, y_pred))
#         metrics['auc'].append(roc_auc_score(y_val, probs))
#
#     # 计算平均指标
#     avg_metrics = {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
#     print("\n交叉验证性能评估:")
#     for metric, (mean, std) in avg_metrics.items():
#         print(f"{metric.capitalize()}: {mean:.4f} ± {std:.4f}")
#
#     return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
#
#
# # 优化点5：改进阈值分析图表
# def plot_metrics_vs_threshold(model, X_test, y_test, save_path=None):
#     """生成并保存阈值-指标图表"""
#     probs = model.predict_proba(X_test)[:, 1]
#     thresholds = np.linspace(0.05, 0.95, 50)
#
#     precisions, recalls, f1s, mccs = [], [], [], []
#
#     for t in thresholds:
#         y_pred = (probs >= t).astype(int)
#         precisions.append(precision_score(y_test, y_pred, zero_division=0))
#         recalls.append(recall_score(y_test, y_pred, zero_division=0))
#         f1s.append(f1_score(y_test, y_pred, zero_division=0))
#         mccs.append(matthews_corrcoef(y_test, y_pred))
#
#     # 计算并标记最佳F1阈值
#     best_f1_idx = np.argmax(f1s)
#     best_threshold = thresholds[best_f1_idx]
#
#     # 创建图表
#     plt.figure(figsize=(12, 8))
#
#     # 主指标曲线
#     plt.plot(thresholds, precisions, 'b-', lw=2, label='Precision')
#     plt.plot(thresholds, recalls, 'g-', lw=2, label='Recall')
#     plt.plot(thresholds, f1s, 'r-', lw=2, label='F1 Score')
#     plt.plot(thresholds, mccs, 'm-', lw=2, label='MCC')
#
#     # 最佳阈值标记
#     plt.axvline(x=best_threshold, color='k', linestyle='--',
#                 label=f'Best F1 Threshold ({best_threshold:.2f})')
#
#     # 添加PR曲线插槽
#     precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
#     plt.text(0.05, 0.05, f"PR-AUC: {auc(recall_curve, precision_curve):.3f}",
#              transform=plt.gca().transAxes, fontsize=12)
#
#     plt.xlabel('Classification Threshold', fontsize=12)
#     plt.ylabel('Score', fontsize=12)
#     plt.title('Metrics vs Classification Threshold', fontsize=14)
#     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300)
#         print(f"阈值分析图已保存至: {save_path}")
#     else:
#         plt.show()
#     plt.close()
#
#
# # 优化点6：添加模型保存和加载功能
# def save_model(model, path="model.pkl"):
#     joblib.dump(model, path)
#     print(f"模型已保存至: {path}")
#
#
# def load_model(path="model.pkl"):
#     if os.path.exists(path):
#         print(f"加载已保存的模型: {path}")
#         return joblib.load(path)
#     return None
#
#
# # 优化点7：完整性能报告
# def evaluate_model(model, X_test, y_test, threshold=None):
#     """综合评估模型性能并返回最佳阈值"""
#     probs = model.predict_proba(X_test)[:, 1]
#
#     # 确定最佳阈值
#     if threshold is None:
#         precision, recall, thresholds = precision_recall_curve(y_test, probs)
#         f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
#         best_idx = np.argmax(f1_scores)
#         threshold = thresholds[best_idx]
#
#     y_pred = (probs >= threshold).astype(int)
#
#     # 计算所有指标
#     results = {
#         'threshold': threshold,
#         'precision': precision_score(y_test, y_pred),
#         'recall': recall_score(y_test, y_pred),
#         'f1': f1_score(y_test, y_pred),
#         'mcc': matthews_corrcoef(y_test, y_pred),
#         'auc_roc': roc_auc_score(y_test, probs),
#         'confusion_matrix': pd.crosstab(y_test, y_pred,
#                                         rownames=['Actual'],
#                                         colnames=['Predicted'])
#     }
#
#     # 打印报告
#     print("\n" + "=" * 50)
#     print("模型性能评估报告")
#     print("=" * 50)
#     print(f"最佳阈值: {threshold:.4f}")
#     print(f"精确率: {results['precision']:.4f}")
#     print(f"召回率: {results['recall']:.4f}")
#     print(f"F1分数: {results['f1']:.4f}")
#     print(f"Matthews系数: {results['mcc']:.4f}")
#     print(f"AUC-ROC: {results['auc_roc']:.4f}")
#     print("\n混淆矩阵:")
#     print(results['confusion_matrix'])
#
#     return results
#
#
# def main():
#     # 1. 加载数据
#     start_time = time.time()
#     X, y = load_data()
#     print(f"数据集大小: {y}")
#     print(f"数据加载完成，用时: {time.time() - start_time:.2f}秒")
#
#     # 显示类别分布
#     class_dist = Counter(y)
#     print(f"\n类别分布: 多数类({class_dist[0]}) vs 少数类({class_dist[1]})")
#
#     # 2. 划分训练测试集
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, stratify=y, random_state=42
#     )
#     print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
#
#     # 3. 降采样方法选择
#     reduction_method = 'knn'  # 可选项: 'knn' 或 'tomek'
#     X_clean, y_clean = reduce_majority_samples(
#         X_train, y_train,
#         method=reduction_method,
#         majority_label=0,
#         keep_ratio=0.7
#     )
#     print("降采样后类别分布:", Counter(y_clean))
#
#     # 4. 训练随机森林
#     rf = RandomForestClassifier(
#         n_estimators=150,  # 增加树的数量
#         class_weight='balanced',  # 使用内置平衡权重
#         max_depth=10,  # 限制深度防止过拟合
#         random_state=42,
#         n_jobs=-1  # 使用所有CPU核心
#     )
#
#     # 使用交叉验证评估
#     cv_metrics = cross_validate_model_with_reduction(X_train, y_train, rf, reduction_method='tomek')
#
#     # 训练最终模型
#     rf.fit(X_clean, y_clean)
#
#     # 5. 在测试集上评估
#     test_results = evaluate_model(rf, X_test, y_test)
#
#     # 6. 可视化分析
#     plot_metrics_vs_threshold(
#         rf,
#         X_test,
#         y_test,
#         save_path="threshold_analysis.png"
#     )
#
#     # 7. 保存模型
#     save_model(rf, "rf_model.pkl")
#
#
# if __name__ == "__main__":
#     main()
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score,
                             precision_recall_curve, auc)
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import joblib
import time
import os


def load_data(cache_path="data_cache.pkl"):
    """加载本地数据集并添加缓存机制"""
    # # 如果缓存存在直接加载
    # if os.path.exists(cache_path):
    #     print("Loading data from cache...")
    #     return joblib.load(cache_path)

    print("Loading and processing data...")
    # 1. 加载原始数据
    df = pd.read_csv(r"G:\pycharm\lutrs\code\data\ant-1.3.csv")

    # 2. 删除前三列字符串列
    print(f"Original columns: {list(df.columns)}")
    columns_to_drop = df.columns[:3].tolist()
    df = df.drop(columns=columns_to_drop)
    print(f"Dropped string columns: {columns_to_drop}")
    print(f"Remaining columns: {list(df.columns)}")

    # 3. 分离特征和标签（假设最后一列为目标变量）
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = (y > 0).astype(int)
    # 4. 处理缺失值（使用列均值填充）
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    print("Missing values imputed.")

    # 5. 特征归一化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("Features normalized.")

    # 6. 保存处理后的数据到缓存
    joblib.dump((X, y), cache_path)
    print(f"Data cached at {cache_path}")

    return X, y


def reduce_majority_samples(X, y, method='knn', majority_label=0, k=5, keep_ratio=0.7):
    """降低多数类样本数量"""
    if method == 'knn':
        return enhanced_knn_reduction(X, y, majority_label, k, keep_ratio)
    elif method == 'tomek':
        return tomek_reduction(X, y, majority_label)
    else:
        raise ValueError(f"未知的降采样方法: {method}")


# def knn_reduction(X, y, majority_label=0, k=5, keep_ratio=0.7):
#     """基于KNN的多数类降采样"""
#     X_major = X[y == majority_label]
#
#     neigh = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X_major)
#     dists, _ = neigh.kneighbors(X_major)
#
#     avg_dists = np.mean(dists[:, 1:], axis=1)
#     threshold = np.percentile(avg_dists, keep_ratio * 100)
#     keep_mask = avg_dists <= threshold
#
#     X_cleaned = np.concatenate([X_major[keep_mask], X[y != majority_label]])
#     y_cleaned = np.concatenate([y[y == majority_label][keep_mask], y[y != majority_label]])
#
#     print(f"[KNN降采样] 原始多数类样本数: {len(X_major)}，保留: {sum(keep_mask)}")
#     return X_cleaned, y_cleaned


def enhanced_knn_reduction(X, y, majority_label=0, k=5, keep_ratio=0.7, reduction_type="border"):
    """
    改进的KNN降采样方法，可选择删除边界或密集区域样本

    参数:
    X: 特征矩阵
    y: 标签数组
    majority_label: 多数类标签(默认0)
    k: KNN邻居数(默认5)
    keep_ratio: 保留的多数类样本比例(默认0.7)
    reduction_type: 降采样类型
        "border" - 删除边界样本(默认)
        "dense" - 删除密集区域样本

    返回:
    X_cleaned: 降采样后的特征矩阵
    y_cleaned: 降采样后的标签数组
    """
    # 分离多数类和少数类
    X_major = X[y == majority_label]
    X_minor = X[y != majority_label]
    y_minor = y[y != majority_label]

    # 创建KNN模型
    neigh = NearestNeighbors(n_neighbors=k + 1).fit(X_major)

    # 计算每个多数类样本的k近邻距离
    distances, _ = neigh.kneighbors(X_major)
    avg_distances = np.mean(distances[:, 1:], axis=1)  # 排除自身距离

    # 计算到最近少数类样本的距离
    if len(X_minor) > 0:
        minority_neigh = NearestNeighbors(n_neighbors=1).fit(X_minor)
        min_dist_to_minority, _ = minority_neigh.kneighbors(X_major)
        min_dist_to_minority = min_dist_to_minority.flatten()
    else:
        min_dist_to_minority = np.full(len(X_major), np.inf)

    # 计算边界系数:平均距离 / 最近少数类距离
    # 值越大表示越接近决策边界
    boundary_scores = avg_distances / (min_dist_to_minority + 1e-10)

    # 计算密度系数:平均距离的倒数
    density_scores = 1 / (avg_distances + 1e-10)

    # 根据降采样类型选择评分指标
    if reduction_type == "dense":
        scores = density_scores  # 密集区域有高分
    else:
        scores = boundary_scores  # 边界区域有高分

    # 确定保留阈值
    threshold = np.percentile(scores, (1 - keep_ratio) * 100)

    # 创建保留掩码
    if reduction_type == "dense":
        keep_mask = scores <= threshold  # 保留低密度区域样本
    else:
        keep_mask = scores <= threshold  # 保留非边界区域样本

    # 可视化评分分布
    # visualize_distribution(scores, reduction_type, threshold, keep_mask)

    # 组合保留的多数类和所有少数类
    X_cleaned = np.concatenate([X_major[keep_mask], X_minor])
    y_cleaned = np.concatenate([np.full(sum(keep_mask), majority_label), y_minor])

    # 打印统计信息
    original_major_count = len(X_major)
    kept_major_count = sum(keep_mask)
    print(f"[改进KNN降采样] 类型: {reduction_type}")
    print(f"  原始多数类样本数: {original_major_count}")
    print(f"  保留多数类样本数: {kept_major_count} ({kept_major_count / original_major_count:.1%})")
    print(f"  保留少数类样本数: {len(X_minor)}")
    print(f"  总样本保留比例: {len(X_cleaned) / len(X):.1%}")

    return X_cleaned, y_cleaned
def tomek_reduction(X, y, majority_label=0):
    X_major = X[y == majority_label]
    X_minor = X[y != majority_label]

    neigh = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X)
    distances, indices = neigh.kneighbors(X_major)

    tomek_links = []
    for i, (x, idx_arr) in enumerate(zip(X_major, indices)):
        neighbor_idx = idx_arr[0]
        if y[neighbor_idx] != majority_label:
            neigh_check = neigh.kneighbors([X[neighbor_idx]], n_neighbors=1)
            if neigh_check[1][0][0] == i:
                tomek_links.append(i)

    keep_mask = np.ones(len(X_major), dtype=bool)
    keep_mask[tomek_links] = False

    X_cleaned = np.concatenate([X_major[keep_mask], X_minor])
    y_cleaned = np.concatenate([y[y == majority_label][keep_mask], y[y != majority_label]])

    print(f"[Tomek降采样] 找到 {len(tomek_links)} 个Tomek links，移除对应多数类样本")
    return X_cleaned, y_cleaned


def cross_validate_knn_with_reduction(X, y, reduction_method='tomek', n_splits=5, k_values=[3, 5, 7, 9]):
    """KNN交叉验证与超参数调优"""
    skf = StratifiedKFold(n_splits=n_splits)
    all_metrics = {}
    best_k_by_fold = []

    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # 降采样
        X_tr_clean, y_tr_clean = reduce_majority_samples(
            X_tr, y_tr, method=reduction_method
        )

        # 特征缩放
        scaler = StandardScaler().fit(X_tr_clean)
        X_tr_scaled = scaler.transform(X_tr_clean)
        X_val_scaled = scaler.transform(X_val)

        best_fold_metrics = {}
        best_f1 = -1
        best_k_fold = None

        # 在当前折叠中评估所有k值
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k).fit(X_tr_scaled, y_tr_clean)

            # 预测概率
            probs = knn.predict_proba(X_val_scaled)[:, 1]

            # 确定最佳阈值
            precision, recall, thresholds = precision_recall_curve(y_val, probs)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_threshold_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_threshold_idx]

            y_pred = (probs >= best_threshold).astype(int)

            metrics = {
                'precision': precision_score(y_val, y_pred, zero_division=0),
                'recall': recall_score(y_val, y_pred, zero_division=0),
                'f1': f1_score(y_val, y_pred, zero_division=0),
                'mcc': matthews_corrcoef(y_val, y_pred),
                'auc': roc_auc_score(y_val, probs)
            }

            best_fold_metrics[k] = metrics

            # 追踪当前折叠中的最佳k值
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_k_fold = k

        # 保存当前折叠的最佳k值
        best_k_by_fold.append(best_k_fold)

        # 聚合所有k值的指标
        for k in k_values:
            metrics = best_fold_metrics[k]
            if k not in all_metrics:
                all_metrics[k] = {m: [] for m in metrics.keys()}

            for metric, value in metrics.items():
                all_metrics[k][metric].append(value)

    # 计算每个k值的平均指标
    avg_metrics = {}
    for k, metrics in all_metrics.items():
        avg_metrics[k] = {m: (np.mean(values), np.std(values)) for m, values in metrics.items()}

    # 打印所有k值的性能
    print("\nK值性能对比:")
    for k, metrics in avg_metrics.items():
        print(f"\nk={k}:")
        for metric, (mean, std) in metrics.items():
            print(f"  {metric.capitalize()}: {mean:.4f} ± {std:.4f}")

    # 选择最常见的最佳k值
    best_k_counter = Counter(best_k_by_fold)
    best_k = best_k_counter.most_common(1)[0][0]
    print(f"\n最佳K值: {best_k} (基于折叠中最佳频率)")

    return avg_metrics, best_k


def plot_metrics_vs_threshold_knn(model, X_test, y_test, save_path=None):
    """生成KNN的阈值-指标图表"""
    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.05, 0.95, 50)

    precisions, recalls, f1s, mccs = [], [], [], []

    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        mccs.append(matthews_corrcoef(y_test, y_pred))

    # 计算并标记最佳F1阈值
    best_f1_idx = np.argmax(f1s)
    best_threshold = thresholds[best_f1_idx]

    # 创建图表
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precisions, 'b-', lw=2, label='Precision')
    plt.plot(thresholds, recalls, 'g-', lw=2, label='Recall')
    plt.plot(thresholds, f1s, 'r-', lw=2, label='F1 Score')
    plt.plot(thresholds, mccs, 'm-', lw=2, label='MCC')
    plt.axvline(x=best_threshold, color='k', linestyle='--',
                label=f'Best F1 Threshold ({best_threshold:.2f})')
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
    plt.text(0.05, 0.05, f"PR-AUC: {auc(recall_curve, precision_curve):.3f}",
             transform=plt.gca().transAxes, fontsize=12)

    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('KNN Metrics vs Classification Threshold', fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"阈值分析图已保存至: {save_path}")
    else:
        plt.show()
    plt.close()


def evaluate_knn_model(model, X_test, y_test, threshold=None):
    """评估KNN模型性能"""
    probs = model.predict_proba(X_test)[:, 1]

    if threshold is None:
        precision, recall, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        threshold = thresholds[best_idx]

    y_pred = (probs >= threshold).astype(int)

    results = {
        'threshold': threshold,
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, probs),
        'confusion_matrix': pd.crosstab(y_test, y_pred,
                                        rownames=['Actual'],
                                        colnames=['Predicted'])
    }

    print("\n" + "=" * 50)
    print("KNN模型性能评估报告")
    print("=" * 50)
    print(f"最佳阈值: {threshold:.4f}")
    print(f"精确率: {results['precision']:.4f}")
    print(f"召回率: {results['recall']:.4f}")
    print(f"F1分数: {results['f1']:.4f}")
    print(f"Matthews系数: {results['mcc']:.4f}")
    print(f"AUC-ROC: {results['auc_roc']:.4f}")
    print("\n混淆矩阵:")
    print(results['confusion_matrix'])

    return results


def main_knn():
    # 1. 加载数据
    start_time = time.time()
    X, y = load_data()
    print(f"数据加载完成，用时: {time.time() - start_time:.2f}秒")

    # 显示类别分布
    class_dist = Counter(y)
    print(f"\n类别分布: 多数类({class_dist[0]}) vs 少数类({class_dist[1]})")

    # 2. 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    # 3. 使用交叉验证选择最佳K值
    cv_metrics, best_k = cross_validate_knn_with_reduction(
        X_train, y_train,
        reduction_method='knn',  # 可改为'knn'
        k_values=[3, 5, 7, 9, 11, 13]
    )

    # 4. 在完整训练集上训练最佳KNN模型
    X_clean, y_clean = reduce_majority_samples(
        X_train, y_train,
        method='knn'
    )

    # 特征缩放
    scaler = StandardScaler().fit(X_clean)
    X_clean_scaled = scaler.transform(X_clean)
    X_test_scaled = scaler.transform(X_test)

    # 训练最终模型
    knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_clean_scaled, y_clean)

    # 5. 评估模型
    test_results = evaluate_knn_model(knn, X_test_scaled, y_test,0.5)

    # 6. 可视化分析
    plot_metrics_vs_threshold_knn(
        knn,
        X_test_scaled,
        y_test,
        save_path="knn_threshold_analysis.png"
    )

    # 7. 保存模型
    save_path = f"knn_model_k{best_k}.pkl"
    joblib.dump((knn, scaler), save_path)
    print(f"KNN模型和标准化器已保存至: {save_path}")


if __name__ == "__main__":
    main_knn()

