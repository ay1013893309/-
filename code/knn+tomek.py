import numpy as np
import pandas as pd
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

# 设置中文字体支持（如果需要显示中文）
try:
    # Windows系统
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    try:
        # Linux系统
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    except:
        # Mac系统
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']


def load_data(cache_path="data_cache.pkl"):
    """加载数据并添加缓存机制"""
    if os.path.exists(cache_path):
        print(f"从缓存加载数据: {cache_path}")
        return joblib.load(cache_path)

    # 使用您的实际数据集路径
    data_path = r"G:\pycharm\lutrs\code\datacunshu\ant-1.7.csv"
    print(f"从CSV文件加载数据: {data_path}")

    # 读取CSV文件
    df = pd.read_csv(data_path)

    # 提取标签列
    if 'bug' not in df.columns:
        raise ValueError("数据集中缺少'bug'列")

    y_raw = df['bug'].values
    y = (y_raw > 0).astype(int)  # 将bug>0的转换为1，否则为0

    # 移除不需要的列
    columns_to_drop = ['name', 'version', 'name1', 'bug']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=col)

    # 只保留数值型特征
    X = df.select_dtypes(include=['int64', 'float64']).values

    # 检查特征数量
    if X.shape[1] == 0:
        raise ValueError("未找到数值型特征列")

    # 添加数据清洗：处理缺失值
    if np.isnan(X).any():
        nan_count = np.isnan(X).sum()
        print(f"发现 {nan_count} 个缺失值，使用中位数填充")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)

    # 保存到缓存
    joblib.dump((X, y), cache_path)
    print(f"数据已缓存至: {cache_path}")
    print(f"数据集大小: {X.shape[0]}样本, {X.shape[1]}特征")
    print(f"类别分布: 多数类({sum(y == 0)}), 少数类({sum(y == 1)})")

    return X, y


def hybrid_reduction(X, y, majority_label=0, knn_k=5, knn_keep_ratio=0.7):
    """
    复合降采样策略：先使用KNN密度采样，再使用Tomek边界清洗
    """
    # 阶段1: KNN密度采样
    X_major = X[y == majority_label]
    X_minor = X[y != majority_label]

    # KNN密度采样
    neigh = NearestNeighbors(n_neighbors=knn_k + 1, algorithm='auto').fit(X_major)
    dists, _ = neigh.kneighbors(X_major)
    avg_dists = np.mean(dists[:, 1:], axis=1)  # 排除自身距离

    # 计算保留阈值（保留密度较高的样本）
    threshold = np.percentile(avg_dists, knn_keep_ratio * 100)
    keep_mask = avg_dists <= threshold

    # 创建KNN采样后的数据集
    X_knn = np.concatenate([X_major[keep_mask], X_minor])
    y_knn = np.concatenate([np.full(np.sum(keep_mask), majority_label),
                            y[y != majority_label]])

    print(f"[KNN降采样] 原始多数类: {len(X_major)}, 保留: {np.sum(keep_mask)}, "
          f"少数类: {len(X_minor)}")

    # 阶段2: Tomek边界清洗 - 修复索引问题
    # 获取全局索引映射
    global_indices = np.arange(len(X_knn))
    major_indices = global_indices[y_knn == majority_label]

    # 构建全局KNN模型
    neigh = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_knn)

    # 识别Tomek链接
    tomek_links = []
    for local_idx, x in enumerate(X_knn[y_knn == majority_label]):
        # 获取当前多数类样本的全局索引
        global_idx = major_indices[local_idx]

        # 查找最近邻
        _, neighbor_idxs = neigh.kneighbors([x])
        neighbor_global_idx = neighbor_idxs[0][0]

        # 确保邻居是少数类样本
        if y_knn[neighbor_global_idx] != majority_label:
            # 验证双向关系
            _, reverse_idxs = neigh.kneighbors([X_knn[neighbor_global_idx]])
            reverse_global_idx = reverse_idxs[0][0]

            # 使用全局索引比较
            if reverse_global_idx == global_idx:
                tomek_links.append(local_idx)
                print(f"发现Tomek链接: 样本{global_idx} <-> 样本{neighbor_global_idx}")

    # 移除边界样本
    tomek_keep_mask = np.ones(len(X_knn[y_knn == majority_label]), dtype=bool)
    tomek_keep_mask[tomek_links] = False

    # 重构最终数据集
    X_clean = np.concatenate([
        X_knn[y_knn == majority_label][tomek_keep_mask],
        X_knn[y_knn != majority_label]
    ])
    y_clean = np.concatenate([
        np.full(np.sum(tomek_keep_mask), majority_label),
        y_knn[y_knn != majority_label]
    ])

    print(f"[Tomek清洗] 移除 {len(tomek_links)} 个边界样本, "
          f"最终多数类: {np.sum(tomek_keep_mask)}, 少数类: {len(X_knn[y_knn != majority_label])}")
    return X_clean, y_clean


def cross_validate_knn_with_reduction(X, y, reduction_method='hybrid', n_splits=5, k_values=[3, 5, 7, 9]):
    """KNN交叉验证与超参数调优"""
    skf = StratifiedKFold(n_splits=n_splits)
    all_metrics = {}
    best_k_by_fold = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # 降采样
        if reduction_method == 'hybrid':
            X_tr_clean, y_tr_clean = hybrid_reduction(X_tr, y_tr)
        else:
            raise ValueError(f"不支持的降采样方法: {reduction_method}")

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
        print(f"Fold {fold + 1}: Best k = {best_k_fold}, F1 = {best_f1:.4f}")

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
    print("\nPerformance comparison by k value:")
    for k, metrics in avg_metrics.items():
        print(f"\nk={k}:")
        for metric, (mean, std) in metrics.items():
            print(f"  {metric.capitalize()}: {mean:.4f} ± {std:.4f}")

    # 选择最常见的最佳k值
    best_k_counter = Counter(best_k_by_fold)
    best_k = best_k_counter.most_common(1)[0][0]
    print(f"\nBest K value: {best_k} (based on fold frequency)")

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

    # 计算PR曲线下面积
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall_curve, precision_curve)
    plt.text(0.05, 0.05, f"PR-AUC: {pr_auc:.3f}",
             transform=plt.gca().transAxes, fontsize=12)

    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('KNN Metrics vs Classification Threshold', fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Threshold analysis plot saved to: {save_path}")
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
    print("KNN Model Performance Evaluation")
    print("=" * 50)
    print(f"Optimal threshold: {threshold:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")
    print(f"Matthews coefficient: {results['mcc']:.4f}")
    print(f"AUC-ROC: {results['auc_roc']:.4f}")
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

    return results


def visualize_sampling_comparison(X_orig, y_orig, X_sampled, y_sampled, title="Sampling Effect"):
    """
    可视化采样前后数据分布（使用英文标签）
    """
    plt.figure(figsize=(15, 6))

    # 原始数据分布
    plt.subplot(121)
    plt.scatter(X_orig[:, 0], X_orig[:, 1], c=y_orig, alpha=0.4,
                cmap='coolwarm', edgecolor='k')
    plt.title(f'Original Data (Majority: {sum(y_orig == 0)}, Minority: {sum(y_orig == 1)})')

    # 采样后分布
    plt.subplot(122)
    plt.scatter(X_sampled[:, 0], X_sampled[:, 1], c=y_sampled, alpha=0.7,
                cmap='coolwarm', edgecolor='k')
    plt.title(f'{title} (Majority: {sum(y_sampled == 0)}, Minority: {sum(y_sampled == 1)})')

    plt.tight_layout()
    plt.show()


def main_knn():
    # 1. 加载数据
    start_time = time.time()
    X, y = load_data()
    print(f"Data loaded in: {time.time() - start_time:.2f} seconds")

    # 显示类别分布
    class_dist = Counter(y)
    print(f"\nClass distribution: Majority({class_dist[0]}) vs Minority({class_dist[1]})")

    # 可视化原始数据分布
    visualize_sampling_comparison(X, y, X, y, "Original Data")

    # 2. 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # 3. 使用交叉验证选择最佳K值
    cv_metrics, best_k = cross_validate_knn_with_reduction(
        X_train, y_train,
        reduction_method='hybrid',
        n_splits=5,
        k_values=[3, 5, 7, 9, 11, 13]
    )

    # 4. 在完整训练集上应用复合降采样
    X_clean, y_clean = hybrid_reduction(X_train, y_train)

    # 可视化降采样效果
    visualize_sampling_comparison(X_train, y_train, X_clean, y_clean, "After Hybrid Reduction")

    # 特征缩放
    scaler = StandardScaler().fit(X_clean)
    X_clean_scaled = scaler.transform(X_clean)
    X_test_scaled = scaler.transform(X_test)

    # 训练最终模型
    knn = KNeighborsClassifier(n_neighbors=best_k).fit(X_clean_scaled, y_clean)

    # 5. 评估模型
    test_results = evaluate_knn_model(knn, X_test_scaled, y_test)

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
    print(f"KNN model and scaler saved to: {save_path}")


if __name__ == "__main__":
    main_knn()