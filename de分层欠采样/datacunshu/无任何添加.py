import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score, \
    confusion_matrix, matthews_corrcoef
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
import matplotlib

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


def preprocess_data(X, y):
    """数据预处理"""
    # 处理缺失值 - 使用中位数填充
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # 标准化
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns)

    return X_scaled, y


def evaluate_model(model, X_test, y_test):
    """评估模型性能并返回指标字典"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'auc_pr': average_precision_score(y_test, y_proba) if y_proba is not None else 0,
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 0,
        'f1': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    return metrics


def plot_class_distribution(y, title_suffix="", save_path=None):
    """可视化类别分布"""
    class_counts = y.value_counts()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')

    # 添加数值标签
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
                f"Overall Performance: MCC = {metrics['mcc']:.4f}, AUC-PR = {metrics['auc_pr']:.4f}",
                ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.3, "pad": 5})

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def baseline_model_evaluation(file_path, target_column='bug', n_runs=5):
    """
    在原始不平衡数据上评估模型性能（基准方法）

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
    plot_class_distribution(y, "(Original)", "original_class_distribution.png")

    # 2. 数据预处理
    print("\nPreprocessing data...")
    # 确保所有特征是数值型
    X = X.apply(pd.to_numeric, errors='coerce')
    X_preprocessed, y = preprocess_data(X, y)

    # 存储结果
    all_metrics = {
        'auc_pr': [],
        'roc_auc': [],
        'f1': [],
        'recall': [],
        'precision': [],
        'mcc': []
    }

    print(f"\nStarting baseline model evaluation with {n_runs} runs...")
    for run in range(n_runs):
        print(f"\n=== Run {run + 1}/{n_runs} ===")

        # 使用不同的随机种子确保结果可重复
        random_state = 42 + run

        # 分割数据集 (80% 训练, 20% 测试)
        X_train, X_test, y_train, y_test = train_test_split(
            X_preprocessed, y,
            test_size=0.2,
            stratify=y,  # 保持类别分布
            random_state=random_state
        )

        # 检查目标列是否有足够的类别
        if len(np.unique(y_train)) < 2:
            print("Warning: Only one class present in training data. Skipping run.")
            continue

        # 3. 训练模型 (使用随机森林)
        print("Training Random Forest model on original imbalanced data...")
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            class_weight='balanced',  # 考虑类别不平衡
            n_jobs=-1  # 使用所有CPU核心
        )
        model.fit(X_train, y_train)

        # 4. 评估模型
        metrics = evaluate_model(model, X_test, y_test)

        # 保存指标
        for key in all_metrics:
            if key in metrics:
                all_metrics[key].append(metrics[key])

        # 5. 可视化结果 (仅第一次运行时)
        if run == 0:
            plot_confusion_matrix(metrics['confusion_matrix'], "(Run 1)", "baseline_confusion_matrix_run1.png")
            plot_metrics_comparison(metrics, "baseline_metrics_run1.png")

            # 计算特征重要性
            feature_importances = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)

            # 可视化前15个重要特征
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature',
                        data=feature_importances.head(15),
                        palette='viridis')
            plt.title('Top 15 Important Features (Baseline Model)')
            plt.tight_layout()
            plt.savefig('baseline_feature_importance.png', dpi=300)
            plt.show()

    # 计算平均指标
    avg_metrics = {f'avg_{key}': np.mean(values) for key, values in all_metrics.items()}
    std_metrics = {f'std_{key}': np.std(values) for key, values in all_metrics.items()}

    # 创建结果数据框
    results_df = pd.DataFrame({
        'Metric': list(all_metrics.keys()),
        'Mean': [np.mean(all_metrics[key]) for key in all_metrics],
        'StdDev': [np.std(all_metrics[key]) for key in all_metrics]
    })

    # 添加性能总结
    best_recall_run = np.argmax(all_metrics['recall'])
    worst_recall_run = np.argmin(all_metrics['recall'])

    summary = {
        'Dataset': file_path,
        'Total Runs': n_runs,
        'Average MCC': results_df.loc[results_df['Metric'] == 'mcc', 'Mean'].values[0],
        'Average Recall': results_df.loc[results_df['Metric'] == 'recall', 'Mean'].values[0],
        'Best Recall (Run)': f"{best_recall_run + 1}: {all_metrics['recall'][best_recall_run]:.4f}",
        'Worst Recall (Run)': f"{worst_recall_run + 1}: {all_metrics['recall'][worst_recall_run]:.4f}",
        'Recall Variation': f"{min(all_metrics['recall']):.4f}-{max(all_metrics['recall']):.4f}",
        'Average AUC-PR': results_df.loc[results_df['Metric'] == 'auc_pr', 'Mean'].values[0]
    }

    # 打印结果
    print("\n" + "=" * 80)
    print("Baseline Model Evaluation Results (Imbalanced Data)")
    print("=" * 80)
    print(f"Dataset: {file_path}")
    print(f"Total runs: {n_runs}")

    # 修改后的性能指标输出
    print("\nPerformance Metrics Summary:")
    print(f"Precision: {results_df.loc[results_df['Metric'] == 'precision', 'Mean'].values[0]:.4f}")
    print(f"Recall: {results_df.loc[results_df['Metric'] == 'recall', 'Mean'].values[0]:.4f}")
    print(f"F1 Score: {results_df.loc[results_df['Metric'] == 'f1', 'Mean'].values[0]:.4f}")
    print(f"MCC: {results_df.loc[results_df['Metric'] == 'mcc', 'Mean'].values[0]:.4f}")
    print(f"AUC: {results_df.loc[results_df['Metric'] == 'roc_auc', 'Mean'].values[0]:.4f}")

    print("\nKey Findings:")
    print(f"- Average Matthews Correlation Coefficient: {summary['Average MCC']:.4f}")
    print(f"- Average Recall: {summary['Average Recall']:.4f}")
    print(f"- Recall range across runs: {summary['Recall Variation']}")
    print(f"- Best recall in run {best_recall_run + 1}: {all_metrics['recall'][best_recall_run]:.4f}")
    print(f"- Worst recall in run {worst_recall_run + 1}: {all_metrics['recall'][worst_recall_run]:.4f}")
    print("=" * 80)

    # 可视化多轮实验结果
    plt.figure(figsize=(14, 8))
    for i, metric in enumerate(['recall', 'mcc', 'auc_pr']):
        plt.subplot(2, 2, i + 1)
        plt.plot(range(1, n_runs + 1), all_metrics[metric], 'o-', linewidth=2)
        plt.title(f'{metric.upper()} over Runs')
        plt.xlabel('Run')
        plt.ylabel(metric.upper())
        plt.ylim(0, 1)
        plt.grid(True)

        # 标注平均线
        avg = np.mean(all_metrics[metric])
        plt.axhline(avg, color='r', linestyle='--')
        plt.text(0.5, avg + 0.02, f'Avg: {avg:.4f}', color='r')

    plt.tight_layout()
    plt.savefig('baseline_performance_over_runs.png', dpi=300)
    plt.show()

    return summary


# ======================
# 使用示例
# ======================
if __name__ == "__main__":
    # 从CSV文件加载真实数据
    data_path = r"G:\pycharm\lutrs\code\datacunshu\ant-1.6.csv"  # 请替换为你的实际路径

    # 运行基准模型评估
    results = baseline_model_evaluation(file_path=data_path, n_runs=5)

    if results:
        print("\nBaseline evaluation completed successfully!")
        print(f"Average recall: {results['Average Recall']:.4f}")
    else:
        print("\nBaseline evaluation encountered an error. See above output for details.")