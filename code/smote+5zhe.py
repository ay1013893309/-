import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import time
import os

from sklearn.neighbors import KNeighborsClassifier

# 检查并创建种子存储文件
SEED_FILE = "last_seed.txt"


def get_next_seed():
    """获取下一个随机种子值，每次运行递增1"""
    try:
        if os.path.exists(SEED_FILE):
            with open(SEED_FILE, "r") as f:
                last_seed = int(f.read().strip())
        else:
            last_seed = 42  # 初始种子

        current_seed = last_seed + 1

        with open(SEED_FILE, "w") as f:
            f.write(str(current_seed))

        return current_seed
    except:
        return int(time.time()) % 10000  # 如果出错，使用时间作为种子


# 画阈值 vs 指标曲线
def plot_metrics_vs_threshold(model, X_test, y_test, fold_idx=None):
    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 1.0, 0.05)

    precisions, recalls, f1s, mccs = [], [], [], []

    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        mccs.append(matthews_corrcoef(y_test, y_pred))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, 'o-', label='Precision')
    plt.plot(thresholds, recalls, 's-', label='Recall')
    plt.plot(thresholds, f1s, 'd-', label='F1 Score')
    plt.plot(thresholds, mccs, '^-', label='MCC')

    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title(f'Metrics vs Classification Threshold (SMOTE + RF) - Fold {fold_idx + 1}')
    plt.legend()
    plt.grid(True)

    if fold_idx is not None:
        plt.savefig(f'metrics_vs_threshold_fold{fold_idx + 1}.png')
    else:
        plt.savefig('metrics_vs_threshold.png')
    plt.close()


# 加载数据
def load_data():
    df = pd.read_csv(r"D:\-\stability-of-smote-main\AEEM\converted_PDE.csv")
    y_raw = df['bug'].values
    y = (y_raw > 0).astype(int)
    df = df.drop(columns=['name', 'version', 'name1', 'bug'], errors='ignore')
    X = df.select_dtypes(include=['int64', 'float64']).values
    return X, y


def evaluate_model(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    metrics = {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'auc': roc_auc_score(y_test, probs),
        'accuracy': accuracy_score(y_test, y_pred)
    }

    print(f"\nThreshold = {threshold}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    return metrics


def run_fold(X_train, y_train, X_test, y_test, current_seed, fold_idx):
    """运行单个折的模型训练和评估"""
    print(f"\n===== Fold {fold_idx + 1}/5 (Seed: {current_seed}) =====")
    print(f"  Original fold size: {len(X_train)} train, {len(X_test)} test")
    print(f"  Class distribution (train):", {label: sum(y_train == label) for label in np.unique(y_train)})

    # 使用SMOTE过采样少数类
    smote = SMOTE(random_state=42,sampling_strategy=1)
    # 创建随机欠采样器
    # rus = RandomUnderSampler(random_state=42, sampling_strategy=1)

    # 应用欠采样
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE (train):", {label: sum(y_res == label) for label in np.unique(y_res)})

    # 训练随机森林
    # clf = RandomForestClassifier(
    #     n_estimators=100,
    #     random_state=42,
    #     n_jobs=-1,
    # )
    clf = KNeighborsClassifier(
        n_neighbors=5,  # 选择5个最近邻（默认值）
        # weights='uniform',  # 所有邻居权重相等
        # algorithm='auto',  # 自动选择最优算法（KDTree/BallTree/Brute）
        # leaf_size=30,  # 树结构叶子节点大小
        # p=2,  # 欧氏距离（p=2）
        # metric='minkowski',  # 闵可夫斯基距离（p=2时等价欧氏距离）
        n_jobs=-1  # 使用所有CPU核心并行计算
    )
    clf.fit(X_res, y_res)

    # 评估模型
    print("\nEvaluation on Test Set:")
    metrics = evaluate_model(clf, X_test, y_test)

    # 绘制阈值曲线
    plot_metrics_vs_threshold(clf, X_test, y_test, fold_idx)

    return metrics


def main():
    # 1. 获取下一个随机种子
    current_seed = get_next_seed()
    print(f"Starting run with seed: {current_seed}")

    # 2. 加载数据
    X, y = load_data()

    # 打印数据集基本信息
    print(f"\nDataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.sum(y == 0)} non-buggy, {np.sum(y == 1)} buggy")

    # 3. 创建分层五折交叉验证器
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 准备收集所有结果
    all_results = []
    fold_accuracies = []

    # 4. 进行五折交叉验证
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        fold_metrics = run_fold(X_train, y_train, X_test, y_test, current_seed, fold_idx)
        all_results.append(fold_metrics)
        fold_accuracies.append(fold_metrics['accuracy'])

    # 5. 计算并打印平均结果
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in all_results]),
        'recall': np.mean([m['recall'] for m in all_results]),
        'f1': np.mean([m['f1'] for m in all_results]),
        'mcc': np.mean([m['mcc'] for m in all_results]),
        'auc': np.mean([m['auc'] for m in all_results]),
        'accuracy': np.mean(fold_accuracies)
    }

    print("\n" + "=" * 50)
    print("Final Average Metrics from 5-Fold Cross-Validation:")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F1 Score: {avg_metrics['f1']:.4f}")
    print(f"MCC: {avg_metrics['mcc']:.4f}")
    print(f"AUC: {avg_metrics['auc']:.4f}")
    print(f"Accuracy: {avg_metrics['accuracy']:.4f}")
    print("=" * 50)

    # 保存平均结果到文件
    with open("smote_rf_results.txt", "w") as f:
        f.write("SMOTE + Random Forest Benchmark Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Random Seed: {current_seed}\n")
        f.write(f"Dataset: ant-1.3.csv\n")
        f.write(f"Total samples: {X.shape[0]}\n")
        f.write(f"Buggy samples: {np.sum(y == 1)}\n")
        f.write("\nAverage Metrics:\n")
        f.write(f"Precision: {avg_metrics['precision']:.4f}\n")
        f.write(f"Recall: {avg_metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {avg_metrics['f1']:.4f}\n")
        f.write(f"MCC: {avg_metrics['mcc']:.4f}\n")
        f.write(f"AUC: {avg_metrics['auc']:.4f}\n")
        f.write(f"Accuracy: {avg_metrics['accuracy']:.4f}\n")
        f.write("=" * 50 + "\n")

    print("\nResults saved to smote_rf_results.txt")


if __name__ == "__main__":
    main()