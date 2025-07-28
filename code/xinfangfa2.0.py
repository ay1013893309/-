import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from collections import Counter


# 画阈值 vs 指标曲线
def plot_metrics_vs_threshold(model, X_test, y_test):
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
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1s, label='F1 Score')
    plt.plot(thresholds, mccs, label='MCC')

    plt.xlabel('Classification Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Classification Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()


# 加载数据
def load_data():
    df = pd.read_csv(r"G:\pycharm\lutrs\code\data\ant-1.3.csv")
    y_raw = df['bug'].values
    y = (y_raw > 0).astype(int)
    df = df.drop(columns=['name', 'version', 'name1', 'bug'], errors='ignore')
    X = df.select_dtypes(include=['int64', 'float64']).values
    return X, y


# 只去多数类冗余样本，少数类不变
def remove_redundant_samples_majority_only(X, y, majority_label=0, k=5, keep_ratio=0.7):
    X_major = X[y == majority_label]
    neigh = NearestNeighbors(n_neighbors=k).fit(X_major)
    dists, _ = neigh.kneighbors(X_major)
    avg_dists = np.mean(dists, axis=1)
    threshold = np.percentile(avg_dists, keep_ratio * 100)
    keep_mask = avg_dists <= threshold
    # 多数类去冗余，少数类全保留
    X_cleaned = np.concatenate([X_major[keep_mask], X[y != majority_label]])
    y_cleaned = np.concatenate([y[y == majority_label][keep_mask], y[y != majority_label]])
    return X_cleaned, y_cleaned


# 计算传统平衡类别权重
def compute_balanced_class_weights(y):
    counts = Counter(y)
    total = len(y)
    weights = {label: total / count for label, count in counts.items()}
    s = sum(weights.values())
    weights = {label: w / s for label, w in weights.items()}
    return weights


# 训练随机森林
def train_rf_with_weights(X_train, y_train, weights, n_estimators=100, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 class_weight=weights,
                                 random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


# 评估模型
def evaluate_model(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    print(f"Threshold = {threshold}")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, probs))


def main():
    # 1. 加载数据
    X, y = load_data()

    # 2. 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        stratify=y,
                                                        random_state=42)

    # 3. 只去多数类冗余样本
    X_clean, y_clean = remove_redundant_samples_majority_only(X_train, y_train,
                                                              majority_label=0,
                                                              k=5,
                                                              keep_ratio=0.7)

    print("去除多数类冗余后训练集类别分布:", Counter(y_clean))

    # 4. 计算平衡类别权重
    class_weights = compute_balanced_class_weights(y_clean)
    print("类别权重:", class_weights)

    # 5. 训练随机森林
    clf = train_rf_with_weights(X_clean, y_clean, class_weights,
                               n_estimators=100,
                               random_state=42)

    # 6. 评估模型，阈值0.5
    evaluate_model(clf, X_test, y_test, threshold=0.5)

    # 7. 绘制阈值与指标曲线
    plot_metrics_vs_threshold(clf, X_test, y_test)


if __name__ == "__main__":
    main()
