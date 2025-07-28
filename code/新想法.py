import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# ---------- Step 6b: 绘制阈值 vs 评价指标曲线 ----------
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


# ---------- Step 1: 加载数据 ----------
def load_data():
    df = pd.read_csv(r"G:\pycharm\lutrs\code\datacunshu\ant-1.3.csv")

    y_raw = df['bug'].values  # 标签列名
    y = (y_raw > 0).astype(int)  # 有缺陷=1，无缺陷=0

    # 删除非数值列，根据数据实际情况修改
    df = df.drop(columns=['name', 'version', 'name1', 'bug'], errors='ignore')
    X = df.select_dtypes(include=['int64', 'float64']).values
    return X, y


# ---------- Step 2: 去除多数类冗余样本 ----------
def remove_redundant_samples(X, y, majority_label=0, k=5, keep_ratio=0.7):
    X_major = X[y == majority_label]
    neigh = NearestNeighbors(n_neighbors=k).fit(X_major)
    dists, _ = neigh.kneighbors(X_major)
    avg_dists = np.mean(dists, axis=1)

    threshold = np.percentile(avg_dists, keep_ratio * 100)
    keep_mask = avg_dists <= threshold

    X_cleaned = np.concatenate([X_major[keep_mask], X[y != majority_label]])
    y_cleaned = np.concatenate([y[y == majority_label][keep_mask], y[y != majority_label]])
    return X_cleaned, y_cleaned


# ---------- Step 3: 类内分组去边缘样本 ----------
def remove_outliers_by_group(X, y, k=5, group_count=4):
    new_X, new_y = [], []
    final_ratios = {}

    for label in np.unique(y):
        X_c = X[y == label]
        n_samples = len(X_c)
        actual_k = min(k, n_samples - 1)
        if actual_k < 1:
            # 样本太少，全部保留
            new_X.append(X_c)
            new_y.append(np.full(len(X_c), label))
            final_ratios[label] = 1.0
            continue

        neigh = NearestNeighbors(n_neighbors=actual_k).fit(X_c)
        dists, _ = neigh.kneighbors(X_c)
        avg_dists = np.mean(dists, axis=1)

        bins = np.linspace(avg_dists.min(), avg_dists.max(), group_count + 1)
        group_ids = np.digitize(avg_dists, bins) - 1

        keep_mask = group_ids < (group_count - 1)
        X_c_filtered = X_c[keep_mask]

        new_X.append(X_c_filtered)
        new_y.append(np.full(len(X_c_filtered), label))
        final_ratios[label] = len(X_c_filtered) / len(X_c)

    X_new = np.vstack(new_X)
    y_new = np.concatenate(new_y)
    return X_new, y_new, final_ratios


# ---------- Step 4: 计算类别权重 ----------
def compute_class_weights_from_ratio(ratios, gamma=1.0):
    weights = {label: 1 / (ratio ** gamma) for label, ratio in ratios.items()}
    total = sum(weights.values())
    for key in weights:
        weights[key] /= total
    return weights


# ---------- Step 5: 训练随机森林 ----------
def train_rf_with_weights(X_train, y_train, weights, n_estimators=100, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 class_weight=weights,
                                 random_state=random_state)
    clf.fit(X_train, y_train)
    return clf


# ---------- Step 6: 评估 ----------
def evaluate_model(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    print(f"Threshold = {threshold}")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, probs))


# ---------- 主流程 ----------
def main():
    # Step 1: 加载数据
    X, y = load_data()

    # Step 2: 划分训练测试集，保持类别比例
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        stratify=y,
                                                        random_state=42)

    # Step 3: 去除多数类冗余样本
    X_clean, y_clean = remove_redundant_samples(X_train, y_train,
                                                majority_label=0,
                                                k=5,
                                                keep_ratio=0.7)

    # Step 4: 类内分组去除边缘样本
    X_final, y_final, ratios = remove_outliers_by_group(X_clean, y_clean,
                                                       k=5,
                                                       group_count=4)
    print("各类别最终样本比例:", ratios)

    # Step 5: 计算类别权重
    class_weights = compute_class_weights_from_ratio(ratios, gamma=1.0)
    print("类别权重:", class_weights)

    # Step 6: 训练随机森林
    clf = train_rf_with_weights(X_final, y_final, class_weights,
                               n_estimators=100,
                               random_state=42)

    # Step 7: 评估模型，阈值0.5
    evaluate_model(clf, X_test, y_test, threshold=0.5)

    # 可视化阈值对各指标影响
    plot_metrics_vs_threshold(clf, X_test, y_test)


if __name__ == "__main__":
    main()
