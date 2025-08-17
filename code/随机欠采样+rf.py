import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler  # 替换为随机欠采样
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


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
    plt.title('Metrics vs Classification Threshold (Random Undersampling + RF)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 加载数据
def load_data():
    df = pd.read_csv(r"G:\pycharm\lutrs\stability-of-smote-main\AEEM\converted_PDE.csv")
    y_raw = df['bug'].values
    y = (y_raw > 0).astype(int)
    df = df.drop(columns=['name', 'version', 'name1', 'bug'], errors='ignore')
    X = df.select_dtypes(include=['int64', 'float64']).values
    return X, y

def evaluate_model(model, X_test, y_test, threshold=0.5):
    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    print(f"Threshold = {threshold}")
    print("Precision:", f"{precision_score(y_test, y_pred):.4f}")
    print("Recall:", f"{recall_score(y_test, y_pred):.4f}")
    print("F1 Score:", f"{f1_score(y_test, y_pred):.4f}")
    print("MCC:", f"{matthews_corrcoef(y_test, y_pred):.4f}")
    print("AUC:", f"{roc_auc_score(y_test, probs):.4f}")

def main():
    # 1. 加载数据
    X, y = load_data()

    # 2. 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        stratify=y,
                                                        random_state=42)

    print("原始训练集类别分布:", {label: sum(y_train == label) for label in np.unique(y_train)})

    # 3. 使用随机欠采样处理不平衡数据
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)

    print("随机欠采样后训练集类别分布:", {label: sum(y_res == label) for label in np.unique(y_res)})

    # 4. 训练随机森林
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    clf.fit(X_res, y_res)

    # 5. 评估模型，阈值0.5
    evaluate_model(clf, X_test, y_test, threshold=0.5)

    # 6. 绘制阈值指标曲线
    plot_metrics_vs_threshold(clf, X_test, y_test)

if __name__ == "__main__":
    main()