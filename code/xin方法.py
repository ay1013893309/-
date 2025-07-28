from scipy.optimize import differential_evolution
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors, tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef
)
import time
import csv
import warnings
import random

# 设置输出目录
output_dir = './results/weighted_smote/'
os.makedirs(output_dir, exist_ok=True)
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)

# 分类器选择
classifier = "knn"
classifier_for_selection = {
    "knn": neighbors.KNeighborsClassifier(),
    "svm": svm.SVC(probability=True, random_state=0),
    "rf": RandomForestClassifier(random_state=0),
    "dt": tree.DecisionTreeClassifier(random_state=0),
    "lr": LogisticRegression(random_state=0),
    "nb": GaussianNB(),
    "mlp": MLPClassifier(random_state=0, max_iter=1000)
}

skf = StratifiedKFold(n_splits=5)


class WeightedSMOTE:
    def __init__(self, k_neighbors=5, weights=None):
        self.k_neighbors = k_neighbors
        self.weights = weights

    def fit_resample(self, X, y, need_number):
        unique_classes, counts = np.unique(y, return_counts=True)
        minority_class = unique_classes[0] if counts[0] < counts[1] else unique_classes[1]

        X_min = X[y == minority_class]
        y_min = y[y == minority_class]
        n_minority = len(X_min)

        if need_number <= 0 or n_minority == 0:
            return X, y

        weights = self.weights
        if weights is None:
            weights = np.ones(X.shape[1]) / X.shape[1]

        actual_k = min(self.k_neighbors, n_minority - 1)
        if actual_k <= 0:
            synthetic_samples = [X_min[0] for _ in range(need_number)]
            return np.vstack([X, synthetic_samples]), np.hstack([y, [minority_class] * need_number])

        tree = KDTree(X_min)
        synthetic_samples = []
        for i in range(n_minority):
            distances, indices = tree.query([X_min[i]], k=actual_k + 1)
            nn_indices = indices[0][1:]
            n_synthetic_for_point = max(1, int(need_number / n_minority))
            for _ in range(n_synthetic_for_point):
                if len(nn_indices) == 0:
                    continue
                nn = X_min[random.choice(nn_indices)]
                diff = nn - X_min[i]
                gap = np.random.rand()
                weighted_gap = gap * (1 + 2 * (weights - 0.5))
                new_sample = X_min[i] + diff * weighted_gap
                synthetic_samples.append(new_sample)

        if len(synthetic_samples) > need_number:
            synthetic_samples = synthetic_samples[:need_number]

        return np.vstack([X, synthetic_samples]), np.hstack([y, [minority_class] * len(synthetic_samples)])


def fit(bound, dataset):
    X = dataset[:, :-1]
    y = dataset[:, -1]
    weights = np.abs(bound) / np.sum(np.abs(bound))
    total_mcc = 0

    for train_idx, _ in skf.split(X, y):
        X_train = dataset[train_idx, :-1]
        y_train = dataset[train_idx, -1]
        for sub_train_idx, sub_val_idx in skf.split(X_train, y_train):
            X_sub_train, y_sub_train = X_train[sub_train_idx], y_train[sub_train_idx]
            X_sub_val, y_sub_val = X_train[sub_val_idx], y_train[sub_val_idx]
            defect_count = np.sum(y_sub_train == 1)
            clean_count = np.sum(y_sub_train == 0)
            n_synthetic = clean_count - defect_count
            sm = WeightedSMOTE(k_neighbors=5, weights=weights)
            X_res, y_res = sm.fit_resample(X_sub_train, y_sub_train, n_synthetic)
            clf = classifier_for_selection[classifier]
            clf.fit(X_res, y_res)
            pred = clf.predict(X_sub_val)
            total_mcc += matthews_corrcoef(y_sub_val, pred)

    return -total_mcc / 25


for iteration in range(1):
    print(f"\nStarting iteration {iteration + 1}")
    result_file = os.path.join(output_dir, f'{classifier}_weighted_smote_{iteration}.csv')
    with open(result_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["inputfile", "mcc", "auc", "balance", "f1", "precision", "recall", "pf", "gmean", "time"])

        for filename in os.listdir("G:\pycharm\lutrs\code\datacunshu"):
            if not filename.endswith('.csv'):
                continue

            filepath = os.path.join("G:\pycharm\lutrs\code\datacunshu", filename)
            try:
                df = pd.read_csv(filepath)
                for col in ["name", "version", "name.1", "Unnamed: 0"]:
                    if col in df.columns:
                        df.drop(columns=col, inplace=True)
                if "bug" not in df.columns:
                    continue

                df["bug"] = df["bug"].apply(lambda x: 1 if x > 0 else 0)
                ratio = df["bug"].mean()
                if ratio < 0.05 or ratio > 0.45:
                    continue

                cols = [c for c in df.columns if c != "bug"]
                for col in cols:
                    max_v, min_v = df[col].max(), df[col].min()
                    df[col] = (df[col] - min_v) / (max_v - min_v) if max_v > min_v else 0

                dataset = df.to_numpy()
                bounds = [(0.1, 2.0)] * (dataset.shape[1] - 1)

                start = time.time()
                result = differential_evolution(fit, bounds, args=(dataset,), maxiter=10, popsize=10, disp=True)
                weights = np.abs(result.x) / np.sum(np.abs(result.x))

                X, y = dataset[:, :-1], dataset[:, -1]
                metrics = dict(mcc=0, auc=0, balance=0, f1=0, precision=0, recall=0, pf=0, gmean=0)

                for train_idx, test_idx in skf.split(X, y):
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[test_idx], y[test_idx]
                    defect_count = np.sum(y_train == 1)
                    clean_count = len(y_train) - defect_count
                    n_synthetic = clean_count - defect_count

                    sm = WeightedSMOTE(k_neighbors=5, weights=weights)
                    X_res, y_res = sm.fit_resample(X_train, y_train, n_synthetic)

                    clf = classifier_for_selection[classifier]
                    clf.fit(X_res, y_res)
                    pred = clf.predict(X_test)
                    prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else pred

                    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
                    recall = recall_score(y_test, pred)
                    pf = fp / (fp + tn) if (fp + tn) > 0 else 0
                    balance = 1 - (((0 - pf)**2 + (1 - recall)**2) / 2) ** 0.5
                    gmean = np.sqrt(recall * (1 - pf)) if recall > 0 and pf < 1 else 0

                    metrics["mcc"] += matthews_corrcoef(y_test, pred)
                    metrics["auc"] += roc_auc_score(y_test, prob)
                    metrics["f1"] += f1_score(y_test, pred)
                    metrics["precision"] += precision_score(y_test, pred)
                    metrics["recall"] += recall
                    metrics["pf"] += pf
                    metrics["balance"] += balance
                    metrics["gmean"] += gmean

                for key in metrics:
                    metrics[key] /= 5

                elapsed = time.time() - start
                writer.writerow([filename] + [metrics[k] for k in ["mcc", "auc", "balance", "f1", "precision", "recall", "pf", "gmean"]] + [f"{elapsed:.2f}s"])
                print(f"Done {filename}: MCC={metrics['mcc']:.4f}, AUC={metrics['auc']:.4f}, G-Mean={metrics['gmean']:.4f}")

            except Exception as e:
                print(f"Error on {filename}: {e}")
