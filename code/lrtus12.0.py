import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import os
import numpy as np
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import time
import csv
import warnings
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from scipy.optimize import differential_evolution
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# 在打开文件前添加以下代码
output_dir = './results/original/'
visualization_dir = './results/visualization/'
os.makedirs(output_dir, exist_ok=True)  # 自动创建目录（如果不存在）
os.makedirs(visualization_dir, exist_ok=True)  # 创建可视化目录

np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')
classifier = "rf"

skf = StratifiedKFold(n_splits=5)
print("test1")


def plot_kde_comparison(original_data, sampled_data, feature_idx, filename, fold_idx):
    """
    绘制单个特征在原始数据和采样数据上的核密度估计图

    参数:
    original_data: 原始数据
    sampled_data: 采样后的数据
    feature_idx: 特征索引
    filename: 数据集文件名
    fold_idx: 交叉验证的折叠索引
    """
    plt.figure(figsize=(10, 6))

    # 提取特定特征的数据
    original_feature = original_data[:, feature_idx]
    sampled_feature = sampled_data[:, feature_idx]

    # 计算核密度估计
    kde_original = gaussian_kde(original_feature)
    kde_sampled = gaussian_kde(sampled_feature)

    # 创建x轴范围
    min_val = min(original_feature.min(), sampled_feature.min())
    max_val = max(original_feature.max(), sampled_feature.max())
    x = np.linspace(min_val - 0.1, max_val + 0.1, 1000)

    # 绘制KDE曲线
    plt.plot(x, kde_original(x), label='Original', linewidth=2, color='blue')
    plt.plot(x, kde_sampled(x), label='Sampled', linewidth=2, color='orange')

    # 添加统计信息
    plt.title(f'KDE Comparison: {filename} - Feature {feature_idx} (Fold {fold_idx})')
    plt.xlabel('Feature Value')
    plt.ylabel('Probability Density')
    plt.legend()

    # 添加统计信息标注
    stats_text = (f"Original: μ={np.mean(original_feature):.3f}, σ={np.std(original_feature):.3f}\n"
                  f"Sampled: μ={np.mean(sampled_feature):.3f}, σ={np.std(sampled_feature):.3f}")
    plt.annotate(stats_text, xy=(0.05, 0.85), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # 保存图片
    plot_filename = f"{filename.split('.')[0]}_feature_{feature_idx}_fold_{fold_idx}.png"
    plt.savefig(os.path.join(visualization_dir, plot_filename))
    plt.close()


def fit(bound, *param):
    de_dataset = []
    for de_i in param:
        de_dataset.append(de_i)

    de_dataset = np.array(de_dataset)
    de_x = de_dataset
    de_y = de_dataset[:, -1]
    de_total_mcc = 0
    for train, test in skf.split(de_x, de_y):
        de_train_x = de_x[train]
        de_train_y = de_y[train]

        for sub_train, sub_test in skf.split(de_train_x, de_train_y):
            de_sub_test_x = de_train_x[sub_test]
            de_sub_test_y = de_train_y[sub_test]
            de_sub_test_x = de_sub_test_x[:, 0:-1]

            de_sub_train_x = de_train_x[sub_train]
            de_sub_defect_x = de_sub_train_x[de_sub_train_x[:, -1] > 0]
            de_sub_clean_x = de_sub_train_x[de_sub_train_x[:, -1] == 0]
            de_need_number = len(de_sub_clean_x) - len(de_sub_defect_x)
            de_sub_clean_x[:, :-1] *= np.array(bound)

            de_total_sum = np.sum(de_sub_clean_x[:, 0:-1], axis=1)
            de_sub_clean_x = np.c_[de_sub_clean_x, de_total_sum]
            de_sub_clean_x = de_sub_clean_x[np.argsort(-de_sub_clean_x[:, -1])]
            de_sub_clean_x = np.delete(de_sub_clean_x, [a for a in range(de_need_number)], axis=0)
            de_sub_clean_x = de_sub_clean_x[:, 0:-1]
            de_sub_clean_x[:, 0:-1] = de_sub_clean_x[:, 0:-1] / bound
            de_sub_train_x = np.r_[de_sub_clean_x, de_sub_defect_x]
            de_sub_train_y = de_sub_train_x[:, -1]
            de_sub_train_x = de_sub_train_x[:, 0:-1]
            de_clf = classifier_for_selection[classifier]
            de_clf.fit(de_sub_train_x, de_sub_train_y)
            de_predict_result = de_clf.predict(de_sub_test_x)
            de_mcc = matthews_corrcoef(de_sub_test_y, de_predict_result)
            de_total_mcc = de_total_mcc + de_mcc

    de_average_mcc = de_total_mcc / 25
    de_average_mcc = - de_average_mcc
    return de_average_mcc


for iteration in range(1):
    print("test2")
    single = open(os.path.join(output_dir, f'{classifier}radius based oversampling {iteration}.csv'), 'w', newline='')
    single_writer = csv.writer(single)
    single_writer.writerow(["inputfile", "mcc", "auc", "balance", "fmeasure", "precision", "pd", "pf"])
    print("test3")
    for inputfile in os.listdir("./data"):
        print("inputfile:", inputfile)
        start_time = time.asctime(time.localtime(time.time()))
        print("start time:", start_time)
        if inputfile == "PC1.csv" or inputfile == "PC2.csv" or inputfile == "PC3.csv" or inputfile == "synapse-1.0.csv" or inputfile == "jedit-4.3.csv" or inputfile == "synapse-1.1.csv" or inputfile == "LICENSE":
            continue
        print("test4")
        dataset = pd.read_csv("./data/" + inputfile)
        dataset = dataset.drop(columns="name")
        dataset = dataset.drop(columns="version")
        dataset = dataset.drop(columns="name.1")
        total_number = len(dataset)
        defect_ratio = len(dataset[dataset["bug"] > 0]) / total_number
        if defect_ratio > 0.45:
            print(inputfile, " defect ratio larger than 0.45")
            continue

        for z in range(total_number):
            if dataset.loc[z, "bug"] > 0:
                dataset.loc[z, "bug"] = 1
        cols = list(dataset.columns)
        for col in cols:
            column_max = dataset[col].max()
            column_min = dataset[col].min()
            dataset[col] = (dataset[col] - column_min) / (column_max - column_min)

        dataset = np.array(dataset)

        classifier_for_selection = {"knn": neighbors.KNeighborsClassifier(), "svm": svm.SVC(),
                                    "rf": RandomForestClassifier(random_state=0),
                                    "dt": tree.DecisionTreeClassifier(random_state=0),
                                    "lr": LogisticRegression(random_state=0), "nb": GaussianNB()}

        # 在转换为numpy数组后添加
        dataset = np.array(dataset)

        # 动态设置边界
        num_features = dataset.shape[1] - 1  # 特征数量 = 总列数 - 1（标签列）
        bound = [(-1, 1)] * num_features  # 根据实际特征数创建边界

        optimal_result = differential_evolution(fit, bound, args=dataset, popsize=10, maxiter=20, mutation=0.3,
                                                recombination=0.9, disp=True)
        optimal_weight = optimal_result.x
        y = dataset[:, -1]
        x = dataset

        total_auc = 0
        total_balance = 0
        total_fmeasure = 0
        total_precision = 0
        total_recall = 0
        total_pf = 0
        total_mcc = 0

        fold_idx = 0  # 用于跟踪折叠索引
        for train, test in skf.split(x, y):
            test_x = x[test]
            test_y = test_x[:, -1]
            test_x = test_x[:, 0:-1]

            train_x = x[train]

            # 保存原始训练数据用于可视化
            original_train_data = train_x.copy()

            defect_x = train_x[train_x[:, -1] > 0]
            clean_x = train_x[train_x[:, -1] == 0]
            need_number = len(clean_x) - len(defect_x)

            clean_x[:, 0:-1] = optimal_weight * clean_x[:, 0:-1]
            total_sum = np.sum(clean_x[:, 0:-1], axis=1)
            clean_x = np.c_[clean_x, total_sum]
            clean_x = clean_x[np.argsort(-clean_x[:, -1])]
            clean_x = np.delete(clean_x, [a for a in range(need_number)], axis=0)
            clean_x = clean_x[:, 0:-1]
            clean_x[:, 0:-1] = clean_x[:, 0:-1] / optimal_weight
            train_x = np.r_[clean_x, defect_x]
            train_y = train_x[:, -1]
            train_x = train_x[:, 0:-1]
            clf = classifier_for_selection[classifier]
            clf.fit(train_x, train_y)
            predict_result = clf.predict(test_x)

            # 可视化特征分布（只对第一次折叠进行可视化）
            if fold_idx == 0:
                # 选择前3个特征进行可视化
                num_features_to_visualize = min(10, train_x.shape[1])
                for feature_idx in range(num_features_to_visualize):
                    plot_kde_comparison(
                        original_train_data[:, :-1],  # 原始训练数据（不包括标签）
                        train_x,  # 采样后的训练数据
                        feature_idx,
                        inputfile,
                        fold_idx
                    )

            auc = roc_auc_score(test_y, predict_result)
            total_auc = total_auc + auc

            mcc = matthews_corrcoef(test_y, predict_result)
            total_mcc = total_mcc + mcc

            fmeasure = f1_score(test_y, predict_result)
            total_fmeasure = total_fmeasure + fmeasure

            true_negative, false_positive, false_negative, true_positive = confusion_matrix(test_y,
                                                                                            predict_result).ravel()

            recall = recall_score(test_y, predict_result)
            total_recall = total_recall + recall

            pf = false_positive / (true_negative + false_positive)
            total_pf = total_pf + pf

            balance = 1 - (((0 - pf) ** 2 + (1 - recall) ** 2) / 2) ** 0.5
            total_balance = total_balance + balance

            precision = precision_score(test_y, predict_result)
            total_precision = total_precision + precision

            fold_idx += 1  # 增加折叠索引

        average_auc = total_auc / 5
        average_balance = total_balance / 5
        average_fmeasure = total_fmeasure / 5
        average_precision = total_precision / 5
        average_recall = total_recall / 5
        average_pf = total_pf / 5
        average_mcc = total_mcc / 5
        single_writer.writerow(
            [inputfile, average_mcc, average_auc, average_balance, average_fmeasure, average_precision, average_recall,
             average_pf])
        print("final auc: ", average_auc)
        print("end time: ", time.asctime(time.localtime(time.time())))
        print("--------------------------------------")