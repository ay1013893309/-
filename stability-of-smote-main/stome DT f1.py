import csv
import os
import random
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import neighbors
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
target_defect_ratio = 0.5
# fold = 5
neighbor = 5
tuned_parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100, 1000]}
classifier_for_selection = {"svm": svm.SVC(), "knn": neighbors.KNeighborsClassifier(), "rf": RandomForestClassifier(), "tree": tree.DecisionTreeClassifier()}

classifier = "tree"


# class stable_SMOTE:
#     def __init__(self, z_nearest=5):
#         self.z_nearest = z_nearest
#
#     def fit_resample(self, x_dataset, y_dataset):
#         # 确保转换为 NumPy 数组格式
#         x_dataset = np.array(x_dataset)
#
#         # 使用索引代替列名
#         bug_col_index = x_dataset.shape[1] - 1  # 最后一列是 bug 列
#
#         # 分离缺陷样本和正常样本
#         defective_mask = x_dataset[:, bug_col_index] > 0
#         clean_mask = x_dataset[:, bug_col_index] == 0
#
#         defective_instance = x_dataset[defective_mask]
#         clean_instance = x_dataset[clean_mask]
#
#         # 后续计算保持不变...
#         defective_number = len(defective_instance)
#         clean_number = len(clean_instance)
#
#         # 计算需要生成的样本数
#         need_number = int((target_defect_ratio * len(x_dataset) - defective_number) / (1 - target_defect_ratio))
#
#         if need_number <= 0:
#             return np.concatenate([clean_instance, defective_instance])
#         generated_dataset = []
#         synthetic_dataset = pd.DataFrame()
#         number_on_each_instance = need_number / defective_number  # 每个实例分摊到了生成几个的任务
#         total_pair = []
#
#         rround = number_on_each_instance / self.z_nearest
#         while rround >= 1:
#             for index, row in defective_instance.iterrows():
#                 temp_defective_instance = defective_instance.copy(deep=True)
#                 subtraction = row - temp_defective_instance
#                 square = subtraction ** 2
#                 row_sum = square.apply(lambda s: s.sum(), axis=1)
#                 distance = row_sum ** 0.5
#                 temp_defective_instance["distance"] = distance
#                 temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
#                 neighbors = temp_defective_instance[1:self.z_nearest + 1]
#                 for a, r in neighbors.iterrows():
#                     selected_pair = [index, a]
#                     selected_pair.sort()
#                     total_pair.append(selected_pair)
#             rround = rround - 1
#         need_number1 = need_number - len(total_pair)
#         number_on_each_instance = need_number1 / defective_number
#
#         for index, row in defective_instance.iterrows():
#             temp_defective_instance = defective_instance.copy(deep=True)
#             subtraction = row - temp_defective_instance
#             square = subtraction ** 2
#             row_sum = square.apply(lambda s: s.sum(), axis=1)
#             distance = row_sum ** 0.5
#
#             temp_defective_instance["distance"] = distance
#             temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
#             neighbors = temp_defective_instance[1:self.z_nearest + 1]
#             neighbors = neighbors.sort_values(by="distance", ascending=False)  # 这里取nearest neighbor里最远的
#             target_sample_instance = neighbors[0: int(number_on_each_instance)]
#             target_sample_instance = target_sample_instance.drop(columns="distance")
#             for a, r in target_sample_instance.iterrows():
#                 selected_pair = [index, a]
#                 selected_pair.sort()
#                 total_pair.append(selected_pair)
#         temp_defective_instance = defective_instance.copy(deep=True)
#         residue_number = need_number - len(total_pair)
#         residue_defective_instance = temp_defective_instance.sample(n=residue_number)
#         for index, row in residue_defective_instance.iterrows():
#             temp_defective_instance = defective_instance.copy(deep=True)
#             subtraction = row - temp_defective_instance
#             square = subtraction ** 2
#             row_sum = square.apply(lambda s: s.sum(), axis=1)
#             distance = row_sum ** 0.5
#
#             temp_defective_instance["distance"] = distance
#             temp_defective_instance = temp_defective_instance.sort_values(by="distance", ascending=True)
#             neighbors = temp_defective_instance[1:self.z_nearest + 1]
#             target_sample_instance = neighbors[-1:]
#             for a in target_sample_instance.index:
#                 selected_pair = [index, a]
#                 selected_pair.sort()
#                 total_pair.append(selected_pair)
#         total_pair_tuple = [tuple(l) for l in total_pair]
#         result = Counter(total_pair_tuple)
#         result_number = len(result)
#         result_keys = result.keys()
#         result_values = result.values()
#         for f in range(result_number):
#             current_pair = list(result_keys)[f]
#             row1_index = current_pair[0]
#             row2_index = current_pair[1]
#             row1 = defective_instance.loc[row1_index]
#             row2 = defective_instance.loc[row2_index]
#             generated_num = list(result_values)[f]
#             generated_instances = np.linspace(row1, row2, generated_num + 2)
#             generated_instances = generated_instances[1:-1]
#             generated_instances = generated_instances.tolist()
#             for w in generated_instances:
#                 generated_dataset.append(w)
#         final_generated_dataset = pd.DataFrame(generated_dataset)
#         final_generated_dataset = final_generated_dataset.rename(
#             columns={0: "wmc", 1: "dit", 2: "noc", 3: "cbo", 4: "rfc", 5: "lcom", 6: "ca", 7: "ce", 8: "npm",
#                      9: "lcom3", 10: "loc", 11: "dam", 12: "moa", 13: "mfa", 14: "cam", 15: "ic", 16: "cbm", 17: "amc",
#                      18: "max_cc", 19: "avg_cc", 20: "bug"}
#         )
#         result = pd.concat([clean_instance, defective_instance, final_generated_dataset])
#         return result
class stable_SMOTE:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest

    def fit_resample(self, x_dataset, y_dataset):
        # 确保输入是 NumPy 数组
        x_dataset = np.array(x_dataset)

        # 获取 bug 列的索引（最后一列）
        bug_col_index = x_dataset.shape[1] - 1

        # 分离缺陷样本和正常样本
        defective_mask = x_dataset[:, bug_col_index] > 0
        clean_mask = x_dataset[:, bug_col_index] == 0

        defective_instance = x_dataset[defective_mask]
        clean_instance = x_dataset[clean_mask]

        defective_number = len(defective_instance)
        clean_number = len(clean_instance)
        total_number = len(x_dataset)

        # 计算需要生成的样本数
        need_number = int((target_defect_ratio * total_number - defective_number) / (1 - target_defect_ratio))

        if need_number <= 0:
            return np.concatenate([clean_instance, defective_instance])

        generated_dataset = []
        total_pair = []

        # 第一轮：尽可能多地生成样本对
        rround = max(1, int(need_number / (defective_number * self.z_nearest)))
        for _ in range(rround):
            for i in range(defective_number):
                current_row = defective_instance[i]

                # 计算当前样本与其他缺陷样本的距离
                distances = np.sqrt(np.sum((defective_instance - current_row) ** 2, axis=1))

                # 排除自身（距离为0）
                valid_indices = np.where(distances > 0)[0]
                if len(valid_indices) == 0:
                    continue

                # 找到最近的邻居
                nearest_indices = valid_indices[np.argsort(distances[valid_indices])[:self.z_nearest]]

                for neighbor_index in nearest_indices:
                    # 创建排序后的样本对标识
                    pair = (min(i, neighbor_index), max(i, neighbor_index))
                    total_pair.append(pair)

        # 第二轮：处理剩余需要生成的样本
        generated_count = len(total_pair)
        if generated_count < need_number:
            remaining_needed = need_number - generated_count
            for i in range(defective_number):
                current_row = defective_instance[i]

                # 计算当前样本与其他缺陷样本的距离
                distances = np.sqrt(np.sum((defective_instance - current_row) ** 2, axis=1))

                # 排除自身
                valid_indices = np.where(distances > 0)[0]
                if len(valid_indices) == 0:
                    continue

                # 找到最近的邻居
                nearest_indices = valid_indices[np.argsort(distances[valid_indices])[:self.z_nearest]]

                # 取最远的邻居
                if len(nearest_indices) > 0:
                    farthest_in_nearest = nearest_indices[-1]
                    pair = (min(i, farthest_in_nearest), max(i, farthest_in_nearest))
                    total_pair.append(pair)
                    remaining_needed -= 1
                    if remaining_needed <= 0:
                        break
            if remaining_needed > 0:
                # 如果还有剩余需要，随机选择样本对
                for _ in range(remaining_needed):
                    i = np.random.randint(0, defective_number)
                    j = np.random.randint(0, defective_number)
                    if i != j:
                        pair = (min(i, j), max(i, j))
                        total_pair.append(pair)

        # 统计每个样本对需要生成的数量
        pair_counter = Counter(total_pair)

        # 生成新样本
        for pair, count in pair_counter.items():
            idx1, idx2 = pair
            row1 = defective_instance[idx1]
            row2 = defective_instance[idx2]

            # 在两点之间生成样本
            for _ in range(count):
                alpha = np.random.random()  # 随机插值因子
                new_sample = row1 * (1 - alpha) + row2 * alpha
                generated_dataset.append(new_sample)

        # 组合所有样本
        final_generated = np.array(generated_dataset)
        return np.concatenate([clean_instance, defective_instance, final_generated])
def separate_data(original_data):
    '''

    用out-of-sample bootstrap方法产生训练集和测试集,参考论文An Empirical Comparison of Model Validation Techniques for DefectPrediction Models
    A bootstrap sample of size N is randomly drawn with replacement from an original dataset that is also of size N .
    The model is instead tested using the rows that do not appear in the bootstrap sample.
    On average, approximately 36.8 percent of the rows will not appear in the bootstrap sample, since the bootstrap sample is drawn with replacement.
    OriginalData:整个数据集

    return: 划分好的 训练集和测试集

    '''
    original_data = np.array(original_data).tolist()
    size = len(original_data)
    train_dataset = []
    train_index = []
    for i in range(size):
        index = random.randint(0, size - 1)
        train_instance = original_data[index]
        train_dataset.append(train_instance)
        train_index.append(index)

    original_index = [z for z in range(size)]
    train_index = list(set(train_index))
    test_index = list(set(original_index).difference(set(train_index)))
    original_data = np.array(original_data)
    train_dataset = original_data[train_index]
    # original_data = pd.DataFrame(original_data)
    # original_data = original_data.rename(
    #     columns={0: "wmc", 1: "dit", 2: "noc", 3: "cbo", 4: "rfc", 5: "lcom", 6: "ca", 7: "ce", 8: "npm",
    #              9: "lcom3", 10: "loc", 11: "dam", 12: "moa", 13: "mfa", 14: "cam", 15: "ic", 16: "cbm", 17: "amc",
    #              18: "max_cc", 19: "avg_cc", 20: "bug"})
    # train_dataset = pd.DataFrame(train_dataset)
    # train_dataset = train_dataset.rename(
    #     columns={0: "wmc", 1: "dit", 2: "noc", 3: "cbo", 4: "rfc", 5: "lcom", 6: "ca", 7: "ce", 8: "npm",
    #              9: "lcom3", 10: "loc", 11: "dam", 12: "moa", 13: "mfa", 14: "cam", 15: "ic", 16: "cbm", 17: "amc",
    #              18: "max_cc", 19: "avg_cc", 20: "bug"}
    # )
    test_dataset = original_data[test_index]
    return train_dataset, test_dataset



# measure = "pf"
# for measure in ["auc", "balance", "recall", "pf", "brier"]:
auc_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'auc_smote_result_on_'+classifier+'.csv', 'w', newline='')
auc_writer = csv.writer(auc_file)
auc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

balance_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'balance_smote_result_on_'+classifier+'.csv', 'w', newline='')
balance_writer = csv.writer(balance_file)
balance_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

recall_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'recall_smote_result_on_'+classifier+'.csv', 'w', newline='')
recall_writer = csv.writer(recall_file)
recall_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

pf_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'pf_smote_result_on_'+classifier+'.csv', 'w', newline='')
pf_writer = csv.writer(pf_file)
pf_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max"])

brier_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'brier_smote_result_on_'+classifier+'.csv', 'w', newline='')
brier_writer = csv.writer(brier_file)
brier_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

mcc_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'mcc_smote_result_on_'+classifier+'.csv', 'w', newline='')
mcc_writer = csv.writer(mcc_file)
mcc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_auc_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'auc_stable_smote_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_auc_writer = csv.writer(stable_auc_file)
stable_auc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_balance_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'balance_stable_smote_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_balance_writer = csv.writer(stable_balance_file)
stable_balance_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_recall_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'recall_stable_smote_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_recall_writer = csv.writer(stable_recall_file)
stable_recall_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_pf_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'pf_stable_smote_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_pf_writer = csv.writer(stable_pf_file)
stable_pf_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_brier_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'brier_stable_smote_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_brier_writer = csv.writer(stable_brier_file)
stable_brier_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_mcc_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'mcc_stable_smote_result_on_'+classifier+'.csv', 'w',
              newline='')
stable_mcc_writer = csv.writer(stable_mcc_file)
stable_mcc_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])
precision_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'precision_smote_result_on_'+classifier+'.csv', 'w', newline='')
precision_writer = csv.writer(precision_file)
precision_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

f1_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'f1_smote_result_on_'+classifier+'.csv', 'w', newline='')
f1_writer = csv.writer(f1_file)
f1_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

gmean_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'gmean_smote_result_on_'+classifier+'.csv', 'w', newline='')
gmean_writer = csv.writer(gmean_file)
gmean_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_precision_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'precision_stable_smote_result_on_'+classifier+'.csv', 'w', newline='')
stable_precision_writer = csv.writer(stable_precision_file)
stable_precision_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_f1_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'f1_stable_smote_result_on_'+classifier+'.csv', 'w', newline='')
stable_f1_writer = csv.writer(stable_f1_file)
stable_f1_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])

stable_gmean_file = open('G:\pycharm\lutrs\stability-of-smote-main\k vlaue'+str(neighbor)+'gmean_stable_smote_result_on_'+classifier+'.csv', 'w', newline='')
stable_gmean_writer = csv.writer(stable_gmean_file)
stable_gmean_writer.writerow(["inputfile", "", "", "", "", "", "", "", "", "", "", "min", "lower", "avg", "median", "upper", "max", "variance"])
for inputfile in os.listdir("G:\pycharm\lutrs\stability-of-smote-main\\ant"):
    print(inputfile)
    start_time = time.asctime(time.localtime(time.time()))
    print("start_time: ", start_time)
    dataset = pd.read_csv("G:\pycharm\lutrs\stability-of-smote-main\\ant\\" + inputfile)
    total_number = len(dataset)
    defect_ratio = len(dataset[dataset["bug"] > 0]) / total_number
    print("defect ratio: ", defect_ratio)
    if defect_ratio > 0.45:
        print(inputfile, " defect ratio larger than 0.45")
        continue
    dataset = dataset.drop(columns="name")
    dataset = dataset.drop(columns="version")
    dataset = dataset.drop(columns="name.1")
    for j in range(total_number):
        if dataset.loc[j, "bug"] > 0:
            dataset.loc[j, "bug"] = 1

    # Normalize data
    cols = list(dataset.columns)
    for col in cols:
        column_max = dataset[col].max()
        column_min = dataset[col].min()
        if column_max != column_min:  # Avoid division by zero
            dataset[col] = (dataset[col] - column_min) / (column_max - column_min)
        else:
            dataset[col] = 0  # Handle constant columns

    for j in range(10):
        # Initialize metric lists for this iteration
        auc_row, balance_row, recall_row, pf_row, brier_row, mcc_row = [], [], [], [], [], []
        precision_row, f1_row, gmean_row = [], [], []
        stable_auc_row, stable_balance_row, stable_recall_row, stable_pf_row = [], [], [], []
        stable_brier_row, stable_mcc_row, stable_precision_row, stable_f1_row, stable_gmean_row = [], [], [], [], []
        # Separate data into training and testing sets
        train_data, test_data = separate_data(dataset)
        while len(train_data[train_data[:, -1] == 0]) == 0 or len(test_data[test_data[:, -1] == 1]) == 0 or len(
                train_data[train_data[:, -1] == 1]) <= neighbor or len(test_data[test_data[:, -1] == 0]) == 0 or len(
                train_data[train_data[:, -1] == 1]) >= len(train_data[train_data[:, -1] == 0]):
            train_data, test_data = separate_data(dataset)
        # print(len(train_data[train_data[:, -1] == 1]))
        train_x = train_data[:, 0:-1]
        # print(train_x)
        train_y = train_data[:, -1]
        # for s in range(10):
        for s in range(10):
            # ========== ORIGINAL SMOTE ==========
            smote = SMOTE(k_neighbors=neighbor)
            smote_train_x, smote_train_y = smote.fit_resample(train_x, train_y)
            test_x = test_data[:, 0:-1]
            test_y = test_data[:, -1]

            # Use Decision Tree directly
            clf = DecisionTreeClassifier(random_state=42)
            clf.fit(smote_train_x, smote_train_y)
            predict_result = clf.predict(test_x)

            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(test_y, predict_result).ravel()
            recall = recall_score(test_y, predict_result)
            pf = fp / (tn + fp) if (tn + fp) > 0 else 0
            balance = 1 - (((0 - pf) ** 2 + (1 - recall) ** 2) / 2) ** 0.5
            auc = roc_auc_score(test_y, predict_result)
            brier = brier_score_loss(test_y, predict_result)
            mcc = matthews_corrcoef(test_y, predict_result)

            # Additional metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = 1 - pf if pf <= 1 else 0
            gmean = np.sqrt(recall * specificity) if (recall >= 0 and specificity >= 0) else 0

            # Store results
            auc_row.append(auc)
            balance_row.append(balance)
            recall_row.append(recall)
            pf_row.append(pf)
            brier_row.append(brier)
            mcc_row.append(mcc)
            precision_row.append(precision)
            f1_row.append(f1)
            gmean_row.append(gmean)

            # ========== STABLE SMOTE ==========
            stable_smote = stable_SMOTE(neighbor)
            stable_smote_train = stable_smote.fit_resample(train_data, train_y)
            stable_smote_train = np.array(stable_smote_train)

            stable_smote_train_x = stable_smote_train[:, 0:-1]
            stable_smote_train_y = stable_smote_train[:, -1]

            # Use Decision Tree directly
            stable_clf = DecisionTreeClassifier(random_state=42)
            stable_clf.fit(stable_smote_train_x, stable_smote_train_y)
            stable_predict_result = stable_clf.predict(test_x)

            # Calculate metrics
            stable_tn, stable_fp, stable_fn, stable_tp = confusion_matrix(test_y, stable_predict_result).ravel()
            stable_recall = recall_score(test_y, stable_predict_result)
            stable_pf = stable_fp / (stable_tn + stable_fp) if (stable_tn + stable_fp) > 0 else 0
            stable_balance = 1 - (((0 - stable_pf) ** 2 + (1 - stable_recall) ** 2) / 2) ** 0.5
            stable_auc = roc_auc_score(test_y, stable_predict_result)
            stable_brier = brier_score_loss(test_y, stable_predict_result)
            stable_mcc = matthews_corrcoef(test_y, stable_predict_result)

            # Additional metrics
            stable_precision = stable_tp / (stable_tp + stable_fp) if (stable_tp + stable_fp) > 0 else 0
            stable_f1 = 2 * (stable_precision * stable_recall) / (stable_precision + stable_recall) if (
                                                                                                                   stable_precision + stable_recall) > 0 else 0
            stable_specificity = 1 - stable_pf if stable_pf <= 1 else 0
            stable_gmean = np.sqrt(stable_recall * stable_specificity) if (
                        stable_recall >= 0 and stable_specificity >= 0) else 0

            # Store results
            stable_auc_row.append(stable_auc)
            stable_balance_row.append(stable_balance)
            stable_recall_row.append(stable_recall)
            stable_pf_row.append(stable_pf)
            stable_brier_row.append(stable_brier)
            stable_mcc_row.append(stable_mcc)
            stable_precision_row.append(stable_precision)
            stable_f1_row.append(stable_f1)
            stable_gmean_row.append(stable_gmean)

        max_brier = max(brier_row)
        min_brier = min(brier_row)
        avg_brier = np.mean(brier_row)
        median_brier = np.median(brier_row)
        quartile_brier = np.percentile(brier_row, (25, 75), interpolation='midpoint')
        lower_quartile_brier = quartile_brier[0]
        upper_quartile_brier = quartile_brier[1]
        variance_brier = np.std(brier_row)
        brier_row.append(min_brier)
        brier_row.append(lower_quartile_brier)
        brier_row.append(avg_brier)
        brier_row.append(median_brier)
        brier_row.append(upper_quartile_brier)
        brier_row.append(max_brier)
        brier_row.append(variance_brier)
        brier_row.insert(0, inputfile + " brier")
        brier_writer.writerow(brier_row)

        # stable smote
        stable_max_brier = max(stable_brier_row)
        stable_min_brier = min(stable_brier_row)
        stable_avg_brier = np.mean(stable_brier_row)
        stable_median_brier = np.median(stable_brier_row)
        stable_quartile_brier = np.percentile(stable_brier_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_brier = stable_quartile_brier[0]
        stable_upper_quartile_brier = stable_quartile_brier[1]
        stable_variance_brier = np.std(stable_brier_row)
        stable_brier_row.append(stable_min_brier)
        stable_brier_row.append(stable_lower_quartile_brier)

        stable_brier_row.append(stable_avg_brier)
        stable_brier_row.append(stable_median_brier)

        stable_brier_row.append(stable_upper_quartile_brier)
        stable_brier_row.append(stable_max_brier)
        stable_brier_row.append(stable_variance_brier)
        stable_brier_row.insert(0, inputfile + " brier")
        stable_brier_writer.writerow(stable_brier_row)

        max_auc = max(auc_row)
        min_auc = min(auc_row)
        avg_auc = np.mean(auc_row)
        median_auc = np.median(auc_row)
        quartile_auc = np.percentile(auc_row, (25, 75), interpolation='midpoint')
        lower_quartile_auc = quartile_auc[0]
        upper_quartile_auc = quartile_auc[1]
        variance_auc = np.std(auc_row)
        auc_row.append(min_auc)
        auc_row.append(lower_quartile_auc)

        auc_row.append(avg_auc)
        auc_row.append(median_auc)

        auc_row.append(upper_quartile_auc)
        auc_row.append(max_auc)
        auc_row.append(variance_auc)
        auc_row.insert(0, inputfile + " auc")
        auc_writer.writerow(auc_row)
        # stable smote
        stable_max_auc = max(stable_auc_row)
        stable_min_auc = min(stable_auc_row)
        stable_avg_auc = np.mean(stable_auc_row)
        stable_median_auc = np.median(stable_auc_row)
        stable_quartile_auc = np.percentile(stable_auc_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_auc = stable_quartile_auc[0]
        stable_upper_quartile_auc = stable_quartile_auc[1]
        stable_variance_auc = np.std(stable_auc_row)
        stable_auc_row.append(stable_min_auc)
        stable_auc_row.append(stable_lower_quartile_auc)
        stable_auc_row.append(stable_avg_auc)
        stable_auc_row.append(stable_median_auc)

        stable_auc_row.append(stable_upper_quartile_auc)
        stable_auc_row.append(stable_max_auc)
        stable_auc_row.append(stable_variance_auc)
        stable_auc_row.insert(0, inputfile + " auc")
        stable_auc_writer.writerow(stable_auc_row)

        max_balance = max(balance_row)
        min_balance = min(balance_row)
        avg_balance = np.mean(balance_row)
        median_balance = np.median(balance_row)
        quartile_balance = np.percentile(balance_row, (25, 75), interpolation='midpoint')
        lower_quartile_balance = quartile_balance[0]
        upper_quartile_balance = quartile_balance[1]
        variance_balance = np.std(balance_row)
        balance_row.append(min_balance)
        balance_row.append(lower_quartile_balance)
        balance_row.append(avg_balance)
        balance_row.append(median_balance)

        balance_row.append(upper_quartile_balance)
        balance_row.append(max_balance)
        balance_row.append(variance_balance)
        balance_row.insert(0, inputfile + " balance")
        balance_writer.writerow(balance_row)

        # stable smote
        stable_max_balance = max(stable_balance_row)
        stable_min_balance = min(stable_balance_row)
        stable_avg_balance = np.mean(stable_balance_row)
        stable_median_balance = np.median(stable_balance_row)
        stable_quartile_balance = np.percentile(stable_balance_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_balance = stable_quartile_balance[0]
        stable_upper_quartile_balance = stable_quartile_balance[1]
        stable_variance_balance = np.std(stable_balance_row)
        stable_balance_row.append(stable_min_balance)
        stable_balance_row.append(stable_lower_quartile_balance)
        stable_balance_row.append(stable_avg_balance)
        stable_balance_row.append(stable_median_balance)

        stable_balance_row.append(stable_upper_quartile_balance)
        stable_balance_row.append(stable_max_balance)
        stable_balance_row.append(stable_variance_balance)
        stable_balance_row.insert(0, inputfile + " balance")
        stable_balance_writer.writerow(stable_balance_row)

        max_recall = max(recall_row)
        min_recall = min(recall_row)
        avg_recall = np.mean(recall_row)
        median_recall = np.median(recall_row)
        quartile_recall = np.percentile(recall_row, (25, 75), interpolation='midpoint')
        lower_quartile_recall = quartile_recall[0]
        upper_quartile_recall = quartile_recall[1]
        variance_recall = np.std(recall_row)
        recall_row.append(min_recall)
        recall_row.append(lower_quartile_recall)
        recall_row.append(avg_recall)
        recall_row.append(median_recall)

        recall_row.append(upper_quartile_recall)
        recall_row.append(max_recall)
        recall_row.append(variance_recall)
        recall_row.insert(0, inputfile + " recall")
        recall_writer.writerow(recall_row)

        # stable smote
        stable_max_recall = max(stable_recall_row)
        stable_min_recall = min(stable_recall_row)
        stable_avg_recall = np.mean(stable_recall_row)
        stable_median_recall = np.median(stable_recall_row)
        stable_quartile_recall = np.percentile(stable_recall_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_recall = stable_quartile_recall[0]
        stable_upper_quartile_recall = stable_quartile_recall[1]
        stable_variance_recall = np.std(stable_recall_row)
        stable_recall_row.append(stable_min_recall)
        stable_recall_row.append(stable_lower_quartile_recall)
        stable_recall_row.append(stable_avg_recall)
        stable_recall_row.append(stable_median_recall)

        stable_recall_row.append(stable_upper_quartile_recall)
        stable_recall_row.append(stable_max_recall)
        stable_recall_row.append(stable_variance_recall)
        stable_recall_row.insert(0, inputfile + " recall")
        stable_recall_writer.writerow(stable_recall_row)

        max_pf = max(pf_row)
        min_pf = min(pf_row)
        avg_pf = np.mean(pf_row)
        median_pf = np.median(pf_row)
        quartile_pf = np.percentile(pf_row, (25, 75), interpolation='midpoint')
        lower_quartile_pf = quartile_pf[0]
        upper_quartile_pf = quartile_pf[1]
        variance_pf = np.std(pf_row)
        pf_row.append(min_pf)
        pf_row.append(lower_quartile_pf)
        pf_row.append(avg_pf)
        pf_row.append(median_pf)

        pf_row.append(upper_quartile_pf)
        pf_row.append(max_pf)
        pf_row.append(variance_pf)
        pf_row.insert(0, inputfile + " pf")
        pf_writer.writerow(pf_row)

        # stable smote
        stable_max_pf = max(stable_pf_row)
        stable_min_pf = min(stable_pf_row)
        stable_avg_pf = np.mean(stable_pf_row)
        stable_median_pf = np.median(stable_pf_row)
        stable_quartile_pf = np.percentile(stable_pf_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_pf = stable_quartile_pf[0]
        stable_upper_quartile_pf = stable_quartile_pf[1]
        stable_variance_pf = np.std(stable_pf_row)
        stable_pf_row.append(stable_min_pf)
        stable_pf_row.append(stable_lower_quartile_pf)
        stable_pf_row.append(stable_avg_pf)
        stable_pf_row.append(stable_median_pf)

        stable_pf_row.append(stable_upper_quartile_pf)
        stable_pf_row.append(stable_max_pf)
        stable_pf_row.append(stable_variance_pf)
        stable_pf_row.insert(0, inputfile + " pf")
        stable_pf_writer.writerow(stable_pf_row)

        max_mcc = max(mcc_row)
        min_mcc = min(mcc_row)
        avg_mcc = np.mean(mcc_row)
        median_mcc = np.median(mcc_row)
        quartile_mcc = np.percentile(mcc_row, (25, 75), interpolation='midpoint')
        lower_quartile_mcc = quartile_mcc[0]
        upper_quartile_mcc = quartile_mcc[1]
        variance_mcc = np.std(mcc_row)
        mcc_row.append(min_mcc)
        mcc_row.append(lower_quartile_mcc)
        mcc_row.append(avg_mcc)
        mcc_row.append(median_mcc)

        mcc_row.append(upper_quartile_mcc)
        mcc_row.append(max_mcc)
        mcc_row.append(variance_mcc)
        mcc_row.insert(0, inputfile + " mcc")
        mcc_writer.writerow(mcc_row)

        # stable smote
        stable_max_mcc = max(stable_mcc_row)
        stable_min_mcc = min(stable_mcc_row)
        stable_avg_mcc = np.mean(stable_mcc_row)
        stable_median_mcc = np.median(stable_mcc_row)
        stable_quartile_mcc = np.percentile(stable_mcc_row, (25, 75), interpolation='midpoint')
        stable_lower_quartile_mcc = stable_quartile_mcc[0]
        stable_upper_quartile_mcc = stable_quartile_mcc[1]
        stable_variance_mcc = np.std(stable_mcc_row)
        stable_mcc_row.append(stable_min_mcc)
        stable_mcc_row.append(stable_lower_quartile_mcc)
        stable_mcc_row.append(stable_avg_mcc)
        stable_mcc_row.append(stable_median_mcc)

        stable_mcc_row.append(stable_upper_quartile_mcc)
        stable_mcc_row.append(stable_max_mcc)
        stable_mcc_row.append(stable_variance_mcc)
        stable_mcc_row.insert(0, inputfile + " mcc")
        stable_mcc_writer.writerow(stable_mcc_row)
    max_precision = max(precision_row)
    min_precision = min(precision_row)
    avg_precision = np.mean(precision_row)
    median_precision = np.median(precision_row)
    quartile_precision = np.percentile(precision_row, (25, 75), interpolation='midpoint')
    lower_quartile_precision = quartile_precision[0]
    upper_quartile_precision = quartile_precision[1]
    variance_precision = np.std(precision_row)
    precision_row.append(min_precision)
    precision_row.append(lower_quartile_precision)
    precision_row.append(avg_precision)
    precision_row.append(median_precision)
    precision_row.append(upper_quartile_precision)
    precision_row.append(max_precision)
    precision_row.append(variance_precision)
    precision_row.insert(0, inputfile + " precision")
    precision_writer.writerow(precision_row)

    # f1指标统计分析
    max_f1 = max(f1_row)
    min_f1 = min(f1_row)
    avg_f1 = np.mean(f1_row)
    median_f1 = np.median(f1_row)
    quartile_f1 = np.percentile(f1_row, (25, 75), interpolation='midpoint')
    lower_quartile_f1 = quartile_f1[0]
    upper_quartile_f1 = quartile_f1[1]
    variance_f1 = np.std(f1_row)
    f1_row.append(min_f1)
    f1_row.append(lower_quartile_f1)
    f1_row.append(avg_f1)
    f1_row.append(median_f1)
    f1_row.append(upper_quartile_f1)
    f1_row.append(max_f1)
    f1_row.append(variance_f1)
    f1_row.insert(0, inputfile + " f1")
    f1_writer.writerow(f1_row)

    # gmean指标统计分析
    max_gmean = max(gmean_row)
    min_gmean = min(gmean_row)
    avg_gmean = np.mean(gmean_row)
    median_gmean = np.median(gmean_row)
    quartile_gmean = np.percentile(gmean_row, (25, 75), interpolation='midpoint')
    lower_quartile_gmean = quartile_gmean[0]
    upper_quartile_gmean = quartile_gmean[1]
    variance_gmean = np.std(gmean_row)
    gmean_row.append(min_gmean)
    gmean_row.append(lower_quartile_gmean)
    gmean_row.append(avg_gmean)
    gmean_row.append(median_gmean)
    gmean_row.append(upper_quartile_gmean)
    gmean_row.append(max_gmean)
    gmean_row.append(variance_gmean)
    gmean_row.insert(0, inputfile + " gmean")
    gmean_writer.writerow(gmean_row)

    # stable precision指标统计分析
    stable_max_precision = max(stable_precision_row)
    stable_min_precision = min(stable_precision_row)
    stable_avg_precision = np.mean(stable_precision_row)
    stable_median_precision = np.median(stable_precision_row)
    stable_quartile_precision = np.percentile(stable_precision_row, (25, 75), interpolation='midpoint')
    stable_lower_quartile_precision = stable_quartile_precision[0]
    stable_upper_quartile_precision = stable_quartile_precision[1]
    stable_variance_precision = np.std(stable_precision_row)
    stable_precision_row.append(stable_min_precision)
    stable_precision_row.append(stable_lower_quartile_precision)
    stable_precision_row.append(stable_avg_precision)
    stable_precision_row.append(stable_median_precision)
    stable_precision_row.append(stable_upper_quartile_precision)
    stable_precision_row.append(stable_max_precision)
    stable_precision_row.append(stable_variance_precision)
    stable_precision_row.insert(0, inputfile + " precision")
    stable_precision_writer.writerow(stable_precision_row)

    # stable f1指标统计分析
    stable_max_f1 = max(stable_f1_row)
    stable_min_f1 = min(stable_f1_row)
    stable_avg_f1 = np.mean(stable_f1_row)
    stable_median_f1 = np.median(stable_f1_row)
    stable_quartile_f1 = np.percentile(stable_f1_row, (25, 75), interpolation='midpoint')
    stable_lower_quartile_f1 = stable_quartile_f1[0]
    stable_upper_quartile_f1 = stable_quartile_f1[1]
    stable_variance_f1 = np.std(stable_f1_row)
    stable_f1_row.append(stable_min_f1)
    stable_f1_row.append(stable_lower_quartile_f1)
    stable_f1_row.append(stable_avg_f1)
    stable_f1_row.append(stable_median_f1)
    stable_f1_row.append(stable_upper_quartile_f1)
    stable_f1_row.append(stable_max_f1)
    stable_f1_row.append(stable_variance_f1)
    stable_f1_row.insert(0, inputfile + " f1")
    stable_f1_writer.writerow(stable_f1_row)

    # stable gmean指标统计分析
    stable_max_gmean = max(stable_gmean_row)
    stable_min_gmean = min(stable_gmean_row)
    stable_avg_gmean = np.mean(stable_gmean_row)
    stable_median_gmean = np.median(stable_gmean_row)
    stable_quartile_gmean = np.percentile(stable_gmean_row, (25, 75), interpolation='midpoint')
    stable_lower_quartile_gmean = stable_quartile_gmean[0]
    stable_upper_quartile_gmean = stable_quartile_gmean[1]
    stable_variance_gmean = np.std(stable_gmean_row)
    stable_gmean_row.append(stable_min_gmean)
    stable_gmean_row.append(stable_lower_quartile_gmean)
    stable_gmean_row.append(stable_avg_gmean)
    stable_gmean_row.append(stable_median_gmean)
    stable_gmean_row.append(stable_upper_quartile_gmean)
    stable_gmean_row.append(stable_max_gmean)
    stable_gmean_row.append(stable_variance_gmean)
    stable_gmean_row.insert(0, inputfile + " gmean")
    stable_gmean_writer.writerow(stable_gmean_row)


    # single_writer.writerow(auc_row)
    # single_writer.writerow(pf_row)
    # single_writer.writerow(recall_row)
    # single_writer.writerow(pf_row)
    # single_writer.writerow([])


precision_file.close()
f1_file.close()
gmean_file.close()
stable_precision_file.close()
stable_f1_file.close()
stable_gmean_file.close()
