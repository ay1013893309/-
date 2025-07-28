from scipy.optimize import differential_evolution
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
# from SMOTE_backup import Smote
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import time
import csv
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.neighbors import KNeighborsRegressor
np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')
class COSTE:
    def __init__(self,
                 population_size=10,
                 generation=10,
                 cf=0.3,
                 f=0.7,
                 fold=5,  # cross validation
                 ):
        self.population_size = population_size
        self.generation = generation
        self.cf = cf
        self.f = f
        self.fold = fold
        self.skf = StratifiedKFold(n_splits=fold)

    def read_csv(self, file):  # 读取数据并进行初步处理，将bug全改为1， 并用min-max标准化 返回处理过的数据集
        dataset = pd.read_csv("G:\pycharm\lutrs\code\data\\" + file)
        defect_ratio = len(dataset[dataset["bug"] > 0]) / len(dataset)
        # defect_number = len(dataset[dataset["bug"] > 0])
        # clean_number = len(dataset[dataset["bug"] == 0])
        total_number = len(dataset)
        if defect_ratio > 0.45:  # 注意这点导致的有几个文件没有选上
            # print(file, " defect ratio larger than 0.5")
            return pd.DataFrame()

        for i in range(total_number):
            if dataset.loc[i, "bug"] > 0:
                dataset.loc[i, "bug"] = 1
        dataset = dataset.drop(columns="name")
        dataset = dataset.drop(columns="version")
        dataset = dataset.drop(columns="name.1")
        # dataset = dataset[0:20]
        cols = list(dataset.columns)
        for col in cols:
            column_max = dataset[col].max()
            column_min = dataset[col].min()
            dataset[col] = (dataset[col] - column_min) / (column_max - column_min)
        return dataset

    def process(self, file):
        print("start time: ", time.asctime(time.localtime(time.time())))
        # bound = [(0.0001, 1), (0.001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 1)]
        bound = [(-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                 (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1),
                 (-1, 1), (-1, 1), (-1, 1), (-1, 1)]
        # bound = [(-0.5, 1), (-0.5, 1), (-0.5, 1), (-0.5, 1), (-0.5, 1), (-0.5, 1), (-0.5, 1), (-0.5, 1),
        #          (-0.5, 1), (-0.5, 1), (-0.5, 1), (-0.5, 1), (-0.5, 1), (-0.5, 1), (-0.5, 1), (-0.5, 1),
        #          (-0.5, 1), (-0.5, 1), (-0.5, 1), (-0.5, 1)]
        dataset = self.read_csv(file)
        if dataset.empty:
            return False
        dataset = dataset.values.tolist()
        # temp_tuple = ()
        # for i in dataset:
            # temp_tuple = temp_tuple + tuple([i])
        result = differential_evolution(self.fit, bound, args=dataset, popsize=10, maxiter=20, mutation=0.3, recombination=0.9)
        print(result)
        generation = result.nit
        optimal_param = result.x

        dataset = np.array(dataset)
        # print(dataset[:, 10])
        y = dataset[:, -1]
        x = dataset

        total_auc = 0
        total_recall = 0
        total_pf = 0
        total_balance = 0
        total_normpopt = 0
        total_acc20 = 0
        for train, test in self.skf.split(x, y):
            train_x = x[train]
            train_y = y[train]
            test_x = x[test]

            # 以下是用来求popt数据的
            total_line_of_code = np.sum(test_x[:, 10])
            total_bug = np.sum(test_x[:, -1])
            original_density = test_x[:, -1] / test_x[:, 10]  #
            popt_test_x = test_x.copy()
            popt_test_x = np.c_[popt_test_x, original_density]
            popt_test_x = popt_test_x[np.argsort(-popt_test_x[:, -1])]
            optimal_x = [0]  # x坐标是累计代码行数
            optimal_y = [0]  # y坐标是累计缺陷密度
            for s in range(len(popt_test_x)):
                optimal_x.append(popt_test_x[s][10] / total_line_of_code + optimal_x[-1])
                optimal_y.append(popt_test_x[s][-2] / total_bug + optimal_y[-1])
            # optimal_auc = auc(optimal_x, optimal_y)
            optimal_auc = 0
            prev_x = 0
            prev_y = 0
            for q, w in zip(optimal_x, optimal_y):
                if q != prev_x:
                    optimal_auc = optimal_auc + (q - prev_x) * (w + prev_y) / 2
                    prev_x = q
                    prev_y = w
            worst_popt_test_x = popt_test_x[np.argsort(popt_test_x[:, -1])]
            worst_x = [0]
            worst_y = [0]
            for s in range(len(worst_popt_test_x)):
                worst_x.append(worst_popt_test_x[s][10] / total_line_of_code + worst_x[-1])
                worst_y.append(worst_popt_test_x[s][-2] / total_bug + worst_y[-1])

            worst_auc = 0
            prev_x = 0
            prev_y = 0
            for q, w in zip(worst_x, worst_y):
                if q != prev_x:
                    worst_auc = worst_auc + (q - prev_x) * (w + prev_y) / 2
                    prev_x = q
                    prev_y = w
            # minority_dataset = minority_dataset[np.argsort(-minority_dataset[:, -1])]
            test_x = np.delete(test_x, -1, axis=1)
            test_y = y[test]
            sampled_train_x = self.COSTE(train_x, optimal_param)
            final_train_y = sampled_train_x[:, -1]
            final_train_x = np.delete(sampled_train_x, -1, axis=1)

            clf = MLPClassifier(random_state=0)  # neighbors.KNeighborsClassifier(n_neighbors=5) # MLPClassifier()  # LogisticRegression()  # svm.SVC()  # RandomForestClassifier()

            clf.fit(final_train_x, final_train_y)
            predict_result = clf.predict(test_x)
            predict_density = predict_result / test_x[:, 10]
            predict_popt_test_x = test_x.copy()
            predict_popt_test_x = np.c_[predict_popt_test_x, test_y]  # 这里添加bug column 方便计算
            predict_popt_test_x = np.c_[predict_popt_test_x, predict_density]
            predict_popt_test_x = predict_popt_test_x[np.argsort(-predict_popt_test_x[:, -1])]
            predict_x = [0]
            predict_y = [0]
            # 下面是求ACC，也就是recall@20%
            total_line_of_code_20 = 0.2 * total_line_of_code
            for s in range(len(predict_popt_test_x)):
                predict_x.append(predict_popt_test_x[s][10] / total_line_of_code + predict_x[-1])
                predict_y.append(predict_popt_test_x[s][-2] / total_bug + predict_y[-1])
            count_line = 0
            count_instance = []
            for s in range(len(predict_popt_test_x)):
                if count_line < total_line_of_code_20:
                    count_instance.append(predict_popt_test_x[s].tolist())
                    count_line = count_line + predict_popt_test_x[s][10]
                else:
                    break
            count_instance = np.array(count_instance)
            bug_20 = np.sum(count_instance[:, -2])
            acc_20 = recall_20 = bug_20 / total_bug
            predict_auc = 0
            prev_x = 0
            prev_y = 0
            for q, w in zip(predict_x, predict_y):
                if q != prev_x:
                    predict_auc = predict_auc + (q - prev_x) * (w + prev_y) / 2
                    prev_x = q
                    prev_y = w
            # popt = 1 - optimal_auc
            # normpopt = (popt - worst_auc) / (optimal_auc - worst_auc)
            normpopt = ((1 - (optimal_auc - predict_auc)) - worst_auc) / (1 - worst_auc)
            # test_popt = (predict_auc - worst_auc) / (optimal_auc - worst_auc)
            total_normpopt = total_normpopt + normpopt
            total_acc20 = total_acc20 + acc_20
            # predict_result[predict_result > 0.5] = 1
            # predict_result[predict_result <= 0.5] = 0
            true_negative, false_positive, false_negative, true_positive = confusion_matrix(test_y, predict_result).ravel()

            coste_auc = roc_auc_score(test_y, predict_result)
            recall = recall_score(test_y, predict_result)

            pf = false_positive / (true_negative + false_positive)
            total_pf = total_pf + pf

            balance = 1 - (((0 - pf) ** 2 + (1 - recall) ** 2) / 2) ** 0.5
            total_balance = total_balance + balance

            total_auc = total_auc + coste_auc
            total_recall = total_recall + recall

        average_auc = total_auc / self.fold
        average_recall = total_recall / self.fold
        average_pf = total_pf / self.fold
        average_balance = total_balance / self.fold
        average_norm_popt = total_normpopt / self.fold
        average_acc20 = total_acc20 / self.fold
        print("\033[1;32m", file, " auc: ", average_auc, "\033[0m")
        print("normpopt:", average_norm_popt)
        print("acc20: ", average_acc20)
        single_writer.writerow([file, average_auc, average_recall, average_pf, average_balance, average_norm_popt, average_acc20,
                                optimal_param[0], optimal_param[1], optimal_param[2], optimal_param[3], optimal_param[4],
                                optimal_param[5], optimal_param[6], optimal_param[7], optimal_param[8], optimal_param[9],
                                optimal_param[10], optimal_param[11], optimal_param[12], optimal_param[13], optimal_param[14],
                                optimal_param[15], optimal_param[16], optimal_param[17], optimal_param[18], optimal_param[19],
                                ])

    # def final_train(self, COSTEparam):

    def fit(self, bound, *param):
        temp_list = []
        for z in param:
            temp_list.append(z)
        temp_list = np.array(temp_list)

        y = temp_list[:, -1]
        x = temp_list
        total_auc = 0
        for train, test in self.skf.split(x, y):
            train_x = x[train]
            train_y = y[train]
            test_x = x[test]
            test_y = y[test]
            for sub_train, sub_test in self.skf.split(train_x, train_y):
                # print("-----------------------------")
                # print(sub_test)
                sub_train_x = train_x[sub_train]
                sub_test_x = train_x[sub_test]
                sub_test_x = np.delete(sub_test_x, -1, axis=1)
                sub_test_y = train_y[sub_test]
                sampled_sub_train_x = self.COSTE(sub_train_x, bound)

                sub_train_y = sampled_sub_train_x[:, -1]
                sub_train_x = np.delete(sampled_sub_train_x, -1, axis=1)
                clf = MLPClassifier(random_state=0) #  svm.SVC()  # neighbors.KNeighborsClassifier(n_neighbors=5)

                clf.fit(sub_train_x, sub_train_y)

                predict_result = clf.predict(sub_test_x)
                auc = roc_auc_score(sub_test_y, predict_result)
                total_auc = total_auc + auc

        average_auc = total_auc / (self.fold * self.fold)
        average_auc = - average_auc
        return average_auc

    def COSTE(self, sampling_dataset, param):
        minority_dataset = sampling_dataset[sampling_dataset[:, -1] > 0]
        majority_dataset = sampling_dataset[sampling_dataset[:, -1] == 0]
        minority_dataset = np.delete(minority_dataset, -1, axis=1)
        minority_dataset = minority_dataset * param
        total_complexity = np.sum(minority_dataset, axis=1)  #  + 20
        minority_dataset = np.insert(minority_dataset, minority_dataset.shape[1], values=total_complexity, axis=1)

        synthetic_minority_dataset = self.recursion(minority_dataset, majority_dataset)

        synthetic_minority_dataset = np.delete(synthetic_minority_dataset, -1, axis=1)
        synthetic_minority_dataset = synthetic_minority_dataset / param
        synthetic_minority_dataset = np.insert(synthetic_minority_dataset, synthetic_minority_dataset.shape[1], values=1, axis=1)

        synthetic_dataset = np.append(synthetic_minority_dataset, majority_dataset, axis=0)
        return synthetic_dataset


    def recursion(self, minority_dataset, majority_dataset):

        majority_number = len(majority_dataset)
        minority_number = len(minority_dataset)
        if minority_number == 1:
            return minority_dataset

        # majority_dataset["bug"] = 0
        # minority_dataset["bug"] = 1
        # temp_dataset = pd.concat([minority_dataset, majority_dataset])
        # temp_dataset = temp_dataset.sort_values(by="average_complexity", ascending=True)
        # temp_dataset = temp_dataset.reset_index()
        # temp_dataset = temp_dataset.drop(columns="index")
        # minority_dataset = temp_dataset[temp_dataset["bug"] > 0]
        minority_dataset = minority_dataset[np.argsort(-minority_dataset[:, -1])]

        # minority_dataset = minority_dataset.sort_values(by="average_complexity", ascending=True)
        need_number = (majority_number * 1) - minority_number
        need_number = int(np.ceil(need_number))

        # index_array = minority_dataset.index.values

        # # 以下是一种方法，现阶段效果最好的方法
        if need_number > (minority_number - 1):  # 如果需要的数量大于原始数据集能生成的数量，那么就需要多次循环，第一次循环用全部的原始数据生成
            for i in range(minority_number):
                if i + 1 == minority_number:
                    continue

                # 这段代码是为了判断两个实例之间间隔是否大于平均间隔
                # i_index = index_array[i]
                # i_1_index = index_array[i + 1]

                # gap = i_1_index - i_index - 1

                # if gap > average_gap:
                    continue

                # 以上代码是为了判断两个实例之间间隔是否大于平均间隔
                temp_instance = (minority_dataset[i] + minority_dataset[i+1]) / 2# pd.DataFrame((minority_dataset.iloc[i] + minority_dataset.iloc[i + 1]) / 2)
                # minority_dataset = pd.concat([minority_dataset, temp_instance.T])
                minority_dataset = np.row_stack((minority_dataset, temp_instance))

            return self.recursion(minority_dataset, majority_dataset)

        elif need_number <= (minority_number - 1) and (need_number != 0):
            for i in range(need_number):
                temp_instance = (minority_dataset[i] + minority_dataset[i+1]) / 2# pd.DataFrame((minority_dataset.iloc[i] + minority_dataset.iloc[i + 1]) / 2)
                minority_dataset = np.row_stack((minority_dataset, temp_instance))
            return self.recursion(minority_dataset, majority_dataset)

        elif need_number == 0:

            return minority_dataset


if __name__ == '__main__':
    # initial guess for variation of parameters
    #             a            b            c
    for iteration in range(10):  # 总体循环次数
        single = open('G:\pycharm\lutrs\code\data' + str(iteration) + '.csv', 'w',
                      newline='')
        single_writer = csv.writer(single)
        single_writer.writerow(
            ["inputfile", "auc", "recall", "pf", "balance", "normpopt", "acc20", "wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", "lcom3", "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc"])
        for inputfile in os.listdir("G:\pycharm\lutrs\code\data"):
            if inputfile == "jedit-4.3.csv" or inputfile == "synapse-1.0.csv" or inputfile == "synapse-1.1.csv":
                continue
            start_time = time.time()  # time.asctime(time.localtime(time.time()))
            de = COSTE()
            de.process(inputfile)
            end_time = time.time()

