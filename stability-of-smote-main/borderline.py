from imblearn.over_sampling import BorderlineSMOTE  # 修改这里
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, matthews_corrcoef, brier_score_loss
import csv
import warnings
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from collections import Counter
import random
import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler

# 忽略特定警告
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=196)
warnings.filterwarnings("ignore", category=UserWarning)

classifier = "tree"  # 决策树分类器
neighbor = 5
# 决策树的超参数网格
tuned_parameters = {
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}
target_defect_ratio = 0.5


class stable_SMOTE:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest

    def fit_sample(self, x_dataset, y_dataset=None):
        # 将输入转换为DataFrame
        if not isinstance(x_dataset, pd.DataFrame):
            x_dataset = pd.DataFrame(x_dataset)

        # 获取特征列和目标列
        feature_cols = list(range(x_dataset.shape[1] - 1))
        target_col = x_dataset.shape[1] - 1

        # 分离缺陷样本和干净样本
        defective_instance = x_dataset[x_dataset[target_col] > 0]
        clean_instance = x_dataset[x_dataset[target_col] == 0]
        defective_number = len(defective_instance)

        if defective_number == 0:
            return pd.concat([clean_instance, defective_instance])

        total_number = len(x_dataset)
        need_number = int((target_defect_ratio * total_number - defective_number) / (1 - target_defect_ratio))

        if need_number <= 0:
            return pd.concat([clean_instance, defective_instance])

        generated_dataset = []
        count = 0
        container = pd.DataFrame()

        # 选择稳定样本
        for idx, row in defective_instance.iterrows():
            copy_dataset = x_dataset.copy()
            distances = np.sqrt(((row[feature_cols] - copy_dataset[feature_cols]) ** 2).sum(axis=1))
            copy_dataset['distance'] = distances
            nearest_neighbors = copy_dataset.nsmallest(self.z_nearest + 1, 'distance')[1:]
            majority_number = len(nearest_neighbors[nearest_neighbors[target_col] == 0])

            if majority_number >= 2.5 and majority_number < 5:
                container = pd.concat([container, row.to_frame().T])

        involve_defective_number = len(container)
        if involve_defective_number == 0:
            return pd.concat([clean_instance, defective_instance])

        number_on_each_instance = need_number / involve_defective_number
        total_pair = []
        rround = number_on_each_instance / self.z_nearest

        # 生成样本对
        while rround >= 1:
            for idx, row in container.iterrows():
                temp_defective = defective_instance.copy()
                distances = np.sqrt(((row[feature_cols] - temp_defective[feature_cols]) ** 2).sum(axis=1))
                temp_defective['distance'] = distances
                neighbors = temp_defective.nsmallest(self.z_nearest + 1, 'distance')[1:]

                for neighbor_idx in neighbors.index:
                    total_pair.append(tuple(sorted([idx, neighbor_idx])))

            rround -= 1

        need_number1 = need_number - len(total_pair)
        number_on_each_instance = need_number1 / involve_defective_number

        for idx, row in container.iterrows():
            temp_defective = defective_instance.copy()
            distances = np.sqrt(((row[feature_cols] - temp_defective[feature_cols]) ** 2).sum(axis=1))
            temp_defective['distance'] = distances
            neighbors = temp_defective.nsmallest(self.z_nearest + 1, 'distance')[1:]
            neighbors = neighbors.nlargest(int(number_on_each_instance), 'distance')

            for neighbor_idx in neighbors.index:
                total_pair.append(tuple(sorted([idx, neighbor_idx])))

        residue_number = need_number - len(total_pair)
        residue_defective = container.sample(n=residue_number, replace=True)

        for idx, row in residue_defective.iterrows():
            temp_defective = defective_instance.copy()
            distances = np.sqrt(((row[feature_cols] - temp_defective[feature_cols]) ** 2).sum(axis=1))
            temp_defective['distance'] = distances
            neighbors = temp_defective.nsmallest(self.z_nearest + 1, 'distance')[1:]
            target_neighbor = neighbors.iloc[-1:].index[0]
            total_pair.append(tuple(sorted([idx, target_neighbor])))

        # 生成合成样本
        pair_counter = Counter(total_pair)
        for pair, count in pair_counter.items():
            row1 = defective_instance.loc[pair[0], feature_cols].values
            row2 = defective_instance.loc[pair[1], feature_cols].values

            for i in np.linspace(0, 1, count + 2)[1:-1]:
                new_sample = row1 * (1 - i) + row2 * i
                new_sample = np.append(new_sample, 1)  # 添加目标值1
                generated_dataset.append(new_sample)

        # 创建最终数据集
        final_generated = pd.DataFrame(generated_dataset)
        result = pd.concat([clean_instance, defective_instance, final_generated], ignore_index=True)
        return result


def preprocess_nasa_data(dataset):
    # 删除不必要的列
    dataset = dataset.drop(columns=['name', 'version', 'name.1'], errors='ignore')

    # 处理目标变量
    dataset['bug'] = dataset['bug'].apply(lambda x: 1 if x > 0 else 0)

    # 删除缺失值过多的列
    missing_cols = dataset.columns[dataset.isnull().mean() > 0.5]
    dataset = dataset.drop(columns=missing_cols)

    # 填充剩余缺失值
    for col in dataset.columns:
        if dataset[col].dtype in ['int64', 'float64']:
            dataset[col] = dataset[col].fillna(dataset[col].median())
        else:
            dataset[col] = dataset[col].fillna(dataset[col].mode()[0])

    # 归一化
    scaler = MinMaxScaler()
    for col in dataset.columns[:-1]:
        if dataset[col].dtype in ['int64', 'float64']:
            dataset[col] = scaler.fit_transform(dataset[[col]])

    return dataset


# 初始化结果文件
metrics = ['auc', 'balance', 'recall', 'pf', 'brier', 'mcc']
writers = {}
stable_writers = {}

for metric in metrics:
    # 标准SMOTE结果文件
    file = open(
        f'G:\pycharm\lutrs\stability-of-smote-main\k value borderline {neighbor}_{metric}_borderline_result_on_{classifier}.csv',
        'w', newline='')
    writer = csv.writer(file)
    writer.writerow(["inputfile", "min", "lower", "avg", "median", "upper", "max", "variance"])
    writers[metric] = writer

    # stable_SMOTE结果文件
    stable_file = open(
        f'G:\pycharm\lutrs\stability-of-smote-main\k value borderline {neighbor}_{metric}_stable_borderline_result_on_{classifier}.csv',
        'w', newline='')
    stable_writer = csv.writer(stable_file)
    stable_writer.writerow(["inputfile", "min", "lower", "avg", "median", "upper", "max", "variance"])
    stable_writers[metric] = stable_writer

# 处理NASA数据集
nasa_dir = "G:\pycharm\lutrs\stability-of-smote-main\\NASA"
for inputfile in os.listdir(nasa_dir):
    if not inputfile.endswith('.csv'):
        continue

    print(f"Processing {inputfile}...")
    start_time = time.time()

    try:
        # 加载并预处理数据
        dataset = pd.read_csv(os.path.join(nasa_dir, inputfile))
        dataset = preprocess_nasa_data(dataset)

        # 检查缺陷比例
        defect_ratio = dataset['bug'].mean()
        if defect_ratio > 0.45 or defect_ratio < 0.05:
            print(f"Skipped {inputfile} due to defect ratio: {defect_ratio:.4f}")
            continue

        # 为当前文件初始化结果存储
        results = {metric: [] for metric in metrics}
        stable_results = {metric: [] for metric in metrics}

        # 10次重复实验
        for j in range(10):
            # 划分训练测试集
            train_data, test_data = train_test_split(dataset, test_size=0.3, stratify=dataset['bug'])

            # 确保有足够的缺陷样本
            if train_data['bug'].sum() < neighbor or test_data['bug'].sum() == 0:
                continue

            # 分离特征和目标
            train_x = train_data.drop(columns='bug')
            train_y = train_data['bug']
            test_x = test_data.drop(columns='bug')
            test_y = test_data['bug']

            # 网格搜索寻找最佳参数（决策树）
            dt_clf = tree.DecisionTreeClassifier(random_state=42)
            validation_clf = GridSearchCV(dt_clf, tuned_parameters, cv=3, n_jobs=-1)
            validation_clf.fit(train_x, train_y)
            best_params = validation_clf.best_params_

            print(f"Best parameters for {inputfile} iteration {j + 1}: {best_params}")

            # 标准BorderlineSMOTE实验 - 修复这里
            smote = BorderlineSMOTE(kind='borderline-1', k_neighbors=neighbor)  # 使用BorderlineSMOTE
            smote_train_x, smote_train_y = smote.fit_resample(train_x, train_y)

            clf = tree.DecisionTreeClassifier(**best_params, random_state=42)
            clf.fit(smote_train_x, smote_train_y)
            predict_result = clf.predict(test_x)

            # 计算指标
            tn, fp, fn, tp = confusion_matrix(test_y, predict_result).ravel()
            results['recall'].append(tp / (tp + fn))
            results['pf'].append(fp / (fp + tn))
            results['balance'].append(
                1 - (((0 - results['pf'][-1]) ** 2 + (1 - results['recall'][-1]) ** 2) / 2) ** 0.5)
            results['auc'].append(roc_auc_score(test_y, predict_result))
            results['brier'].append(brier_score_loss(test_y, predict_result))
            results['mcc'].append(matthews_corrcoef(test_y, predict_result))

            # stable_SMOTE实验
            train_data_with_target = pd.concat([train_x, train_y], axis=1)
            stable_smote = stable_SMOTE(z_nearest=neighbor)
            stable_train_data = stable_smote.fit_sample(train_data_with_target.values)

            # 检查stable_train_data是否有效
            if stable_train_data is not False and len(stable_train_data) > 0:
                if isinstance(stable_train_data, pd.DataFrame):
                    stable_train_x = stable_train_data.iloc[:, :-1]
                    stable_train_y = stable_train_data.iloc[:, -1]
                else:  # 处理numpy数组情况
                    stable_train_x = stable_train_data[:, :-1]
                    stable_train_y = stable_train_data[:, -1]

                stable_clf = tree.DecisionTreeClassifier(**best_params, random_state=42)
                stable_clf.fit(stable_train_x, stable_train_y)
                stable_predict_result = stable_clf.predict(test_x)

                # 计算指标
                tn, fp, fn, tp = confusion_matrix(test_y, stable_predict_result).ravel()
                stable_results['recall'].append(tp / (tp + fn))
                stable_results['pf'].append(fp / (fp + tn))
                stable_results['balance'].append(
                    1 - (((0 - stable_results['pf'][-1]) ** 2 + (1 - stable_results['recall'][-1]) ** 2) / 2) ** 0.5)
                stable_results['auc'].append(roc_auc_score(test_y, stable_predict_result))
                stable_results['brier'].append(brier_score_loss(test_y, stable_predict_result))
                stable_results['mcc'].append(matthews_corrcoef(test_y, stable_predict_result))
            else:
                # 如果stable_SMOTE没有生成新样本，使用原始数据
                stable_clf = tree.DecisionTreeClassifier(**best_params, random_state=42)
                stable_clf.fit(train_x, train_y)
                stable_predict_result = stable_clf.predict(test_x)

                # 计算指标
                tn, fp, fn, tp = confusion_matrix(test_y, stable_predict_result).ravel()
                stable_results['recall'].append(tp / (tp + fn))
                stable_results['pf'].append(fp / (fp + tn))
                stable_results['balance'].append(
                    1 - (((0 - stable_results['pf'][-1]) ** 2 + (1 - stable_results['recall'][-1]) ** 2) / 2) ** 0.5)
                stable_results['auc'].append(roc_auc_score(test_y, stable_predict_result))
                stable_results['brier'].append(brier_score_loss(test_y, stable_predict_result))
                stable_results['mcc'].append(matthews_corrcoef(test_y, stable_predict_result))

        # 保存结果
        for metric in metrics:
            if results[metric]:
                # 计算统计量
                min_val = min(results[metric])
                max_val = max(results[metric])
                avg_val = np.mean(results[metric])
                median_val = np.median(results[metric])
                q1, q3 = np.percentile(results[metric], [25, 75])
                std_val = np.std(results[metric])

                # 写入标准SMOTE结果
                writers[metric].writerow([
                    inputfile, min_val, q1, avg_val, median_val, q3, max_val, std_val
                ])

                # 计算stable_SMOTE统计量
                if stable_results[metric]:
                    min_val = min(stable_results[metric])
                    max_val = max(stable_results[metric])
                    avg_val = np.mean(stable_results[metric])
                    median_val = np.median(stable_results[metric])
                    q1, q3 = np.percentile(stable_results[metric], [25, 75])
                    std_val = np.std(stable_results[metric])

                    # 写入stable_SMOTE结果
                    stable_writers[metric].writerow([
                        inputfile, min_val, q1, avg_val, median_val, q3, max_val, std_val
                    ])

    except Exception as e:
        print(f"Error processing {inputfile}: {str(e)}")
        import traceback

        traceback.print_exc()

    print(f"Completed {inputfile} in {time.time() - start_time:.2f} seconds")

# 关闭所有文件
for writer in writers.values():
    writer.writerow([])
for stable_writer in stable_writers.values():
    stable_writer.writerow([])

print("All experiments completed!")