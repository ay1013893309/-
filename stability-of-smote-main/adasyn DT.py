from imblearn.over_sampling import ADASYN
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, matthews_corrcoef, brier_score_loss
import csv
import warnings
from sklearn import neighbors
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import random
import time
from collections import Counter

# 忽略警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 使用决策树分类器
classifier = "tree"
np.set_printoptions(suppress=True)
neighbor = 5
target_defect_ratio = 0.5

# 决策树的超参数网格
tuned_parameters = {
    'max_depth': [3, 5, 7, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}


class stable_ADASYN:
    def __init__(self, z_nearest=5):
        self.z_nearest = z_nearest

    def fit_sample(self, x_dataset, y_dataset=None):
        # 将输入转换为DataFrame
        if not isinstance(x_dataset, pd.DataFrame):
            x_dataset = pd.DataFrame(x_dataset)

        # 获取特征列和目标列
        n_features = x_dataset.shape[1] - 1
        feature_cols = list(range(n_features))
        target_col = n_features

        # 分离缺陷样本和干净样本
        clean_dataset = x_dataset[x_dataset[target_col] == 0]
        defect_dataset = x_dataset[x_dataset[target_col] > 0]
        defect_number = len(defect_dataset)
        clean_number = len(clean_dataset)

        if defect_number == 0:
            return pd.concat([clean_dataset, defect_dataset])

        total_number = len(x_dataset)
        need_number = int((target_defect_ratio * total_number - defect_number) / (1 - target_defect_ratio))

        if need_number <= 0:
            return pd.concat([clean_dataset, defect_dataset])

        total_ratio = 0
        container = pd.DataFrame()
        generated_dataset = []
        total_pair = []

        # 计算每个缺陷样本的困难程度
        for idx, row in defect_dataset.iterrows():
            copy_dataset = x_dataset.copy()
            distances = np.sqrt(((row[feature_cols] - copy_dataset[feature_cols]) ** 2).sum(axis=1))
            copy_dataset['distance'] = distances
            nearest_neighbors = copy_dataset.nsmallest(self.z_nearest + 1, 'distance')[1:]
            majority_number = len(nearest_neighbors[nearest_neighbors[target_col] == 0])
            ratio = majority_number / self.z_nearest
            total_ratio += ratio

        # 生成合成样本
        for idx, row in defect_dataset.iterrows():
            copy_dataset = x_dataset.copy()
            distances = np.sqrt(((row[feature_cols] - copy_dataset[feature_cols]) ** 2).sum(axis=1))
            copy_dataset['distance'] = distances
            nearest_neighbors = copy_dataset.nsmallest(self.z_nearest + 1, 'distance')[1:]
            majority_number = len(nearest_neighbors[nearest_neighbors[target_col] == 0])
            ratio = majority_number / self.z_nearest
            normalized_ratio = ratio / total_ratio if total_ratio > 0 else 0
            single_need_number = round(normalized_ratio * need_number)

            if single_need_number <= 0:
                continue

            # 在缺陷样本中寻找最近邻
            defect_copy = defect_dataset.copy()
            distances = np.sqrt(((row[feature_cols] - defect_copy[feature_cols]) ** 2).sum(axis=1))
            defect_copy['distance'] = distances
            neighbors = defect_copy.nsmallest(self.z_nearest + 1, 'distance')[1:]

            # 生成样本对
            rround = single_need_number / self.z_nearest
            while rround >= 1:
                for neighbor_idx in neighbors.index:
                    total_pair.append(tuple(sorted([idx, neighbor_idx])))
                rround -= 1

            # 处理剩余需要生成的样本
            number_on_each_instance = round(rround * self.z_nearest)
            if number_on_each_instance > 0:
                target_samples = neighbors.sample(n=number_on_each_instance)
                for neighbor_idx in target_samples.index:
                    total_pair.append(tuple(sorted([idx, neighbor_idx])))

        # 生成合成样本
        pair_counter = Counter(total_pair)
        for pair, count in pair_counter.items():
            row1 = defect_dataset.loc[pair[0], feature_cols].values
            row2 = defect_dataset.loc[pair[1], feature_cols].values

            for i in np.linspace(0, 1, count + 2)[1:-1]:
                new_sample = row1 * (1 - i) + row2 * i
                new_sample = np.append(new_sample, 1)  # 添加目标值1
                generated_dataset.append(new_sample)

        # 创建最终数据集
        final_generated = pd.DataFrame(generated_dataset)
        result = pd.concat([clean_dataset, defect_dataset, final_generated], ignore_index=True)
        return result


def preprocess_nasa_data(dataset):
    # 删除不必要的列
    dataset = dataset.drop(columns=['name', 'version', 'name.1'], errors='ignore')

    # 处理目标变量
    dataset['bug'] = dataset['bug'].apply(lambda x: 1 if x > 0 else 0)

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
    # 标准ADASYN结果文件
    file = open(
        f'G:\pycharm\lutrs\stability-of-smote-main\k value adasyn {neighbor}_{metric}_adasyn_result_on_{classifier}.csv',
        'w', newline='')
    writer = csv.writer(file)
    writer.writerow(["inputfile", "min", "lower", "avg", "median", "upper", "max", "variance"])
    writers[metric] = writer

    # stable_ADASYN结果文件
    stable_file = open(
        f'G:\pycharm\lutrs\stability-of-smote-main\k value adasyn {neighbor}_{metric}_stable_adasyn_result_on_{classifier}.csv',
        'w', newline='')
    stable_writer = csv.writer(stable_file)
    stable_writer.writerow(["inputfile", "min", "lower", "avg", "median", "upper", "max", "variance"])
    stable_writers[metric] = stable_writer

# 处理NASA数据集
nasa_dir = "G:\pycharm\lutrs\stability-of-smote-main\\ant"
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
            # 划分训练测试集 - 使用train_test_split代替bootstrap
            train_data, test_data = train_test_split(dataset, test_size=0.3, stratify=dataset['bug'])

            # 确保有足够的缺陷样本
            if train_data['bug'].sum() < neighbor or test_data['bug'].sum() == 0:
                continue

            # 分离特征和目标
            train_x = train_data.drop(columns='bug').values
            train_y = train_data['bug'].values
            test_x = test_data.drop(columns='bug').values
            test_y = test_data['bug'].values

            # 网格搜索寻找最佳参数（决策树）
            dt_clf = tree.DecisionTreeClassifier(random_state=42)
            validation_clf = GridSearchCV(dt_clf, tuned_parameters, cv=3, n_jobs=-1)
            validation_clf.fit(train_x, train_y)
            best_params = validation_clf.best_params_

            print(f"Best parameters for {inputfile} iteration {j + 1}: {best_params}")

            # 标准ADASYN实验
            smote = ADASYN(n_neighbors=neighbor)
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

            # stable_ADASYN实验
            train_data_with_target = pd.concat([pd.DataFrame(train_x), pd.Series(train_y)], axis=1)
            stable_adasyn = stable_ADASYN(z_nearest=neighbor)
            stable_train_data = stable_adasyn.fit_sample(train_data_with_target.values)

            if stable_train_data is not None and len(stable_train_data) > 0:
                stable_train_x = stable_train_data.iloc[:, :-1].values
                stable_train_y = stable_train_data.iloc[:, -1].values

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

                # 写入标准ADASYN结果
                writers[metric].writerow([
                    inputfile, min_val, q1, avg_val, median_val, q3, max_val, std_val
                ])

                # 计算stable_ADASYN统计量
                if stable_results[metric]:
                    min_val = min(stable_results[metric])
                    max_val = max(stable_results[metric])
                    avg_val = np.mean(stable_results[metric])
                    median_val = np.median(stable_results[metric])
                    q1, q3 = np.percentile(stable_results[metric], [25, 75])
                    std_val = np.std(stable_results[metric])

                    # 写入stable_ADASYN结果
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