import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer

# 读取数据并处理标签
dataset_url = "./EQ.csv"
# dataset_url1="./PC4_test.csv"
full_data = pd.read_csv(dataset_url)
# full_data['Defective'] = full_data['Defective'].apply(lambda x: 1 if x > 1 else x)

# 划分数据集
train_data, test_data = train_test_split(
    full_data, test_size=0.2, random_state=42, stratify=full_data["class"]
)

# train_data=full_data = pd.read_csv(dataset_url)
# test_data=pd.read_csv(dataset_url1)
# 分离特征和标签
X_train = train_data.drop(columns='class')
X_test = test_data.drop(columns='class')
Y_train = train_data["class"]
Y_test = test_data["class"]

# ------------------ 数据预处理 ------------------#
# 1. 缺失值填补（使用训练集的统计量处理测试集）
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# 2. 归一化处理（使用训练集的参数处理测试集）
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ 模型训练 ------------------#
voting_classifiers = [
    ['RF:', RandomForestClassifier(criterion="gini", max_depth=10,
                                  n_estimators=500, max_features=None, random_state=0)],
    ['SVM:', svm.SVC(kernel='poly', C=2, probability=True, random_state=0)],
    ['ANN:', MLPClassifier(hidden_layer_sizes=(2, 10, 10), max_iter=1000)],
    ['NBG:', GaussianNB()]
]

clf = VotingClassifier(estimators=voting_classifiers, voting='soft', verbose=1)
clf.fit(X_train, Y_train)

# ------------------ 训练集评估 ------------------#
training_predictions = clf.predict(X_train)
buggy_index = list(clf.classes_).index('buggy')
train_prob = clf.predict_proba(X_train)[:, buggy_index]

# 计算评估指标
precision = precision_score(Y_train, training_predictions, pos_label='buggy')
recall = recall_score(Y_train, training_predictions, pos_label='buggy')
specificity = recall_score(Y_train, training_predictions, pos_label='clean')
gmean_train = np.sqrt(recall * specificity)
auc = metrics.roc_auc_score(Y_train, train_prob)
f_measure = f1_score(Y_train, training_predictions, pos_label='buggy')
accuracy = accuracy_score(Y_train, training_predictions)

# 绘制混淆矩阵 (使用实际标签名称)
cm = confusion_matrix(Y_train, training_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['clean', 'buggy'])
disp.plot(cmap='Blues')
plt.title('Training Confusion Matrix')
plt.show()

# ------------------ 测试集评估 ------------------#
predictions = clf.predict(X_test)
buggy_index = list(clf.classes_).index('buggy')
test_prob = clf.predict_proba(X_test)[:, buggy_index]

# 计算评估指标
test_precision = precision_score(Y_test, predictions, pos_label='buggy')
test_recall = recall_score(Y_test, predictions, pos_label='buggy')
test_specificity = recall_score(Y_test, predictions, pos_label='clean')
gmean_test = np.sqrt(test_recall * test_specificity)
test_auc = metrics.roc_auc_score(Y_test, test_prob)
test_f_measure = f1_score(Y_test, predictions, pos_label='buggy')
test_accuracy = accuracy_score(Y_test, predictions)

# 绘制混淆矩阵 (使用实际标签名称)
cm_test = confusion_matrix(Y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=['clean', 'buggy'])
disp.plot(cmap='Blues')
plt.title('Test Confusion Matrix')
plt.show()

# ------------------ 打印评估结果 ------------------#
print("\n================ Training Results ================")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"G-Mean: {gmean_train:.4f}")
print(f"F1-Score: {f_measure:.4f}")
print(f"AUC: {auc:.4f}")
print(classification_report(Y_train, training_predictions))

print("\n================ Test Results ================")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall (Sensitivity): {test_recall:.4f}")
print(f"Specificity: {test_specificity:.4f}")
print(f"G-Mean: {gmean_test:.4f}")
print(f"F1-Score: {test_f_measure:.4f}")
print(f"AUC: {test_auc:.4f}")
print(classification_report(Y_test, predictions))