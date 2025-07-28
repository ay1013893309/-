import pandas as pd
import re

# 读取数据，header=None表示数据没有表头
df = pd.read_csv('G:\pycharm\lutrs\stability-of-smote-main\k vlaue5recall_stable_smote_result_on_tree.csv', header=None)

# 获取第一列和倒数第五列的索引
first_col_index = 0
fifth_last_col_index = df.shape[1] - 5  # 倒数第五列的索引

# 提取需要的列
selected_data = df.iloc[:, [first_col_index, fifth_last_col_index]].copy()

# 设置列名以便于理解
selected_data.columns = ['first_column', 'fifth_last_column']


# 定义函数从字符串中提取数值
def extract_number(value):
    # 如果已经是数值类型，直接返回
    if isinstance(value, (int, float)):
        return value

    # 使用正则表达式提取数值
    match = re.search(r'[-+]?\d*\.\d+|\d+', str(value))
    if match:
        return float(match.group())
    return None  # 无法提取数值时返回None


# 对倒数第五列进行数据清洗，提取数值
selected_data['fifth_last_column'] = selected_data['fifth_last_column'].apply(extract_number)

# 移除无法提取数值的行
selected_data = selected_data.dropna(subset=['fifth_last_column'])

# 初始化结果列表
result = []

# 每10行计算一次平均值
for i in range(0, len(selected_data), 10):
    chunk = selected_data.iloc[i:i + 10]

    # 检查是否有足够的行数（至少10行）
    if len(chunk) >= 10:
        # 计算第一列的众数（如果有多个众数，取第一个）
        first_col_mode = chunk['first_column'].mode().iloc[0]

        # 计算倒数第五列的平均值
        fifth_last_col_mean = chunk['fifth_last_column'].mean()

        # 添加到结果列表
        result.append({
            'first_column': first_col_mode,
            'fifth_last_column_mean': fifth_last_col_mean
        })

# 创建结果DataFrame
result_df = pd.DataFrame(result)

# 保存结果到CSV文件
result_df.to_csv('auc.csv', index=False)

print(f"处理完成，共生成 {len(result_df)} 条合并数据。")