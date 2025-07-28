import os
import pandas as pd
from scipy.io import arff

# 指定 NASA 数据集目录路径
nasa_dir = 'Datasets/PROMISE'

# 检查目录是否存在
if not os.path.exists(nasa_dir):
    print(f"错误: 目录 '{nasa_dir}' 不存在!")
else:
    # 获取目录下所有 .arff 文件
    arff_files = [f for f in os.listdir(nasa_dir) if f.endswith('.arff')]

    if not arff_files:
        print(f"错误: 在目录 '{nasa_dir}' 中未找到 ARFF 文件!")
    else:
        # 处理每个 ARFF 文件
        for arff_file in arff_files:
            try:
                # 构建完整的文件路径
                arff_path = os.path.join(nasa_dir, arff_file)

                # 读取 ARFF 文件
                data, meta = arff.loadarff(arff_path)

                # 转换为 DataFrame
                df = pd.DataFrame(data)

                # 处理字节字符串（若存在）
                for col in df.columns:
                    if df[col].dtype == 'object':
                        df[col] = df[col].str.decode('utf-8')  # 解码字节字符串

                # 构建输出 CSV 文件路径（使用相同的文件名，替换扩展名）
                csv_file = os.path.splitext(arff_file)[0] + '.csv'
                csv_path = os.path.join(nasa_dir, csv_file)

                # 导出为 CSV
                df.to_csv(csv_path, index=False)

                print(f"成功: 将 '{arff_file}' 转换为 '{csv_file}'")
            except Exception as e:
                print(f"错误: 处理文件 '{arff_file}' 时出错: {e}")