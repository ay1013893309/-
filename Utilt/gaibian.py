import pandas as pd
import os


def convert_csv_labels(input_file, output_file=None):
    """
    将CSV文件最后一列重命名为'bug'，并将'clean'替换为0、'buggy'替换为1

    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径（默认在输入文件同目录创建新文件）
    """
    # 设置默认输出文件名
    if output_file is None:
        dir_name, file_name = os.path.split(input_file)
        output_file = os.path.join(dir_name, f"converted_{file_name}")

    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 重命名最后一列为'bug'
    if len(df.columns) > 0:
        last_col_name = df.columns[-1]
        df = df.rename(columns={last_col_name: 'bug'})
    else:
        print(f"警告: 文件 {input_file} 无有效列，跳过重命名")

    # 对'bug'列执行标签替换
    if 'bug' in df.columns:
        df['bug'] = df['bug'].apply(lambda x:
                                    0 if str(x).strip().lower() == "n"
                                    else 1 if str(x).strip().lower() == "y"
                                    else x)
    else:
        print(f"警告: 文件 {input_file} 重命名后无'bug'列，跳过替换")

    # 保存结果
    df.to_csv(output_file, index=False)
    print(f"文件已转换并保存至: {output_file}")
    return output_file


# 示例使用
if __name__ == "__main__":
    # 指定CSV文件所在目录路径
    csv_dir = 'Datasets/NASA'  # 修改为实际的目录路径

    # 检查目录是否存在
    if not os.path.exists(csv_dir):
        print(f"错误: 目录 '{csv_dir}' 不存在!")
    else:
        # 获取目录下所有 .csv 文件
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

        if not csv_files:
            print(f"错误: 在目录 '{csv_dir}' 中未找到 CSV 文件!")
        else:
            # 处理每个 CSV 文件
            for csv_file in csv_files:
                try:
                    # 构建完整的文件路径
                    input_path = os.path.join(csv_dir, csv_file)

                    # 转换文件
                    convert_csv_labels(input_path)
                except Exception as e:
                    print(f"错误: 处理文件 '{csv_file}' 时出错: {e}")