import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_numeric_distributions(df, output_dir="output", prefix="dist"):
    os.makedirs(output_dir, exist_ok=True)

    # 提取所有数值型列
    df_num = df.select_dtypes(include=[np.number])
    num_cols = df_num.shape[1]

    if num_cols == 0:
        print("没有可绘制的数值型列。")
        return

    # 每行最多 3 个图
    cols_per_row = 3
    rows = math.ceil(num_cols / cols_per_row)

    # 设置图像大小（每图宽 6，高 4）
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(6 * cols_per_row, 4 * rows))
    axes = axes.flatten()  # 变成一维索引

    for i, col in enumerate(df_num.columns):
        axes[i].hist(df_num[col].dropna(), bins=30, color='skyblue', edgecolor='black')
        axes[i].set_title(col)
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
        axes[i].grid(True, linestyle='--', alpha=0.5)

    # 删除多余的子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{prefix}_numeric_distribution.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"图像已保存至：{output_path}")

def main():
    parser = argparse.ArgumentParser(description="绘制所有数值型数据的分布图")
    parser.add_argument("csv_path", type=str, help="输入CSV文件路径")
    parser.add_argument("--output_dir", default=os.getcwd() , help="输出图片的文件夹路径")
    parser.add_argument("--prefix", default="dist", help="输出文件名前缀")

    args = parser.parse_args()

    # 读取 CSV 文件
    df = pd.read_csv(args.csv_path)

    # 绘图
    plot_numeric_distributions(df, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
