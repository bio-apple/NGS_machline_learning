import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_missing_data(df, output_dir=None, prefix="output"):
    missing_count = df.isnull().sum()
    missing_percent = df.isnull().mean() * 100

    missing_info = pd.DataFrame({
        'MissingCount': missing_count,
        'MissingPercent': missing_percent
    })
    missing_info = missing_info[missing_info['MissingCount'] > 0]

    if missing_info.empty:
        print("✅ 数据中没有缺失值。")
        return

    plt.figure(figsize=(max(12, len(missing_info)), 6))
    bars = plt.bar(missing_info.index, missing_info['MissingPercent'], color='skyblue')

    for bar, (pct, cnt) in zip(bars, zip(missing_info['MissingPercent'], missing_info['MissingCount'])):
        height = bar.get_height()
        label = f"{pct:.1f}% ({cnt})"
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                 label, ha='center', va='bottom', fontsize=9)

    plt.ylabel("Missing Percentage (%)")
    plt.title("Missing Data Percentage per Column")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, min(100, missing_info['MissingPercent'].max() + 10))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{prefix}_missing_data.png")
    plt.savefig(filename, dpi=300)
    print(f"📁 图像已保存至：{filename}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot missing data percentage per column.")
    parser.add_argument("csv_path", type=str, help="输入CSV文件路径")
    parser.add_argument("--output_dir", type=str, default=os.getcwd(), help="输出图像文件夹（可选）")
    parser.add_argument("--prefix", type=str, default="output", help="输出图像文件名前缀（可选）")

    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    plot_missing_data(df, output_dir=args.output_dir, prefix=args.prefix)
