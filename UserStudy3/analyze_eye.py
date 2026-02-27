import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import glob
import os

# --- 配置区 ---
INPUT_DIR = "input/eye_tracing"
OUTPUT_DIR = "output"
# 严格按照你要求的坐标范围
X_RANGE = (-2200, 2200)
Y_RANGE = (-2000, 2000)

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_individual_files():
    # 检查输入路径
    if not os.path.exists(INPUT_DIR):
        print(f"【错误】找不到 '{INPUT_DIR}' 文件夹。请在脚本同级目录下创建它。")
        return
        
    ensure_directory(OUTPUT_DIR)
    
    # 获取所有 txt 文件
    files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    
    if not files:
        print(f"【提醒】在 '{INPUT_DIR}' 文件夹中没有找到任何 .txt 文件。")
        return

    # 正则表达式：匹配 (x, y) 格式
    pattern = re.compile(r"\(([-0-9.]+),\s+([-0-9.]+)\)")

    # 设置 Seaborn 风格
    sns.set_theme(style="white")

    for file_path in files:
        # 获取不带后缀的文件名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 为每个用户/文件创建独立的文件夹
        user_folder = os.path.join(OUTPUT_DIR, base_name)
        ensure_directory(user_folder)

        print(f"正在处理文件: {base_name} ...")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = pattern.findall(content)
            
        if not matches:
            print(f"  > 跳过: {base_name} 中未提取到有效坐标。")
            continue

        # 转换为 DataFrame
        df = pd.DataFrame(matches, columns=["x", "y"], dtype=float)
        
        # 过滤指定坐标范围
        df_filtered = df[(df['x'] >= X_RANGE[0]) & (df['x'] <= X_RANGE[1]) & 
                         (df['y'] >= Y_RANGE[0]) & (df['y'] <= Y_RANGE[1])]
        
        if df_filtered.empty:
            print(f"  > 跳过: {base_name} 的点全部在范围外。")
            continue

        # 解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
        plt.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题

        # --- 1. 绘制并保存散点图 ---
        plt.figure(figsize=(10, 8))
        plt.scatter(df_filtered['x'], df_filtered['y'], alpha=0.5, s=12, color='#3498db', edgecolors='none')
        plt.axhline(0, color='gray', lw=0.8, ls='--')
        plt.axvline(0, color='gray', lw=0.8, ls='--')
        plt.xlim(X_RANGE)
        plt.ylim(Y_RANGE)
        plt.title(f"Scatter Plot: {base_name}\n(Points: {len(df_filtered)})", fontsize=14)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        
        scatter_save_path = os.path.join(user_folder, f"{base_name}_scatter.png")
        plt.savefig(scatter_save_path, dpi=300, bbox_inches='tight')
        plt.close() # 必须关闭当前图表，否则会叠加到下一个文件

        # --- 2. 绘制并保存热力图 ---
        plt.figure(figsize=(10, 8))
        try:
            sns.kdeplot(
                data=df_filtered, x="x", y="y", 
                fill=True, thresh=0, levels=100, cmap="magma",
                cbar=True
            )
            plt.xlim(X_RANGE)
            plt.ylim(Y_RANGE)
            plt.title(f"Heatmap: {base_name}\n(Points: {len(df_filtered)})", fontsize=14)
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            
            heatmap_save_path = os.path.join(user_folder, f"{base_name}_heatmap.png")
            plt.savefig(heatmap_save_path, dpi=300, bbox_inches='tight')
            print(f"  > 已成功保存至: {user_folder}")
        except Exception as e:
            print(f"  > 热力图生成失败: {e}")
        finally:
            plt.close()

    print("\n【全部完成】请在 output 文件夹中查看结果。")

if __name__ == "__main__":
    process_individual_files()