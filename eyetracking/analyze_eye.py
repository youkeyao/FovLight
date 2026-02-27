import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import os

# --- 配置区 ---
INPUT_DIR = "input"
OUTPUT_DIR = "output"
# 严格按照你要求的坐标范围
X_RANGE = (-2200, 2200)
Y_RANGE = (-2000, 2000)

# 颜色配置
COLOR_VALID = '#3498db'   # 有效数据颜色 (蓝色)
COLOR_INVALID = '#e74c3c' # 无效数据颜色 (红色)

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

    # 设置 Seaborn 风格
    sns.set_theme(style="white")
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
    plt.rcParams['axes.unicode_minus'] = False     # 解决保存图像是负号'-'显示为方块的问题


    for file_path in files:
        # 获取不带后缀的文件名
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # 为每个用户/文件创建独立的文件夹
        user_folder = os.path.join(OUTPUT_DIR, base_name)
        ensure_directory(user_folder)

        print(f"正在处理文件: {base_name} ...")

        # --- 修改1：读取逻辑更新 ---
        data_points = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过空行或表头
                if not line or line.startswith("==="):
                    continue
                
                parts = line.split(',')
                # 确保有足够的数据列 (Time, IsValid, X, Y)
                if len(parts) >= 4:
                    is_valid_str = parts[1].strip()
                    # 将字符串 "True" 转为布尔值 True，其他情况为 False
                    is_valid_bool = (is_valid_str == "True")
                    
                    try:
                        # 提取 X 和 Y
                        x = float(parts[2])
                        y = float(parts[3])
                        # 【重要】同时存储坐标和有效性状态
                        data_points.append((x, y, is_valid_bool))
                    except ValueError:
                        continue

        if not data_points:
            print(f"  > 跳过: {base_name} 中未提取到任何坐标数据。")
            continue

        # --- 修改2：创建包含有效性列的 DataFrame ---
        df = pd.DataFrame(data_points, columns=["x", "y", "is_valid"])
        # 确保 is_valid 列是布尔类型，方便后续筛选
        df['is_valid'] = df['is_valid'].astype(bool)
        
        # 过滤指定坐标范围（无论有效无效，先确保在画布范围内）
        df_filtered = df[(df['x'] >= X_RANGE[0]) & (df['x'] <= X_RANGE[1]) & 
                         (df['y'] >= Y_RANGE[0]) & (df['y'] <= Y_RANGE[1])]
        
        if df_filtered.empty:
            print(f"  > 跳过: {base_name} 的点全部在范围外。")
            continue

        # --- 修改3：分离有效和无效数据 ---
        df_valid = df_filtered[df_filtered['is_valid'] == True]
        df_invalid = df_filtered[df_filtered['is_valid'] == False]
        
        valid_count = len(df_valid)
        invalid_count = len(df_invalid)

        # --- 1. 绘制并保存散点图 (区分颜色) ---
        plt.figure(figsize=(10, 8))
        
        # 绘制有效点 (蓝色圆点)
        if not df_valid.empty:
            plt.scatter(df_valid['x'], df_valid['y'], 
                        alpha=0.5, s=15, color=COLOR_VALID, edgecolors='none', 
                        label=f'Valid ({valid_count})')
            
        # 绘制无效点 (红色叉号，稍微突出一点)
        if not df_invalid.empty:
            plt.scatter(df_invalid['x'], df_invalid['y'], 
                        alpha=0.7, s=25, color=COLOR_INVALID, marker='x', linewidth=1,
                        label=f'Invalid ({invalid_count})')

        plt.axhline(0, color='gray', lw=0.8, ls='--')
        plt.axvline(0, color='gray', lw=0.8, ls='--')
        plt.xlim(X_RANGE)
        plt.ylim(Y_RANGE)
        # 添加图例
        plt.legend(loc='upper right')
        plt.title(f"Scatter Plot: {base_name}\n(Total Points: {len(df_filtered)})", fontsize=14)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        
        scatter_save_path = os.path.join(user_folder, f"{base_name}_scatter.png")
        plt.savefig(scatter_save_path, dpi=300, bbox_inches='tight')
        plt.close() 

        # --- 2. 绘制并保存热力图 (仅使用有效数据) ---
        # 热力图依然只应该显示有效的注视区域密度
        plt.figure(figsize=(10, 8))
        try:
            # 检查是否有足够的有效点位
            if len(df_valid.drop_duplicates()) < 3:
                 print(f"  > 跳过 {base_name} 热力图: 有效坐标点太少，无法计算密度。")
            else:
                sns.kdeplot(
                    data=df_valid, x="x", y="y", 
                    fill=True, 
                    thresh=0.05,      
                    levels=20,        
                    cmap="magma",
                    cbar=True
                )
                plt.xlim(X_RANGE)
                plt.ylim(Y_RANGE)
                plt.title(f"Heatmap (Valid Only): {base_name}\n(Valid Points: {valid_count})", fontsize=14)
                plt.xlabel("X Position")
                plt.ylabel("Y Position")
                
                heatmap_save_path = os.path.join(user_folder, f"{base_name}_heatmap.png")
                plt.savefig(heatmap_save_path, dpi=300, bbox_inches='tight')
                print(f"  > 已成功保存至: {user_folder} (Valid:{valid_count}, Invalid:{invalid_count})")
        except Exception as e:
            print(f"  > 热力图生成失败 [{base_name}]: {e}")
        finally:
            plt.close()

    print("\n【全部完成】请在 output 文件夹中查看结果。")

if __name__ == "__main__":
    process_individual_files()