"""主观评测统计分析脚本。

基于 Thurstone Case V 与 bootstrap，输出各模型分数与显著性可视化。
"""

import pandas as pd
import numpy as np
from scipy.stats import norm, binomtest
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import io

# =========================
# 1. 数据准备
# =========================
INPUT_FILE = "metric4.csv"

# =========================
# 2. 配置参数
# =========================
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

N_BOOT = 2000        # Bootstrap 模拟次数

# 绘图配色
DEFAULT_COLOR = '#9C9C9C' 

# 图像尺寸与分辨率
FIG_WIDTH = 5.0
FIG_HEIGHT = 5.0
FIG_DPI = 300

# ================ 字体大小配置 =================
# 基准字体大小，可根据需要调整
FONT_BASE = 14
# 用于柱子上数值的字体大小
FONT_SIZE_BAR_LABEL = FONT_BASE
# 用于 p-value / 显著性标签的字体大小
FONT_SIZE_PVALUE = 11
# 用于 x 轴刻度标签的字体大小；设为 None 时会根据模型数量自动计算
FONT_SIZE_XTICKS = FONT_BASE + 2
# 用于 y 轴标签字体大小
FONT_SIZE_YLABEL = FONT_BASE + 2
# 用于 y 轴刻度标签字体大小；设为 None 时使用 `FONT_BASE`
FONT_SIZE_YTICKS = FONT_BASE
# ==============================================
# 自适应图像与柱状参数
AUTO_FIG = True
PER_BAR_SPACE = 1.0 
MIN_FIG_WIDTH = 5.0
MAX_FIG_WIDTH = 30.0 
BAR_WIDTH = 0.4

# =========================
# 3. 核心算法 
# =========================
def thurstone_case_v(C):
    """根据成对比较矩阵估计 Thurstone z-score。"""
    N_total = C + C.T
    mask = N_total > 0
    P = np.zeros_like(C, dtype=float)
    P[mask] = C[mask] / N_total[mask]
    n_obs = np.max(N_total) if np.max(N_total) > 0 else 1
    delta = 1.0 / (2 * n_obs) 
    P = np.clip(P, delta, 1 - delta)
    Z = norm.ppf(P)
    np.fill_diagonal(Z, 0)
    z_scores = np.mean(Z, axis=1)
    z_scores -= np.mean(z_scores)
    return z_scores

def parametric_bootstrap(C, n_boot=1000):
    """参数化 bootstrap 估计分数不确定性。"""
    N_total = C + C.T
    P_obs = np.divide(C, N_total, out=np.zeros_like(C, dtype=float), where=N_total!=0)
    boot_z_scores = []
    for _ in range(n_boot):
        C_sim = np.zeros_like(C)
        rows, cols = C.shape
        for i in range(rows):
            for j in range(i+1, cols):
                n_trials = int(N_total[i, j])
                if n_trials > 0:
                    p = P_obs[i, j]
                    wins = np.random.binomial(n_trials, p)
                    C_sim[i, j] = wins
                    C_sim[j, i] = n_trials - wins
        boot_z_scores.append(thurstone_case_v(C_sim))
    return np.array(boot_z_scores)

def get_significance_heights(intervals):
    """为显著性连线分配不重叠的显示层级。"""
    # 简单的贪心算法分配层级
    levels = []
    for start, end in intervals:
        placed = False
        for level_intervals in levels:
            conflict = False
            for l_start, l_end in level_intervals:
                # 碰撞检测：间距设为 0.5 避免文字重叠
                if not (end < l_start - 0.5 or start > l_end + 0.5):
                    conflict = True
                    break
            if not conflict:
                level_intervals.append((start, end))
                placed = True
                break
        if not placed:
            levels.append([(start, end)])
    results = []
    for start, end in intervals:
        for i, level_intervals in enumerate(levels):
            if (start, end) in level_intervals:
                results.append(i)
                break
    return results

# =========================
# 4. 数据解析
# =========================
print("Parsing data...")
df_raw = pd.read_csv(INPUT_FILE)
df_raw = df_raw.dropna(how='all')
df_raw = df_raw[df_raw['材质'].notna()]

# 提取模型列名
model_columns = [c for c in df_raw.columns if c not in ['材质', '行胜过列']]
model_ids = sorted(model_columns)
model2idx = {m: i for i, m in enumerate(model_ids)}
n_models = len(model_ids)

PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
    '#4E79A7', '#F28E2B'
]
MODEL_PALETTE = {}
tab20 = plt.get_cmap('tab20')
for i, m in enumerate(model_ids):
    if i < len(PALETTE):
        color = PALETTE[i]
    else:
        color = mcolors.to_hex(tab20((i - len(PALETTE)) % tab20.N))
    MODEL_PALETTE[str(m)] = color

# =========================
# 5. 分析与绘图
# =========================
unique_materials = df_raw['材质'].unique()

for material in unique_materials:
    print(f"Analyzing {material}...")

    sub_df = df_raw[df_raw['材质'] == material]
    C = np.zeros((n_models, n_models))

    # 构建矩阵
    for _, row in sub_df.iterrows():
        try:
            row_model = str(row['行胜过列']).strip()
            if row_model.endswith('.0'): row_model = row_model[:-2]
            if row_model not in model2idx: continue
            
            idx_i = model2idx[row_model]
            
            for col_model in model_columns:
                val = row[col_model]
                if str(val).strip() == '-': continue
                idx_j = model2idx[col_model]
                C[idx_i, idx_j] = float(val)
        except: pass

    # 计算分数
    z_scores = thurstone_case_v(C)
    boot_scores = parametric_bootstrap(C, N_BOOT)
    print("Z-scores:", z_scores)

    # --- 使用 SE (标准误) ---
    se = np.std(boot_scores, axis=0)
    print("Standard errors:", se)
    err_low = se
    err_high = se

    # 准备绘图数据
    plot_data = []
    for i, m_id in enumerate(model_ids):
        plot_data.append({
            'model': str(m_id),
            'mean': z_scores[i],
            'err_low': err_low[i],
            'err_high': err_high[i]
        })

    # 排序
    plot_df = pd.DataFrame(plot_data)
    plot_df.to_csv(OUTPUT_DIR / f"{material.replace('.', 'p')}_scores.csv", index=False)
    # plot_df = plot_df.sort_values(by='mean', ascending=True)
    
    sorted_models = plot_df['model'].tolist()
    sorted_indices = [model2idx[m] for m in sorted_models]
    bar_colors = [MODEL_PALETTE.get(m, DEFAULT_COLOR) for m in sorted_models]

    # ================= 计算所有 P 值 =================
    sig_pairs = []
    n_sorted = len(sorted_models)
    for i in range(n_sorted):
        for j in range(i + 1, n_sorted):
            idx_i = sorted_indices[i]
            idx_j = sorted_indices[j]
            wins = C[idx_i, idx_j]
            total = wins + C[idx_j, idx_i]
            if total > 0:
                res = binomtest(int(wins), int(total), 0.5, alternative='two-sided')
                sig_pairs.append({'i': i, 'j': j, 'p': res.pvalue})

    # 绘图设置
    n_bars = n_sorted
    if AUTO_FIG:
        margins = 1.0
        auto_w = n_bars * PER_BAR_SPACE + margins
        fig_w = max(MIN_FIG_WIDTH, min(MAX_FIG_WIDTH, auto_w))
    else:
        fig_w = FIG_WIDTH

    fig_h = FIG_HEIGHT
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x_pos = np.arange(n_sorted)

    bars = ax.bar(x_pos, plot_df['mean'], yerr=[plot_df['err_low'], plot_df['err_high']],
                    align='center', alpha=0.9, ecolor='#333333', capsize=5,
                    color=bar_colors, width=BAR_WIDTH, edgecolor='black')

    # 1. 追踪最高/最低点
    max_element_y = -999.0
    min_element_y = 999.0

    # 2. 绘制柱子数值
    for rect, e_low, e_high in zip(bars, plot_df['err_low'], plot_df['err_high']):
        height = rect.get_height()
        padding = 0.08

        if height >= 0:
            pos_y = height + e_high + padding
            va = 'bottom'
            current_top = pos_y + 0.1 
            if current_top > max_element_y: max_element_y = current_top
        else:
            pos_y = height - e_low - padding
            va = 'top'
            current_bottom = pos_y - 0.1
            if current_bottom < min_element_y: min_element_y = current_bottom
            
        ax.text(rect.get_x() + rect.get_width() / 2.0, pos_y,
            f'{height:.2f}',
            ha='center', va=va, fontsize=FONT_SIZE_BAR_LABEL, color='black')

    # 3. 绘制所有括号 + P值标记 (更新了 P<0.1 的逻辑)
    if sig_pairs:
        pair_intervals = [(item['i'], item['j']) for item in sig_pairs]
        levels = get_significance_heights(pair_intervals)
        
        # 紧凑设置
        bracket_base_h = max(max_element_y, 0) + 0.3
        bracket_step = 0.4

        for idx, item in enumerate(sig_pairs):
            x1, x2 = item['i'], item['j']
            level = levels[idx]
            y_h = bracket_base_h + level * bracket_step
            
            # 画括号线条 (扁平化 leg_len=0.02)
            leg_len = 0.04
            ax.plot([x1, x1, x2, x2], [y_h - leg_len, y_h, y_h, y_h - leg_len], lw=1.0, c='k')
            
            # --- 文本格式化逻辑 (新增 <0.1) ---
            p_val = item['p']
            if p_val < 0.001:
                sig_marker = "***"
                p_text = "p<0.001"
                font_weight = 'bold'
            elif p_val < 0.01:
                sig_marker = "**"
                p_text = f"p={p_val:.3f}"
                font_weight = 'bold'
            elif p_val < 0.05:
                sig_marker = "*"
                p_text = f"p={p_val:.3f}"
                font_weight = 'bold'
            elif p_val < 0.1:  # <--- 新增逻辑
                sig_marker = "."
                p_text = f"p={p_val:.3f}"
                font_weight = 'bold' # 让边缘显著也稍微突出一点，如需普通字体可改为 'normal'
            else:
                sig_marker = "ns"
                p_text = f"p={p_val:.3f}"
                font_weight = 'normal'
            
            label_text = f"{sig_marker} {p_text}"
            
            mid_x = (x1 + x2) / 2
            text_y = y_h + 0.04
            
            ax.text(mid_x, text_y, label_text, 
                ha='center', va='bottom', fontsize=FONT_SIZE_PVALUE, color='black', weight=font_weight)

            # 更新最高点
            current_top = text_y + 0.25
            if current_top > max_element_y: max_element_y = current_top

    # 4. Y 轴范围
    data_max = plot_df['mean'].max()
    final_top = max_element_y + 0.1
    final_bottom = min(min_element_y, 0) - 0.3
    
    ax.set_ylim(bottom=final_bottom, top=final_top)
    
    ax.set_ylabel('z-score', fontsize=FONT_SIZE_YLABEL)
    ax.set_xticks(x_pos)
    
    # X 轴刻度字体大小：支持手动设置或自动根据模型数缩放
    if FONT_SIZE_XTICKS is not None:
        xt_fs = FONT_SIZE_XTICKS
        if n_sorted > 20:
            rotation = 45
            ha = 'right'
        else:
            rotation = 0
            ha = 'center'
    else:
        if n_sorted > 20:
            xt_fs = max(6, FONT_BASE - (n_sorted - 20) // 5)
            rotation = 45
            ha = 'right'
        else:
            xt_fs = FONT_BASE
            rotation = 0
            ha = 'center'
    ax.set_xticklabels(sorted_models, fontsize=xt_fs, rotation=rotation, ha=ha)
    # Y 轴刻度字体大小：支持手动设置或自动使用 FONT_BASE
    if FONT_SIZE_YTICKS is not None:
        yt_fs = FONT_SIZE_YTICKS
    else:
        yt_fs = FONT_BASE
    ax.tick_params(axis='y', labelsize=yt_fs)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    safe_name = material.replace(".", "p")
    plt.savefig(OUTPUT_DIR / f"{safe_name}.png", dpi=FIG_DPI, bbox_inches='tight')
    print(f"Saved {safe_name}")
    plt.close()

print("All done.")