"""FOV 与 JND 关系分析脚本。

解析评测数据并进行心理测量曲线拟合与回归可视化。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from sklearn.metrics import r2_score
import io
import re
import os

# ==========================================
# 1. 基础设置与数据录入
# ==========================================

model_metric_map = {
    '1': -0.80611277,
    '3': -0.41304126,
    '6': 0.84129552
}

csv_path = os.path.join(os.path.dirname(__file__), 'result2.csv')
if os.path.exists(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()
else:
    raw_data = ""

# ==========================================
# 2. 核心函数定义
# ==========================================

def psychometric_function(x, beta, mu):
    """心理测量函数（S 形）。"""
    # S 型曲线
    gamma = 0.5
    return gamma + (1 - gamma) / (1 + np.exp(-beta * (x - mu)))


def fit_psychometric(x_data, y_data):
    """稳健拟合心理测量函数，失败时返回 None。"""
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    if x.size == 0:
        return None
    # 若响应恒定，协方差不可估计，拟合会退化。
    if np.allclose(y, y.flat[0]):
        return None

    mu_min = float(np.min(x)) - 5.0
    mu_max = float(np.max(x)) + 5.0
    bounds = ([-np.inf, mu_min], [np.inf, mu_max])

    p0_candidates = [
        [-5.0, np.median(x)],
        [5.0, np.median(x)],
        [-1.0, np.median(x)],
        [1.0, np.median(x)]
    ]

    for p0 in p0_candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', OptimizeWarning)
                popt, pcov = curve_fit(psychometric_function, x, y, p0=p0, bounds=bounds, maxfev=20000)
            return popt
        except OptimizeWarning:
            # 协方差不可估计或优化异常时，尝试下一组初值。
            continue
        except Exception:
            continue

    # 最后一次尝试：忽略优化告警并放宽迭代上限。
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', OptimizeWarning)
            popt, pcov = curve_fit(psychometric_function, x, y, p0=[-5.0, np.median(x)], maxfev=50000)
        return popt
    except Exception:
        return None

def get_threshold_from_params(beta, mu):
    return mu

def model_linear(x, a, b):
    return a * x + b

def model_quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def model_exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def parse_and_aggregate(text):
    """解析原始文本并聚合为材质/FOV/模型统计表。"""
    pattern = r'--- FOV_材质:\s*(\d+)_(\w+) ---\s*\n(.*?)(?=\n---|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    agg_dict = {} 
    
    for fov_str, mat_str, csv_content in matches:
        fov = int(fov_str)
        material = mat_str
        if not csv_content.strip(): continue
        try:
            df = pd.read_csv(io.StringIO(csv_content.strip()), index_col=0)
        except pd.errors.EmptyDataError:
            continue
        
        for m_id in ['1', '3', '6']:
            if m_id not in df.columns or m_id not in df.index: continue
            gt_wins = pd.to_numeric(df.loc['gt', m_id], errors='coerce')
            model_wins = pd.to_numeric(df.loc[m_id, 'gt'], errors='coerce')
            if pd.isna(gt_wins): gt_wins = 0
            if pd.isna(model_wins): model_wins = 0
            
            key = (material, fov, m_id)
            if key not in agg_dict: agg_dict[key] = {'gt_wins': 0, 'total': 0}
            agg_dict[key]['gt_wins'] += gt_wins
            agg_dict[key]['total'] += (gt_wins + model_wins)
            
    final_data = []
    for (mat, fov, m_id), stats in agg_dict.items():
        if stats['total'] == 0: continue
        final_data.append({
            'Material': mat,
            'FOV': fov,
            'Model': m_id,
            'Metric': model_metric_map[m_id],
            'Accuracy': stats['gt_wins'] / stats['total']
        })
    return pd.DataFrame(final_data)

# ==========================================
# 3. 处理与绘图 (分开保存)
# ==========================================

def main():
    """执行完整分析与绘图流程。"""
    df = parse_and_aggregate(raw_data)
    PSYCHO_FONT_SIZE = 18.0
    FIT_FONT_SIZE = 10.0
    plt.rcParams.update({'font.size': PSYCHO_FONT_SIZE})

    if df.empty:
        print("错误：无有效数据。")
        return

    materials = sorted(df['Material'].unique())
    colors_fov = {0: '#e41a1c', 15: '#377eb8', 30: '#4daf4a', 45: '#984ea3'}
    
    # 存储用于第二步拟合的数据
    regression_results = []

    # -------------------------------------------------------
    # 图表 1: 心理测量曲线 (分开画)
    # -------------------------------------------------------
    print(">>> 开始绘制 Step 1: 心理测量曲线 (每个材质单独保存)...")
    
    styles_fov = {0: '-', 15: '--', 30: '-.', 45: ':'}
    # 定义标记点形状
    markers_fov = {0: 'o', 15: 's', 30: '^', 45: 'D'}

    for mat in materials:
        plt.figure(figsize=(10, 7)) # 尺寸加大
        
        mat_df = df[df['Material'] == mat]
        fov_levels = sorted(mat_df['FOV'].unique())
        
        current_jnds = []
        current_fovs = []
        
        # 用于调整文字标签高度，防止文字重叠
        text_y_pos = 0.4 
        
        for fov in fov_levels:
            subset = mat_df[mat_df['FOV'] == fov].sort_values('Metric')
            x_data = subset['Metric'].values
            y_data = subset['Accuracy'].values
            
            if len(x_data) < 2: continue

            try:
                # 1. 拟合 (使用更稳健的封装函数)
                popt = fit_psychometric(x_data, y_data)
                if popt is None:
                    raise RuntimeError('Fit failed or data degenerate (constant responses)')

                # 2. 获取 JND
                jnd = get_threshold_from_params(*popt)
                current_jnds.append(jnd)
                current_fovs.append(fov)
                
                # 3. 准备绘图样式
                c = colors_fov.get(fov, 'k')
                ls = styles_fov.get(fov, '-')
                mk = markers_fov.get(fov, 'o')
                
                # 4. 绘制光滑曲线
                x_smooth = np.linspace(min(df['Metric'])-1, max(df['Metric'])+1, 400)
                y_smooth = psychometric_function(x_smooth, *popt)
                
                # 计算 RMSE 用于图例
                y_pred = psychometric_function(x_data, *popt)
                rmse = np.sqrt(np.mean((y_data - y_pred)**2))
                
                plt.plot(x_smooth, y_smooth, color=c, linestyle=ls, linewidth=2, 
                         label=f'FOV {fov}° (RMSE={rmse:.2f})')
                
                # 5. 绘制原始散点 (带边框，防重叠)
                plt.scatter(x_data, y_data, color=c, marker=mk, s=80, edgecolors='white', zorder=5, alpha=0.8)
                
                # 6. 绘制 JND 垂线
                plt.vlines(jnd, 0.35, 0.75, colors=c, linestyles=ls, alpha=0.5)
                
                # 7. 【关键】添加 JND 文字标注 (错开高度)
                plt.text(jnd, text_y_pos, f'{jnd:.2f}', color=c, fontweight='bold', ha='center', fontsize=PSYCHO_FONT_SIZE, 
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))
                text_y_pos += 0.05 # 每次循环让文字向上挪一点，防止文字重叠
                
            except Exception as e:
                print(f"[{mat}] FOV {fov} 拟合失败: {e}")

        # 装饰图表
        plt.axhline(0.75, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        plt.text(-1.5, 0.755, 'Threshold (75%)', color='gray', fontsize=PSYCHO_FONT_SIZE)
        
        plt.xlabel('Metric Value', fontsize=PSYCHO_FONT_SIZE+2)
        plt.ylabel('Accuracy (Probability)', fontsize=PSYCHO_FONT_SIZE+2)
        plt.ylim(0.35, 1.05)
        # plt.title(f'Psychometric Curves: {mat}')
        plt.legend(fontsize=PSYCHO_FONT_SIZE, loc='lower left')
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        filename = f'1_psychometric_curves_{mat}.png'
        plt.savefig(filename, dpi=300)
        plt.close()

        if len(current_fovs) > 1:
            regression_results.append({
                'Material': mat,
                'FOV': np.array(current_fovs),
                'JND': np.array(current_jnds)
            })

    # -------------------------------------------------------
    # 图表 2: 最佳拟合模型 (分开画)
    # -------------------------------------------------------
    print("\n>>> 开始绘制 Step 2: JND 拟合分析 (每个材质单独保存)...")
    
    candidates = [
        # {
        #     'name': 'Linear', 
        #     'func': model_linear, 
        #     'label': lambda p: f"${p[0]:.3f}E + {p[1]:.3f}$", 
        #     'p0': [0.01, 0]
        #     # 线性模型通常不需要强行限制，但如果你想让它也必须下降，可以加 bounds
        # },
        {
            'name': 'Quadratic', 
            'func': model_quadratic, 
            'label': lambda p: f"${p[0]:.4f}E^2 + {p[1]:.3f}E + {p[2]:.3f}$", 
            'p0': [-0.01, -0.01, 0.5], # 修改初始猜测 p0，引导它往负方向拟合
            # --- 核心修改在这里 ---
            # 参数顺序: a, b, c
            # bounds: ([a下限, b下限, c下限], [a上限, b上限, c上限])
            # 限制 a<=0 (开口向下), b<=0 (顶点不向右偏移), c无限制
            'bounds': ([-np.inf, -np.inf, -np.inf], [0, 0, np.inf]) 
        },
        # {
        #     'name': 'Exponential', 
        #     'func': model_exponential, 
        #     'label': lambda p: f"${p[0]:.3f}e^{{{p[1]:.3f}E}} + {p[2]:.3f}$", 
        #     'p0': [0.1, 0.01, 0]
        # }
    ]

    for res in regression_results:
        mat = res['Material']
        x = res['FOV']
        y = res['JND']
        
        # 为每个材质单独创建一张图。
        plt.figure(figsize=(7, 5))
        
        best_score = -np.inf
        best_model = None
        best_popt = None
        
        # 寻找最佳模型
        for model in candidates:
            try:
                fit_kwargs = {
                    'p0': model['p0'],
                    'maxfev': 10000
                }
                # 如果模型定义了 bounds，则传入 curve_fit
                if 'bounds' in model:
                    fit_kwargs['bounds'] = model['bounds']
                popt, _ = curve_fit(model['func'], x, y, **fit_kwargs)
                y_pred = model['func'](x, *popt)
                score = r2_score(y, y_pred)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_popt = popt
            except:
                continue
        
        # 绘图
        plt.scatter(x, y, color='#d62728', s=80, edgecolors='k', zorder=5, label='Measured JND')
        
        title_str = f"Material: {mat}"
        if best_model:
            x_smooth = np.linspace(0, 50, 100)
            y_smooth = best_model['func'](x_smooth, *best_popt)
            eq_str = best_model['label'](best_popt)
            label_str = f"{eq_str}\n$R^2={best_score:.3f}$"
            
            plt.plot(x_smooth, y_smooth, color='#1f77b4', linewidth=2.5, label=label_str)
            print(f"   [{mat}] 最佳模型: {best_model['name']} (R2={best_score:.3f})")
        else:
            print(f"   [{mat}] 拟合失败")

        plt.xlabel('Eccentricity E (Degrees)', fontsize=FIT_FONT_SIZE)
        plt.ylabel('Metric Threshold $\omega$', fontsize=FIT_FONT_SIZE)
        # plt.title(title_str, fontsize=14)
        plt.legend(fontsize=FIT_FONT_SIZE, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f'2_best_fit_model_{mat}.png'
        plt.savefig(filename, dpi=300)
        plt.close() # 关闭画布

    print("\n所有图表绘制完成。")


if __name__ == '__main__':
    main()