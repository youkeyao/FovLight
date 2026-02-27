import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ==========================================
# 1. 定义模型参数 (来自文中表 5-4 材质1)
# ==========================================
# f(E) = a * E^2 + b * E + c
params = {
    'a': -0.0001,
    'b': -0.005,
    'c': 0.658
}

def perceived_quality(e_deg):
    """
    根据偏心角 E (度) 计算感知阈值 omega
    """
    val = params['a'] * e_deg**2 + params['b'] * e_deg + params['c']
    return val

# ==========================================
# 2. 定义积分计算逻辑
# ==========================================
def calculate_normalized_cost(half_fov_deg):
    """
    计算给定半视场角下的归一化计算耗时 (相对于全分辨率 f(0))
    
    Formula:
    Ratio = ( Integral[ f(E) * sin(E) dE ] ) / ( f(0) * (1 - cos(E_max)) )
    """
    # 避免除以零
    if half_fov_deg < 0.1:
        return 1.0
    
    half_fov_rad = np.radians(half_fov_deg)
    f0 = params['c'] # f(0) 就是截距 c
    
    # 定义被积函数: f(E_deg) * sin(E_rad)
    # 注意：积分变量 x 是弧度
    integrand = lambda x_rad: perceived_quality(np.degrees(x_rad)) * np.sin(x_rad)
    
    # 计算分子：加权质量总和
    numerator, _ = quad(integrand, 0, half_fov_rad)
    
    # 计算分母：归一化因子 (总立体角权重 * 基准质量)
    # 积分 sin(x) dx 从 0 到 theta = 1 - cos(theta)
    denominator = f0 * (1 - np.cos(half_fov_rad))
    
    return numerator / denominator

# ==========================================
# 3. 生成数据
# ==========================================
fov_x = np.linspace(0.1, 60, 100) # 半视场角从 0 到 60 度
costs = [calculate_normalized_cost(x) for x in fov_x]

# 计算特定点的数值用于标注
cost_at_60 = calculate_normalized_cost(60)
savings_at_60 = (1 - cost_at_60) * 100

# ==========================================
# 4. 绘图
# ==========================================
plt.rcParams['font.family'] = 'sans-serif' # 设置字体，防止乱码
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# ========== Font size settings ==========
# Change this value to increase/decrease font sizes across the plot
FONT_SIZE = 16
plt.rcParams['font.size'] = FONT_SIZE

plt.figure(figsize=(8, 6), dpi=120)

# 绘制基准线 (Baseline)
plt.plot([0, 60], [1, 1], 'r--', linewidth=2, label='Full Resolution (Baseline)')

# 绘制自适应曲线 (Adaptive)
plt.plot(fov_x, costs, 'b-', linewidth=2.5, label='Foveated Adaptive (Ours)')
plt.fill_between(fov_x, costs, 1, color='gray', alpha=0.1, label='Saved Computation')

# 添加标注
plt.scatter([60], [cost_at_60], color='blue', zorder=5)
annot_fs = max(8, FONT_SIZE - 2)
plt.annotate(f'Normalized Cost: {cost_at_60:.2f}\n(Saving {savings_at_60:.1f}%)',
             xy=(60, cost_at_60), xytext=(40, 0.55),
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black'),
             fontsize=annot_fs, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8))

# 设置坐标轴
plt.xlim(0, 62)
plt.ylim(0.4, 1.1)
plt.xlabel('Half Field of View ($E_{max}$ in degrees)', fontsize=FONT_SIZE)
plt.ylabel('Normalized Computational Cost', fontsize=FONT_SIZE)
# plt.title('Theoretical Performance Analysis\n(Based on Material 1 Threshold Model)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower left', fontsize=FONT_SIZE)

# 保存或显示
plt.tight_layout()
plt.savefig('foveated_performance_analysis.png', dpi=300, bbox_inches='tight')

# 打印具体数值验证
print(f"Half-FOV 60度时的归一化耗时: {cost_at_60:.4f}")
print(f"理论节省比例: {savings_at_60:.4f}%")