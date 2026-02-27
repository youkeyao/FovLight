import numpy as np
import matplotlib.pyplot as plt

# --- 设置中文字体 (如果在本地运行且有字体文件，可取消注释) ---
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 使用适合学术论文的绘图风格
plt.style.use('seaborn-v0_8-whitegrid') 

# --- 1. 定义参数 ---
f_max = 60      # 注视点最大频率 (Hz)
f_min = 1       # 边缘最小频率 (Hz)
lam = 0.05      # 衰减常数
E_max = 60      # 偏心角范围 (度)

# --- 2. 生成数据 ---
E = np.linspace(0, E_max, 500)
fs = f_min + (f_max - f_min) * np.exp(-lam * E)

# --- 3. 绘制曲线 ---
fig, ax = plt.subplots(figsize=(8, 5))

# 绘制主曲线
line, = ax.plot(E, fs, label=r'$f_s(E) = f_{min} + (f_{max} - f_{min})e^{-\lambda E}$', 
                color='#2b7bba', linewidth=2.5)

# --- 4. 添加标注与辅助线 ---

# 标注 f_min 和 f_max
# ax.axhline(y=f_min, color='#e74c3c', linestyle='--', alpha=0.7, label=f'Min Frequency ({f_min} Hz)')
# ax.axhline(y=f_max, color='green', linestyle=':', alpha=0.5, label=f'Max Frequency ({f_max} Hz)')

# 标注注视点 (0度)
# ax.scatter([0], [f_max], color='green', s=50, zorder=5)
# ax.text(1.5, f_max, 'Central Fovea (0°)', verticalalignment='center', fontsize=10, color='green')

# 标注典型边缘区域 (例如 45度)
E_mark = 45
fs_mark = f_min + (f_max - f_min) * np.exp(-lam * E_mark)
# ax.scatter([E_mark], [fs_mark], color='#e74c3c', s=50, zorder=5)
# ax.vlines(E_mark, 0, fs_mark, colors='#e74c3c', linestyles=':', alpha=0.5)
# ax.text(E_mark + 1, fs_mark + 2, f'Peripheral ({E_mark}°)\n~{fs_mark:.1f} Hz', 
#         color='#e74c3c', fontsize=9)

# --- 5. 设置标签与图例 ---
# ax.set_title('Dynamic Sampling Frequency vs. Eccentricity', fontsize=14, pad=15)
ax.set_xlabel('Eccentricity (degrees)', fontsize=12)
ax.set_ylabel('Sampling Frequency (Hz)', fontsize=12)

# 设置坐标轴范围
ax.set_xlim(0, E_max)
ax.set_ylim(0, 65)

# 图例
# ax.legend(fontsize=11, frameon=True, fancybox=True, framealpha=0.9)

# 网格
ax.grid(True, linestyle='--', alpha=0.6)

# 保存图片
plt.tight_layout()
plt.savefig('fs_curve.png', dpi=300)