import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches

# ================= 配置区域 =================
img_path = '/mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/fov.png'
fov_horizontal = 120 

# 材质2模型参数: ω = aE² + bE + c
MODEL_PARAMS = {'a': -0.0001, 'b': -0.001, 'c': 0.123}

# 阈值列表 (从敏感 -> 迟钝)
# 对应颜色逻辑：红(中心) -> 黄 -> 绿 -> 青(最外层)
variable_thresholds = [0.237, 0.107, -0.075, -0.224]

# 文本字体大小配置（修改此处以调整所有标签字体大小）
FONT_SIZE = 18

# ===========================================

def solve_angle_for_omega(target_omega, params):
    a, b, c = params['a'], params['b'], params['c']
    if target_omega > c: return 0.0
    delta = b**2 - 4*a*(c - target_omega)
    if delta < 0: return None
    return max((-b + np.sqrt(delta))/(2*a), (-b - np.sqrt(delta))/(2*a))

def get_pixel_radius(angle_deg, width, fov_deg):
    if angle_deg <= 0: return 0
    if angle_deg >= 89.9: return float('inf')
    half_fov_rad = np.radians(fov_deg / 2)
    # r = (w/2) * tan(theta) / tan(fov/2)
    return (width / 2) * np.tan(np.radians(angle_deg)) / np.tan(half_fov_rad)

# 1. 初始化
img = Image.open(img_path)
width, height = img.size
center_x, center_y = width / 2, height / 2
diag_len = np.sqrt(width**2 + height**2) / 2 # 屏幕最大可见半径

fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(img)

# 2. 计算所有阈值的半径
zones = []
# 颜色映射：我们需要反向匹配，因为我们要从大圆画到小圆
# variable_thresholds[1] (0.107) -> 对应红色边界
# variable_thresholds[2] (-0.125) -> 对应黄色边界
# variable_thresholds[3] (-0.54) -> 对应绿色边界
# 最外层底色 -> 青色
mapping_colors = ['#FF0000', "#FF0000", '#00BFFF', '#00FF00']

for i, omega in enumerate(variable_thresholds):
    angle = solve_angle_for_omega(omega, MODEL_PARAMS)
    if angle is None: continue
    r = get_pixel_radius(angle, width, fov_horizontal)
    
    # 获取对应填充色 (如果i对应阈值列表的索引)
    # 逻辑：Zone[i] 是由 Threshold[i] 围成的
    # 例如 -0.54 (idx 3) 围成的是 Green 区域
    color = mapping_colors[i] if i < len(mapping_colors) else '#FFFFFF'
    
    zones.append({'omega': omega, 'r': r, 'angle': angle, 'color': color})

# 3. 绘制叠加层 (Painter's Algorithm: 从大到小画)
overlay = np.zeros((height, width, 4))
y_grid, x_grid = np.ogrid[:height, :width]
dist_map = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)

# 3.1 先铺最底层背景 (对应比最小阈值还外面的区域 - Zone 4 Cyan)
# 如果所有圆都在画面内，角落会显示这个颜色；如果圆超大，这个颜色被覆盖
overlay[:] = [0, 1, 1, 0.2] # Cyan, alpha 0.2

# 3.2 按照半径从大到小排序，依次叠加绘制
# 这样大圆先画，小圆后画在上面，形成层级
zones_sorted = sorted(zones, key=lambda x: x['r'], reverse=True)

for zone in zones_sorted:
    r = zone['r']
    # 转换Hex颜色为RGB并设置透明度
    c_rgb = list(int(zone['color'].lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    c_rgba = c_rgb + [0.25] # Alpha 0.25
    
    # 核心：即使 r > diag_len (超大)，这里依然会执行填充
    # 效果就是该颜色铺满全屏，作为新的背景
    mask = dist_map <= r
    overlay[mask] = c_rgba

ax.imshow(overlay)

# 4. 绘制线条和标签 (仅绘制在画面内的)
print(f"{'Threshold':<10} {'Angle':<10} {'Radius(px)':<12} {'Visible?'}")
print("-" * 45)

for zone in zones:
    r = zone['r']
    color = zone['color']
    
    # 边界判断：如果半径大于对角线，说明完全在画幅外
    is_visible = r < diag_len
    
    print(f"{zone['omega']:<10} {zone['angle']:<10.1f} {int(r):<12} {is_visible}")
    
    if is_visible and r > 0:
        # 画圆环
        circle = patches.Circle((center_x, center_y), r, 
                                fill=False, edgecolor=color, linewidth=2, linestyle='--')
        ax.add_patch(circle)
        
        # 画标签
        label_y = center_y - r - 20
        # 防溢出逻辑
        if label_y < 0: label_y = center_y + r + 40
            
        ax.text(center_x, label_y, f"ω ≥ {zone['omega']}", 
            color=color, fontsize=FONT_SIZE, fontweight='bold', ha='center',
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

# 5. 中心点
# ax.scatter([center_x], [center_y], c='red', marker='+', s=150, zorder=10)

ax.axis('off')
plt.tight_layout()
plt.savefig('exp3impl.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
# plt.show()