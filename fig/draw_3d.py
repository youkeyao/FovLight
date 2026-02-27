import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.interpolate import RectBivariateSpline  # 引入插值函数

def draw_smooth_surface():
    # 1. 原始数据定义
    models = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
    roughness_labels = ['0.0', '0.1', '0.4', '1.0']

    # 原始 4x4 数据点
    Z = np.array([
        [-0.91, -0.51, 0.48, 0.94],
        [-0.99, -0.28, 0.55, 0.72],
        [-0.69,  0.09, 0.47, 0.13],
        [-0.49, -0.08, 0.31, 0.27]
    ])

    # 原始坐标索引 (0, 1, 2, 3)
    x = np.arange(len(models))
    y = np.arange(len(roughness_labels))

    # 2. 数据平滑处理 (插值)
    # 创建插值函数对象 (kx=3, ky=3 表示三次样条插值，曲线最平滑)
    interpolator = RectBivariateSpline(y, x, Z, kx=3, ky=3)

    # 生成更密集的网格坐标 (比如把 0-3 之间细分成 100 份)
    x_dense = np.linspace(0, 3, 100)
    y_dense = np.linspace(0, 3, 100)
    
    # 计算密集网格对应的 Z 值
    Z_dense = interpolator(y_dense, x_dense)
    
    # 生成用于绘图的网格矩阵
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)

    # 3. 绘图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制曲面
    # rstride 和 cstride 设置为 1 保证采样的精细度
    # antialiased=True 开启抗锯齿，让线条更柔和
    surf = ax.plot_surface(X_dense, Y_dense, Z_dense, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True, alpha=0.9)

    # 4. 细节修饰
    ax.set_zlim(-1.2, 1.2)

    # 还原坐标轴刻度 (虽然网格密了，但刻度还是要显示原来的 4 个点)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, fontsize=10, rotation=-15)

    ax.set_yticks(np.arange(len(roughness_labels)))
    ax.set_yticklabels(roughness_labels, fontsize=10, rotation=-15, verticalalignment='baseline')

    ax.set_xlabel('\nModel type', fontsize=12)
    ax.set_ylabel('\nRoughness', fontsize=12)
    ax.set_zlabel('\nz-score', fontsize=12)

    ax.set_title('Smoothed Model Performance Surface', fontsize=15, pad=20)

    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, pad=0.1)
    cbar.set_label('Value', rotation=270, labelpad=15)
    xx, yy = np.meshgrid(np.linspace(0, 3, 10), np.linspace(0, 3, 10))
    zz = np.zeros_like(xx)
    ax.plot_wireframe(xx, yy, zz, color='gray', alpha=0.3, linewidth=0.5)

    # 调整视角
    ax.view_init(elev=30, azim=-60)

    # plt.tight_layout()
    # plt.savefig('3d_chart.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    draw_smooth_surface()