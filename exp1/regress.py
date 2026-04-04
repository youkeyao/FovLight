"""客观指标回归脚本。

提取多种图像质量指标并训练 SVR 拟合主观偏好分数。
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import glob
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import argparse

# === 新增：绘图和统计库 ===
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# 引入必要的计算库
from skimage.metrics import structural_similarity as ssim
from skimage import color
from skimage.metrics import peak_signal_noise_ratio as psnr

# 第三方库 (请确保已安装)
try:
    import pyfvvdp
    import pyiqa
    import flip_evaluator as flip
except ImportError as e:
    print(f"Warning: 缺少部分依赖库 ({e})，相关指标将跳过或返回0。")

# ==========================================
# 第一部分：用户提供的指标计算模块
# ==========================================

METRIC_MODELS = {}

def init_metrics(device='cuda'):
    """初始化深度学习质量评估模型。"""
    # if not torch.cuda.is_available():
    device = 'cpu'
    device = torch.device(device)
    print(f"正在加载 IQA 模型到 {device} ...")
    
    model_names = {
        'vif': 'vif',
        'pieapp': 'pieapp',
        'lpips': 'lpips',
        'brisque': 'brisque',
        'niqe': 'niqe',
        'hyperiqa': 'hyperiqa',
        'unique': 'unique', 
    }
    
    for key, name in model_names.items():
        try:
            # 尝试加载 pyiqa 模型
            if 'pyiqa' in globals():
                METRIC_MODELS[key] = pyiqa.create_metric(name, device=device, as_loss=False)
                print(f"  [Loaded] {name}")
            else:
                METRIC_MODELS[key] = None
        except Exception as e:
            print(f"  [Warning] 无法加载模型 {name}: {e}")
            METRIC_MODELS[key] = None

    return device

def safe_predict(model_key, img_tensor_x, img_tensor_y=None):
    """安全执行指标模型推理，失败时返回 0。"""
    model = METRIC_MODELS.get(model_key)
    if model is None:
        print(f"  [Warning] 模型 {model_key} 未加载")
        return 0.0
    try:
        with torch.no_grad():
            if img_tensor_y is not None:
                return model(img_tensor_x, img_tensor_y).item()
            else:
                return model(img_tensor_x).item()
    except Exception as e:
        print(f"  [Warning] {model_key} 预测失败: {e}")
        return 0.0

# --- 传统指标函数 ---
def calc_rmse(img1, img2):
    return np.sqrt(np.mean((img1 - img2) ** 2))

def calc_si_rmse(img1, img2):
    pred = img1.reshape(-1)
    target = img2.reshape(-1)
    dot_pred = np.dot(pred, pred)
    if dot_pred < 1e-6: return calc_rmse(img1, img2)
    alpha = np.dot(pred, target) / dot_pred
    return np.sqrt(np.mean((alpha * img1 - img2) ** 2))

def calc_angular_error(img1, img2):
    norm1 = np.linalg.norm(img1, axis=2)
    norm2 = np.linalg.norm(img2, axis=2)
    mask = (norm1 > 1e-6) & (norm2 > 1e-6)
    dot_product = np.sum(img1 * img2, axis=2)
    cosine = np.clip(dot_product[mask] / (norm1[mask] * norm2[mask]), -1.0, 1.0)
    angles = np.arccos(cosine)
    if angles.size == 0: return 0.0
    return np.mean(angles)

def calc_delta_e(img1_bgr, img2_bgr):
    lab1 = color.rgb2lab(img1_bgr[:, :, ::-1]) 
    lab2 = color.rgb2lab(img2_bgr[:, :, ::-1])
    return np.mean(color.deltaE_ciede2000(lab1, lab2))

def calc_hdr_vdp(img1, img2):
    if 'pyfvvdp' not in globals(): return 0.0
    try:
        fv = pyfvvdp.fvvdp(display_name='standard_4k', heatmap='threshold')
        Q_JOD_noise, _ = fv.predict(img1, img2, dim_order="HWC")
        return Q_JOD_noise.cpu().numpy()
    except Exception as e:
        print(f"  [Warning] HDR-VDP 预测失败: {e}")
        return 0.0

def calc_flip(img_path, gt_path):
    if 'flip' not in globals(): return 0.0
    try:
        # 假设 flip 库已正确配置
        flipErrorMap, meanFLIPError, parameters = flip.evaluate(gt_path, img_path, "HDR")
        return meanFLIPError
    except Exception as e:
        print(f"  [Warning] FLIP 预测失败: {e}")
        return 0.0

def calculate_15_metrics(img_path, gt_path, device):
    """计算单个图像对的 15 维质量特征。"""
    img_bgr = cv2.imread(img_path)
    gt_bgr = cv2.imread(gt_path)
    
    if img_bgr is None or gt_bgr is None:
        raise ValueError(f"Image not found: {img_path} or {gt_path}")

    # 如尺寸不一致，先重采样到真值尺寸。
    if img_bgr.shape != gt_bgr.shape:
        img_bgr = cv2.resize(img_bgr, (gt_bgr.shape[1], gt_bgr.shape[0]))

    img_np = img_bgr.astype(np.float32) / 255.0
    gt_np = gt_bgr.astype(np.float32) / 255.0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
    
    img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    gt_tensor = torch.from_numpy(gt_rgb.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    
    img_tensor = img_tensor.to(device)
    gt_tensor = gt_tensor.to(device)

    # 计算列表
    metrics = [
        calc_rmse(img_np, gt_np),                 # 1. RMSE
        calc_si_rmse(img_np, gt_np),              # 2. si-RMSE
        calc_angular_error(img_np, gt_np),        # 3. Angular
        psnr(gt_np, img_np, data_range=1.0),      # 4. PSNR
        ssim(gt_np, img_np, data_range=1.0, channel_axis=2), # 5. SSIM
        safe_predict('vif', img_tensor, gt_tensor),       # 6. VIF
        safe_predict('pieapp', img_tensor, gt_tensor),    # 7. PieAPP
        calc_flip(img_path, gt_path),             # 8. FLIP
        safe_predict('lpips', img_tensor, gt_tensor),     # 9. LPIPS
        calc_delta_e(img_bgr, gt_bgr),            # 10. Delta E
        calc_hdr_vdp(img_np, gt_np),              # 11. HDR-VDP
        safe_predict('brisque', img_tensor),      # 12. BRISQUE
        safe_predict('niqe', img_tensor),         # 13. NIQE
        safe_predict('unique', img_tensor),       # 14. UNIQUE
        safe_predict('hyperiqa', img_tensor)      # 15. HyperIQA
    ]
    
    # 清理 NaN
    return [float(x) if not np.isnan(x) else 0.0 for x in metrics]


# ==========================================
# 第二部分：数据加载与训练
# ==========================================

class MetricLearner:
    """指标学习器：数据准备、训练与可视化。"""

    def __init__(self, root_dir, csv_path, mat_type, font_size=12):
        self.root_dir = root_dir
        self.df_labels = pd.read_csv(csv_path)
        self.device = init_metrics()
        self.feature_names = [
            "RMSE", "si-RMSE", "Angular", "PSNR", "SSIM", 
            "VIF", "PieAPP", "FLIP", "LPIPS", "DeltaE", 
            "HDR-VDP", "BRISQUE", "NIQE", "UNIQUE", "HyperIQA"
        ]
        self.mat_type = mat_type
        # 新增：全局字体大小配置（用于绘图）
        self.font_size = font_size

    def prepare_data(self):
        """
        遍历目录，匹配 CSV 中的模型 ID，构建训练数据集。
        """
        X = [] # 特征矩阵
        y = [] # 标签 (Z-score)
        
        # 将 CSV 转换为字典: {模型ID: z_score_mean}
        model_score_map = dict(zip(self.df_labels['model'], self.df_labels['mean']))
        target_models = list(model_score_map.keys()) 
        
        print(f"目标模型 ID: {target_models}")
        print("开始遍历场景并计算指标...")

        # 遍历 exp1/scene_xxx
        scene_dirs = sorted(glob.glob(os.path.join(self.root_dir, 'scene_*')))
        
        count = 0
        for scene_dir in scene_dirs:
            base_path = os.path.join(scene_dir, self.mat_type)
            gt_path = os.path.join(base_path, 'gt.png')
            
            # 如果不存在 png gt，尝试 exr (虽然你的代码主要用 png，但保留鲁棒性)
            if not os.path.exists(gt_path) and os.path.exists(os.path.join(base_path, 'gt.exr')):
                 gt_path = os.path.join(base_path, 'gt.exr')

            for model_id in target_models:
                img_name = f"{model_id}.png"
                img_path = os.path.join(base_path, img_name)
                
                if os.path.exists(img_path) and os.path.exists(gt_path):
                    print(f"正在处理: {scene_dir} - {img_name}")
                    try:
                        # 计算 15 个指标
                        feats = calculate_15_metrics(img_path, gt_path, self.device)
                        
                        X.append(feats)
                        y.append(model_score_map[model_id])
                        count += 1
                    except Exception as err:
                        print(f"Error processing {img_path}: {err}")
        
        print(f"特征提取完成。共收集 {count} 组数据。")
        return np.array(X), np.array(y)

    def plot_results(self, y_true, y_pred, output_path):
        """
        绘制拟合结果散点图：横坐标为 Ground Truth (Human)，纵坐标为 Predicted
        """
        plt.figure(figsize=(8, 7))

        # 1. 绘制散点（点大小可以与字体大小适当关联）
        scatter_size = max(20, int(self.font_size * 4))
        plt.scatter(y_true, y_pred, c='blue', alpha=0.6, edgecolors='w', s=scatter_size, label='Data Points')

        # 2. 绘制拟合直线 (红色虚线)
        # 使用 polyfit 做简单线性回归，仅用于显示趋势线。
        if len(y_true) > 1:
            z = np.polyfit(y_true, y_pred, 1)
            p = np.poly1d(z)
            min_x, max_x = min(y_true), max(y_true)
            plt.plot([min_x, max_x], [p(min_x), p(max_x)], "r--", linewidth=2, label='Trend Line')

        # 3. 计算相关系数
        if len(y_true) > 1:
            p_corr, _ = pearsonr(y_true, y_pred)
            s_corr, _ = spearmanr(y_true, y_pred)
            stats_text = f"Pearson (PLCC): {p_corr:.3f}\nSpearman (SROCC): {s_corr:.3f}"
        else:
            stats_text = "Not enough data"

        # 4. 图表装饰（使用配置的字体大小）
        title_fs = self.font_size + 2
        label_fs = self.font_size + 2
        tick_fs = max(8, self.font_size - 4)
        legend_fs = max(8, self.font_size)

        # plt.title(f"Metric Fitting Results: {self.mat_type}", fontsize=title_fs)
        plt.xlabel("Human Preference Score (Z-Score)", fontsize=label_fs)
        plt.ylabel("Learned Metric Score", fontsize=label_fs)
        plt.legend(fontsize=legend_fs)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 调整刻度字体
        plt.xticks(fontsize=tick_fs)
        plt.yticks(fontsize=tick_fs)

        # 在图上显示统计数据
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                 fontsize=self.font_size, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 保存
        save_name = output_path.replace(".pkl", "_result.png")
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.close()
        print(f"结果图已保存至: {save_name}")

    def train(self, X, y, output_model_path):
        """训练 SVR 并保存模型与标准化器。"""
        if len(X) == 0:
            print("没有数据，无法训练。请检查路径和文件名是否匹配。")
            return

        # 1. 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. 训练 SVR
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        
        print("开始训练 SVR ...")
        svr.fit(X_scaled, y)
        
        # 3. 评估拟合程度
        y_pred = svr.predict(X_scaled)
        
        # 计算 R^2
        score = svr.score(X_scaled, y)
        print(f"训练集 R^2 分数: {score:.4f}")

        # 4. 绘制并保存结果图
        self.plot_results(y, y_pred, output_model_path)

        # 5. 保存模型
        save_dict = {
            'model': svr,
            'scaler': scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(save_dict, output_model_path)
        print(f"模型已保存至: {output_model_path}")
        
        return svr, scaler

# ==========================================
# 主程序运行（支持通过命令行参数设置 CSV_PATH 和 MAT_TYPE）
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train metric learner from scenes and human scores.")
    parser.add_argument("--root_path", type=str,
                        default="/mnt/data/youkeyao/Sig25_4DLighting/exp1",
                        help="Root path containing scene_* folders")
    parser.add_argument("--csv_path", type=str,
                        default="/mnt/data/youkeyao/FovLight/exp1/results/roughness_0_scores.csv",
                        help="CSV file with columns 'model' and 'mean' for target models")
    parser.add_argument("--mat_type", type=str, default="0", help="Subfolder name under each scene (e.g. '0' or '0.1')")
    parser.add_argument("--font_size", type=int, default=22, help="Font size to use in result plots")

    args = parser.parse_args()

    ROOT_PATH = args.root_path
    CSV_PATH = args.csv_path
    MAT_TYPE = args.mat_type
    FONT_SIZE = args.font_size
    OUTPUT_MODEL_PATH = MAT_TYPE + ".pkl"

    # 运行流程
    learner = MetricLearner(ROOT_PATH, CSV_PATH, MAT_TYPE, font_size=FONT_SIZE)

    # 1. 准备数据
    X, y = learner.prepare_data()

    # 2. 训练、画图并保存
    learner.train(X, y, OUTPUT_MODEL_PATH)

    # 使用说明（示例命令）
    print("\n=== 使用方法 ===")
    print("模型已保存。要预测新图片的得分：")
    print("1. 加载模型: data = joblib.load('<MAT_TYPE>.pkl')")
    print("2. 计算15个指标: feats = calculate_15_metrics(img, gt, ...)")
    print("3. 标准化: feats_scaled = data['scaler'].transform([feats])")
    print("4. 预测: score = data['model'].predict(feats_scaled)")
    print("")
    print("示例: 在 bash 中运行:")
    print("python regress.py --csv_path /path/to/scores.csv --mat_type 0.1 --root_path /path/to/exp --font_size 14")