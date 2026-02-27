import os
import cv2
import numpy as np
import pandas as pd
import argparse
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
import pickle
from regress import init_metrics, calculate_15_metrics
import joblib

# ================= 配置区域 =================
ROOT_PATH = '/mnt/data/youkeyao/Sig25_4DLighting/exp2'  # 根目录
IGNORE_FILES = ['gt.png'] # 不参与对比的文件
# ===========================================

def get_args():
    parser = argparse.ArgumentParser(description="聚合计算指定目录下模型的对比矩阵")
    parser.add_argument('--target', type=str, default='0/0', 
                        help='指定要进入的子目录名称，例如: 0, 0.1, 0.4, 1')
    parser.add_argument('--model_path', type=str, default='/mnt/data/youkeyao/FovLight/exp1/0/0.pkl')
    return parser.parse_args()

def predict_score_with_model(model, measures1, measures2):
    left = np.asarray(measures1).copy()
    right = np.asarray(measures2).copy()

    prediction = 1 if model['model'].predict(model['scaler'].transform([left])) > model['model'].predict(model['scaler'].transform([right])) else 0
    
    return prediction

def main():
    args = get_args()
    target_dir = args.target # 获取命令行参数，比如 "0"

    device = init_metrics()
    
    # 用于存储累加分数和计数
    # 结构: score_accumulator[(model_a, model_b)] = total_score
    score_accumulator = defaultdict(float)
    count_accumulator = defaultdict(int)
    
    # 用于收集所有出现过的模型名字，确保矩阵完整
    all_model_names = set()
    
    # 1. 寻找所有符合条件的 scene 路径
    scene_dirs = []
    # 排除的 scene 列表
    # exclude_scenes = {"scene_003", "scene_005"}
    exclude_scenes = {}
    if os.path.exists(ROOT_PATH):
        for d in os.listdir(ROOT_PATH):
            full_path = os.path.join(ROOT_PATH, d)
            # 只处理以 scene_ 开头的目录，且排除指定场景
            if os.path.isdir(full_path) and d.startswith("scene_") and d not in exclude_scenes:
                scene_dirs.append(full_path)
    
    scene_dirs.sort()
    print(f"检测到参数: {target_dir}")
    print(f"将遍历以下 Scene: {[os.path.basename(s) for s in scene_dirs]}")
    print("-" * 30)

    processed_scene_count = 0

    # 2. 遍历每个 Scene
    for scene_path in scene_dirs:
        scene_name = os.path.basename(scene_path) # e.g., "scene_001"
        target_path = os.path.join(scene_path, target_dir)
        
        # 1. 检查目录是否存在
        if not os.path.exists(target_path):
            continue

        files = [f for f in os.listdir(target_path) if f.endswith('.png') and f not in IGNORE_FILES]
        files.sort()
        
        if len(files) < 2:
            continue
            
        print(f"处理: {scene_name} / {target_dir} | 图片数: {len(files)}")

        # 2. 加载该 Scene 对应的模型
        # 假设 scene_type 就是文件夹名字 (或者你需要在这里做映射)
        model = joblib.load(args.model_path)
        
        if model is None:
            print(f"  -> 跳过: 缺少对应模型")
            continue

        # 3. [优化] 预先计算该目录下所有图片的 Metrics，避免重复计算
        # 缓存: { '1.png': [v1...v15], '2.png': [...] }
        metrics_cache = {}
        for fname in files:
            fpath = os.path.join(target_path, fname)
            gt_path = os.path.join(target_path, 'gt.png')
            metric = np.asarray(calculate_15_metrics(fpath, gt_path, device))
            metrics_cache[fname] = model['model'].predict(model['scaler'].transform([metric]))
            print(f"  计算并预测: {fname} -> {metrics_cache[fname]}")
        processed_scene_count += 1
        all_model_names.update(files)

        # 4. 两两对比并预测
        for i in range(len(files)):
            for j in range(len(files)): # 这是一个全矩阵遍历 (因为 model(A-B) 可能不等于 model(B-A) ?)
                # 通常差值预测模型是对称的 abs(A-B)，但如果你的模型是有方向的 (A比B好?)，则需要全遍历
                # 如果是对称矩阵，可以优化为 j range(i, len)
                
                name_a = files[i]
                name_b = files[j]
                
                # 矩阵 Key
                key = (name_a, name_b) 
                
                if name_a == name_b:
                    score = 0.0 # 自我对比默认为 1 (或者根据你的模型逻辑修改)
                else:
                    m1 = metrics_cache[name_a]
                    m2 = metrics_cache[name_b]
                    score = 1 if m1 > m2 else 0
                
                score_accumulator[key] += score
                count_accumulator[key] += 1

    # ================= 输出结果 =================
    if processed_scene_count == 0:
        print("未处理任何场景，请检查路径或模型文件。")
        return

    sorted_models = sorted(list(all_model_names))
    n = len(sorted_models)
    final_matrix = np.zeros((n, n))

    print("\n正在生成聚合矩阵...")
    for i in range(n):
        for j in range(n):
            name_a = sorted_models[i]
            name_b = sorted_models[j]
            key = (name_a, name_b)
            
            total = score_accumulator.get(key, 0.0)
            count = count_accumulator.get(key, 0)
            
            final_matrix[i][j] = total

    df = pd.DataFrame(final_matrix, index=sorted_models, columns=sorted_models)
    # pd.set_option('display.float_format', lambda x: '%.4f' % x)
    
    # print(df)
    
    out_file = f"ml_score_matrix.csv"
    df.to_csv(out_file)
    print(f"\n保存至: {out_file}")

if __name__ == '__main__':
    # main()
    model = joblib.load('/mnt/data/youkeyao/FovLight/exp1/0.pkl')
    image_path = '/mnt/data/youkeyao/FovLight/output_envmap.png'
    gt_path = '/mnt/data/youkeyao/Sig25_4DLighting/exp1/scene_006/0/gt.png'
    device = init_metrics()
    metric = np.asarray(calculate_15_metrics(image_path, gt_path, device))
    print(model['model'].predict(model['scaler'].transform([metric])))