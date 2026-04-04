"""回归模型快速测试脚本。"""

import numpy as np
from regress import init_metrics, calculate_15_metrics
import joblib

if __name__ == '__main__':
    model = joblib.load('/mnt/data/youkeyao/FovLight/exp1/models/0.pkl')
    image_path = '/mnt/data/youkeyao/FovLight/output_envmap.png'
    gt_path = '/mnt/data/youkeyao/Sig25_4DLighting/exp1/scene_006/0/gt.png'
    device = init_metrics()
    metric = np.asarray(calculate_15_metrics(image_path, gt_path, device))
    print(model['model'].predict(model['scaler'].transform([metric])))