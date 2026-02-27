import numpy as np
import math
import torch
from ObjectRenderer import ObjectRenderer
import cv2
import mitsuba as mi
import re
from datetime import datetime

# ==========================================
# 1. 系统参数与辅助类定义
# ==========================================

class SystemConfig:
    """对应文中的实验环境配置"""
    SCREEN_WIDTH = 4400
    SCREEN_HEIGHT = 3850
    FOCAL_LENGTH = 148
    
    # 采样频率参数 (Equation 6.1)
    F_MAX = 60.0  # 中央凹最高帧率
    F_MIN = 1.0   # 边缘最低帧率
    LAMBDA = 0.05 # 衰减系数
    
    # 时域平滑参数
    ALPHA_STATIC = 0.9      # 稳态混合因子 (Temporal Denoising)
    ALPHA_TRANSIENT = 0.9   # 瞬态切换初始因子 (Cross-fading start)
    TRANSITION_FRAMES = 60  # 过渡窗口长度


class OneEuroFilter:
    """
    One Euro Filter implementation for smoothing gaze points.
    Reference: "The One Euro Filter: Minimalistic, efficient smoothing".
    """
    def __init__(self, freq=60.0, min_cutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)

        self.x_prev = None
        self.dx_prev = None
        self.last_time = None

    def __alpha(self, cutoff, dt):
        # Alpha calculation from cutoff frequency
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, timestamp=None):
        # determine dt
        if timestamp is None:
            dt = 1.0 / max(1e-6, self.freq)
        else:
            if self.last_time is None:
                dt = 1.0 / max(1e-6, self.freq)
            else:
                dt = max(1e-6, timestamp - self.last_time)
        self.last_time = timestamp

        # derivative
        if self.x_prev is None:
            dx = 0.0
        else:
            dx = (x - self.x_prev) / dt

        # filter the derivative
        a_d = self.__alpha(self.dcutoff, dt)
        if self.dx_prev is None:
            edx = dx
        else:
            edx = a_d * dx + (1 - a_d) * self.dx_prev

        # cutoff
        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self.__alpha(cutoff, dt)

        # filter signal
        if self.x_prev is None:
            x_hat = x
        else:
            x_hat = a * x + (1 - a) * self.x_prev

        # update state
        self.x_prev = x_hat
        self.dx_prev = edx

        return x_hat

# 尝试加载模型，如果路径不存在则提示（这里假设环境与之前一致）
try:
    models = {
        'M1': torch.load("/mnt/data/youkeyao/Sig25_4DLighting/results/exp2/scene_006/1/optimized_envmap.pth"),
        'M2': torch.load("/mnt/data/youkeyao/Sig25_4DLighting/results/exp2/scene_006/3/optimized_envmap.pth"),
        'M4': torch.load("/mnt/data/youkeyao/Sig25_4DLighting/results/exp2/scene_006/6/optimized_envmap.pth")
    }
except FileNotFoundError:
    print("Warning: Model files not found at specified paths. Please check paths.")
    models = {} # 防止崩溃，实际运行时需要正确路径

# ==========================================
# 2. 核心渲染器实现 (GCIE System)
# ==========================================

class GCIERenderer:
    def __init__(self):        
        # 系统状态变量
        self.last_update_time = -10000
        self.last_img = None
        self.last_model_name = 'M4'
        self.current_alpha = SystemConfig.ALPHA_STATIC

        # 用于处理瞬态切换的计数器
        self.transition_counter = 0

    def _get_eccentricity(self, gaze_pos):
        """
        计算注视偏心角 E
        注意：这里假设 gaze_pos 是相对于系统预设的“屏幕中心”或“模型中心”的坐标。
        原代码中使用 (256, 256) 作为模型中心。
        """
        # 如果输入数据是屏幕绝对坐标，这里可能需要根据屏幕尺寸进行归一化或映射
        # 这里保持原逻辑：计算到 (256, 256) 的欧氏距离
        dist = np.linalg.norm(np.array([256, 256, 2]) - np.array(gaze_pos))
        E_rad = np.arctan(dist / SystemConfig.FOCAL_LENGTH)
        return np.degrees(E_rad) # 返回角度值

    def _calculate_sampling_freq(self, E):
        """
        计算连续空间采样频率 f_s(E)
        Equation 6.1
        """
        decay = math.exp(-SystemConfig.LAMBDA * E)
        f_s = SystemConfig.F_MIN + (SystemConfig.F_MAX - SystemConfig.F_MIN) * decay
        return f_s

    def _select_model(self, E):
        """
        基于感知的动态选择
        """
        if E > 45:  # 远周边视野
            return models.get('M1'), 'M1'
        elif E > 8.2288: # 中周边视野
            return models.get('M2'), 'M2'
        else:          # 中央凹区域
            return models.get('M4'), 'M4'

    def update(self, current_time, gaze_pos, device):
        """
        算法 6.1: 基于注视点的时空自适应光照更新算法
        """
        if not models:
            return None, "Error", "No Models Loaded"

        # 1. 计算偏心角
        E = self._get_eccentricity(gaze_pos)
        
        # 2. 计算当前位置的采样频率与更新周期
        f_s = self._calculate_sampling_freq(E)
        update_period = 1.0 / f_s
        
        # 3. 判断是否需要更新 (时间采样)
        if (current_time - self.last_update_time) < update_period:
            return self.last_img, self.last_model_name, "Skipped (Freq limit)"

        # 4. 模型选择
        selected_model, model_name = self._select_model(E)
        if selected_model is None:
             return self.last_img, "Error", "Model not found"

        # 5. 推理 (模拟)
        # 注意：这里将 gaze_pos 直接传入 model.to_image。如果坐标范围不匹配（如负值很大），模型可能会输出奇怪的结果。
        # 实际应用中可能需要 clamp 或 normalize。
        raw_illumination = selected_model.to_image(
            *gaze_pos, t=0, device=device, depth_ratio=1.0
        ).squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
        
        # 6. 动态混合因子控制
        if model_name != self.last_model_name:
            self.transition_counter = SystemConfig.TRANSITION_FRAMES
            target_alpha = SystemConfig.ALPHA_TRANSIENT
        else:
            if self.transition_counter > 0:
                ratio = 1.0 - (self.transition_counter / SystemConfig.TRANSITION_FRAMES)
                target_alpha = SystemConfig.ALPHA_TRANSIENT + ratio * (SystemConfig.ALPHA_STATIC - SystemConfig.ALPHA_TRANSIENT)
                self.transition_counter -= 1
            else:
                target_alpha = SystemConfig.ALPHA_STATIC
        
        # 7. 更新光照贴图
        if self.last_img is None:
            current_img = raw_illumination
        else:
            current_img = target_alpha * self.last_img + (1 - target_alpha) * raw_illumination
        
        # 更新状态
        self.last_update_time = current_time
        self.last_img = current_img
        self.last_model_name = model_name
        
        return current_img, model_name, f"Updated (Alpha={target_alpha:.2f})"

# ==========================================
# 3. 数据解析函数
# ==========================================

def load_and_parse_gaze_data(filepath):
    """
    解析特定的 gaze txt 格式
    """
    gaze_points = []
    start_time = None
    end_time = None
    
    print(f"Loading gaze data from {filepath}...")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 解析头部时间戳
        if "数据记录开始于" in line:
            # 提取时间字符串，例如 "2026-01-03 16:39:58"
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                start_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            continue
            
        # 解析尾部时间戳
        if "数据记录结束于" in line:
            match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            if match:
                end_time = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
            continue
            
        # 解析坐标点 (x, y)
        if line.startswith('(') and line.endswith(')'):
            content = line[1:-1] # 去掉括号
            parts = content.split(',')
            if len(parts) == 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    gaze_points.append((x, y))
                except ValueError:
                    pass
                    
    duration = 10.0 # 默认值
    if start_time and end_time:
        duration = (end_time - start_time).total_seconds()
        
    print(f"Parsed {len(gaze_points)} gaze points.")
    print(f"Duration: {duration} seconds (Start: {start_time}, End: {end_time})")
    
    return gaze_points, duration

# ==========================================
# 4. 模拟与测试
# ==========================================

def run_simulation():
    def _build_transform(pos, rot_deg, scale=None):
        pitch, yaw, roll = rot_deg
        t = mi.ScalarTransform4f().translate(pos) @ mi.ScalarTransform4f().rotate([1, 0, 0], pitch) @ mi.ScalarTransform4f().rotate([0, 1, 0], yaw) @ mi.ScalarTransform4f().rotate([0, 0, 1], roll)
        if scale is not None:
            t = t @ mi.ScalarTransform4f().scale(scale)
        return t

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    renderer = GCIERenderer()
    objectRenderer = ObjectRenderer((SystemConfig.SCREEN_HEIGHT, SystemConfig.SCREEN_WIDTH), 136.4)
    cam_pos = (0, 0, 0)
    cam_rot = (90, 0, 90)
    objectRenderer.camera_transform = _build_transform(cam_pos, cam_rot)
    
    # === 加载外部数据 ===
    gaze_data_points, duration_sec = load_and_parse_gaze_data("eyedata.txt")
    
    if not gaze_data_points:
        print("Error: No valid gaze data found.")
        return

    # 计算帧率和总帧数
    total_frames = len(gaze_data_points)
    # 如果数据点太少，避免 FPS 为 0
    FPS = total_frames / duration_sec if duration_sec > 0 else 30
    print(f"Calculated FPS: {FPS:.2f}")

    # === One-Euro filters for gaze smoothing ===
    # 调整 min_cutoff/beta/dcutoff 参数可以控制平滑程度与响应速度
    filter_x = OneEuroFilter(freq=FPS, min_cutoff=1.0, beta=0.007, dcutoff=1.0)
    filter_y = OneEuroFilter(freq=FPS, min_cutoff=1.0, beta=0.007, dcutoff=1.0)

    # 背景加载
    try:
        bg_img = cv2.imread("/mnt/data/youkeyao/Sig25_4DLighting/exp2/scene_006/bg.png", -1)
        if bg_img is None:
            raise FileNotFoundError
        background = bg_img[:, :, 0:3][:, :, ::-1].astype(np.float32) / 255.0
        background = cv2.flip(background, 1)
    except Exception as e:
        print(f"Warning: Could not load background image ({e}). Using black background.")
        background = np.zeros((SystemConfig.SCREEN_HEIGHT, SystemConfig.SCREEN_WIDTH, 3), dtype=np.float32)

    output_video_path = 'output_simulation_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    # 缩小分辨率以加快写入速度
    frame_size = (600, 600)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, FPS, frame_size)

    if not video_writer.isOpened():
        print(f"错误：无法打开视频文件 {output_video_path} 进行写入。")
        return

    print(f"开始模拟并保存视频到 {output_video_path}\n")
    print(f"{'Frame':<5} {'Time':<6} | {'Gaze X':<8} | {'Gaze Y':<8} | {'Model':<8} | {'Status'}")
    print("-" * 65)
    
    # 动画变量
    last_pos = -1.95 # 兔子的 Z 轴起始位置
    SCALE_FACTOR = 0.2 # 绘图缩放因子
    
    for frame_idx in range(total_frames):
        current_time = frame_idx / FPS
        
        # === 从文件中获取注视点 ===
        raw_x, raw_y = gaze_data_points[frame_idx]

        # Apply one-euro filter to raw gaze coordinates
        filt_x = filter_x.filter(raw_x, timestamp=current_time) / 1500 * 2200
        filt_y = filter_y.filter(raw_y, timestamp=current_time) / 1200 * 1900
        
        # 构造 gaze_pos。
        # 注意：原代码逻辑假设中心为 256。如果输入数据是相对于屏幕中心(0,0)的坐标，
        # 我们可能需要偏移它，或者直接使用它（如果模型支持负坐标）。
        # 这里直接使用 raw 数据，Z轴设为 2.0 (原逻辑)
        gaze_pos = (filt_x / 2200.0 * 256 + 256, filt_y / 1900.0 * 256 + 256, 2.0)
        
        # 简单的对象动画，随时间线性移动
        # current_obj_z = last_pos * (1 - frame_idx / total_frames) # 简单线性插值
        current_obj_z = 0
        obj_pos = torch.tensor((0.3, -1.5, current_obj_z))
        
        # 平面位置
        plane_pos = (0.03, -2, current_obj_z)
        plane_rot = (-90, -70, 0)
        plane_scale = [1, 1, 1]
        objectRenderer.plane_transform = _build_transform(plane_pos, plane_rot, scale=plane_scale)
        
        # 执行渲染管线
        envmap_np, model_name, status = renderer.update(current_time, gaze_pos, device)
        
        if envmap_np is None:
            # 如果没有模型加载，无法继续渲染
            continue

        img_np = objectRenderer.render_bunny(
            obj_pos,
            torch.tensor(envmap_np).permute(2, 0, 1),
            torch.tensor(background).permute(2, 0, 1),
            "Bunny.obj",
            2,
            [0.7, 0.7, 0.7],
            0.0,
            0.0
        )
        img_np = objectRenderer.render_bunny_shadow(
            obj_pos,
            torch.tensor(envmap_np).permute(2, 0, 1),
            torch.tensor(background).permute(2, 0, 1),
            torch.tensor(img_np).permute(2, 0, 1),
            "Bunny.obj",
            2,
            [0.7, 0.7, 0.7]
        )[:, :, ::-1]

        frame_np = (img_np * 255).astype(np.uint8)
        frame_np = cv2.flip(frame_np, 1)
        envmap_np = np.clip(envmap_np, 0.0, 1.0)
        envmap_np = (envmap_np * 255).astype(np.uint8)
        
        pixel_x_orig = SystemConfig.SCREEN_WIDTH / 2 + filt_x
        pixel_y_orig = SystemConfig.SCREEN_HEIGHT / 2 - filt_y
        draw_x = int(pixel_x_orig)
        draw_y = int(pixel_y_orig)
        # if -50 < draw_x < frame_size[0] + 50 and -50 < draw_y < frame_size[1] + 50:
        # cv2.circle(frame_np, (draw_x, draw_y), 40, (0, 0, 255), 10)  # 外圈 (红色)
        # 中心注视点：用黄色十字表示
        cross_size = 30
        cross_thickness = 10
        # cv2.line(frame_np, (draw_x - cross_size, draw_y), (draw_x + cross_size, draw_y), (0, 255, 255), cross_thickness)
        # cv2.line(frame_np, (draw_x, draw_y - cross_size), (draw_x, draw_y + cross_size), (0, 255, 255), cross_thickness)

        frame_np = frame_np[1625:2225, 1900:2500]

        if frame_idx == 60:
            cv2.imwrite("exp3imp0.png", frame_np)
            cv2.imwrite("exp3env0.png", envmap_np)
        if frame_idx == 80:
            cv2.imwrite("exp3imp1.png", frame_np)
            cv2.imwrite("exp3env1.png", envmap_np)
        if frame_idx == 100:
            cv2.imwrite("exp3imp2.png", frame_np)
            cv2.imwrite("exp3env2.png", envmap_np)

        # frame_np = cv2.resize(frame_np, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR, interpolation=cv2.INTER_AREA)
        video_writer.write(frame_np)
        
        # 只有发生状态变化或特定帧才打印，避免刷屏
        if frame_idx % 10 == 0 or "Updated" in status:
            print(f"{frame_idx:<5} {current_time:6.2f} | {filt_x:<8.1f} | {filt_y:<8.1f} | {model_name:<8} | {status}")

    print("\nSimulation Finished.")
    video_writer.release()
    print(f"视频已成功保存到：{output_video_path}")
    
if __name__ == "__main__":
    run_simulation()