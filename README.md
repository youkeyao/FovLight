## dependency
```bash
conda create -n FovLight python=3.9
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
wget https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2
conda install pytorch3d-0.7.8-py39_cu118_pyt241.tar.bz2
pip install opencv-python accelerate tensorboard pyiqa lpips iopath mitsuba openpyxl colour-science pyfvvdp flip-evaluator
```

## train
```bash
# create new session
tmux new -s FovLight
./train.sh
# exist Ctrl + b -> d
# connect to session
tmux attach -t FovLight
# kill session
tmux kill-session -t FovLight
# show tensorboard
tensorboard --logdir=logs
```

## 虚拟物体渲染

### 基本功能
- 读取背景图与环境贴图（HDR/EXR 等）。
- 按参数设置相机、地面平面和球体材质。
- 输出合成结果图，并额外输出一张主频带可视化图用于空间指标分析。

### 命令行参数
- `--background, -b`：背景图路径（必填）。
- `--envmap, -e`：环境贴图路径（必填）。
- `--out, -o`：输出图路径（必填）。
- `--resolution, -r`：输出分辨率，格式为 `WIDTH HEIGHT`（可选，不填时使用背景图原始分辨率）。
- `--camera_pos`：相机位置 `x y z`。
- `--camera_rot`：相机旋转 `pitch yaw roll`（度）。
- `--fov`：相机视场角（度）。
- `--plane_pos`：平面位置 `x y z`。
- `--plane_rot`：平面旋转 `pitch yaw roll`（度）。
- `--plane_scale`：平面缩放系数。
- `--sphere_pos`：球体中心位置 `x y z`。
- `--sphere_radius`：球体半径。
- `--sphere_roughness`：球体粗糙度。
- `--sphere_metallic`：球体金属度。
- `--sphere_color`：球体颜色 `r g b`（0~1）。
- `--flip`：水平翻转背景和输出（布尔开关，不带值）。

### 使用示例

```bash
python ObjectRenderer.py \
	-b input/image.png \
	-e input/envmap.exr \
	-o output/render.png \
	-r 4400 3850 \
	--camera_pos 0 0.51303 -1.4095 \
	--camera_rot 20 0 0 \
	--fov 120 \
	--plane_pos 0 -0.1 0 \
	--plane_rot -90 0 0 \
	--plane_scale 2.0 \
	--sphere_pos 0 0 0 \
	--sphere_radius 0.1 \
	--sphere_roughness 0.05 \
	--sphere_metallic 1.0 \
	--sphere_color 0.3 0.3 0.3 \
	--flip
```
