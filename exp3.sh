#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/bg.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_001/1/390/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/390_1.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.34 -1.7 1.5 \
    --plane_rot 0 -90 0 \
    --sphere_pos 0.14 -1.2 1.5 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True

python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/390_1.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_001/1/145/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/145_1.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.67 -2 -1.6 \
    --plane_rot 0 -90 0 \
    --sphere_pos 0.47 -1.5 -1.6 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True

python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/145_1.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_001/1/307/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/all_1.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.58 -2.5 1 \
    --plane_rot 0 -90 0 \
    --sphere_pos 0.38 -2 1 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True



python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/390_1.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_001/3/145/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/145_3.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.67 -2 -1.6 \
    --plane_rot 0 -90 0 \
    --sphere_pos 0.47 -1.5 -1.6 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True

python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/145_3.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_001/4/307/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/fov.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.58 -2.5 1 \
    --plane_rot 0 -90 0 \
    --sphere_pos 0.38 -2 1 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True



python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/bg.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_001/4/390/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/390_4.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.34 -1.7 1.5 \
    --plane_rot 0 -90 0 \
    --sphere_pos 0.14 -1.2 1.5 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True

python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/390_4.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_001/4/145/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/145_4.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.67 -2 -1.6 \
    --plane_rot 0 -90 0 \
    --sphere_pos 0.47 -1.5 -1.6 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True

python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/145_4.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_001/4/307/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_001/all_4.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.58 -2.5 1 \
    --plane_rot 0 -90 0 \
    --sphere_pos 0.38 -2 1 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True



# -----------------------------------------------------------------------------------------------



python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/bg.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_002/1/407/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/407_1.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos -0.02 -1.4 1.3 \
    --plane_rot -90 -70 0 \
    --sphere_pos 0.02 -0.9 1.3 \
    --sphere_radius 0.15 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True

python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/407_1.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_002/1/167/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/167_1.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.03 -2 -1.3 \
    --plane_rot -90 -70 0 \
    --sphere_pos 0 -1.5 -1.3 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True

python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/167_1.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_002/1/239/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/all_1.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.63 -1.2 -0.2 \
    --plane_rot -90 -70 0 \
    --sphere_pos 0.42 -1.2 -0.2 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True



python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/407_1.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_002/3/167/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/167_3.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.03 -2 -1.3 \
    --plane_rot -90 -70 0 \
    --sphere_pos 0 -1.5 -1.3 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True

python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/167_3.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_002/4/239/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/fov.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.63 -1.2 -0.2 \
    --plane_rot -90 -70 0 \
    --sphere_pos 0.42 -1.2 -0.2 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True



python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/bg.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_002/4/407/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/407_4.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos -0.02 -1.4 1.3 \
    --plane_rot -90 -70 0 \
    --sphere_pos 0.02 -0.9 1.3 \
    --sphere_radius 0.15 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True

python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/407_4.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_002/4/167/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/167_4.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.03 -2 -1.3 \
    --plane_rot -90 -70 0 \
    --sphere_pos 0 -1.5 -1.3 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True

python ./ObjectRenderer.py \
    --background /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/167_4.png \
    --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp3/scene_002/4/239/env.exr \
    --out /mnt/data/youkeyao/Sig25_4DLighting/exp3/scene_002/all_4.png \
    --resolution 4400 3850 \
    --camera_rot 90 0 90 \
    --fov 136.4 \
    --plane_pos 0.63 -1.2 -0.2 \
    --plane_rot -90 -70 0 \
    --sphere_pos 0.42 -1.2 -0.2 \
    --sphere_radius 0.2 \
    --sphere_roughness 0 \
    --sphere_metallic 0.0 \
    --sphere_color 0.7 0.7 0.7 \
    --flip True