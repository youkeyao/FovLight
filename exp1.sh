#!/bin/bash

# CUDA_VISIBLE_DEVICES=$DEVICE_ID python ./ObjectRenderer.py \
#     --background /mnt/data/youkeyao/Sig25_4DLighting/exp1/scene_001/bg.png \
#     --envmap /mnt/data/youkeyao/Sig25_4DLighting/results/exp1/scene_001/6/env.exr \
#     --out /mnt/data/youkeyao/Sig25_4DLighting/exp1/scene_001/0/6.png \
#     --resolution 4400 3850 \
#     --camera_rot 90 0 90 \
#     --plane_pos 0.7 -2 0 \
#     --plane_rot 0 -90 0 \
#     --sphere_pos 0.5 -1.5 0 \
#     --sphere_radius 0.2 \
#     --sphere_roughness 0 \
#     --sphere_metallic 0.0 \
#     --flip True

export CUDA_VISIBLE_DEVICES=0
BASE_DIR="/mnt/data/youkeyao/Sig25_4DLighting"
SCRIPT_PATH="./ObjectRenderer.py"

# 通用渲染参数 (所有命令共用的参数)
# COMMON_ARGS=(
#     "--resolution" "4400" "3850"
#     "--camera_rot" "90" "0" "90"
#     "--plane_pos" "0.7" "-2" "0"
#     "--plane_rot" "0" "-90" "0"
#     "--sphere_pos" "0.5" "-1.5" "0"
#     "--sphere_radius" "0.2"
# )
COMMON_ARGS=(
    "--resolution" "4400" "3850"
    "--camera_rot" "90" "0" "90"
    "--plane_pos" "-0.07" "-2" "0"
    "--plane_rot" "-90" "-70" "0"
    "--sphere_pos" "0.0" "-1.5" "0"
    "--sphere_radius" "0.1"
)

# 1. 从实际目录中生成场景列表 (按 scene_* 排序)
SCENES=()
for scene_dir in "$BASE_DIR/exp1"/scene_*; do
    [ -d "$scene_dir" ] || continue
    scene=$(basename "$scene_dir")
    SCENES+=("$scene")
done

# 如果没有找到任何 scene_* 目录，则退出并提示
if [ ${#SCENES[@]} -eq 0 ]; then
    echo "No scene_* directories found under $BASE_DIR/exp1"
    exit 1
fi

# 2. 定义预测结果的 ID 列表 (这些需要 --flip True)
IDS=("1" "3" "4" "6")
# IDS=("6")

# 3. 定义材质配置
# 格式: "输出文件夹名 粗糙度(Roughness) 金属度(Metallic)"
# 注意: 这里的顺序对应原脚本中的不同组合
MATERIAL_CONFIGS=(
    "0_0   0    0.0" # Folder 0:  Roughness 0, Metallic 0.0
    "0.2_0 0.2  0.0" # Folder 0.2: Roughness 0.2, Metallic 0.0
    "0.3_0 0.3  0.0" # Folder 0.3: Roughness 0.3, Metallic 0.0
    "1_0   1    0.0" # Folder 1:  Roughness 1, Metallic 0.0
    "0_0.5   0    0.5" # Folder 0:  Roughness 0, Metallic 0.0
    "0.2_0.5 0.2  0.5" # Folder 0.2: Roughness 0.2, Metallic 0.0
    "0.3_0.5 0.3  0.5" # Folder 0.3: Roughness 0.3, Metallic 0.0
    "1_0.5   1    0.5" # Folder 1:  Roughness 1, Metallic 0.0
    "0_0.8   0    0.8" # Folder 0:  Roughness 0, Metallic 0.0
    "0.2_0.8 0.2  0.8" # Folder 0.2: Roughness 0.2, Metallic 0.0
    "0.3_0.8 0.3  0.8" # Folder 0.3: Roughness 0.3, Metallic 0.0
    "1_0.8   1    0.8" # Folder 1:  Roughness 1, Metallic 0.0
    "0_1   0    1" # Folder 0:  Roughness 0, Metallic 0.0
    "0.2_1 0.2  1" # Folder 0.2: Roughness 0.2, Metallic 0.0
    "0.3_1 0.3  1" # Folder 0.3: Roughness 0.3, Metallic 0.0
    "1_1   1    1" # Folder 1:  Roughness 1, Metallic 0.0
)

# ================= 执行逻辑 =================

for scene in "${SCENES[@]}"; do
    echo "Processing Scene: $scene"
    
    # 背景图路径对每个场景是固定的
    BG_PATH="${BASE_DIR}/exp1/${scene}/bg.png"

    for config in "${MATERIAL_CONFIGS[@]}"; do
        # 读取配置变量
        read -r folder_name roughness metallic <<< "$config"
        OUT_DIR="${BASE_DIR}/exp1/${scene}/${folder_name}"

        # ----------------------------------------
        # A. 渲染预测结果 (ID: 1, 3, 4, 6)
        # 特点: envmap在results目录, 需要 --flip True
        # ----------------------------------------
        for id in "${IDS[@]}"; do
            ENVMAP="${BASE_DIR}/results/exp1/${scene}/${id}/env.exr"
            OUT_FILE="${OUT_DIR}/${id}.png"

            echo "  [Render] ID: $id | Folder: $folder_name"
            
            python "$SCRIPT_PATH" \
                --background "$BG_PATH" \
                --envmap "$ENVMAP" \
                --out "$OUT_FILE" \
                "${COMMON_ARGS[@]}" \
                --sphere_roughness "$roughness" \
                --sphere_metallic "$metallic" \
                --flip True
        done

        # ----------------------------------------
        # B. 渲染 GT (Ground Truth)
        # 特点: envmap在exp1目录, 不需要 --flip 参数
        # ----------------------------------------
        ENVMAP_GT="${BASE_DIR}/exp1/${scene}/gt.exr"
        OUT_FILE_GT="${OUT_DIR}/gt.png"

        echo "  [Render] GT    | Folder: $folder_name"

        python "$SCRIPT_PATH" \
            --background "$BG_PATH" \
            --envmap "$ENVMAP_GT" \
            --out "$OUT_FILE_GT" \
            "${COMMON_ARGS[@]}" \
            --sphere_roughness "$roughness" \
            --sphere_metallic "$metallic"
    done
done

echo "Batch rendering complete."