#!/bin/bash

# 1. 设置基础路径
export CUDA_VISIBLE_DEVICES=3
BASE_DIR="/mnt/data/youkeyao/Sig25_4DLighting"
SCENE="scene_006"
BG_PATH="${BASE_DIR}/exp2/${SCENE}/bg.png"

# 2. 定义通用的参数
COMMON_ARGS=(
    --background "$BG_PATH"
    --resolution 4400 3850
    --camera_rot 90 0 90
    --fov 136.4
    --plane_rot -90 -70 0
    --sphere_radius 0.2
)

# 3. 定义变化的参数 (Frame & Z-coord)
FRAMES=(0 15 30 45)
Z_COORDS=(0 -0.65 -1.3 -1.95)
INDICES=(1 3 6)

# 4. 定义材质配置
# 格式: "文件夹名 粗糙度 金属度 R G B"
MATERIAL_CONFIGS=(
    "0   0    0.0  0.7 0.7 0.7"   # Roughness 0, Metallic 0.0, Color [0.5,0.5,0.5]
    "metal  0    1.0  1.0 1.0 1.0" # Roughness 0, Metallic 1.0, Color [1,1,1]
)

# 5. 开始循环处理
for i in "${!FRAMES[@]}"; do
    FRAME=${FRAMES[$i]}
    Z_POS=${Z_COORDS[$i]}
    
    # --- 新增材质循环 ---
    for MAT_CONF in "${MATERIAL_CONFIGS[@]}"; do
        # 读取配置字符串: 文件夹名, 粗糙度, 金属度, R, G, B
        read -r MAT_NAME MAT_ROUGH MAT_METAL MAT_R MAT_G MAT_B <<< "$MAT_CONF"
        
        # 构建包含材质名称的输出目录
        # 结构例如: .../scene_006/0/01/ (Frame/Material_Folder)
        OUTPUT_DIR="${BASE_DIR}/exp2/${SCENE}/${FRAME}/${MAT_NAME}"

        echo "------------------------------------------------------"
        echo "Frame: $FRAME (Z=$Z_POS) | Material: $MAT_NAME (R: $MAT_ROUGH, M: $MAT_METAL)"
        echo "------------------------------------------------------"

        # 循环 envmap indices
        for IDX in "${INDICES[@]}"; do
            echo "  > Rendering Index $IDX..."
            
            python ./ObjectRenderer.py \
                "${COMMON_ARGS[@]}" \
                --sphere_roughness "$MAT_ROUGH" \
                --sphere_metallic "$MAT_METAL" \
                --sphere_color "$MAT_R" "$MAT_G" "$MAT_B" \
                --envmap "${BASE_DIR}/results/exp2/${SCENE}/${IDX}/${FRAME}/env.exr" \
                --out "${OUTPUT_DIR}/${IDX}.png" \
                --plane_pos 0.03 -2 "${Z_POS}" \
                --sphere_pos 0 -1.5 "${Z_POS}" \
                --flip True
        done

        # 渲染 GT (Ground Truth)
        echo "  > Rendering GT..."
        Z_POS_INVERT=$(awk -v val="$Z_POS" 'BEGIN {print -1 * val}')
        python ./ObjectRenderer.py \
            "${COMMON_ARGS[@]}" \
            --sphere_roughness "$MAT_ROUGH" \
            --sphere_metallic "$MAT_METAL" \
            --sphere_color "$MAT_R" "$MAT_G" "$MAT_B" \
            --envmap "${BASE_DIR}/exp2/${SCENE}/${FRAME}.exr" \
            --out "${OUTPUT_DIR}/gt.png" \
            --plane_pos 0.03 -2 "${Z_POS_INVERT}" \
            --sphere_pos 0 -1.5 "${Z_POS_INVERT}"
            
    done
done