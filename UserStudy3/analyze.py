import csv
import os
from collections import defaultdict

# --- 配置区 ---
INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
TARGET_IDS = ['1', '3', '6', 'gt'] # 保持排序一致性

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def analyze_file(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = [l.strip() for l in content.split('\n') if l.strip()]

    for line in lines:
        if any(key in line for key in ["PairIndex", "数据记录", "Timestamp", "Experimenter"]):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6: continue
        try:
            path_info = parts[1].split('/')
            fov, mat = path_info[0], (path_info[1] if len(path_info) > 1 else "default")
            img_l, img_r, selected, duration = parts[2], parts[3], parts[4], float(parts[5])
            
            if img_l in TARGET_IDS and img_r in TARGET_IDS:
                data_list.append({
                    'fov': fov, 'mat': mat, 
                    'pair': tuple(sorted([img_l, img_r])),
                    'img_l': img_l, 'img_r': img_r, 
                    'selected': selected, 'duration': duration
                })
        except: continue
    return data_list

def process_stats_and_matrices(data_list):
    """处理统计数据和对战矩阵"""
    # 结构: [维度Key][图片ID/PairKey]
    res = {
        'full': {'stats': defaultdict(lambda: defaultdict(lambda: {'展示': 0, '胜出': 0})),
                 'matrix': defaultdict(lambda: defaultdict(int))},
        'mat':  {'stats': defaultdict(lambda: defaultdict(lambda: {'展示': 0, '胜出': 0})),
                 'matrix': defaultdict(lambda: defaultdict(int))},
        'fov':  {'stats': defaultdict(lambda: defaultdict(lambda: {'展示': 0, '胜出': 0})),
                 'matrix': defaultdict(lambda: defaultdict(int))}
    }

    for d in data_list:
        keys = [
            ('full', f"{d['fov']}_{d['mat']}"),
            ('mat',  d['mat']),
            ('fov',  d['fov'])
        ]
        
        for level, k in keys:
            # 更新胜率统计
            for img in [d['img_l'], d['img_r']]:
                res[level]['stats'][k][img]['展示'] += 1
            res[level]['stats'][k][d['selected']]['胜出'] += 1
            # 更新矩阵 (记录: 在该维度下，该对组合中，selected 胜出的次数)
            res[level]['matrix'][k][(d['pair'], d['selected'])] += 1
            
    return res

def write_matrix_section(writer, label, category_key, matrix_data, all_imgs):
    """通用的矩阵写入逻辑"""
    writer.writerow([f"--- {label}: {category_key} ---"])
    writer.writerow(["胜者(行) \ 败者(列)"] + all_imgs)
    for row_img in all_imgs:
        row = [row_img]
        for col_img in all_imgs:
            if row_img == col_img:
                row.append("-")
            else:
                pair = tuple(sorted([row_img, col_img]))
                win_count = matrix_data.get((pair, row_img), 0)
                row.append(win_count)
        writer.writerow(row)
    writer.writerow([])

def export_all(data_list, output_dir, prefix):
    processed = process_stats_and_matrices(data_list)
    all_imgs = TARGET_IDS

    # --- 1. 导出胜率统计 (三个维度合一或分开) ---
    stats_path = os.path.join(output_dir, f"{prefix}_胜率统计汇总.csv")
    with open(stats_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 交叉维度
        writer.writerow(["[1. FOV与材质交叉维度]"])
        writer.writerow(['FOV', '材质', '图片ID', '展示次数', '胜出次数', '胜率%'])
        for fov_mat in sorted(processed['full']['stats'].keys()):
            f_val, m_val = fov_mat.split('_')
            for img in all_imgs:
                d = processed['full']['stats'][fov_mat][img]
                wr = (d['胜出']/d['展示']*100) if d['展示']>0 else 0
                writer.writerow([f_val, m_val, img, d['展示'], d['胜出'], f"{wr:.2f}%"])
        # 材质维度
        writer.writerow(["", "[2. 仅按材质汇总]"])
        for mat in sorted(processed['mat']['stats'].keys()):
            for img in all_imgs:
                d = processed['mat']['stats'][mat][img]
                wr = (d['胜出']/d['展示']*100) if d['展示']>0 else 0
                writer.writerow([mat, img, d['展示'], d['胜出'], f"{wr:.2f}%"])
        # FOV 维度
        writer.writerow(["", "[3. 仅按FOV汇总]"])
        for fov in sorted(processed['fov']['stats'].keys(), key=lambda x: int(x)):
            for img in all_imgs:
                d = processed['fov']['stats'][fov][img]
                wr = (d['胜出']/d['展示']*100) if d['展示']>0 else 0
                writer.writerow([fov, img, d['展示'], d['胜出'], f"{wr:.2f}%"])

    # --- 2. 导出对战矩阵 (三个维度) ---
    matrix_path = os.path.join(output_dir, f"{prefix}_对战矩阵汇总.csv")
    with open(matrix_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        
        writer.writerow(["===== 分材质汇总矩阵 ====="])
        for mat in sorted(processed['mat']['matrix'].keys()):
            write_matrix_section(writer, "材质", mat, processed['mat']['matrix'][mat], all_imgs)
            
        writer.writerow(["", "===== 分FOV汇总矩阵 ====="])
        for fov in sorted(processed['fov']['matrix'].keys(), key=lambda x: int(x)):
            write_matrix_section(writer, "FOV", fov, processed['fov']['matrix'][fov], all_imgs)

        writer.writerow(["", "===== 交叉维度详细矩阵 ====="])
        for fov_mat in sorted(processed['full']['matrix'].keys()):
            write_matrix_section(writer, "FOV_材质", fov_mat, processed['full']['matrix'][fov_mat], all_imgs)

def main():
    ensure_directory(OUTPUT_FOLDER)
    global_data = []
    if not os.path.exists(INPUT_FOLDER): return
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.txt')]
    for filename in files:
        u_name = os.path.splitext(filename)[0]
        u_data = analyze_file(os.path.join(INPUT_FOLDER, filename))
        if u_data:
            u_dir = os.path.join(OUTPUT_FOLDER, u_name)
            ensure_directory(u_dir)
            export_all(u_data, u_dir, u_name)
            global_data.extend(u_data)
            print(f"处理完成: {u_name}")
    if global_data:
        g_dir = os.path.join(OUTPUT_FOLDER, "GLOBAL_TOTAL")
        ensure_directory(g_dir)
        export_all(global_data, g_dir, "全员汇总报告")
        print(f"\n[全部完成] 汇总报告已生成至: {g_dir}")

if __name__ == "__main__":
    main()