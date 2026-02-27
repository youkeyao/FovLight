import re
import csv
import os
from collections import defaultdict

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def analyze_full_data(data, target_ids={'1', '3', '4', '6'}):
    lines = [l.strip() for l in data.strip().split('\n') if l.strip()]
    
    # 统计容器
    stats_full = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'出现次数': 0, '胜出次数': 0, '真实感票数': 0})))
    matrix_full = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    
    stats_mat_only = defaultdict(lambda: defaultdict(lambda: {'出现次数': 0, '胜出次数': 0, '真实感票数': 0}))
    matrix_mat_only = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # 时间统计容器: time_data[场景][材质][pair_key] = [duration1, duration2...]
    time_data_full = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    time_data_mat_only = defaultdict(lambda: defaultdict(list))

    valid_count = 0

    for i in range(len(lines)):
        line = lines[i]
        match = re.search(r'scene_(\d+)_([\w\.]+)_pair_(\w+)_vs_(\w+)选择了(\w+),用时([\d\.]+)', line)
        
        if match:
            sid = match.group(1)
            raw_mat = match.group(2)
            mat = "metal" if (raw_mat == "01" or raw_mat.lower() == "metal") else f"粗糙度_{raw_mat}"
            imgA, imgB = match.group(3), match.group(4)
            selected = match.group(5)
            duration = float(match.group(6))
            
            if imgA in target_ids and imgB in target_ids:
                is_real = (i + 1 < len(lines) and lines[i+1] == "是")
                pair_key = tuple(sorted([imgA, imgB]))

                # 基础统计累加
                for s_dict, m_dict in [(stats_full[sid][mat], matrix_full[sid][mat]), (stats_mat_only[mat], matrix_mat_only[mat])]:
                    s_dict[imgA]['出现次数'] += 1
                    s_dict[imgB]['出现次数'] += 1
                    s_dict[selected]['胜出次数'] += 1
                    if is_real: s_dict[selected]['真实感票数'] += 1
                    m_dict[pair_key][selected] += 1
                
                # 时间累加
                time_data_full[sid][mat][pair_key].append(duration)
                time_data_mat_only[mat][pair_key].append(duration)
                
                valid_count += 1

    return stats_full, matrix_full, stats_mat_only, matrix_mat_only, time_data_full, time_data_mat_only, valid_count

def material_sort_key(val):
    if val == "metal": return float('inf')
    try: return float(val.split('_')[-1])
    except: return 999.0

def export_time_report(time_full, time_mat, output_dir, file_basename):
    """生成耗时统计报告"""
    txt_path = os.path.join(output_dir, f"{file_basename}_时间分析报告.txt")
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"决策耗时分析报告: {file_basename}\n")
            f.write("="*60 + "\n\n")

            f.write("一、跨场景汇总：每种材质的平均选择时间\n")
            f.write("-" * 50 + "\n")
            for mat in sorted(time_mat.keys(), key=material_sort_key):
                f.write(f"材质: {mat}\n")
                all_durations = []
                for pair, durations in time_mat[mat].items():
                    avg = sum(durations) / len(durations)
                    f.write(f"  - {pair[0]} vs {pair[1]}: 平均用时 {avg:.2f}s (样本数: {len(durations)})\n")
                    all_durations.extend(durations)
                if all_durations:
                    total_avg = sum(all_durations) / len(all_durations)
                    f.write(f"  >> {mat} 整体平均用时: {total_avg:.2f}s\n\n")

            f.write("\n二、详细数据：按场景+材质拆分\n")
            f.write("-" * 50 + "\n")
            for sid in sorted(time_full.keys()):
                f.write(f"【场景 {sid}】\n")
                for mat in sorted(time_full[sid].keys(), key=material_sort_key):
                    f.write(f"  材质: {mat}\n")
                    for pair, durations in time_full[sid][mat].items():
                        avg = sum(durations) / len(durations)
                        f.write(f"    - {pair[0]} vs {pair[1]}: {avg:.2f}s\n")
                f.write("\n")
    except Exception as e:
        print(f"时间报告导出失败: {e}")

def export_to_csv(stats, matrix, output_dir, file_basename, target_ids, include_scene=True):
    # 此函数逻辑保持不变（同上个版本）
    all_imgs = sorted(list(target_ids))
    suffix = "按场景材质" if include_scene else "仅按材质汇总"
    stats_csv = os.path.join(output_dir, f"{file_basename}_统计_{suffix}.csv")
    matrix_csv = os.path.join(output_dir, f"{file_basename}_矩阵_{suffix}.csv")
    try:
        with open(stats_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f); h = ['场景', '材质'] if include_scene else ['材质']
            writer.writerow(h + ['图片ID', '展示次数', '胜出次数', '胜率%', '真实感票数'])
            if include_scene:
                for sid in sorted(stats.keys()):
                    for mat in sorted(stats[sid].keys(), key=material_sort_key):
                        for img in all_imgs:
                            d = stats[sid][mat][img]; wr = (d['胜出次数']/d['出现次数']*100) if d['出现次数']>0 else 0
                            writer.writerow([sid, mat, img, d['出现次数'], d['胜出次数'], round(wr, 2), d['真实感票数']])
            else:
                for mat in sorted(stats.keys(), key=material_sort_key):
                    for img in all_imgs:
                        d = stats[mat][img]; wr = (d['胜出次数']/d['出现次数']*100) if d['出现次数']>0 else 0
                        writer.writerow([mat, img, d['出现次数'], d['胜出次数'], round(wr, 2), d['真实感票数']])
        with open(matrix_csv, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f); h = ['场景', '材质'] if include_scene else ['材质']
            writer.writerow(h + ['行胜过列', *all_imgs])
            if include_scene:
                for sid in sorted(matrix.keys()):
                    for mat in sorted(matrix[sid].keys(), key=material_sort_key):
                        for r in all_imgs:
                            row = [sid, mat, r]
                            for c in all_imgs: row.append("-" if r==c else matrix[sid][mat][tuple(sorted([r,c]))][r])
                            writer.writerow(row)
                        writer.writerow([])
            else:
                for mat in sorted(matrix.keys(), key=material_sort_key):
                    for r in all_imgs:
                        row = [mat, r]
                        for c in all_imgs: row.append("-" if r==c else matrix[mat][tuple(sorted([r,c]))][r])
                        writer.writerow(row)
                    writer.writerow([])
    except Exception as e: print(f"CSV导出失败: {e}")

def main():
    input_folder, output_folder = "input", "output"
    target_ids = {'1', '3', '4', '6'}
    ensure_directory(output_folder)
    if not os.path.exists(input_folder): return

    # 全局容器
    g_stats_f = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'出现次数': 0, '胜出次数': 0, '真实感票数': 0})))
    g_matrix_f = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))
    g_stats_m = defaultdict(lambda: defaultdict(lambda: {'出现次数': 0, '胜出次数': 0, '真实感票数': 0}))
    g_matrix_m = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    g_time_f = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    g_time_m = defaultdict(lambda: defaultdict(list))

    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    for filename in files:
        with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as f:
            content = f.read()
        u_name = os.path.splitext(filename)[0]; u_dir = os.path.join(output_folder, u_name)
        ensure_directory(u_dir)

        s_f, m_f, s_m, m_m, t_f, t_m, count = analyze_full_data(content, target_ids)
        # 排除指定场景的结果（scene_003, scene_006）
        # exclude_scenes = {'003', '006', '3', '6'}
        exclude_scenes = {}
        for ex in exclude_scenes:
            s_f.pop(ex, None)
            m_f.pop(ex, None)
            t_f.pop(ex, None)

        if count > 0:
            export_to_csv(s_f, m_f, u_dir, u_name, target_ids, True)
            export_to_csv(s_m, m_m, u_dir, u_name, target_ids, False)
            export_time_report(t_f, t_m, u_dir, u_name)
            print(f"处理成功: {u_name}")

            # 累加到全局
            for sid, mats in s_f.items():
                for mat, imgs in mats.items():
                    for img, d in imgs.items():
                        g_stats_f[sid][mat][img]['出现次数'] += d['出现次数']
                        g_stats_f[sid][mat][img]['胜出次数'] += d['胜出次数']
                        g_stats_f[sid][mat][img]['真实感票数'] += d['真实感票数']
                        g_stats_m[mat][img]['出现次数'] += d['出现次数']
                        g_stats_m[mat][img]['胜出次数'] += d['胜出次数']
                        g_stats_m[mat][img]['真实感票数'] += d['真实感票数']
            for sid, mats in m_f.items():
                for mat, pairs in mats.items():
                    for pair, wins in pairs.items():
                        for winner, c in wins.items():
                            g_matrix_f[sid][mat][pair][winner] += c
                            g_matrix_m[mat][pair][winner] += c
            for sid, mats in t_f.items():
                for mat, pairs in mats.items():
                    for pair, durs in pairs.items():
                        g_time_f[sid][mat][pair].extend(durs)
                        g_time_m[mat][pair].extend(durs)

    if files:
        g_dir = os.path.join(output_folder, "GLOBAL_TOTAL_ALL_USERS"); ensure_directory(g_dir)
        export_to_csv(g_stats_f, g_matrix_f, g_dir, "全员总汇总", target_ids, True)
        export_to_csv(g_stats_m, g_matrix_m, g_dir, "全员总汇总", target_ids, False)
        export_time_report(g_time_f, g_time_m, g_dir, "全员总汇总")
        print(f"\n所有数据汇总已生成至: {g_dir}")

if __name__ == "__main__":
    main()