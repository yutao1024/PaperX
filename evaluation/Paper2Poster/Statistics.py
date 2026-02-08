import os
import json
import math

def calculate_statistics(root_dir):
    # ================= 1. 初始化存储列表 =================
    # judge_result 相关
    scores_element = []
    scores_engage = []
    scores_layout = []
    scores_aesthetic_avg = []
    
    scores_clarity = [] # 对应 information_low_level
    scores_content = []
    scores_logic = []
    scores_info_avg = []
    
    scores_overall_avg = []

    # overall_qa_result 相关
    scores_detail_acc = []
    scores_understand_acc = []
    scores_qa_combined_avg = []

    # stats_result 相关
    scores_clip_sim = []
    scores_visual_rel = []
    scores_ppl_sum = []

    # ================= 2. 遍历目录 =================
    # 获取根目录下所有的子文件夹（即 paper 文件夹）
    try:
        subdirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    except FileNotFoundError:
        print(f"错误：找不到目录 {root_dir}")
        return

    print(f"找到 {len(subdirs)} 个文件夹，开始处理...")

    for paper_dir in subdirs:
        # ----- 处理 judge_result.json -----
        judge_path = os.path.join(paper_dir, 'P2S_generated_posters', 'judge_result.json')
        if os.path.exists(judge_path):
            with open(judge_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    res = data.get('results', {})
                    
                    # 提取 Aesthetic 细项
                    scores_element.append(res.get('aesthetic_element', {}).get('score', 0))
                    scores_engage.append(res.get('aesthetic_engagement', {}).get('score', 0))
                    scores_layout.append(res.get('aesthetic_layout', {}).get('score', 0))
                    scores_aesthetic_avg.append(data.get('aesthetic_average', 0))

                    # 提取 Information 细项 (Clarity 对应 information_low_level)
                    scores_clarity.append(res.get('information_low_level', {}).get('score', 0))
                    scores_logic.append(res.get('information_logic', {}).get('score', 0))
                    scores_content.append(res.get('information_content', {}).get('score', 0))
                    scores_info_avg.append(data.get('information_average', 0))

                    # 提取 Overall
                    scores_overall_avg.append(data.get('overall_average', 0))
                except Exception as e:
                    print(f"读取 {judge_path} 出错: {e}")

        # ----- 处理 overall_qa_result.json -----
        qa_path = os.path.join(paper_dir, 'P2S_generated_posters', 'overall_qa_result.json')
        if os.path.exists(qa_path):
            with open(qa_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    d_acc = data.get('avg_detail_accuracy', 0)
                    u_acc = data.get('avg_understanding_accuracy', 0)
                    
                    scores_detail_acc.append(d_acc)
                    scores_understand_acc.append(u_acc)
                    # 计算二者的平均值
                    scores_qa_combined_avg.append((d_acc + u_acc) / 2)
                except Exception as e:
                    print(f"读取 {qa_path} 出错: {e}")

        # ----- 处理 stats_result.json -----
        stats_path = os.path.join(paper_dir, 'P2S_generated_posters', 'stats_result.json')
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    scores_clip_sim.append(data.get('CLIP_similarity', 0))
                    scores_visual_rel.append(data.get('visual_relevance', 0))

                    # 计算所有 PPL 之和
                    # 识别所有以 _ppl 结尾的键
                    current_ppl_sum = sum(value for key, value in data.items() if key.endswith('_ppl'))
                    scores_ppl_sum.append(current_ppl_sum)
                except Exception as e:
                    print(f"读取 {stats_path} 出错: {e}")

    # ================= 3. 计算辅助函数 =================
    def calc_avg(lst):
        # 过滤掉 nan 值
        valid_values = [x for x in lst if not math.isnan(x)]
        return sum(valid_values) / len(valid_values) if valid_values else 0.0

    # ================= 4. 输出结果 =================
    print("\n" + "="*30)
    print("      统计结果汇报      ")
    print("="*30)
    
    print(f"【Judge Result 统计】(样本数: {len(scores_overall_avg)})")
    print(f"  - Element Average:   {calc_avg(scores_element):.4f}")
    print(f"  - Layout Average:    {calc_avg(scores_layout):.4f}")
    print(f"  - Engage Average:    {calc_avg(scores_engage):.4f}")
    print(f"  - Aesthetic Avg:     {calc_avg(scores_aesthetic_avg):.4f}")
    print("-" * 20)
    print(f"  - Clarity (Low-level): {calc_avg(scores_clarity):.4f}")
    print(f"  - Content Average:     {calc_avg(scores_content):.4f}")
    print(f"  - Logic Average:       {calc_avg(scores_logic):.4f}")
    print(f"  - Information Avg:     {calc_avg(scores_info_avg):.4f}")
    print("-" * 20)
    print(f"  - Overall Average:     {calc_avg(scores_overall_avg):.4f}")

    print(f"\n【QA Result 统计】(样本数: {len(scores_detail_acc)})")
    # print(f"  - Detail Accuracy Avg:        {calc_avg(scores_detail_acc):.4f}")
    print(f"  - V-Avg:        {calc_avg(scores_detail_acc):.4f}")
    # print(f"  - Understanding Accuracy Avg: {calc_avg(scores_understand_acc):.4f}")
    print(f"  - I-Avg: {calc_avg(scores_understand_acc):.4f}")
    print(f"  - QA Avg:            {calc_avg(scores_qa_combined_avg):.4f}")

    print(f"\n【Stats Result 统计】(样本数: {len(scores_clip_sim)})")
    print(f"  - Vis.Sim. :   {calc_avg(scores_clip_sim):.4f}")
    print(f"  - PPL:       {calc_avg(scores_ppl_sum)/5:.4f}")
    print(f"  - Fig.Rel. :  {calc_avg(scores_visual_rel):.4f}")

    # 返回字典形式的结果，便于后续调用
    return {
        "judge": {
            "element": calc_avg(scores_element),
            "engage": calc_avg(scores_engage),
            "layout": calc_avg(scores_layout),
            "aesthetic_avg": calc_avg(scores_aesthetic_avg),
            "clarity": calc_avg(scores_clarity),
            "content": calc_avg(scores_content),
            "logic": calc_avg(scores_logic),
            "info_avg": calc_avg(scores_info_avg),
            "overall": calc_avg(scores_overall_avg)
        },
        "qa": {
            "detail": calc_avg(scores_detail_acc),
            "understanding": calc_avg(scores_understand_acc),
            "combined": calc_avg(scores_qa_combined_avg)
        },
        "stats": {
            "clip": calc_avg(scores_clip_sim),
            "visual_relevance": calc_avg(scores_visual_rel),
            "ppl_sum": calc_avg(scores_ppl_sum)
        }
    }

# ================= 使用示例 =================
if __name__ == "__main__":
# 将下面的路径替换为你实际的文件夹路径
    root_directory = "/home/yutao/agent/Paper2Poster/eval_results"
    result_path = "/home/yutao/agent/Paper2Poster/statistics_result_addition.json"

    statistics = calculate_statistics(root_directory)
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=4, ensure_ascii=False)
    print(f"\n统计结果已保存到 {result_path}")
