import json
import os

def calculate_scores_from_file(file_path, exist_metric_file_path=None):
    """
    从指定的 JSON 文件中读取数据，并计算 logic, vision, content 的平均分（可选：PPL和ROUGE-L）。
    """
    
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"[跳过] 找不到文件: {file_path}")
        return None

    try:
        # 2. 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 3. 计算 Logic 分数
        logic_avg = data.get("logic", {}).get("score", 0)

        # 4. 计算 Vision 平均分
        vision_items = data.get("vision", {}).values()
        vision_scores = [item["score"] for item in vision_items if isinstance(item, dict) and "score" in item]
        vision_avg = sum(vision_scores) / len(vision_scores) if vision_scores else 0

        # 5. 计算 Content 平均分
        content_items = data.get("content", {}).values()
        content_scores = [item["score"] for item in content_items if isinstance(item, dict) and "score" in item]
        content_avg = sum(content_scores) / len(content_scores) if content_scores else 0

        # 6. 可选：计算 exist_metric_file_path 中的 PPL 和 ROUGE-L 平均分数
        ppl_score = None
        rouge_l_score = None
        
        if exist_metric_file_path:
                print(f"正在检查路径: '{exist_metric_file_path}'")
                print(f"该路径是否存在: {os.path.exists(exist_metric_file_path)}")
                if not os.path.exists(exist_metric_file_path):
                    # 打印出目录下的实际文件列表，对比一下
                    dir_name = os.path.dirname(exist_metric_file_path)
                    if os.path.exists(dir_name):
                        print(f"实际目录下有哪些文件: {os.listdir(dir_name)}")
                    else:
                        print(f"甚至连目录都不存在: {dir_name}")

        if exist_metric_file_path and os.path.exists(exist_metric_file_path):
            with open(exist_metric_file_path, 'r', encoding='utf-8') as f:
                exist_data = json.load(f)
            ppl_score = exist_data.get("ppl", None)
            rouge_l_score = exist_data.get("rouge_l", None)

        return {
            "logic": logic_avg,
            "vision": vision_avg,
            "content": content_avg,
            "ppl": ppl_score,       # 可能是 None
            "rouge_l": rouge_l_score # 可能是 None
        }

    except json.JSONDecodeError:
        print(f"[错误] 文件 '{file_path}' 不是有效的 JSON 格式。")
        return None
    except Exception as e:
        print(f"[错误] 处理文件 '{file_path}' 时发生异常: {e}")
        return None

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    # -------------------------------------------------
    # 配置区域
    # -------------------------------------------------
    from pathlib import Path
    current_path = Path(__file__).resolve().parent
    root_path = os.path.join(current_path, "data_eval")  # 批量处理的根路径
    # -------------------------------------------------

    if not os.path.exists(root_path):
        print(f"[错误] 根路径不存在: {root_path}")
    else:
        print(f"开始批量处理路径: {root_path}")
        print("=" * 50)

        all_items = sorted(os.listdir(root_path))
        
        found_count = 0
        
        # 初始化总分
        logic_sum = 0
        vision_sum = 0
        content_sum = 0
        rouge_l_sum = 0
        ppl_sum = 0
        
        # 初始化计数器 (注意：PPL/ROUGE 需要单独计数)
        folder_count = 0
        ppl_count = 0
        rouge_count = 0

        for folder_name in all_items:
            full_folder_path = os.path.join(root_path, folder_name)

            if os.path.isdir(full_folder_path):
                target_json_path = os.path.join(full_folder_path, "eval_results.json")
                target_exist_metric_json_path = os.path.join(full_folder_path, "eval_result_exist_metric.json")

                results = calculate_scores_from_file(target_json_path, target_exist_metric_json_path)

                if results:
                    found_count += 1
                    folder_count += 1
                    
                    # 累加基础分数
                    logic_sum += results["logic"]
                    vision_sum += results["vision"]
                    content_sum += results["content"]
                    
                    # 累加 PPL (仅当非 None 时)
                    if results["ppl"] is not None:
                        ppl_sum += results["ppl"]
                        ppl_count += 1
                        
                    # 累加 ROUGE (仅当非 None 时)
                    if results["rouge_l"] is not None:
                        rouge_l_sum += results["rouge_l"]
                        rouge_count += 1

                    print(f"文件夹: [{folder_name}]")
                    print(f"  Logic   平均分: {results['logic']:.4f}")
                    print(f"  Vision  平均分: {results['vision']:.4f}")
                    print(f"  Content 平均分: {results['content']:.4f}")
                    
                    if results["ppl"] is not None:
                        print(f"  PPL 分数: {results['ppl']:.4f}")
                    if results["rouge_l"] is not None:
                        print(f"  ROUGE-L 分数: {results['rouge_l']:.4f}")
                    print("-" * 30)
        
        if found_count == 0:
            print("未在任何子文件夹中找到 'eval_results.json'。")
        else:
            print(f"处理完成，共处理 {found_count} 个文件夹。")
            print("=" * 50)
            
            # 计算总体平均分
            overall_logic_avg = logic_sum / folder_count if folder_count else 0
            overall_vision_avg = vision_sum / folder_count if folder_count else 0
            overall_content_avg = content_sum / folder_count if folder_count else 0
            overall_ppteval_avg = (overall_logic_avg + overall_vision_avg + overall_content_avg) / 3
            
            # 使用各自独立的计数器计算平均分
            overall_ppl_avg = ppl_sum / ppl_count if ppl_count else 0
            overall_rouge_l_avg = rouge_l_sum / rouge_count if rouge_count else 0

            print("总体平均分:")
            print(f"  Logic   平均分: {overall_logic_avg:.4f}")
            print(f"  Vision  平均分: {overall_vision_avg:.4f}")
            print(f"  Content 平均分: {overall_content_avg:.4f}")
            print(f"  PPTEval 平均分: {overall_ppteval_avg:.4f} ")
            
            if rouge_count > 0:
                print(f"  ROUGE-L 平均分: {overall_rouge_l_avg:.4f} (基于 {rouge_count} 个文件)")
            if ppl_count > 0:
                print(f"  PPL     平均分: {overall_ppl_avg:.4f} (基于 {ppl_count} 个文件)")