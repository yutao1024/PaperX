import asyncio
import os
import sys
import json
import argparse
import traceback

try:
    from pptagent.ppteval import eval_ppt
    print("✅ pptagent.ppteval 模块导入成功")
    from html2pptx import convert_to_pptx
    print("✅ html2pptx 模块导入成功")
except ImportError as e:
    print(f"导入失败！请检查路径是否正确: {e}")
    sys.exit(1)


async def run_ppt_evaluation_pipeline(html_dir, paper_dir):
    """
    执行核心转换和评测逻辑
    html_dir: HTML 文件所在的目录 (PAPER_DIR/html)
    paper_dir: 论文根目录，也是 PPTX 存放的目录 (PAPER_DIR)
    """
    output_filename = "merged_presentation.pptx"
    pptx_full_path = os.path.join(paper_dir, output_filename)

    print(f"\n  [转换] 开始转换 HTML -> PPTX...")
    print(f"       输入: {html_dir}")
    print(f"       输出: {pptx_full_path}")

    # 清理可能存在的错误文件夹
    if os.path.isdir(pptx_full_path):
        import shutil
        print(f"       检测到同名文件夹，正在清理: {pptx_full_path}")
        shutil.rmtree(pptx_full_path)

    try:
        convert_to_pptx(html_dir, paper_dir, output_filename)
        print(f"       转换成功！")
    except Exception as e:
        return {"error": f"Conversion failed: {str(e)}"}

    if not os.path.exists(pptx_full_path):
        return {"error": f"PPTX file was not found at {pptx_full_path}"}

    print(f"  [评测] 开始评测 PPT...")
    try:
        results = await eval_ppt(pptx_full_path)
        return results
    except Exception as e:
        return {"error": f"Evaluation failed: {str(e)}"}


async def process_single_project(paper_dir):
    """
    封装单个项目的完整处理流程：路径检查 -> 运行管道 -> 保存结果
    """
    paper_dir = os.path.abspath(paper_dir)
    html_dir = os.path.join(paper_dir, "html")
    save_path = os.path.join(paper_dir, "eval_results.json")
    project_name = os.path.basename(paper_dir)

    print(f"\n{'='*10} 处理项目: {project_name} {'='*10}")

    # 预检查：如果连html目录都没有，直接跳过
    if not os.path.exists(html_dir):
        print(f"❌ 错误: 目录中未找到 'html' 文件夹，跳过: {paper_dir}")
        return False

    try:
        # 运行核心管道
        results = await run_ppt_evaluation_pipeline(html_dir, paper_dir)

        # 保存 JSON 结果
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        if results and "error" not in results:
            print(f"✅ 完成！结果已存入: {save_path}")
            return True
        else:
            print(f"❌ 流程中断: {results.get('error')}")
            return False

    except Exception as e:
        print(f"❌ 程序运行异常: {e}")
        traceback.print_exc()
        return False


async def main():
    parser = argparse.ArgumentParser(description="PPT 生成与评测工具")
    parser.add_argument("path", type=str, help="单个项目的 PAPER_DIR 路径，或者包含多个项目的根目录")
    parser.add_argument("--batch", action="store_true", help="是否启用批量模式。如果启用，将遍历 path 下的所有子目录进行处理")
    
    args = parser.parse_args()
    input_path = os.path.abspath(args.path)

    if not os.path.exists(input_path):
        print(f"错误: 路径不存在 -> {input_path}")
        return

    # --- 模式 A: 批量处理模式 ---
    if args.batch:
        print(f"🚀 === 启动批量处理模式 ===")
        print(f"📂 根目录: {input_path}")
        
        # 获取根目录下的所有一级子目录
        sub_dirs = [os.path.join(input_path, d) for d in os.listdir(input_path) 
                    if os.path.isdir(os.path.join(input_path, d))]
        
        sub_dirs.sort() # 排序以保证处理顺序一致
        
        total = len(sub_dirs)
        success_count = 0
        
        print(f"🔍 发现 {total} 个潜在子项目目录")

        for idx, sub_dir in enumerate(sub_dirs, 1):
            print(f"\n>>> 进度 [{idx}/{total}]")
            # 只有当子目录下确实有 html 文件夹时，才被视为有效项目
            if os.path.exists(os.path.join(sub_dir, "html")):
                is_success = await process_single_project(sub_dir)
                if is_success:
                    success_count += 1
            else:
                print(f"⚠️  跳过: {os.path.basename(sub_dir)} (无 html 文件夹)")
        
        print(f"\n{'='*30}")
        print(f"📊 批量任务结束: 总计 {total}, 成功 {success_count}, 失败/跳过 {total - success_count}")

    # --- 模式 B: 单个项目模式 (原有逻辑) ---
    else:
        print(f"🚀 === 启动单项目模式 ===")
        await process_single_project(input_path)

if __name__ == "__main__":
    asyncio.run(main())

# python run_benchmark.py /path/to/specific_paper_folder

# python run_benchmark.py /path/to/category_root_folder --batch