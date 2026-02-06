import os
import glob
import shutil  # 引入shutil，处理文件夹移动更稳健

def move_html_files(src_dir, dest_dir):
    """将 src_dir 下的所有 HTML 文件移动到 dest_dir"""
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 1. 移动 HTML 文件
    html_files = glob.glob(os.path.join(src_dir, "*.html"))
    for file_path in html_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, filename)
        # 使用 shutil.copy 替换 os.rename，跨文件系统也能工作，且能覆盖
        shutil.copy(file_path, dest_path)
    
    print(f"  -> 已将 {len(html_files)} 个 HTML 文件复制到 {dest_dir}")

    # 2. 移动 images 文件夹
    image_dir = os.path.join(src_dir, "images")
    if os.path.exists(image_dir):
        dest_image_dir = os.path.join(dest_dir, "images")

        # 如果目标 images 文件夹已存在，shutil.copytree 可能会报错或嵌套
        # 这里逻辑：如果目标存在，先删除目标(或者你可以选择合并)，确保复制成功
        if os.path.exists(dest_image_dir):
            shutil.rmtree(dest_image_dir)

        shutil.copytree(image_dir, dest_image_dir)
        print(f"  -> 已将 images 文件夹复制到 {dest_image_dir}")
    else:
        print(f"  -> 未找到 images 文件夹")

def move_paper_pdf(src_pdf_dir, dest_pdf_dir):
    """将 src_pdf_dir 下的以origin结尾的 PDF 文件复制到 dest_pdf_dir"""
    if not os.path.exists(dest_pdf_dir):
        os.makedirs(dest_pdf_dir)

    pdf_files = glob.glob(os.path.join(src_pdf_dir, "*origin.pdf"))
    for file_path in pdf_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dest_pdf_dir, filename)
        shutil.copy(file_path, dest_path)

    print(f"  -> 已将 {len(pdf_files)} 个 PDF 文件复制到 {dest_pdf_dir}")


if __name__ == "__main__":
    # 设定输入和输出的根目录
    from pathlib import Path
    current_path = Path(__file__).resolve().parent
    # 输入根路径 (包含多个 paper 文件夹)
    INPUT_ROOT = current_path.parents[2] / "PaperX" / "mineru_outputs"
    # 输出根路径
    OUTPUT_ROOT = current_path / "data_eval" # 就在当前目录下


    if not os.path.exists(INPUT_ROOT):
        print(f"错误：输入根目录不存在 -> {INPUT_ROOT}")
        exit()

    # 获取 INPUT_ROOT 下的所有子目录（即 paper_index）
    paper_dirs = [d for d in os.listdir(INPUT_ROOT) if os.path.isdir(os.path.join(INPUT_ROOT, d))]
    
    print(f"开始处理，共发现 {len(paper_dirs)} 个项目...\n")

    for paper_index in paper_dirs:
        # 构造当前 paper 的具体源路径：~/auto/final
        current_src_dir = os.path.join(INPUT_ROOT, paper_index, "auto", "final")
        pdf_src_dir = os.path.join(INPUT_ROOT, paper_index, "auto")
        
        # 构造当前 paper 的具体目标路径：~/html
        current_dest_dir = os.path.join(OUTPUT_ROOT, paper_index, "html")
        pdf_dest_dir = os.path.join(OUTPUT_ROOT, paper_index)

        # 检查源路径是否存在，存在则处理
        if os.path.exists(current_src_dir):
            print(f"正在处理项目: {paper_index}")
            move_html_files(current_src_dir, current_dest_dir)
            move_paper_pdf(pdf_src_dir, pdf_dest_dir)

            # move_html_files(current_dest_dir, current_src_dir)
            # move_paper_pdf(pdf_dest_dir, pdf_src_dir)

            print("-" * 30)
        else:
            # 如果某个 paper 还没生成到 final 阶段，跳过并提示
            print(f"[跳过] {paper_index}: 源路径不存在 ({current_src_dir})")