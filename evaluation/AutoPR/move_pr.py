import os
import shutil
import re

def process_papers(src_root, dst_root):
    """
    从源目录处理Paper文件并复制到目标目录
    """
    # 检查源目录是否存在
    if not os.path.exists(src_root):
        print(f"错误: 源目录 '{src_root}' 不存在。")
        return

    # 遍历源目录下的所有子目录
    for paper_name in os.listdir(src_root):
        paper_src_dir = os.path.join(src_root, paper_name)

        # 确保是文件夹而非文件
        if not os.path.isdir(paper_src_dir):
            continue

        # 定义源文件的具体路径
        # 假设结构为 source/paper_name/auto/markdown.md
        auto_dir = os.path.join(paper_src_dir, 'auto')
        src_md_path = os.path.join(auto_dir, 'markdown_refined.md')
        src_images_dir = os.path.join(auto_dir, 'images')

        # 检查markdown文件是否存在，不存在则跳过该文件夹
        if not os.path.exists(src_md_path):
            # print(f"跳过: '{paper_name}' (未找到 auto/markdown.md)")
            continue

        print(f"正在处理: {paper_name}...")

        # 定义目标路径
        # 结构为 dst/paper_name/
        paper_dst_dir = os.path.join(dst_root, paper_name)
        dst_img_dir = os.path.join(paper_dst_dir, 'img')

        # 创建目标文件夹 (包括 img 子文件夹)
        os.makedirs(dst_img_dir, exist_ok=True)

        # --- 步骤 1: 读取 Markdown 并解析图片 ---
        try:
            with open(src_md_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"  读取 Markdown 失败: {e}")
            continue

        # 使用正则匹配图片文件名
        # 匹配格式: ![](images/文件名)
        # 捕获组 (group 1) 将提取出文件名
        image_pattern = re.compile(r'!\[.*?\]\(images/([^)]+)\)')
        found_images = image_pattern.findall(content)

        # --- 步骤 2: 复制并修正 Markdown ---
        # 因为目标图片文件夹变成了 'img'，我们需要把 markdown 内容里的 'images/' 替换为 'img/'
        # 这样复制过去后的 markdown 才能正确显示图片
        new_content = content.replace('(images/', '(img/')
        
        dst_md_path = os.path.join(paper_dst_dir, 'markdown.md')
        
        try:
            with open(dst_md_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"  [√] Markdown 已复制并修正链接")
        except Exception as e:
            print(f"  Markdown 写入失败: {e}")

        # --- 步骤 3: 复制匹配到的图片 ---
        copied_count = 0
        if not os.path.exists(src_images_dir):
            print(f"  注意: 源图片目录不存在 ({src_images_dir})")
            continue

        for img_name in set(found_images): # 使用 set 去重
            src_img_path = os.path.join(src_images_dir, img_name)
            dst_img_path = os.path.join(dst_img_dir, img_name)

            if os.path.exists(src_img_path):
                try:
                    shutil.copy2(src_img_path, dst_img_path)
                    copied_count += 1
                except Exception as e:
                    print(f"  复制图片 {img_name} 失败: {e}")
            else:
                print(f"  警告: 找不到引用的图片 {img_name}")

        print(f"  [√] 已复制 {copied_count} 张图片到 {dst_img_dir}")

    print("-" * 30)
    print("所有任务处理完成。")

if __name__ == "__main__":
    input_source_root = "PaperX/mineru_outputs"
    input_dst_root = "PaperX/eveluation/AutoPR/eval/output_dir"

    process_papers(input_source_root, input_dst_root)
