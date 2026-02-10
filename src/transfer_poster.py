# 这个脚本用于将每一个文件夹（01到10）的poster_final.png复制到/Paper2Poster/P2S_generated_posters/Paper2Poster-data目录下，参与评测
import os
import shutil

def transfer_posters(input_root: str, output_root: str):
    """
    从输入目录下的各个论文文件夹中提取 poster_final.png，
    并将下划线替换为冒号后，复制到输出目录下的对应文件夹中。
    """
    
    # 1. 检查输入目录是否存在
    if not os.path.exists(input_root):
        print(f"错误: 输入目录 '{input_root}' 不存在。")
        return

    # 2. 遍历输入根目录下的所有内容
    for paper_folder in os.listdir(input_root):
        input_folder_path = os.path.join(input_root, paper_folder)

        # 确保处理的是文件夹
        if os.path.isdir(input_folder_path):
            
            # 3. 构建源文件的路径
            src_file_path = os.path.join(input_folder_path, "auto", "final", "poster_final.png")
            
            # 4. 检查源文件是否存在
            if os.path.exists(src_file_path):
                
                # --- [关键修改点] ---
                # 将输入文件夹名中的 下划线 '_' 替换为 冒号 ':'
                target_folder_name = paper_folder.replace('_', ':') 
                
                # 构建输出文件夹路径
                dest_folder_path = os.path.join(output_root, target_folder_name)
                
                # 检查输出目录下的目标文件夹是否存在
                if not os.path.exists(dest_folder_path):
                    print(f"[失败] 目标文件夹不存在: {target_folder_name} (原名: {paper_folder})")
                    # 如果需要自动创建带冒号的文件夹(仅限Linux/Mac)，取消下面注释:
                    # os.makedirs(dest_folder_path, exist_ok=True)
                    continue
                
                # 构建输出文件路径
                dest_file_path = os.path.join(dest_folder_path, "poster.png")
                
                try:
                    # 5. 复制并重命名文件
                    shutil.copy2(src_file_path, dest_file_path)
                    print(f"[成功] {paper_folder} -> {target_folder_name}/poster.png")
                except Exception as e:
                    print(f"[失败] 复制出错 '{src_file_path}': {e}")
            else:
                # print(f"[跳过] {paper_folder} 中未找到图片")
                pass


if __name__ == "__main__":
    # --- 示例用法 ---
    
    # 请修改为你的实际路径
    in_dir = r""  # 输入根目录
    out_dir = r"" # 输出根目录

    print("开始执行文件转移...")
    transfer_posters(in_dir, out_dir)
    print("执行完毕。")
