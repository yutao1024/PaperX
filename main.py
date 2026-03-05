import argparse
import json
import os
import sys
from pathlib import Path

from src import *

import yaml

# ============= 读取配置 ===============
def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {config_path}")
        if not os.path.exists("config.example.yaml"):
            sys.exit(1)
        print("  Copy the example config and edit it:")
        print("    cp config.example.yaml config.yaml")
        sys.exit(1)
    return config

# ========== 读取 prompt 模板 ==========
def load_prompt(prompt_path="prompt.json", prompt_name="poster_prompt"):
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(prompt_name, "")


def validate_config(config):
    """Check required config keys and warn if paths missing."""
    required = {
        "path": ["root_folder"],
        "model_settings": ["generation_model"],
        "api_keys": ["gemini_api_key", "openai_api_key"],
    }
    for section, keys in required.items():
        if section not in config:
            print(f"Warning: config missing section '{section}'")
            continue
        for k in keys:
            if k not in config[section]:
                print(f"Warning: config['{section}']['{k}'] not set")
    root = config.get("path", {}).get("root_folder")
    if root and not os.path.isdir(root):
        print(f"Warning: root_folder does not exist: {root}")


# ========== 主流程 ==========
def main(config_path="config.yaml", paper=None, dry_run=False, force=False):
    config = load_config(config_path)
    validate_config(config)
    model_name = config["model_settings"]["generation_model"]

    # 输入总文件夹路径（包含多个论文子文件夹）
    root_folder = config["path"]["root_folder"]
    prompt_path = "prompt.json"


    print("📘 Loading prompts...")
    # dag prompt:
    section_split_prompt = load_prompt(prompt_path, prompt_name="section_split_prompt")
    clean_prompt = load_prompt(prompt_path, prompt_name="clean_prompt")
    initialize_dag_prompt = load_prompt(prompt_path, prompt_name="initialize_dag_prompt")
    visual_dag_prompt = load_prompt(prompt_path, prompt_name="visual_dag_prompt")
    section_dag_generation_prompt = load_prompt(prompt_path, prompt_name="section_dag_generation_prompt")
    
    # ppt prompt:
    outline_initialize_prompt = load_prompt(prompt_path, prompt_name="outline_initialize_prompt")
    generate_complete_outline_prompt = load_prompt(prompt_path, prompt_name="generate_complete_outline_prompt")
    arrange_template_prompt= load_prompt(prompt_path, prompt_name="arrange_template_prompt")
    commenter_prompt= load_prompt(prompt_path, prompt_name="commenter_prompt")
    reviser_prompt= load_prompt(prompt_path, prompt_name="reviser_prompt")
    
    # poster prompt:
    poster_outline_prompt= load_prompt(prompt_path, prompt_name="poster_outline_prompt")
    poster_refinement_prompt= load_prompt(prompt_path, prompt_name="poster_refinement_prompt")
    modified_poster_logic_prompt=load_prompt(prompt_path, prompt_name="modified_poster_logic_prompt")

    # pr prompt:
    extract_basic_information_prompt = load_prompt("prompt.json", "extract_basic_information_prompt")
    generate_pr_prompt = load_prompt(prompt_path, prompt_name="generate_pr_prompt")
    add_title_and_hashtag_prompt = load_prompt(prompt_path, prompt_name="add_title_and_hashtag_prompt")
    pr_refinement_prompt = load_prompt(prompt_path, prompt_name="pr_refinement_prompt")

    # === dry-run: only list papers that would be processed ===
    if dry_run:
        would_process = []
        for subdir in sorted(os.listdir(root_folder)):
            subdir_path = os.path.join(root_folder, subdir)
            auto_path = os.path.join(subdir_path, "auto")
            if not os.path.isdir(auto_path):
                continue
            if not force and os.path.isfile(os.path.join(auto_path, "success.txt")):
                continue
            if paper and subdir != paper:
                continue
            pdf_path = os.path.join(auto_path, f"{subdir}_origin.pdf")
            md_path = os.path.join(auto_path, f"{subdir}.md")
            if not os.path.exists(pdf_path) or not os.path.exists(md_path):
                continue
            would_process.append(subdir)
        print("Dry-run: would process", len(would_process), "paper(s):", would_process)
        return

    # === 遍历每个子文件夹 ===
    processed, failed = 0, 0
    for subdir in sorted(os.listdir(root_folder)):
        subdir_path = os.path.join(root_folder, subdir)
        auto_path = os.path.join(subdir_path, "auto")

        if not os.path.isdir(auto_path):
            print(f"⚠️ No 'auto' folder found in {subdir_path}, skipping...")
            continue

        if paper and subdir != paper:
            continue

        success_flag = os.path.join(auto_path, "success.txt")
        if not force and os.path.isfile(success_flag):
            print(f"✅ success.txt exists in {auto_path}, skipping...")
            continue

        print(f"\n🚀 Processing paper folder: {auto_path}")

        try:
            # === 根据子文件夹名精确匹配文件 ===
            target_pdf = f"{subdir}_origin.pdf"
            target_md = f"{subdir}.md"

            pdf_path = os.path.join(auto_path, target_pdf)
            md_path = os.path.join(auto_path, target_md)

            # === 检查文件是否存在 ===
            if not os.path.exists(pdf_path):
                print(f"⚠️ Missing expected PDF: {target_pdf} in {auto_path}, skipping...")
                continue
            if not os.path.exists(md_path):
                print(f"⚠️ Missing expected Markdown: {target_md} in {auto_path}, skipping...")
                continue

            print(f"📄 Matched files:\n   PDF: {target_pdf}\n   MD:  {target_md}")

            # === 输出文件路径 ===
            output_html = os.path.join(auto_path, "poster.html")
            graph_json_path = os.path.join(auto_path, "graph.json")

            # === 清理 markdown ===
            print("🧹 Cleaning markdown before splitting...")
            cleaned_md_path = clean_paper(md_path, clean_prompt, model="gemini-3-pro-preview", config=config)

            # === 利用gpt将论文分段 ===
            paths = split_paper(cleaned_md_path, section_split_prompt, model="gemini-3-pro-preview" ,config=config)
            # === 利用gpt初始化dag ===
            dag = initialize_dag(markdown_path=cleaned_md_path,initialize_dag_prompt=initialize_dag_prompt,model="gemini-3-pro-preview", config=config)
            dag_path = os.path.join(auto_path, "dag.json")
            # === 生成visual_dag ===
            visual_dag_path=os.path.join(auto_path, "visual_dag.json")
            extract_and_generate_visual_dag(markdown_path=cleaned_md_path,prompt_for_gpt=visual_dag_prompt,output_json_path=visual_dag_path,model="gemini-3-pro-preview", config=config)
            add_resolution_to_visual_dag(auto_path, visual_dag_path)
            # === 生成section_dag ===
            section_split_output_path=os.path.join(subdir_path, "section_split_output")
            build_section_dags(folder_path=section_split_output_path,base_prompt=section_dag_generation_prompt,model="gemini-3-pro-preview", config=config)
            # === 向dag.json添加section_dag ===
            section_dag_path=os.path.join(subdir_path, "section_dag")
            merged_path = add_section_dag(section_dag_folder=section_dag_path,main_dag_path=dag_path,output_path=None)
            # === 向dag.json添加visual_dag ===
            add_visual_dag(dag_path=dag_path,visual_dag_path=visual_dag_path)
            # === 完善dag中每一个结点的visual_node ===
            refine_visual_node(dag_path)




            # =============================  PPT部分  ==============================
            selected_node_path=os.path.join(auto_path, "selected_node.json")
            generate_selected_nodes(dag_json_path=dag_path, max_len=15,output_path=selected_node_path)
            outline_path= os.path.join(auto_path, "outline.json")
            outline = outline_initialize(dag_json_path=dag_path,outline_initialize_prompt=outline_initialize_prompt,model=model_name, config=config)
            outline_data=generate_complete_outline(selected_node_path,outline_path,generate_complete_outline_prompt,model=model_name, config=config)
            arrange_template(outline_path,arrange_template_prompt,model=model_name, config=config)
            ppt_template_path="./ppt_template"
            generate_ppt_prompt = {"role":"system","content":"You are given (1) a slide node (JSON) and (2) an HTML slide template. Your task: revise the HTML template to produce the final slide HTML using the node content. Node fields: text (slide textual content), figure (list of images to display, each has name, caption, resolution), formula (list of formula images to display, each has name, caption, resolution), template (template filename). IMPORTANT RULES: 1) Only modify places in the HTML that are marked by comments like <!-- You need to revise the following parts. X: ... -->. 2) For 'Subjects' sections: replace the placeholder title with ONE concise summary sentence for this slide. 3) For 'Image' sections (<img ...>): replace src with the relative path extracted from node.figure/formula[i].name; the node image name may be markdown like '![](images/abc.jpg)', use only 'images/abc.jpg'. 4) For 'Text' sections: replace the placeholder text with the node.text content, formatted cleanly in HTML; keep it readable and you may use <p>, <ul><li>, <br/> appropriately. 5) If the template expects more images/text blocks than provided by the node, leave the missing positions unchanged and do not invent content. 6) If the node provides more images than the template has slots, fill slots in order and ignore the rest. 7) Preserve all other HTML, CSS, and structure exactly. OUTPUT FORMAT: Return ONLY the revised HTML as plain text. Do NOT wrap it in markdown fences. Do NOT add explanations."}
            generate_ppt(outline_path,ppt_template_path,generate_ppt_prompt,model=model_name, config=config)
            refinement_ppt(input_index=auto_path, prompts=[commenter_prompt, reviser_prompt], model=model_name, max_iterations=3, config=config)

            # =============================  Poster部分  ==============================
            poster_outline_path = os.path.join(auto_path, "poster_outline.txt")
            print(f"📝 Generating poster outline at: {poster_outline_path}")
            generate_poster_outline_txt(dag_path=dag_path,poster_outline_path=poster_outline_path,poster_outline_prompt=poster_outline_prompt,model=model_name, config=config)
            print (f"✅ Poster outline generated.")
            poster_path=os.path.join(auto_path, "poster.html")
            subdir_name = Path(subdir_path).name
            poster_outline_path_modified  = os.path.join(auto_path, "poster_outline_modified.txt")
            print(f"📝 Modifying poster outline for paper: {subdir_name}")
            modify_poster_outline(poster_outline_path=poster_outline_path,poster_paper_name=subdir_name,modified_poster_outline_path=poster_outline_path_modified)
            print (f"✅ Poster outline modified.")
            modified_poster_logic(poster_outline_path_modified, modified_poster_logic_prompt, model=model_name, config=config)
            print(f"🖼️ Building poster HTML at: {poster_path}")
            build_poster_from_outline(poster_outline_path=poster_outline_path_modified,poster_template_path="./poster_template/poster_template.html",poster_path=poster_path,)
            print (f"✅ Poster HTML built.")
            print(f"🖊️ Modifying title and authors in poster HTML...")
            modify_title_and_author(dag_path=dag_path,poster_path=poster_path)
            print (f"✅ Title and authors modified.")
            poster_final_index = os.path.join(auto_path, "final")
            os.makedirs(poster_final_index, exist_ok=True)
            poster_final_output_path = os.path.join(poster_final_index, "poster_final.html")
            print(f"🖊️ Refining poster HTML with Gemini...")
            out = inject_img_section_to_poster(figure_path="./poster_template/expore_our_work_in_detail.jpg",auto_path=auto_path,poster_path=poster_path)
            refinement_poster(input_html_path=poster_path, prompts=poster_refinement_prompt,output_html_path=poster_final_output_path,model=model_name, config=config)
            print (f"✅ Poster HTML refined. Final poster at: {poster_final_output_path}")


            # =============================  PR部分  ================================
            pr_template_path="./pr_template.md"
            basic_information_path = extract_basic_information(dag_path=dag_path, auto_path=auto_path,extract_basic_information_prompt=extract_basic_information_prompt,model=model_name, config=config)
            initialize_pr_markdown(basic_information_path=basic_information_path,auto_path=auto_path,pr_template_path=pr_template_path)
            pr_path=os.path.join(auto_path, "markdown.md")
            generate_pr_from_dag(dag_path=dag_path, pr_path=pr_path, generate_pr_prompt=generate_pr_prompt, model=model_name, config=config)
            print(f"📝 PR generated at: {pr_path}")
            add_title_and_hashtag(pr_path=pr_path, add_title_and_hashtag_prompt=add_title_and_hashtag_prompt, model=model_name, config=config)
            add_institution_tag(pr_path=pr_path)
            dedup_consecutive_markdown_images(pr_path, inplace=True)
            print(f"✅ PR markdown post-processed.")
            print(f"🖊️ Refining PR markdown with LLM...")
            pr_refine_path=os.path.join(auto_path, "markdown_refined.md")
            refinement_pr(pr_path=pr_path, pr_refine_path=pr_refine_path, prompts=pr_refinement_prompt, model=model_name, config=config)
            print (f"✅ PR markdown refined.")
            success_file_path = os.path.join(auto_path, "success.txt")
            open(success_file_path, "w").close()
            processed += 1
            print(f"✅ Finished processing: {subdir}\n{'-' * 80}")
        except Exception as e:
            failed += 1
            failed_path = os.path.join(auto_path, "failed.txt")
            with open(failed_path, "w", encoding="utf-8") as f:
                f.write(f"{type(e).__name__}: {e}\n")
            print(f"❌ Failed {subdir}: {e}\n  See {failed_path}")

    print("\n🎉 Batch done. Processed:", processed, "Failed:", failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaperX: from paper PDFs to PPTs, Posters, and PRs")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--paper", metavar="SUBDIR", help="Process only this paper subfolder (name under root_folder).")
    parser.add_argument("--dry-run", action="store_true", help="Only list papers that would be processed, then exit.")
    parser.add_argument("--force", action="store_true", help="Re-run even if success.txt exists.")
    args = parser.parse_args()
    main(config_path=args.config, paper=args.paper, dry_run=args.dry_run, force=args.force)
