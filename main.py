import json
import os
from pathlib import Path

from src import *
# ========== 读取 prompt 模板 ==========
def load_prompt(prompt_path="prompt.json", prompt_name="poster_prompt"):
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(prompt_name, "")


# ========== 主流程 ==========
def main():
    # 输入总文件夹路径（包含多个论文子文件夹）
    root_folder = "/home/yutao/agent/P2S/Paper2Poster_Benchmark/output_addition/01"  # ⚠️ 修改为你的实际路径
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




    # === 遍历每个子文件夹 ===
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        auto_path = os.path.join(subdir_path, "auto")  # ✅ 进入 auto 子文件夹

        if not os.path.isdir(auto_path):
            print(f"⚠️ No 'auto' folder found in {subdir_path}, skipping...")
            continue  # 只处理存在 auto 文件夹的目录
        
        # ✅ 如果 success.txt 已存在，跳过该目录
        success_flag = os.path.join(auto_path, "success.txt")
        if os.path.isfile(success_flag):
            print(f"✅ success.txt exists in {auto_path}, skipping...")
            continue
        
        print(f"\n🚀 Processing paper folder: {auto_path}")

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

        # === 清理 markdown ===                                              去除无意义的段落，如relative work，reference，appendix等等
        print("🧹 Cleaning markdown before splitting...")
        cleaned_md_path = clean_paper_with_gpt(md_path, clean_prompt)
        
        # === 利用gpt将论文分段 === 
        paths = split_paper_with_gpt(cleaned_md_path, section_split_prompt, model="gemini-3-pro-preview")
        
        # === 利用gpt初始化dag === 
        dag = initialize_dag_with_gpt(markdown_path=cleaned_md_path,initialize_dag_prompt=initialize_dag_prompt,model="gemini-3-pro-preview")
        dag_path = os.path.join(auto_path, "dag.json")


        # === 生成visual_dag === 
        visual_dag_path=os.path.join(auto_path, "visual_dag.json")
        extract_and_generate_visual_dag_gemini_xj(markdown_path=cleaned_md_path,prompt_for_gpt=visual_dag_prompt,output_json_path=visual_dag_path,model="gemini-3-pro-preview")
        # extract_and_generate_visual_dag_gemini(markdown_path=cleaned_md_path,prompt_for_gpt=visual_dag_prompt,output_json_path=visual_dag_path)
        add_resolution_to_visual_dag(auto_path, visual_dag_path)

        # === 生成section_dag ===
        section_split_output_path=os.path.join(subdir_path, "section_split_output")
        build_section_dags_with_gemini_xj(folder_path=section_split_output_path,base_prompt=section_dag_generation_prompt,model_name="gemini-3-pro-preview")


        # === 向dag.json添加section_dag ===
        section_dag_path=os.path.join(subdir_path, "section_dag")
        merged_path = add_section_dag(section_dag_folder=section_dag_path,main_dag_path=dag_path,output_path=None)

        # === 向dag.json添加visual_dag ===
        add_visual_dag(dag_path=dag_path,visual_dag_path=visual_dag_path)

        # ===  完善dag中每一个结点的visual_node ===
        refine_visual_node(dag_path)




        # =============================  PPT部分  ================================

        # ===  按照算法选出结点，以便后续生成outline ===
        selected_node_path=os.path.join(auto_path, "selected_node.json")
        generate_selected_nodes(dag_json_path=dag_path, max_len=15,output_path=selected_node_path)

        # ===  初始化ouline ===
        outline_path= os.path.join(auto_path, "outline.json")
        outline = outline_initialize_with_gpt(dag_json_path=dag_path,outline_initialize_prompt=outline_initialize_prompt,model="gemini-3-pro-preview")
        
        # ===  生成完整ouline ===
        outline_data=generate_complete_outline(selected_node_path,outline_path,generate_complete_outline_prompt,model="gemini-3-pro-preview")
        
        # ===  配模板 ===
        arrange_template_with_gpt(outline_path,arrange_template_prompt,model="gemini-3-pro-preview")
        
        # ===  生成最终的PPT ===
        ppt_template_path="./ppt_template"
        generate_ppt_with_gemini_prompt = {"role":"system","content":"You are given (1) a slide node (JSON) and (2) an HTML slide template. Your task: revise the HTML template to produce the final slide HTML using the node content. Node fields: text (slide textual content), figure (list of images to display, each has name, caption, resolution), formula (list of formula images to display, each has name, caption, resolution), template (template filename). IMPORTANT RULES: 1) Only modify places in the HTML that are marked by comments like <!-- You need to revise the following parts. X: ... -->. 2) For 'Subjects' sections: replace the placeholder title with ONE concise summary sentence for this slide. 3) For 'Image' sections (<img ...>): replace src with the relative path extracted from node.figure/formula[i].name; the node image name may be markdown like '![](images/abc.jpg)', use only 'images/abc.jpg'. 4) For 'Text' sections: replace the placeholder text with the node.text content, formatted cleanly in HTML; keep it readable and you may use <p>, <ul><li>, <br/> appropriately. 5) If the template expects more images/text blocks than provided by the node, leave the missing positions unchanged and do not invent content. 6) If the node provides more images than the template has slots, fill slots in order and ignore the rest. 7) Preserve all other HTML, CSS, and structure exactly. OUTPUT FORMAT: Return ONLY the revised HTML as plain text. Do NOT wrap it in markdown fences. Do NOT add explanations."}
        
        # ===下面的函数是用来调用智增增api的，当官方api充足时，可以替换掉这个函数（把zzz删除）===
        # generate_ppt_with_gemini(outline_path,ppt_template_path,generate_ppt_with_gemini_prompt)
        # generate_ppt_with_gemini_zzz(outline_path,ppt_template_path,generate_ppt_with_gemini_prompt)
        generate_ppt_with_gemini_xj(outline_path,ppt_template_path,generate_ppt_with_gemini_prompt,model="gemini-3-pro-preview")

        # ===  Refiner  ===                
        refinement_ppt(input_index=auto_path, prompts=[commenter_prompt, reviser_prompt], model="gemini-3-pro-preview", max_iterations=3)

        # =============================  Poster部分  ================================
        
        poster_outline_path = os.path.join(auto_path, "poster_outline.txt")
        
        print(f"📝 Generating poster outline at: {poster_outline_path}")
        generate_poster_outline_txt(dag_path=dag_path,poster_outline_path=poster_outline_path,poster_outline_prompt=poster_outline_prompt,model_name="gemini-3-pro-preview")
        print (f"✅ Poster outline generated.")


        poster_path=os.path.join(auto_path, "poster.html")


        subdir_name = Path(subdir_path).name
        poster_outline_path_modified  = os.path.join(auto_path, "poster_outline_modified.txt")
        print(f"📝 Modifying poster outline for paper: {subdir_name}")
        modify_poster_outline(poster_outline_path=poster_outline_path,poster_paper_name=subdir_name,modified_poster_outline_path=poster_outline_path_modified)
        print (f"✅ Poster outline modified.")
        
        modified_poster_logic(poster_outline_path_modified, modified_poster_logic_prompt, model="gemini-3-pro-preview")


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




        refinement_poster(input_html_path=poster_path, prompts=poster_refinement_prompt,output_html_path=poster_final_output_path,model="gemini-3-pro-preview")
        
        print (f"✅ Poster HTML refined. Final poster at: {poster_final_output_path}")


        # =============================  PR部分  ================================
        pr_template_path="./pr_template.md"
        basic_information_path = extract_basic_information(dag_path=dag_path, auto_path=auto_path,extract_basic_information_prompt=extract_basic_information_prompt,model="gemini-3-pro-preview")
        
        initialize_pr_markdown(basic_information_path=basic_information_path,auto_path=auto_path,pr_template_path=pr_template_path)

        pr_path=os.path.join(auto_path, "markdown.md")


        generate_pr_from_dag(dag_path=dag_path, pr_path=pr_path, generate_pr_prompt=generate_pr_prompt, model="gemini-3-pro-preview")
        print(f"📝 PR generated at: {pr_path}")
        
        add_title_and_hashtag(pr_path=pr_path, add_title_and_hashtag_prompt=add_title_and_hashtag_prompt, model="gemini-3-pro-preview")
        add_institution_tag(pr_path=pr_path)
        


        dedup_consecutive_markdown_images(pr_path, inplace=True)

        print(f"✅ PR markdown post-processed.")

        print(f"🖊️ Refining PR markdown with LLM...")
        pr_refine_path=os.path.join(auto_path, "markdown_refined.md")

        refinement_pr(pr_path=pr_path, pr_refine_path=pr_refine_path, prompts=pr_refinement_prompt, model="gemini-3-pro-preview")
        print (f"✅ PR markdown refined.")



        # =============================================================
        # 在auto目录下创建success.txt作为标志
        success_file_path = os.path.join(auto_path, "success.txt")
        open(success_file_path, "w").close()



        print(f"✅ Finished processing: {subdir}\n{'-' * 80}")

    print("\n🎉 All papers processed successfully!")


if __name__ == "__main__":
    main()