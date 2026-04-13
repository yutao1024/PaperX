import json
import os
import re
import base64
from PIL import Image
from typing import Optional
from openai import OpenAI
from google import genai
from google.genai import types

from .llm_icl import get_instruction_with_icl


# ==========  调用 Gemini 删除无用段落 ==========
def clean_paper(markdown_path, clean_prompt, model, config):
    """
    使用 Google Gemini 清理论文 Markdown 文件：
    删除 Abstract / Related Work / Appendix / References 等部分，
    保留标题、作者、Introduction、Methods、Experiments、Conclusion。
    """
    # === 初始化 Client  ===
    client = genai.Client(
        api_key=config['api_keys']['gemini_api_key']
    )

    # === 读取 markdown 文件 ===
    with open(markdown_path, "r", encoding="utf-8") as f:
        md_text = f.read().strip()

    full_prompt = (
        f"{clean_prompt}\n\n"
        "=== PAPER MARKDOWN TO CLEAN ===\n"
        f"\"\"\"{md_text}\"\"\"\n\n"
        "Return only the cleaned markdown, keeping all formatting identical to the original."
    )

    print("🧹 Sending markdown to Gemini for cleaning...")

    try:
        # === 调用 Gemini API (Client 模式) ===
        resp = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        cleaned_text = resp.text.strip()
        
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        return None

    # === 提取纯 markdown（防止模型返回 ```markdown ``` 块） ===
    m = re.search(r"```markdown\s*([\s\S]*?)```", cleaned_text)
    if not m:
        m = re.search(r"```\s*([\s\S]*?)```", cleaned_text)
    cleaned_text = m.group(1).strip() if m else cleaned_text

    # === 生成输出文件路径 ===
    dir_path = os.path.dirname(markdown_path)
    base_name = os.path.basename(markdown_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(dir_path, f"{name}_cleaned{ext}")

    # === 保存结果 ===
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f"✅ Cleaned markdown saved to: {output_path}")

    return output_path


# ==========  调用 Gemini 划分段落 ==========
SECTION_RE = re.compile(r'^#\s+\d+(\s|$)', re.MULTILINE)


def sanitize_filename(name: str) -> str:
    """移除非法字符：/ \ : * ? \" < > |"""
    unsafe = r'\/:*?"<>|'
    return "".join(c for c in name if c not in unsafe).strip()


def split_paper(
    cleaned_md_path: str,      # 输入: A/auto/clean_paper.md
    prompt: str,
    separator: str = "===SPLIT===",
    model: str = "gemini-3.0-pro-preview",
    config: dict = None
):
    """
    使用 Gemini 拆分论文，并将所有拆分后的 markdown 保存在：
        <parent_of_auto>/section_split_output/
    """
    # 1️⃣ 输入文件所在的 auto 文件夹
    auto_dir = os.path.dirname(os.path.abspath(cleaned_md_path))

    # 2️⃣ auto 的上一级目录（即 A/）
    parent_dir = os.path.dirname(auto_dir)

    # 3️⃣ 最终输出目录 A/section_split_output/
    output_dir = os.path.join(parent_dir, "section_split_output")
    os.makedirs(output_dir, exist_ok=True)

    # === 读取 markdown ===
    with open(cleaned_md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read()

    # === 2. 初始化 Gemini Client ===
    client = genai.Client(
        api_key=config['api_keys']['gemini_api_key']
    )

    # === 提取一级 section 信息供参考 (假设 SECTION_RE 已在外部定义) ===
    # 注意：确保 SECTION_RE 在此函数作用域内可用
    section_positions = [(m.start(), m.group()) for m in SECTION_RE.finditer(markdown_text)]

    auto_analysis = "Detected top-level sections:\n"
    for pos, sec in section_positions:
        auto_analysis += f"- position={pos}, heading='{sec.strip()}'\n"
    auto_analysis += "\nThese are for your reference. You MUST still split strictly by the rules.\n"

    # === 构建 prompt ===
    final_prompt = (
        prompt
        + "\n\n---\nBelow is an automatic analysis of top-level sections (for reference only):\n"
        + auto_analysis
        + "\n---\nHere is the FULL MARKDOWN PAPER:\n\n"
        + markdown_text
    )

    # === 3. Gemini 调用 ===
    try:
        response = client.models.generate_content(
            model=model,
            contents=final_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0
            )
        )
        output_text = response.text
    except Exception as e:
        print(f"❌ Gemini Split Error: {e}")
        return []

    # === 按分隔符拆分 ===
    # 简单的容错处理，防止模型没有完全按格式输出
    if not output_text:
        print("❌ Empty response from Gemini")
        return []

    chunks = [c.strip() for c in output_text.split(separator) if c.strip()]
    saved_paths = []

    # === 保存拆分后的 chunks ===
    for chunk in chunks:
        lines = chunk.splitlines()

        # 找第一行有效内容
        first_line = next((ln.strip() for ln in lines if ln.strip()), "")

        # 解析标题
        if first_line.startswith("#"):
            title = first_line.lstrip("#").strip()
        else:
            title = first_line[:20].strip()

        # 注意：sanitize_filename 需要在外部定义或引入
        filename = sanitize_filename(title) + ".md"
        filepath = os.path.join(output_dir, filename)

        # 写文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chunk)

        saved_paths.append(filepath)

    print(f"✅ Paper is splitted successfully")
    return saved_paths


# ==========  调用 Gemini 初始化dag.json ==========
def initialize_dag(markdown_path, initialize_dag_prompt, model, config=None):
    """
    使用 Gemini 初始化论文 DAG。
    
    输入:
        markdown_path: markdown 文件路径
        initialize_dag_prompt: prompt 字符串
        model: 模型名称 (建议使用 gemini-2.0-flash 或 pro)
        config: 包含 api_keys 的配置字典
    
    输出:
        dag.json: 保存在 markdown 文件同目录
        返回 python 字典形式的 DAG
    """
    # --- load markdown ---
    if not os.path.exists(markdown_path):
        raise FileNotFoundError(f"Markdown not found: {markdown_path}")

    with open(markdown_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # --- Gemini Client Init ---
    client = genai.Client(
        api_key=config['api_keys']['gemini_api_key']
    )

    # --- Gemini Call ---
    # 将 Prompt 和 文本合并作为用户输入，System Prompt 放入 config
    full_content = f"{initialize_dag_prompt}\n\n{md_text}"

    try:
        response = client.models.generate_content(
            model=model,
            contents=full_content,
            config=types.GenerateContentConfig(
                system_instruction="You are an expert academic document parser and structural analyzer.",
                temperature=0.0,
                response_mime_type="application/json" # <--- 强制输出 JSON 模式
            )
        )
        raw_output = response.text.strip()
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        raise e

    # --- Extract JSON (remove possible markdown fences) ---
    # Gemini 在 JSON 模式下通常只返回纯 JSON，但保留此逻辑以防万一
    cleaned = raw_output

    # Remove ```json ... ```
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.split("\n", 1)[1]

    # Last safety: locate JSON via first { and last }
    try:
        first = cleaned.index("{")
        last = cleaned.rindex("}")
        cleaned = cleaned[first:last+1]
    except Exception:
        pass

    try:
        dag_data = json.loads(cleaned)
    except json.JSONDecodeError:
        print("⚠️ Standard JSON parsing failed. Attempting regex repair for backslashes...")
        try:
            # 这里保留原有的重试逻辑 (通常是为了处理转义字符)
            dag_data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"Gemini output is not valid JSON:\n{raw_output}")

    # --- Save dag.json ---
    out_dir = os.path.dirname(markdown_path)
    out_path = os.path.join(out_dir, "dag.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dag_data, f, indent=4, ensure_ascii=False)

    print(f"✅ DAG saved to: {out_path}")

    return dag_data


# ==========  调用 大模型 添加视觉结点 ==========
def extract_and_generate_visual_dag(
    markdown_path: str,
    prompt_for_gpt: str,
    output_json_path: str,
    model="gemini-3.0-pro-preview",
    config=None
):
    """
    输入:
        markdown_path: 原论文 markdown 文件路径
        prompt_for_gpt: 给 GPT 使用的 prompt
        output_json_path: 生成的 visual_dag.json 存放路径
        model: 默认 gemini-3.0-pro-preview

    输出:
        生成 visual_dag.json
        返回 Python dict
    """
    # === 1. 读取 markdown ===
    if not os.path.exists(markdown_path):
        raise FileNotFoundError(f"Markdown not found: {markdown_path}")

    with open(markdown_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # === 2. 正则提取所有图片相对引用 ===
    pattern = r"!\[[^\]]*\]\(([^)]+)\)"
    matches = re.findall(pattern, md_text)

    # 过滤为相对路径（不包含http）
    relative_imgs = [m for m in matches if not m.startswith("http")]

    # 生成标准格式 name 字段使用的写法 "![](xxx)"
    normalized_refs = [f"![]({m})" for m in relative_imgs]

    # === 3. 发送给 Gemini ===
    # 初始化 Client
    client = genai.Client(
        api_key=config['api_keys']['gemini_api_key']
    )

    gpt_input = prompt_for_gpt + "\n\n" + \
        "### Extracted Image References:\n" + \
        json.dumps(normalized_refs, indent=2) + "\n\n" + \
        "### Full Markdown:\n" + md_text

    try:
        response = client.models.generate_content(
            model=model,
            contents=gpt_input,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"  # 强制 JSON 输出
            )
        )
        visual_dag_str = response.text.strip()
    except Exception as e:
        print(f"❌ Gemini API Error: {e}")
        raise e

    # === JSON 解析兜底修复逻辑（不做任何语义改写） ===
    def _strip_fenced_code_block(s: str) -> str:
        s = (s or "").strip()
        if not s.startswith("```"):
            return s
        lines = s.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _sanitize_json_string_minimal(s: str) -> str:
        s = (s or "")
        s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        s = re.sub(r"\s{2,}", " ", s)
        return s
    
    # 必须把 _remove_one_offending_backslash 函数定义放回来，否则后面调用会报错
    def _remove_one_offending_backslash(s: str, err: Exception) -> str:
        if not isinstance(err, json.JSONDecodeError):
            return ""
        msg = str(err)
        if "Invalid \\escape" not in msg and "Invalid \\u" not in msg:
            return ""
        pos = getattr(err, "pos", None)
        if pos is None or pos <= 0 or pos > len(s):
            return ""
        candidates = []
        if pos < len(s) and s[pos] == "\\":
            candidates.append(pos)
        if pos - 1 >= 0 and s[pos - 1] == "\\":
            candidates.append(pos - 1)
        start = max(0, pos - 16)
        window = s[start:pos + 1]
        last_bs = window.rfind("\\")
        if last_bs != -1:
            candidates.append(start + last_bs)
        seen = set()
        candidates = [i for i in candidates if not (i in seen or seen.add(i))]
        for idx in candidates:
            if 0 <= idx < len(s) and s[idx] == "\\":
                return s[:idx] + s[idx + 1:]
        return ""

    # 解析 JSON（要求 GPT/Gemini 返回纯 JSON）
    try:
        # Gemini 的 response_mime_type 已经很大程度保证了 json，但保留打印方便调试
        # print("====== RAW GEMINI OUTPUT ======")
        # print(visual_dag_str)
        # print("====== END ======")
        visual_dag = json.loads(visual_dag_str)
    except Exception as e1:
        # 下面的修复逻辑保持原样
        try:
            unwrapped = _strip_fenced_code_block(visual_dag_str)
            fixed_str = _sanitize_json_string_minimal(unwrapped).strip()

            if not fixed_str:
                raise ValueError("Gemini returned empty/whitespace-only JSON content after repair.")

            try:
                visual_dag = json.loads(fixed_str)
            except Exception as e2:
                working = fixed_str
                last_err = e2
                max_backslash_removals = 50 
                removed_times = 0

                while removed_times < max_backslash_removals:
                    new_working = _remove_one_offending_backslash(working, last_err)
                    if not new_working:
                        break 
                    working = new_working
                    removed_times += 1
                    try:
                        visual_dag = json.loads(working)
                        fixed_str = working 
                        last_err = None
                        break
                    except Exception as e_next:
                        last_err = e_next
                        continue
                
                if last_err is not None:
                     raise ValueError(
                        "Gemini returned invalid JSON: " + str(e1) +
                        " | After repair still invalid: " + str(e2) +
                        f" | Tried removing offending backslashes up to {max_backslash_removals} times "
                        f"(actually removed {removed_times}) but still invalid: " + str(last_err)
                    )

        except Exception as e_final:
            if isinstance(e_final, ValueError):
                raise
            raise ValueError(
                "Gemini returned invalid JSON: " + str(e1) +
                " | After repair still invalid: " + str(e_final)
            )

    # === 4. 保存 visual_dag.json ===
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(visual_dag, f, indent=2, ensure_ascii=False)

    print(f"\n📂 visual_dag.json is generated successfully")
    return visual_dag


# ==========  计算每一个视觉结点的分辨率 ==========
def add_resolution_to_visual_dag(auto_path, visual_dag_path):
    """
    遍历 visual_dag.json，提取图片路径，计算分辨率并添加到结点属性中。
    
    Args:
        auto_path (str): 图片所在的根目录路径。
        visual_dag_path (str): visual_dag.json 文件的路径。
        
    Returns:
        list: 更新后的节点列表。
    """
    
    # 1. 读取 JSON 文件
    try:
        with open(visual_dag_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {visual_dag_path}")
        return []
    except json.JSONDecodeError:
        print(f"错误: 文件 {visual_dag_path} 不是有效的 JSON 格式")
        return []

    nodes = data.get("nodes", [])
    
    # 正则表达式用于匹配 ![](path) 中的 path
    # 解释: !\[\]\((.*?)\) 匹配 ![](...) 括号内的所有内容
    pattern = re.compile(r'!\[\]\((.*?)\)')

    for node in nodes:
        name_str = node.get("name", "")
        
        # 2. 从 name 中提取路径
        match = pattern.search(name_str)
        if match:
            # 获取括号内的路径部分，例如 images/xxx.jpg
            relative_image_path = match.group(1)
            
            # 3. 拼接完整路径
            full_image_path = os.path.join(auto_path, relative_image_path)
            
            # 4. 读取图片并计算分辨率
            try:
                # 使用 Pillow 打开图片
                with Image.open(full_image_path) as img:
                    width, height = img.size
                    resolution_str = f"{width}x{height}"
                    
                    # 5. 添加 resolution 字段
                    node["resolution"] = resolution_str
                    # print(f"成功处理: {relative_image_path} -> {resolution_str}")
                    
            except FileNotFoundError:
                print(f"警告: 找不到图片文件 {full_image_path}，跳过该节点。")
                node["resolution"] = "Unknown" # 或者可以选择不添加该字段
            except Exception as e:
                print(f"警告: 处理图片 {full_image_path} 时发生错误: {e}")
                node["resolution"] = "Error"
        else:
            print(f"警告: 节点 name 格式不匹配: {name_str}")

    # (可选) 将更新后的数据写回文件，或者另存为新文件
    # 这里演示将数据写回原文件
    try:
        with open(visual_dag_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"处理完成，已更新文件: {visual_dag_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

    return nodes


# ==========  调用 gemini-3-pro-preview 生成每一个section_dag ==========
def build_section_dags(
    folder_path: str,
    base_prompt: str,
    model: str = "gemini-3.0-pro-preview", # 建议使用 flash 或 pro
    config: dict = None
):
    """
    Traverse all markdown files in a folder, send each section to Gemini,
    and save <SectionName>_dag.json.
    Includes robust JSON repair and retry logic.
    """
    # -----------------------------
    # Tunables (safe defaults)
    # -----------------------------
    ENABLE_FALLBACK_CONTENT_BACKSLASH_STRIP = True
    FALLBACK_STRIP_BACKSLASH_ONLY_IN_CONTENT = True
    MAX_RETRIES_ON_FAIL = 2

    # === Init Client (Gemini) ===
    # 使用 config 中的 key 
    client = genai.Client(
        api_key=config['api_keys']['gemini_api_key']
    )

    base_prompt_effective = get_instruction_with_icl(
        "section_dag_generation_prompt", base_prompt, config
    )

    def build_full_prompt(instruction: str, section_name: str, md_text: str) -> str:
        return (
            f"{instruction}\n\n"
            "=== SECTION NAME ===\n"
            f"{section_name}\n\n"
            "=== SECTION MARKDOWN (FULL) ===\n"
            f"\"\"\"{md_text}\"\"\""
        )

    # === Helper Functions (Keep exactly as is) ===
    def remove_invisible_control_chars(s: str) -> str:
        if not s: return s
        s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
        s = re.sub(r"[\uFEFF\u200B\u200C\u200D\u2060\u00AD\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069]", "", s)
        return s

    def sanitize_json_literal_newlines(s: str) -> str:
        out = []
        in_string = False
        escape = False
        for ch in s:
            if in_string:
                if escape:
                    out.append(ch); escape = False
                else:
                    if ch == '\\': out.append(ch); escape = True
                    elif ch == '"': out.append(ch); in_string = False
                    elif ch in ('\n', '\r', '\t'): out.append(' ')
                    else: out.append(ch)
            else:
                if ch == '"': out.append(ch); in_string = True; escape = False
                else: out.append(ch)
        return ''.join(out)

    def sanitize_invalid_backslashes_in_strings(s: str) -> str:
        out = []
        in_string = False
        i = 0
        valid_esc = set(['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'])
        while i < len(s):
            ch = s[i]
            if not in_string:
                out.append(ch)
                if ch == '"': in_string = True
                i += 1; continue
            if ch == '"':
                out.append(ch); in_string = False
                i += 1; continue
            if ch != '\\':
                out.append(ch); i += 1; continue
            if i == len(s) - 1:
                out.append('\\\\'); i += 1; continue
            nxt = s[i + 1]
            if nxt in valid_esc:
                out.append('\\'); out.append(nxt); i += 2
            else:
                out.append('\\\\'); out.append(nxt); i += 2
        return ''.join(out)

    def force_content_single_line(dag_obj):
        if not isinstance(dag_obj, dict): return dag_obj
        nodes = dag_obj.get("nodes", None)
        if not isinstance(nodes, list): return dag_obj
        for node in nodes:
            if isinstance(node, dict) and "content" in node and isinstance(node["content"], str):
                node["content"] = re.sub(r"[\r\n]+", " ", node["content"])
        return dag_obj

    def fallback_strip_backslashes_in_content(dag_obj):
        if not isinstance(dag_obj, dict): return dag_obj
        nodes = dag_obj.get("nodes", None)
        if not isinstance(nodes, list): return dag_obj
        for node in nodes:
            if isinstance(node, dict) and "content" in node and isinstance(node["content"], str):
                node["content"] = node["content"].replace("\\", "")
        return dag_obj

    def extract_first_json_object_substring(s: str):
        start = s.find("{")
        if start < 0: return None
        in_string = False; escape = False; depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if in_string:
                if escape: escape = False
                elif ch == "\\": escape = True
                elif ch == '"': in_string = False
            else:
                if ch == '"': in_string = True
                elif ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0: return s[start:i + 1]
        return None

    def robust_load_json(raw: str):
        raw0 = remove_invisible_control_chars(raw)
        try: return json.loads(raw0), raw0, "A_raw"
        except json.JSONDecodeError: pass
        
        b = sanitize_json_literal_newlines(raw0)
        try: return json.loads(b), b, "B_newlines_fixed"
        except json.JSONDecodeError: pass
        
        c = sanitize_invalid_backslashes_in_strings(b)
        try: return json.loads(c), c, "C_backslashes_fixed"
        except json.JSONDecodeError: pass
        
        sub = extract_first_json_object_substring(raw0)
        if sub:
            d0 = remove_invisible_control_chars(sub)
            d1 = sanitize_json_literal_newlines(d0)
            d2 = sanitize_invalid_backslashes_in_strings(d1)
            try: return json.loads(d2), d2, "D_extracted_object_repaired"
            except json.JSONDecodeError: pass
            
        return None, raw0, "FAIL"

    # === Modified: Call Gemini ===
    def call_llm(full_prompt: str) -> str:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    # 可以在这里加 response_mime_type="application/json" 进一步增强稳定性
                )
            )
            return resp.text.strip()
        except Exception as e:
            print(f"❌ Gemini API Error: {e}")
            return ""

    def preprocess_llm_output(raw_content: str) -> str:
        raw_content = remove_invisible_control_chars(raw_content)
        fence_match = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)```", raw_content)
        if fence_match:
            raw_content = fence_match.group(1).strip()
            raw_content = remove_invisible_control_chars(raw_content)
        return raw_content

    outputs = {}

    # === Main Loop ===
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return outputs

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".md", ".markdown")):
            continue

        markdown_path = os.path.join(folder_path, filename)
        if not os.path.isfile(markdown_path):
            continue

        section_name = filename
        with open(markdown_path, "r", encoding="utf-8") as f:
            md_text = f.read().strip()

        full_prompt = build_full_prompt(base_prompt_effective, section_name, md_text)
        print(f"📐 Sending section '{section_name}' to Gemini for DAG generation...")

        dag_obj = None
        used_text = ""
        stage = "INIT"

        # Retry Loop
        for attempt_idx in range(1 + MAX_RETRIES_ON_FAIL):
            if attempt_idx > 0:
                print(f"🔁 Retry LLM for section '{section_name}' (retry={attempt_idx}/{MAX_RETRIES_ON_FAIL})...")

            raw_content = call_llm(full_prompt)
            if not raw_content: continue # 如果 API 调用报错返回空，直接重试

            raw_content = preprocess_llm_output(raw_content)
            dag_obj, used_text, stage = robust_load_json(raw_content)

            if dag_obj is not None:
                break
            
            print(f"⚠️ JSON parse failed for section '{section_name}' after repairs. Stage={stage}")

        if dag_obj is None:
            print(f"{section_name} 处理失败超过两次，已清除")
            dag_obj = {}
        else:
            dag_obj = force_content_single_line(dag_obj)
            if ENABLE_FALLBACK_CONTENT_BACKSLASH_STRIP and FALLBACK_STRIP_BACKSLASH_ONLY_IN_CONTENT:
                if stage in ("D_extracted_object_repaired",):
                    dag_obj = fallback_strip_backslashes_in_content(dag_obj)

        # Output
        safe_section_name = re.sub(r"[\\/:*?\"<>|]", "_", section_name)
        output_filename = f"{safe_section_name}_dag.json"
        subdir_path = os.path.dirname(folder_path)
        section_dag_path = os.path.join(subdir_path, "section_dag")
        os.makedirs(section_dag_path, exist_ok=True)
        output_path = os.path.join(section_dag_path, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dag_obj, f, ensure_ascii=False, indent=4)

        print(f"✅ DAG for section '{section_name}' saved to: {output_path} (parse_stage={stage})")
        outputs[section_name] = output_path

    return outputs


# ==========  合并 section_dag 到 dag ==========
def add_section_dag(
    section_dag_folder: str,
    main_dag_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Merge all section DAGs under `section_dag_folder` into the main DAG at `main_dag_path`.

    For each section DAG JSON:
      - Take its root node name (nodes[0]["name"]) and append that name
        to the edge list of the main DAG's root node (main_dag["nodes"][0]["edge"]).
      - Append ALL nodes from that section DAG to the end of main_dag["nodes"],
        preserving their original order.

    Compatibility patch:
      - If a section JSON is a single node object (missing the top-level "nodes" wrapper),
        automatically wrap it into:
            {"nodes": [<that_node_obj>]}
        so the downstream merge logic can proceed.

    Notes:
      - This function does NOT call GPT, it only manipulates JSON.
      - The main DAG is assumed to have the same format:
        {
            "nodes": [
                {
                    "name": "...",
                    "content": "...",
                    "edge": [],
                    "level": 0 or 1,
                    "visual_node": []
                },
                ...
            ]
        }

    Args:
        section_dag_folder: Path to a folder that contains per-section DAG JSON files.
        main_dag_path:      Path to the main DAG JSON file (original).
        output_path:        Path to save the merged DAG. If None, overwrite main_dag_path.

    Returns:
        The path of the merged DAG JSON file.
    """

    def _coerce_section_dag_to_nodes_wrapper(obj, section_path: str) -> dict:
        """
        If `obj` is already a valid {"nodes": [...]} dict, return as-is.
        If `obj` looks like a single node dict (has "name"/"content"/"edge"/etc. but no "nodes"),
        wrap it into {"nodes": [obj]}.
        Otherwise, raise ValueError.
        """
        # Case 1: already in expected format
        if isinstance(obj, dict) and "nodes" in obj:
            return obj

        # Case 2: single-node object (missing wrapper)
        if isinstance(obj, dict) and "nodes" not in obj:
            # Heuristic: if it has at least "name" and "content" (common node keys), treat it as node.
            has_name = isinstance(obj.get("name"), str) and obj.get("name").strip()
            has_content = isinstance(obj.get("content"), str)
            if has_name and has_content:
                # Wrap into nodes list
                return {"nodes": [obj]}

        raise ValueError(
            f"Section DAG JSON at '{section_path}' is neither a valid DAG wrapper "
            f"nor a recognizable single-node object."
        )

    # === Load main DAG ===
    with open(main_dag_path, "r", encoding="utf-8") as f:
        main_dag = json.load(f)

    if "nodes" not in main_dag or not isinstance(main_dag["nodes"], list) or len(main_dag["nodes"]) == 0:
        raise ValueError("main_dag JSON is invalid: missing non-empty 'nodes' array.")

    # Root node is assumed to be the first node
    root_node = main_dag["nodes"][0]

    # Ensure 'edge' field exists and is a list
    if "edge" not in root_node or not isinstance(root_node["edge"], list):
        root_node["edge"] = []

    # === Traverse section DAG folder ===
    # To keep deterministic order, sort filenames
    for filename in sorted(os.listdir(section_dag_folder)):
        # Only process *.json files
        if not filename.lower().endswith(".json"):
            continue

        section_path = os.path.join(section_dag_folder, filename)

        # Skip if it's the same file as main_dag_path, just in case
        if os.path.abspath(section_path) == os.path.abspath(main_dag_path):
            continue

        if not os.path.isfile(section_path):
            continue

        # Load section DAG
        with open(section_path, "r", encoding="utf-8") as f:
            try:
                section_raw = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Section DAG JSON invalid at '{section_path}': {e}")

        # NEW: coerce into {"nodes":[...]} if missing wrapper
        section_dag = _coerce_section_dag_to_nodes_wrapper(section_raw, section_path)

        # Validate nodes array
        if "nodes" not in section_dag or not isinstance(section_dag["nodes"], list) or len(section_dag["nodes"]) == 0:
            raise ValueError(f"Section DAG JSON at '{section_path}' has no valid 'nodes' array.")

        section_nodes = section_dag["nodes"]
        section_root = section_nodes[0]

        # Get section root name
        section_root_name = section_root.get("name")
        if not isinstance(section_root_name, str) or not section_root_name.strip():
            raise ValueError(f"Section DAG root node at '{section_path}' has invalid or empty 'name'.")

        # Append section root name into main root's edge
        # (avoid duplicates, in case of reruns)
        if section_root_name not in root_node["edge"]:
            root_node["edge"].append(section_root_name)

        # Append all section nodes to the end of main_dag["nodes"]
        main_dag["nodes"].extend(section_nodes)

    # === Save merged DAG ===
    if output_path is None:
        output_path = main_dag_path  # overwrite by default

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(main_dag, f, ensure_ascii=False, indent=4)

    return output_path


# ==========  向原dag中添加visual_dag ==========
def add_visual_dag(dag_path: str, visual_dag_path: str) -> str:
    """
    Append all nodes from a visual DAG JSON file into an existing DAG JSON file.

    Both JSON files must share the same structure, e.g.:

        {
            "nodes": [
                {
                    "name": "...",
                    "content": "...",
                    "edge": [],
                    "level": 0,
                    "visual_node": []
                }
            ]
        }

    Behavior:
        - Load the main DAG from `dag_path`.
        - Load the visual DAG from `visual_dag_path`.
        - Append ALL nodes from visual_dag["nodes"] to the end of main_dag["nodes"],
          preserving their original order.
        - Overwrite `dag_path` with the merged DAG.
        - Does NOT modify any edge relationships automatically.

    Args:
        dag_path:        Path to the main DAG JSON (will be overwritten).
        visual_dag_path: Path to the visual DAG JSON whose nodes will be appended.

    Returns:
        The `dag_path` of the merged DAG.
    """
    # === Load main DAG ===
    with open(dag_path, "r", encoding="utf-8") as f:
        main_dag = json.load(f)

    if "nodes" not in main_dag or not isinstance(main_dag["nodes"], list):
        raise ValueError(f"Main DAG at '{dag_path}' is invalid: missing 'nodes' array.")

    # === Load visual DAG ===
    with open(visual_dag_path, "r", encoding="utf-8") as f:
        visual_dag = json.load(f)

    if "nodes" not in visual_dag or not isinstance(visual_dag["nodes"], list):
        raise ValueError(f"Visual DAG at '{visual_dag_path}' is invalid: missing 'nodes' array.")

    # === Append visual nodes to main DAG (to the bottom) ===
    main_dag["nodes"].extend(visual_dag["nodes"])

    # === Save merged DAG back to dag_path (overwrite) ===
    with open(dag_path, "w", encoding="utf-8") as f:
        json.dump(main_dag, f, ensure_ascii=False, indent=4)

    return dag_path


# ==========  完善dag中每一个结点的visual_node ==========
from typing import List
def refine_visual_node(dag_path: str) -> None:
    """
    Refine `visual_node` for each node in the DAG JSON at `dag_path`.

    Behavior:
        - Load the DAG JSON from `dag_path`, whose structure is:
            {
                "nodes": [
                    {
                        "name": "...",
                        "content": "...",
                        "edge": [],
                        "level": 0,
                        "visual_node": []
                    },
                    ...
                ]
            }

        - For each node in `nodes`:
            * If node["visual_node"] == 1:
                - Treat this as a special marker meaning the node is already
                  a visual node; skip it and do NOT modify `visual_node`.
            * Else:
                - Look at node["content"] (if it's a string).
                - Find all markdown image references of the form:
                      ![alt text](path)
                  using a regex.
                - Filter to keep only relative paths (e.g., 'images/xxx.jpg'):
                      - path does NOT start with 'http://', 'https://', 'data:', or '//'.
                - For each such match, append the full markdown snippet
                  (e.g., '![something](images/xxx.jpg)') into node["visual_node"].
                - If `visual_node` is missing or not a list (and not equal to 1),
                  it will be overwritten as a list of these strings.

        - The function overwrites the original `dag_path` with the refined DAG.
    """
    # === Load DAG ===
    with open(dag_path, "r", encoding="utf-8") as f:
        dag = json.load(f)

    if "nodes" not in dag or not isinstance(dag["nodes"], list):
        raise ValueError(f"DAG JSON at '{dag_path}' is invalid: missing 'nodes' array.")

    nodes: List[dict] = dag["nodes"]

    # Regex to match markdown images: ![alt](path)
    # group(0) = full match, group(1) = alt, group(2) = path
    img_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def is_relative_path(path: str) -> bool:
        """Return True if path looks like a relative path, not URL or absolute."""
        lowered = path.strip().lower()
        if lowered.startswith("http://"):
            return False
        if lowered.startswith("https://"):
            return False
        if lowered.startswith("data:"):
            return False
        if lowered.startswith("//"):
            return False
        # You can optionally reject absolute filesystem paths too:
        # if lowered.startswith("/") or re.match(r"^[a-zA-Z]:[\\/]", lowered):
        #     return False
        return True

    for node in nodes:
        # Skip if this is not a dict
        if not isinstance(node, dict):
            continue

        # If visual_node == 1, this is a special visual node -> skip
        if node.get("visual_node") == 1:
            continue

        content = node.get("content")
        if not isinstance(content, str) or not content:
            # No textual content to search
            # But if visual_node should still be a list, ensure that
            if "visual_node" not in node or not isinstance(node["visual_node"], list):
                node["visual_node"] = []
            continue

        # Find all markdown image references
        matches = img_pattern.findall(content)  # returns list of (alt, path)
        full_matches = img_pattern.finditer(content)  # to get exact substrings

        # Ensure visual_node is a list (since we already filtered out ==1)
        visual_list = node.get("visual_node")
        if not isinstance(visual_list, list):
            visual_list = []
        else:
            # create a copy to safely modify
            visual_list = list(visual_list)

        # To keep consistent mapping, use the iterator to get full strings
        for match in full_matches:
            full_str = match.group(0)        # e.g., '![alt](images/xxx.jpg)'
            path_str = match.group(2).strip()  # inside parentheses

            if not is_relative_path(path_str):
                continue  # skip URLs / absolute paths

            if full_str not in visual_list:
                visual_list.append(full_str)

        # Update node
        node["visual_node"] = visual_list

    # === Save back to disk (overwrite) ===
    with open(dag_path, "w", encoding="utf-8") as f:
        json.dump(dag, f, ensure_ascii=False, indent=4)
