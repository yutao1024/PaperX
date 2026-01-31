from openai import OpenAI
import json
import os
import re
import base64
from PIL import Image

# ========== 1. 读取 prompt 模板 ==========
def load_prompt(prompt_path="prompt.json", prompt_name="poster_prompt"):
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(prompt_name, "")

# ==========  调用 gpt 删除无用段落 ==========
def clean_paper_with_gpt(markdown_path, clean_prompt, model_name="gpt-4o"):
    """
    使用 GPT 清理论文 Markdown 文件：
    删除 Abstract / Related Work / Appendix / References 等部分，
    保留标题、作者、Introduction、Methods、Experiments、Conclusion。
    只允许删除操作，不允许修改原文。
    """
    import os, re

    # === 初始化 client ===
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", None)
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

    print("🧹 Sending markdown to GPT for cleaning...")
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.0
    )
    cleaned_text = resp.choices[0].message.content.strip()

    # === 提取纯 markdown（防止 GPT 返回 ```markdown ``` 块） ===
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



# ==========  调用 gpt 划分段落 ==========
client = OpenAI()

SECTION_RE = re.compile(r'^#\s+\d+(\s|$)', re.MULTILINE)


def sanitize_filename(name: str) -> str:
    """移除非法字符：/ \ : * ? \" < > |"""
    unsafe = r'\/:*?"<>|'
    return "".join(c for c in name if c not in unsafe).strip()


def split_paper_with_gpt(
    cleaned_md_path: str,      # 输入: A/auto/clean_paper.md
    prompt: str,
    separator: str = "===SPLIT===",
    model: str = "gemini-3-pro-preview"
):
    """
    使用 GPT 拆分论文，并将所有拆分后的 markdown 保存在：

        <parent_of_auto>/section_split_output/

    如果输入为：
        A/auto/clean_paper.md

    则输出为：
        A/section_split_output/*.md
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

    # === 提取一级 section 信息供 GPT 参考 ===
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

    # === GPT 调用 ===
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0
    )

    output_text = response.choices[0].message.content

    # === 按分隔符拆分 ===
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

        filename = sanitize_filename(title) + ".md"
        filepath = os.path.join(output_dir, filename)

        # 写文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(chunk)

        saved_paths.append(filepath)

    print(f"✅ Paper is splitted successfully")
    return saved_paths





# ==========  调用 gpt 初始化dag.json ==========
def initialize_dag_with_gpt(markdown_path, initialize_dag_prompt, model="gemini-3-pro-preview"):
    """
    使用 GPT-4o 初始化论文 DAG。
    
    输入:
        markdown_path: markdown 文件路径
        initialize_dag_prompt: 适合 gpt-4o 的 prompt 字符串
        model: 默认使用 gpt-4o（支持多模态，但这里用于纯文本）
    
    输出:
        dag.json: 保存在 markdown 文件同目录
        返回 python 字典形式的 DAG
    """

    # --- load markdown ---
    if not os.path.exists(markdown_path):
        raise FileNotFoundError(f"Markdown not found: {markdown_path}")

    with open(markdown_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # --- GPT call ---
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert academic document parser and structural analyzer."},
            {"role": "user", "content": initialize_dag_prompt},
            {"role": "user", "content": md_text}
        ],
        temperature=0,
        response_format={"type": "json_object"} # <--- 强制输出合法的 JSON 结构
    )

    raw_output = response.choices[0].message.content.strip()

    # --- Extract JSON (remove possible markdown fences) ---
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
            dag_data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"GPT output is not valid JSON:\n{raw_output}")

    # --- Save dag.json ---
    out_dir = os.path.dirname(markdown_path)
    out_path = os.path.join(out_dir, "dag.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dag_data, f, indent=4, ensure_ascii=False)

    print(f"✅ DAG saved to: {out_path}")

    return dag_data


# ==========  调用 大模型 添加视觉结点 ==========
import os
import re
import json
from google import genai
from google.genai import types  # 用于配置类型

def extract_and_generate_visual_dag_gemini(
    markdown_path: str,
    prompt_for_gpt: str,
    output_json_path: str,
    model="gemini-3-pro-preview"  # 这里建议填具体的模型名，比如 gemini-1.5-pro 或 gemini-2.0-flash
):
    """
    输入:
        markdown_path: 原论文 markdown 文件路径
        prompt_for_gpt: 给 LLM 使用的 prompt
        output_json_path: 生成的 visual_dag.json 存放路径
        model: 默认使用 Gemini 模型

    输出:
        生成 visual_dag.json
        返回 Python dict
    """

    # === 1. 读取 markdown (逻辑不变) ===
    if not os.path.exists(markdown_path):
        raise FileNotFoundError(f"Markdown not found: {markdown_path}")

    with open(markdown_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # === 2. 正则提取所有图片相对引用 (逻辑不变) ===
    pattern = r"!\[[^\]]*\]\(([^)]+)\)"
    matches = re.findall(pattern, md_text)

    # 过滤为相对路径（不包含http）
    relative_imgs = [m for m in matches if not m.startswith("http")]
    
    # 生成标准格式 name 字段使用的写法 "![](xxx)"
    normalized_refs = [f"![]({m})" for m in relative_imgs]

    # === 3. 发送给 Gemini (适配 google-genai 新版 SDK) ===
    
    # 获取 API Key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 GOOGLE_API_KEY")

    # 实例化客户端 (这是新版 SDK 的写法，不再使用 configure)
    client = genai.Client(api_key=api_key)

    # 拼接输入内容
    gemini_input = prompt_for_gpt + "\n\n" + \
        "### Extracted Image References:\n" + \
        json.dumps(normalized_refs, indent=2) + "\n\n" + \
        "### Full Markdown:\n" + md_text

    try:
        # 调用模型 (使用 models.generate_content)
        response = client.models.generate_content(
            model=model,
            contents=gemini_input,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"  # 强制 JSON 输出
            )
        )
        
        # 获取文本
        visual_dag_str = response.text.strip()

    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}")

    # === 解析 JSON ===
    # 清洗可能存在的 markdown 代码块标记
    if visual_dag_str.startswith("```json"):
        visual_dag_str = visual_dag_str.replace("```json", "").replace("```", "").strip()
    elif visual_dag_str.startswith("```"):
        visual_dag_str = visual_dag_str.replace("```", "").strip()
    
    try:
        visual_dag = json.loads(visual_dag_str)
    except Exception as e:
        print("Raw response from Gemini:", visual_dag_str) 
        raise ValueError("Gemini returned invalid JSON: " + str(e))

    # === 4. 保存 visual_dag.json (逻辑不变) ===
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(visual_dag, f, indent=2, ensure_ascii=False)

    print(f"\n📂 visual_dag.json is generated successfully using {model}")
    return visual_dag

def extract_and_generate_visual_dag_gemini_xj(
    markdown_path: str,
    prompt_for_gpt: str,
    output_json_path: str,
    model="gemini-3-pro-preview"
):
    """
    输入:
        markdown_path: 原论文 markdown 文件路径
        prompt_for_gpt: 给 GPT 使用的 prompt（你下面会看到完整版本）
        output_json_path: 生成的 visual_dag.json 存放路径
        model: 默认 gpt-4o

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

    # === 3. 发送给 GPT ===
    client = OpenAI()

    gpt_input = prompt_for_gpt + "\n\n" + \
        "### Extracted Image References:\n" + \
        json.dumps(normalized_refs, indent=2) + "\n\n" + \
        "### Full Markdown:\n" + md_text

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": gpt_input}
        ]
    )

    visual_dag_str = response.choices[0].message.content.strip()

    # === 新增：JSON 解析兜底修复逻辑（不做任何语义改写） ===
    def _strip_fenced_code_block(s: str) -> str:
        """
        If response is wrapped in a fenced code block like:
            ```json
            {...}
            ```
        then extract the inside part.
        No semantic rewrite, only unwrap.
        """
        s = (s or "").strip()
        if not s.startswith("```"):
            return s

        lines = s.splitlines()
        # drop first fence line: ``` or ```json
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        # drop trailing fence lines
        while lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        return "\n".join(lines).strip()

    def _sanitize_json_string_minimal(s: str) -> str:
        """
        Minimal sanitizer:
        - Replace raw control characters \r \n \t with spaces
        - Collapse consecutive whitespace into a single space
        IMPORTANT: no semantic rewrite, keep name like '![](images/xxx.jpg)'.
        """
        s = (s or "")
        s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        s = re.sub(r"\s{2,}", " ", s)
        return s

    def _remove_one_offending_backslash(s: str, err: Exception) -> str:
        """
        If JSONDecodeError indicates an invalid escape caused by a backslash,
        locate the error position and remove exactly ONE backslash near that position.
        Return "" if not applicable.
        """
        if not isinstance(err, json.JSONDecodeError):
            return ""

        # 只针对 Invalid \escape 这种典型反斜杠错误做处理，避免误删
        msg = str(err)
        if "Invalid \\escape" not in msg and "Invalid \\u" not in msg:
            return ""

        pos = getattr(err, "pos", None)
        if pos is None or pos <= 0 or pos > len(s):
            return ""

        candidates = []

        # 常见情况：pos 指向反斜杠本身
        if pos < len(s) and s[pos] == "\\":
            candidates.append(pos)

        # 或 pos 指向反斜杠后的字符，那么 pos-1 是反斜杠
        if pos - 1 >= 0 and s[pos - 1] == "\\":
            candidates.append(pos - 1)

        # 再退一步：在 pos 左侧一个小窗口里找最近的 '\'
        start = max(0, pos - 16)
        window = s[start:pos + 1]
        last_bs = window.rfind("\\")
        if last_bs != -1:
            candidates.append(start + last_bs)

        # 去重保序
        seen = set()
        candidates = [i for i in candidates if not (i in seen or seen.add(i))]

        for idx in candidates:
            if 0 <= idx < len(s) and s[idx] == "\\":
                return s[:idx] + s[idx + 1:]

        return ""

    # 解析 JSON（要求 GPT 返回纯 JSON）
    try:
        print("====== RAW GPT OUTPUT ======")
        print(visual_dag_str)
        print("====== END ======")
        visual_dag = json.loads(visual_dag_str)
    except Exception as e1:
        try:
            # 1) unwrap fenced code block if present
            unwrapped = _strip_fenced_code_block(visual_dag_str)
            # 2) minimal sanitize (control chars + whitespace only)
            fixed_str = _sanitize_json_string_minimal(unwrapped).strip()

            # 若修复后为空，直接给出明确错误（不改变主逻辑，仅更明确）
            if not fixed_str:
                raise ValueError("GPT returned empty/whitespace-only JSON content after repair.")

            try:
                visual_dag = json.loads(fixed_str)
            except Exception as e2:
                # === 新增逻辑：如果仍不合法，且由反斜杠引起，则“迭代删除反斜杠”直到成功或达到上限 ===
                working = fixed_str
                last_err = e2
                max_backslash_removals = 50  # 上限：防止极端情况无限循环（只影响该新增分支）
                removed_times = 0

                while removed_times < max_backslash_removals:
                    # 每次根据当前错误定位并删除一个反斜杠
                    new_working = _remove_one_offending_backslash(working, last_err)
                    if not new_working:
                        break  # 说明当前错误不适用反斜杠删除策略，退出

                    working = new_working
                    removed_times += 1

                    try:
                        visual_dag = json.loads(working)
                        fixed_str = working  # 保持后续逻辑一致
                        last_err = None
                        break
                    except Exception as e_next:
                        last_err = e_next
                        continue

                if last_err is not None:
                    # 仍然失败：保持原来的报错风格，但把迭代删除的信息加进去
                    raise ValueError(
                        "GPT returned invalid JSON: " + str(e1) +
                        " | After repair still invalid: " + str(e2) +
                        f" | Tried removing offending backslashes up to {max_backslash_removals} times "
                        f"(actually removed {removed_times}) but still invalid: " + str(last_err)
                    )
        except Exception as e_final:
            # 保持原有异常外观（只是在必要时更详细）
            if isinstance(e_final, ValueError):
                raise
            raise ValueError(
                "GPT returned invalid JSON: " + str(e1) +
                " | After repair still invalid: " + str(e_final)
            )

    # === 4. 保存 visual_dag.json ===
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(visual_dag, f, indent=2, ensure_ascii=False)

    print(f"\n📂 visual_dag.json is generated successfully")
    return visual_dag


# ==========  计算每一个视觉结点的分辨率 ==========
import json
import os
import re
from PIL import Image

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
def build_section_dags_with_gemini_xj(
    folder_path: str,
    base_prompt: str,
    model_name: str = "gemini-3-pro-preview"
):
    """
    Traverse all markdown files in a folder (each file is one paper section),
    send each section to LLM with a DAG-generation prompt, and save
    <SectionName>_dag.json for each section.

    Protections included:
    1) Extract fenced JSON if present (```json ... ```).
    2) Remove invisible control characters from LLM output (both ASCII controls and common Unicode invisibles).
    3) Robust JSON repair before json.loads:
       - Fix literal newline/tab characters inside JSON string literals.
       - Fix invalid JSON escape sequences (e.g., "\alpha") by converting them
         to literal backslashes ("\\alpha") inside JSON strings.
       - Fix lone trailing backslash inside JSON string by escaping it.
    4) If still failing, optional "fallback" parsing attempts:
       - Parse only the first balanced JSON object substring.
       - As last resort, write raw output for manual inspection.
    5) After parsing, enforce node["content"] single-line and (fallback-only) optionally
       remove backslashes from node["content"] if you decide to sacrifice fidelity.

    NEW:
    - If JSON parsing ultimately FAILs, re-call LLM up to 2 retries.
      If still FAIL after 2 retries, clear JSON to {} and print:
      "<section_name> 处理失败超过两次，已清除"
    """
    import os
    import re
    import json
    from openai import OpenAI

    # -----------------------------
    # Tunables (safe defaults)
    # -----------------------------
    ENABLE_FALLBACK_CONTENT_BACKSLASH_STRIP = True
    FALLBACK_STRIP_BACKSLASH_ONLY_IN_CONTENT = True

    # NEW: retry config
    MAX_RETRIES_ON_FAIL = 2  # "最多两次"（指失败后的额外重试次数）

    # === Init client ===
    base_url = os.getenv("OPENAI_BASE_URL", None)
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=base_url if base_url else None
    )

    def build_full_prompt(base_prompt: str, section_name: str, md_text: str) -> str:
        return (
            f"{base_prompt}\n\n"
            "=== SECTION NAME ===\n"
            f"{section_name}\n\n"
            "=== SECTION MARKDOWN (FULL) ===\n"
            f"\"\"\"{md_text}\"\"\""
        )

    def remove_invisible_control_chars(s: str) -> str:
        if not s:
            return s
        s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", s)
        s = re.sub(
            r"[\uFEFF\u200B\u200C\u200D\u2060\u00AD\u061C\u200E\u200F\u202A-\u202E\u2066-\u2069]",
            "",
            s
        )
        return s

    def sanitize_json_literal_newlines(s: str) -> str:
        out = []
        in_string = False
        escape = False

        for ch in s:
            if in_string:
                if escape:
                    out.append(ch)
                    escape = False
                else:
                    if ch == '\\':
                        out.append(ch)
                        escape = True
                    elif ch == '"':
                        out.append(ch)
                        in_string = False
                    elif ch == '\n' or ch == '\r' or ch == '\t':
                        out.append(' ')
                    else:
                        out.append(ch)
            else:
                if ch == '"':
                    out.append(ch)
                    in_string = True
                    escape = False
                else:
                    out.append(ch)

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
                if ch == '"':
                    in_string = True
                i += 1
                continue

            if ch == '"':
                out.append(ch)
                in_string = False
                i += 1
                continue

            if ch != '\\':
                out.append(ch)
                i += 1
                continue

            if i == len(s) - 1:
                out.append('\\')
                out.append('\\')
                i += 1
                continue

            nxt = s[i + 1]
            if nxt in valid_esc:
                out.append('\\')
                out.append(nxt)
                i += 2
            else:
                out.append('\\')
                out.append('\\')
                out.append(nxt)
                i += 2

        return ''.join(out)

    def force_content_single_line(dag_obj):
        if not isinstance(dag_obj, dict):
            return dag_obj
        nodes = dag_obj.get("nodes", None)
        if not isinstance(nodes, list):
            return dag_obj

        for node in nodes:
            if not isinstance(node, dict):
                continue
            if "content" in node and isinstance(node["content"], str):
                node["content"] = re.sub(r"[\r\n]+", " ", node["content"])
        return dag_obj

    def fallback_strip_backslashes_in_content(dag_obj):
        if not isinstance(dag_obj, dict):
            return dag_obj
        nodes = dag_obj.get("nodes", None)
        if not isinstance(nodes, list):
            return dag_obj

        for node in nodes:
            if not isinstance(node, dict):
                continue
            if "content" in node and isinstance(node["content"], str):
                node["content"] = node["content"].replace("\\", "")
        return dag_obj

    def extract_first_json_object_substring(s: str):
        start = s.find("{")
        if start < 0:
            return None

        in_string = False
        escape = False
        depth = 0

        for i in range(start, len(s)):
            ch = s[i]
            if in_string:
                if escape:
                    escape = False
                else:
                    if ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return s[start:i + 1]
        return None

    def robust_load_json(raw: str):
        raw0 = remove_invisible_control_chars(raw)

        try:
            return json.loads(raw0), raw0, "A_raw"
        except json.JSONDecodeError:
            pass

        b = sanitize_json_literal_newlines(raw0)
        try:
            return json.loads(b), b, "B_newlines_fixed"
        except json.JSONDecodeError:
            pass

        c = sanitize_invalid_backslashes_in_strings(b)
        try:
            return json.loads(c), c, "C_backslashes_fixed"
        except json.JSONDecodeError:
            pass

        sub = extract_first_json_object_substring(raw0)
        if sub:
            d0 = remove_invisible_control_chars(sub)
            d1 = sanitize_json_literal_newlines(d0)
            d2 = sanitize_invalid_backslashes_in_strings(d1)
            try:
                return json.loads(d2), d2, "D_extracted_object_repaired"
            except json.JSONDecodeError:
                pass

        return None, raw0, "FAIL"

    def call_llm(full_prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.2
        )
        raw_content = (resp.choices[0].message.content or "").strip()
        return raw_content

    def preprocess_llm_output(raw_content: str) -> str:
        raw_content = remove_invisible_control_chars(raw_content)

        fence_match = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)```", raw_content)
        if fence_match:
            raw_content = fence_match.group(1).strip()
            raw_content = remove_invisible_control_chars(raw_content)

        return raw_content

    outputs = {}

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".md", ".markdown")):
            continue

        markdown_path = os.path.join(folder_path, filename)
        if not os.path.isfile(markdown_path):
            continue

        section_name = filename

        with open(markdown_path, "r", encoding="utf-8") as f:
            md_text = f.read().strip()

        full_prompt = build_full_prompt(base_prompt, section_name, md_text)

        print(f"📐 Sending section '{section_name}' to GPT for DAG generation...")

        dag_obj = None
        used_text = ""
        stage = "INIT"

        # 初次调用 + 失败后最多重试 2 次（总尝试次数 = 1 + 2 = 3）
        for attempt_idx in range(1 + MAX_RETRIES_ON_FAIL):
            if attempt_idx > 0:
                print(f"🔁 Retry LLM for section '{section_name}' (retry={attempt_idx}/{MAX_RETRIES_ON_FAIL})...")

            raw_content = call_llm(full_prompt)
            raw_content = preprocess_llm_output(raw_content)

            dag_obj, used_text, stage = robust_load_json(raw_content)

            if dag_obj is not None:
                # 解析成功则退出重试循环
                break

            # 本次尝试失败：打印一次失败信息
            print(f"⚠️ JSON parse failed for section '{section_name}' after repairs. Stage={stage}")

        # 尝试结束后，如果仍失败：清空并提示
        if dag_obj is None:
            print(f"{section_name} 处理失败超过两次，已清除")
            dag_obj = {}  # 清空 JSON，保证后续读取不再报 JSON 格式错

        else:
            # Enforce content single-line
            dag_obj = force_content_single_line(dag_obj)

            # Optional last-resort: only apply if heavy fallback
            if ENABLE_FALLBACK_CONTENT_BACKSLASH_STRIP and FALLBACK_STRIP_BACKSLASH_ONLY_IN_CONTENT:
                if stage in ("D_extracted_object_repaired",):
                    dag_obj = fallback_strip_backslashes_in_content(dag_obj)

        # Output file
        safe_section_name = re.sub(r"[\\/:*?\"<>|]", "_", section_name)
        output_filename = f"{safe_section_name}_dag.json"

        subdir_path = os.path.dirname(folder_path)
        section_dag_path = os.path.join(subdir_path, "section_dag")
        os.makedirs(section_dag_path, exist_ok=True)
        output_path = os.path.join(section_dag_path, output_filename)

        # 始终写合法 JSON（成功则写 dag_obj；失败则写 {}）
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dag_obj, f, ensure_ascii=False, indent=4)

        print(f"✅ DAG for section '{section_name}' saved to: {output_path} (parse_stage={stage})")
        outputs[section_name] = output_path

    return outputs




# ==========  合并 section_dag 到 dag ==========
import os
import json
from typing import Optional
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



# # ==========  生成selected_nodes.json ==========
# import json


# def generate_selected_nodes(dag_json_path, max_len, output_path='selected_node.json'):

#     # 1. 读取 dag.json
#     with open(dag_json_path, 'r', encoding='utf-8') as f:
#         dag_data = json.load(f)
    
#     all_nodes = dag_data.get('nodes', [])
    
#     # 2. 构建辅助字典，方便通过 name 快速查找节点信息
#     # 同时区分普通节点和视觉节点
#     node_map = {node['name']: node for node in all_nodes}
    
#     # 3. 初始化队列
#     # 找到根节点 (level=0)
#     root_node = next((node for node in all_nodes if node.get('level') == 0), None)
    
#     if not root_node:
#         raise ValueError("Root node (level 0) not found in dag.json")
    
#     # 获取根节点的子节点 (Sections) 作为初始队列
#     # 注意：这里队列存储的是节点的 name
#     current_queue = list(root_node.get('edge', []))
    
#     # 初始化计数器
#     node_num = len(current_queue)
#     level_num = 1
    
#     # 4. 循环处理队列，直到 level_num 达到 5
#     while level_num < 5:
#         i = 0
#         while i < len(current_queue):
#             node_name = current_queue[i]
#             node_info = node_map.get(node_name)
            
#             if not node_info:
#                 # 异常情况：队列里的节点在map里找不到
#                 i += 1
#                 continue
                
#             # 这里的 level 属性可能缺失，默认给个非当前level的值
#             current_node_level = node_info.get('level', -1)
            
#             # 判断这个结点的level是否等于level_num
#             if current_node_level != level_num:
#                 i += 1
#                 continue
            
#             # 获取子节点
#             children_names = node_info.get('edge', [])
#             num_children = len(children_names)
            
#             if num_children == 0:
#                 # 没有子节点，无法展开
#                 i += 1
#                 continue
            
#             potential_total_num = len(current_queue) + num_children 
#             if len(current_queue) + num_children <= max_len:
#                 # 执行展开操作
#                 current_queue[i:i+1] = children_names
#             else:
#                 # 大于 max_num，不展开，处理下一个
#                 i += 1
        
#         # 当处理完当前队列的最后一个结点时，level+1
#         level_num += 1
        
#     # 5. 生成最终结果
#     final_nodes_list = []
    
#     for node_name in current_queue:
#         original_node = node_map.get(node_name)
#         if not original_node:
#             continue
            
#         # 深拷贝以避免修改原始数据（也可以直接构建新字典）
#         # 这里为了安全起见构建新字典
#         export_node = original_node.copy()
        

        
#         original_visual_list = export_node.get('visual_node', [])
        
#         # 某些节点可能 visual_node 字段是空的或者不存在
#         if original_visual_list:
#             expanded_visual_nodes = []
            
#             # 确保它是列表，有些脏数据可能不是列表
#             if isinstance(original_visual_list, list):
#                 for v_name in original_visual_list:
#                     # 根据 name 查找视觉节点详细信息
#                     v_node_full = node_map.get(v_name)
#                     if v_node_full:
#                         expanded_visual_nodes.append(v_node_full)
#                     else:
#                         # 如果找不到，保留原名或者忽略，这里选择保留原结构提醒缺失
#                         expanded_visual_nodes.append({"name": v_name, "error": "Node not found"})
            
#             # 替换原有属性
#             export_node['visual_node'] = expanded_visual_nodes
            
#         final_nodes_list.append(export_node)
        
#     # 6. 写入文件
#     output_data = {"selected_nodes": final_nodes_list}
    

    
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(final_nodes_list, f, ensure_ascii=False, indent=4)
        
#     print(f"Successfully generated {output_path} with {len(final_nodes_list)} nodes.")


# ==========  生成selected_nodes.json ==========
import json


def generate_selected_nodes(dag_json_path, max_len, output_path='selected_node.json'):

    # 1. 读取 dag.json
    with open(dag_json_path, 'r', encoding='utf-8') as f:
        dag_data = json.load(f)

    all_nodes = dag_data.get('nodes', [])

    # 2. 构建辅助字典，方便通过 name 快速查找节点信息
    # 同时区分普通节点和视觉节点
    node_map = {node['name']: node for node in all_nodes}

    # 3. 初始化队列
    # 找到根节点 (level=0)
    root_node = next((node for node in all_nodes if node.get('level') == 0), None)

    if not root_node:
        raise ValueError("Root node (level 0) not found in dag.json")

    # 获取根节点的子节点 (Sections) 作为初始队列
    # 注意：这里队列存储的是节点的 name
    current_queue = list(root_node.get('edge', []))

    # 初始化计数器
    node_num = len(current_queue)
    level_num = 1

    # 4. 循环处理队列，直到 level_num 达到 5
    while level_num < 5:
        i = 0
        while i < len(current_queue):
            node_name = current_queue[i]
            node_info = node_map.get(node_name)

            if not node_info:
                # 异常情况：队列里的节点在map里找不到
                i += 1
                continue

            # ===== 新增逻辑：如果结点 name 含有 "introduction"/"INTRODUCTION"，则跳过该结点 =====
            # 注意：不修改其他逻辑，仅在处理该结点时直接跳过
            if "introduction" in node_name.lower():
                i += 1
                continue

            # 这里的 level 属性可能缺失，默认给个非当前level的值
            current_node_level = node_info.get('level', -1)

            # 判断这个结点的level是否等于level_num
            if current_node_level != level_num:
                i += 1
                continue

            # 获取子节点
            children_names = node_info.get('edge', [])
            num_children = len(children_names)

            if num_children == 0:
                # 没有子节点，无法展开
                i += 1
                continue

            potential_total_num = len(current_queue) + num_children
            if len(current_queue) + num_children <= max_len:
                # 执行展开操作
                current_queue[i:i+1] = children_names
            else:
                # 大于 max_num，不展开，处理下一个
                i += 1

        # 当处理完当前队列的最后一个结点时，level+1
        level_num += 1

    # 5. 生成最终结果
    final_nodes_list = []

    for node_name in current_queue:
        original_node = node_map.get(node_name)
        if not original_node:
            continue

        # 深拷贝以避免修改原始数据（也可以直接构建新字典）
        # 这里为了安全起见构建新字典
        export_node = original_node.copy()

        original_visual_list = export_node.get('visual_node', [])

        # 某些节点可能 visual_node 字段是空的或者不存在
        if original_visual_list:
            expanded_visual_nodes = []

            # 确保它是列表，有些脏数据可能不是列表
            if isinstance(original_visual_list, list):
                for v_name in original_visual_list:
                    # 根据 name 查找视觉节点详细信息
                    v_node_full = node_map.get(v_name)
                    if v_node_full:
                        expanded_visual_nodes.append(v_node_full)
                    else:
                        # 如果找不到，保留原名或者忽略，这里选择保留原结构提醒缺失
                        expanded_visual_nodes.append({"name": v_name, "error": "Node not found"})

            # 替换原有属性
            export_node['visual_node'] = expanded_visual_nodes

        final_nodes_list.append(export_node)

    # 6. 写入文件
    output_data = {"selected_nodes": final_nodes_list}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_nodes_list, f, ensure_ascii=False, indent=4)

    print(f"Successfully generated {output_path} with {len(final_nodes_list)} nodes.")




# ==========  初始化outline ==========
def outline_initialize_with_gpt(
    dag_json_path,
    outline_initialize_prompt,
    model="gpt-4o"
):
    """
    使用 GPT 初始化 outline.json（仅创建两个节点：Title + Contents）

    输入:
        dag_json_path: dag.json 文件路径
        outline_initialize_prompt: 传给 GPT 的 prompt（字符串）
        model: 默认使用 gpt-4o

    输出:
        outline.json: 保存在 dag.json 同目录
        返回 python list（outline 结构）
    """

    # --- load dag.json ---
    if not os.path.exists(dag_json_path):
        raise FileNotFoundError(f"dag.json not found: {dag_json_path}")

    with open(dag_json_path, "r", encoding="utf-8") as f:
        dag_data = json.load(f)

    # --- extract first node ---
    if isinstance(dag_data, list):
        first_node = dag_data[0]
    elif isinstance(dag_data, dict) and "nodes" in dag_data:
        first_node = dag_data["nodes"][0]
    else:
        raise ValueError("Unsupported dag.json format")

    first_node_text = json.dumps(first_node, ensure_ascii=False, indent=2)

    # --- GPT call ---
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are an expert academic presentation outline generator."
            },
            {
                "role": "user",
                "content": outline_initialize_prompt
            },
            {
                "role": "user",
                "content": first_node_text
            }
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()

    # --- Extract JSON (remove possible markdown fences) ---
    cleaned = raw_output

    # Remove ```json ... ```
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.split("\n", 1)[1]

    # Last safety: locate JSON via first [ and last ]
    try:
        first = cleaned.index("[")
        last = cleaned.rindex("]")
        cleaned = cleaned[first:last + 1]
    except Exception:
        pass

    try:
        outline_data = json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError(f"GPT output is not valid JSON:\n{raw_output}")

    # --- Save outline.json ---
    out_dir = os.path.dirname(dag_json_path)
    out_path = os.path.join(out_dir, "outline.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(outline_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Outline saved to: {out_path}")

    return outline_data


# ==========  调用 gpt 生成完整 outline ==========
def generate_complete_outline(
    selected_node_path,
    outline_path,
    generate_complete_outline_prompt,
    model="gpt-4o"
):
    """
    逐个 selected_node 调用 GPT，生成 outline 节点并追加到 outline.json

    输入:
        selected_node_path: selected_node.json 路径
        outline_path: outline.json 路径
        generate_complete_outline_prompt: 给 GPT 的 prompt（字符串）
        model: 默认 gpt-4o

    输出:
        更新后的 outline.json
        返回 outline（list）
    """

    # --- load selected_node.json ---
    if not os.path.exists(selected_node_path):
        raise FileNotFoundError(f"selected_node.json not found: {selected_node_path}")

    with open(selected_node_path, "r", encoding="utf-8") as f:
        selected_nodes = json.load(f)

    if not isinstance(selected_nodes, list):
        raise ValueError("selected_node.json must be a list")

    # --- load outline.json ---
    if not os.path.exists(outline_path):
        raise FileNotFoundError(f"outline.json not found: {outline_path}")

    with open(outline_path, "r", encoding="utf-8") as f:
        outline_data = json.load(f)

    if not isinstance(outline_data, list):
        raise ValueError("outline.json must be a list")

    client = OpenAI()

    # --- iterate selected nodes ---
    for idx, node in enumerate(selected_nodes):

        payload = {
            "name": node.get("name"),
            "content": node.get("content"),
            "visual_node": node.get("visual_node", [])
        }

        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert academic presentation outline generator."
                },
                {
                    "role": "user",
                    "content": generate_complete_outline_prompt
                },
                {
                    "role": "user",
                    "content": payload_text
                }
            ],
            temperature=0
        )

        raw_output = response.choices[0].message.content.strip()

        # --- clean JSON ---
        cleaned = raw_output

        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lstrip().startswith("json"):
                cleaned = cleaned.split("\n", 1)[1]

        try:
            first = cleaned.index("{")
            last = cleaned.rindex("}")
            cleaned = cleaned[first:last + 1]
        except Exception:
            pass

        try:
            outline_node = json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(
                f"GPT output is not valid JSON for selected_node index {idx}:\n{raw_output}"
            )

        # --- append to outline ---
        outline_data.append(outline_node)

    # --- save outline.json ---
    with open(outline_path, "w", encoding="utf-8") as f:
        json.dump(outline_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Complete outline updated: {outline_path}")

    return outline_data





# ==========  调用 gpt 为每一张ppt配模板 ==========
from typing import Any, Dict, List, Union
client = OpenAI()
SlideType = Dict[str, Any]
OutlineType = List[SlideType]
JsonType = Union[Dict[str, Any], List[Any], str, int, float, bool, None]
def arrange_template_with_gpt(
    outline_path: str,
    arrange_template_prompt: str,
    model: str = "gpt-4o-mini",
) -> OutlineType:
    """
    Read an outline.json (which is a list of slide nodes), iterate over each slide,
    and for every slide whose 'template' field is NULL/None, call GPT to choose a PPT template.

    Expected outline.json format (top-level is a list):
    [
      {
        "text": "...",
        "figure": [
          {
            "name": "...",
            "caption": "...",
            "resolution": "widthxheight"   # e.g. "602x350"
          },
          ...
        ],
        "formula": [
          {
            "latex": "...",                # optional, name can vary
            "resolution": "widthxheight"
          },
          ...
        ],
        "template": null                   # or a template filename string
      },
      ...
    ]

    Parameters
    ----------
    outline_path : str
        Path to outline.json.
    arrange_template_prompt : str
        System-level prompt describing how to choose a template
        (in English, and including the template list).
    model : str, optional
        Model name, by default "gpt-4o-mini".

    Returns
    -------
    OutlineType
        The modified outline (list of slide nodes) with 'template' filled in.
    """

    # 读取 outline.json
    with open(outline_path, "r", encoding="utf-8") as f:
        outline: OutlineType = json.load(f)

    def is_null_template(value: Any) -> bool:
        """
        Treat Python None or explicit string 'NULL' / 'null' / ''
        as empty template that needs to be filled.
        """
        if value is None:
            return True
        if isinstance(value, str) and value.strip().lower() in {"null", ""}:
            return True
        return False

    def select_template_for_slide(slide: SlideType, index: int) -> None:
        """
        If slide['template'] is NULL/None, call GPT to select a template,
        then write back the chosen template into slide['template'].
        """
        if not is_null_template(slide.get("template")):
            return  # already has a template, skip

        # 整个 slide 作为 JSON 发给 GPT（包含 text / figure / formula / template 等）
        slide_json_str = json.dumps(slide, ensure_ascii=False, indent=2)

        # 可选：在 user 内容中附上一些简单统计信息，帮 GPT 更好理解
        figures = slide.get("figure", []) or []
        formulas = slide.get("formula", []) or []

        summary_info = {
            "slide_index": index,
            "num_figures": len(figures),
            "num_formulas": len(formulas),
        }
        summary_json_str = json.dumps(summary_info, ensure_ascii=False, indent=2)

        messages = [
            {
                "role": "system",
                "content": arrange_template_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Below is one slide node from outline.json.\n"
                    "First, read the raw slide JSON.\n"
                    "Then, use the template selection rules in the system message to choose "
                    "exactly one template for this slide.\n\n"
                    "A small auto-generated summary is also provided to help you:\n"
                    f"Summary:\n```json\n{summary_json_str}\n```\n\n"
                    "Full slide node (JSON):\n```json\n"
                    + slide_json_str
                    + "\n```"
                ),
            },
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
            )
        except Exception as e:
            print(f"[WARN] Failed to call OpenAI for slide {index}: {e}")
            return

        content = (response.choices[0].message.content or "").strip()

        # 期望 GPT 返回 JSON：{"template": "T2_ImageRight.html"}
        template_name: Union[str, None] = None

        # 1) 尝试直接解析为 JSON
        try:
            # 去掉可能的代码块包装 ```json ... ```
            if "```" in content:
                parts = content.split("```")
                candidate = parts[1]  # e.g. 'json\n{...}' 或 '{...}'
                candidate = candidate.split("\n", 1)[-1]
                content_for_json = candidate
            else:
                content_for_json = content

            parsed = json.loads(content_for_json)

            if isinstance(parsed, dict) and "template" in parsed:
                template_name = parsed["template"]
            elif isinstance(parsed, str):
                template_name = parsed
        except Exception:
            # 2) 如果 JSON 解析失败，当作纯文本处理
            cleaned = content.strip()
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1].strip()
            template_name = cleaned or None

        if isinstance(template_name, str) and template_name:
            slide["template"] = template_name
        else:
            print(
                f"[WARN] Could not parse template from model output for slide {index}, "
                "leaving 'template' unchanged."
            )

    # 顶层是一个列表，每个元素是一张 slide
    if not isinstance(outline, list):
        raise ValueError("outline.json must be a list of slide nodes at top level.")

    for idx, slide in enumerate(outline):
        if isinstance(slide, dict):
            select_template_for_slide(slide, idx)

    # 写回文件
    with open(outline_path, "w", encoding="utf-8") as f:
        json.dump(outline, f, ensure_ascii=False, indent=2)

    return outline







# ========== 生成最终的PPT ==========
import os
import json
import re
from typing import Any, Dict, List, Optional, Union
from google import genai
_MD_IMAGE_RE = re.compile(r"!\[\s*.*?\s*\]\(\s*([^)]+?)\s*\)")
def _extract_md_image_path(name_field: str) -> str:
    """
    Extracts relative image path from a markdown image string like:
      '![](images/abc.jpg)' -> 'images/abc.jpg'
    If not markdown format, returns the original string stripped.
    """
    if not isinstance(name_field, str):
        return ""
    s = name_field.strip()
    m = _MD_IMAGE_RE.search(s)
    if m:
        return m.group(1).strip()
    return s


def _normalize_node(node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize node fields and extract clean image paths for figure/formula name fields.
    """
    text = node.get("text", "")
    template = node.get("template", "")
    figure = node.get("figure", []) or []
    formula = node.get("formula", []) or []

    def norm_imgs(imgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for it in imgs:
            if not isinstance(it, dict):
                continue
            name = it.get("name", "")
            out.append({
                "name": name,
                "path": _extract_md_image_path(name),
                "caption": it.get("caption", ""),
                "resolution": it.get("resolution", "")
            })
        return out

    return {
        "text": text if isinstance(text, str) else str(text),
        "template": template if isinstance(template, str) else str(template),
        "figure": norm_imgs(figure if isinstance(figure, list) else []),
        "formula": norm_imgs(formula if isinstance(formula, list) else [])
    }


def generate_ppt_with_gemini(
    outline_path: str,
    ppt_template_path: str,
    generate_ppt_with_gemini_prompt: Union[Dict[str, str], List[Dict[str, str]]],
    api_key: Optional[str] = None,
    model: str = "gemini-3-pro-preview",
) -> List[str]:
    """
    Traverse outline JSON nodes, load corresponding HTML templates, send (prompt + node + template)
    to Gemini, then save revised HTML to the outline.json directory as:
        <PPT编号>_ppt.html

    Args:
        outline_path: path to outline json file.
        ppt_template_path: folder containing html templates.
        generate_ppt_with_gemini_prompt: JSON-like prompt (dict or list of messages).
        api_key: optional Google API Key, otherwise reads env GOOGLE_API_KEY.
        model: gemini model name, default 'gemini-3-pro-preview'.

    Returns:
        List of saved HTML file paths (one per node).
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("请提供 Google API Key，或者设置环境变量 GOOGLE_API_KEY")

    client = genai.Client(api_key=api_key)

    outline_path = os.path.abspath(outline_path)
    ppt_template_path = os.path.abspath(ppt_template_path)

    if not os.path.isfile(outline_path):
        raise FileNotFoundError(f"outline_path 不存在或不是文件: {outline_path}")
    if not os.path.isdir(ppt_template_path):
        raise NotADirectoryError(f"ppt_template_path 不存在或不是文件夹: {ppt_template_path}")

    with open(outline_path, "r", encoding="utf-8") as f:
        outline = json.load(f)

    if not isinstance(outline, list):
        raise ValueError("outline_path 的 JSON 顶层必须是 list（每个元素代表一页 PPT 结点）")

    out_dir = os.path.dirname(outline_path)
    saved_files: List[str] = []

    # Allow prompt to be either a single message dict or a list of messages.
    if isinstance(generate_ppt_with_gemini_prompt, dict):
        base_messages = [generate_ppt_with_gemini_prompt]
    elif isinstance(generate_ppt_with_gemini_prompt, list):
        base_messages = generate_ppt_with_gemini_prompt
    else:
        raise TypeError("generate_ppt_with_gemini_prompt 必须是 dict 或 list[dict] 的 JSON 形式")

    for idx, node in enumerate(outline, start=1):
        if not isinstance(node, dict):
            # Skip invalid node but keep position consistent
            continue

        norm_node = _normalize_node(node)
        template_file = norm_node.get("template", "").strip()
        if not template_file:
            # No template: skip
            continue

        template_full_path = os.path.join(ppt_template_path, template_file)
        if not os.path.isfile(template_full_path):
            raise FileNotFoundError(f"找不到模板文件: {template_full_path}")

        with open(template_full_path, "r", encoding="utf-8") as tf:
            template_html = tf.read()

        user_payload = {
            "ppt_index": idx,
            "node": norm_node,
            "template_html": template_html,
        }

        messages = list(base_messages) + [
            {
                "role": "user",
                "content": (
                    "Here is the slide node JSON and the HTML template. "
                    "Revise the HTML per instructions and return ONLY the final HTML.\n\n"
                    f"{json.dumps(user_payload, ensure_ascii=False)}"
                ),
            }
        ]

        # ---- FIX: google-genai expects contents as a string (or specific SDK types), not OpenAI-style messages list ----
        # Convert the JSON-like prompt/messages into a single string prompt.
        final_prompt_parts: List[str] = []
        for m in messages:
            if isinstance(m, dict):
                c = m.get("content", "")
                if c:
                    final_prompt_parts.append(str(c))
            elif isinstance(m, str):
                final_prompt_parts.append(m)
        final_prompt = "\n\n".join(final_prompt_parts)
        # -----------------------------------------------------------------------------------------------

        # Gemini call
        resp = client.models.generate_content(
            model=model,
            contents=final_prompt
        )

        # Try to robustly extract text
        revised_html = getattr(resp, "text", None)
        if revised_html is None:
            # Fallback: attempt stringify
            revised_html = str(resp)

        revised_html = revised_html.strip()

        # Save
        out_name = f"{idx}_ppt.html"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as wf:
            wf.write(revised_html)

        saved_files.append(out_path)

    return saved_files




def generate_ppt_with_gemini_zzz(
    outline_path: str,
    ppt_template_path: str,
    generate_ppt_with_gemini_prompt: Union[Dict[str, str], List[Dict[str, str]]],
    api_key: Optional[str] = None,
    model: str = "gemini-3-pro-preview",
) -> List[str]:
    """
    Traverse outline JSON nodes, load corresponding HTML templates, send (prompt + node + template)
    to Gemini, then save revised HTML to the outline.json directory as:
        <PPT编号>_ppt.html

    Args:
        outline_path: path to outline json file.
        ppt_template_path: folder containing html templates.
        generate_ppt_with_gemini_prompt: JSON-like prompt (dict or list of messages).
        api_key: optional Google API Key, otherwise reads env GOOGLE_API_KEY.
        model: gemini model name, default 'gemini-3-pro-preview'.

    Returns:
        List of saved HTML file paths (one per node).
    """
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("请提供 Google API Key，或者设置环境变量 GOOGLE_API_KEY")

    # client = genai.Client(api_key=api_key)
    client = genai.Client(
        api_key= api_key,
        http_options={
            "base_url": "https://api.zhizengzeng.com/google"
        }
    )
    

    outline_path = os.path.abspath(outline_path)
    ppt_template_path = os.path.abspath(ppt_template_path)

    if not os.path.isfile(outline_path):
        raise FileNotFoundError(f"outline_path 不存在或不是文件: {outline_path}")
    if not os.path.isdir(ppt_template_path):
        raise NotADirectoryError(f"ppt_template_path 不存在或不是文件夹: {ppt_template_path}")

    with open(outline_path, "r", encoding="utf-8") as f:
        outline = json.load(f)

    if not isinstance(outline, list):
        raise ValueError("outline_path 的 JSON 顶层必须是 list（每个元素代表一页 PPT 结点）")

    out_dir = os.path.dirname(outline_path)
    saved_files: List[str] = []

    # Allow prompt to be either a single message dict or a list of messages.
    if isinstance(generate_ppt_with_gemini_prompt, dict):
        base_messages = [generate_ppt_with_gemini_prompt]
    elif isinstance(generate_ppt_with_gemini_prompt, list):
        base_messages = generate_ppt_with_gemini_prompt
    else:
        raise TypeError("generate_ppt_with_gemini_prompt 必须是 dict 或 list[dict] 的 JSON 形式")

    for idx, node in enumerate(outline, start=1):
        if not isinstance(node, dict):
            # Skip invalid node but keep position consistent
            continue

        norm_node = _normalize_node(node)
        template_file = norm_node.get("template", "").strip()
        if not template_file:
            # No template: skip
            continue

        template_full_path = os.path.join(ppt_template_path, template_file)
        if not os.path.isfile(template_full_path):
            raise FileNotFoundError(f"找不到模板文件: {template_full_path}")

        with open(template_full_path, "r", encoding="utf-8") as tf:
            template_html = tf.read()

        user_payload = {
            "ppt_index": idx,
            "node": norm_node,
            "template_html": template_html,
        }

        messages = list(base_messages) + [
            {
                "role": "user",
                "content": (
                    "Here is the slide node JSON and the HTML template. "
                    "Revise the HTML per instructions and return ONLY the final HTML.\n\n"
                    f"{json.dumps(user_payload, ensure_ascii=False)}"
                ),
            }
        ]

        # ---- FIX: google-genai expects contents as a string (or specific SDK types), not OpenAI-style messages list ----
        # Convert the JSON-like prompt/messages into a single string prompt.
        final_prompt_parts: List[str] = []
        for m in messages:
            if isinstance(m, dict):
                c = m.get("content", "")
                if c:
                    final_prompt_parts.append(str(c))
            elif isinstance(m, str):
                final_prompt_parts.append(m)
        final_prompt = "\n\n".join(final_prompt_parts)
        # -----------------------------------------------------------------------------------------------

        # Gemini call
        resp = client.models.generate_content(
            model=model,
            contents=final_prompt
        )

        # Try to robustly extract text
        revised_html = getattr(resp, "text", None)
        if revised_html is None:
            # Fallback: attempt stringify
            revised_html = str(resp)

        revised_html = revised_html.strip()

        # Save
        out_name = f"{idx}_ppt.html"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as wf:
            wf.write(revised_html)

        saved_files.append(out_path)

    return saved_files


def generate_ppt_with_gemini_xj(
    outline_path: str,
    ppt_template_path: str,
    generate_ppt_with_gemini_prompt: Union[Dict[str, str], List[Dict[str, str]]],
    api_key: Optional[str] = None,
    model: str = "gemini-3-pro-preview",
) -> List[str]:
    """
    Traverse outline JSON nodes, load corresponding HTML templates, send (prompt + node + template)
    to Gemini, then save revised HTML to the outline.json directory as:
        <PPT编号>_ppt.html

    Args:
        outline_path: path to outline json file.
        ppt_template_path: folder containing html templates.
        generate_ppt_with_gemini_prompt: JSON-like prompt (dict or list of messages).
        api_key: optional Google API Key, otherwise reads env GOOGLE_API_KEY.
        model: gemini model name, default 'gemini-3-pro-preview'.

    Returns:
        List of saved HTML file paths (one per node).
    """
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("请提供 Google API Key，或者设置环境变量 GOOGLE_API_KEY")

    # client = genai.Client(api_key=api_key)
    client = genai.Client(
        api_key= api_key,
        http_options={
            "base_url": "https://open.xiaojingai.com/"
        }
    )
    

    outline_path = os.path.abspath(outline_path)
    ppt_template_path = os.path.abspath(ppt_template_path)

    if not os.path.isfile(outline_path):
        raise FileNotFoundError(f"outline_path 不存在或不是文件: {outline_path}")
    if not os.path.isdir(ppt_template_path):
        raise NotADirectoryError(f"ppt_template_path 不存在或不是文件夹: {ppt_template_path}")

    with open(outline_path, "r", encoding="utf-8") as f:
        outline = json.load(f)

    if not isinstance(outline, list):
        raise ValueError("outline_path 的 JSON 顶层必须是 list（每个元素代表一页 PPT 结点）")

    out_dir = os.path.dirname(outline_path)
    saved_files: List[str] = []

    # Allow prompt to be either a single message dict or a list of messages.
    if isinstance(generate_ppt_with_gemini_prompt, dict):
        base_messages = [generate_ppt_with_gemini_prompt]
    elif isinstance(generate_ppt_with_gemini_prompt, list):
        base_messages = generate_ppt_with_gemini_prompt
    else:
        raise TypeError("generate_ppt_with_gemini_prompt 必须是 dict 或 list[dict] 的 JSON 形式")

    for idx, node in enumerate(outline, start=1):
        if not isinstance(node, dict):
            # Skip invalid node but keep position consistent
            continue

        norm_node = _normalize_node(node)
        template_file = norm_node.get("template", "").strip()
        if not template_file:
            # No template: skip
            continue

        template_full_path = os.path.join(ppt_template_path, template_file)
        if not os.path.isfile(template_full_path):
            raise FileNotFoundError(f"找不到模板文件: {template_full_path}")

        with open(template_full_path, "r", encoding="utf-8") as tf:
            template_html = tf.read()

        user_payload = {
            "ppt_index": idx,
            "node": norm_node,
            "template_html": template_html,
        }

        messages = list(base_messages) + [
            {
                "role": "user",
                "content": (
                    "Here is the slide node JSON and the HTML template. "
                    "Revise the HTML per instructions and return ONLY the final HTML.\n\n"
                    f"{json.dumps(user_payload, ensure_ascii=False)}"
                ),
            }
        ]

        # ---- FIX: google-genai expects contents as a string (or specific SDK types), not OpenAI-style messages list ----
        # Convert the JSON-like prompt/messages into a single string prompt.
        final_prompt_parts: List[str] = []
        for m in messages:
            if isinstance(m, dict):
                c = m.get("content", "")
                if c:
                    final_prompt_parts.append(str(c))
            elif isinstance(m, str):
                final_prompt_parts.append(m)
        final_prompt = "\n\n".join(final_prompt_parts)
        # -----------------------------------------------------------------------------------------------

        # Gemini call
        resp = client.models.generate_content(
            model=model,
            contents=final_prompt
        )

        # Try to robustly extract text
        revised_html = getattr(resp, "text", None)
        if revised_html is None:
            # Fallback: attempt stringify
            revised_html = str(resp)

        revised_html = revised_html.strip()

        # Save
        out_name = f"{idx}_ppt.html"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as wf:
            wf.write(revised_html)

        saved_files.append(out_path)

    return saved_files




# ========== Generate poster_outline.txt via Gemini (section-by-section) ==========
def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dag_schema(dag_obj: Any) -> Dict[str, Any]:
    """
    Robustness: if LLM/other step produced a single node dict instead of {"nodes":[...]},
    wrap it into the expected schema.
    """
    if isinstance(dag_obj, dict) and "nodes" in dag_obj and isinstance(dag_obj["nodes"], list):
        return dag_obj
    if isinstance(dag_obj, dict) and "name" in dag_obj and "content" in dag_obj:
        return {"nodes": [dag_obj]}
    raise ValueError("Invalid dag.json schema: expected {'nodes': [...]} or a single node dict.")


def _resolution_area(resolution: Any) -> int:
    """
    resolution can be like "536x86" (string) or [536, 86] etc.
    Returns area; invalid -> 0
    """
    if resolution is None:
        return 0
    if isinstance(resolution, str):
        m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", resolution)
        if not m:
            return 0
        w = int(m.group(1))
        h = int(m.group(2))
        return w * h
    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        try:
            w = int(resolution[0])
            h = int(resolution[1])
            return w * h
        except Exception:
            return 0
    if isinstance(resolution, dict):
        # sometimes {"width":..., "height":...}
        try:
            w = int(resolution.get("width", 0))
            h = int(resolution.get("height", 0))
            return w * h
        except Exception:
            return 0
    return 0


def _strip_md_image(s: str) -> str:
    return (s or "").strip()


def _extract_image_src_from_md(md_image: str) -> Optional[str]:
    """
    md_image example: "![](images/xxx.jpg)" or " ![](images/xxx.jpg) "
    returns "images/xxx.jpg" (without surrounding spaces)
    """
    if not md_image:
        return None
    m = re.search(r"!\[[^\]]*\]\(([^)]+)\)", md_image.strip())
    if not m:
        return None
    return m.group(1).strip()


def _safe_section_title(name: str) -> str:
    """
    Optional cleanup: remove trailing .md if present.
    """
    if not name:
        return ""
    name = name.strip()
    if name.lower().endswith(".md"):
        name = name[:-3]
    return name.strip()


def _remove_key_deep(obj: Any, key_to_remove: str) -> Any:
    """
    Create a JSON-serializable copy of obj with a top-level key removed if dict.
    (We only need to remove section_node["visual_node"] at top-level, but keep it safe.)
    """
    if isinstance(obj, dict):
        return {k: _remove_key_deep(v, key_to_remove) for k, v in obj.items() if k != key_to_remove}
    if isinstance(obj, list):
        return [_remove_key_deep(x, key_to_remove) for x in obj]
    return obj


def generate_poster_outline_txt(
    dag_path: str,
    poster_outline_path: str,
    poster_outline_prompt: str,
    model_name: str = "gemini-2.5-pro",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional[OpenAI] = None,
    overwrite: bool = True,
) -> None:
    """
    Read dag.json from dag_path, iterate root->section nodes, and for each section:
      - choose the largest-resolution visual node referenced by section["visual_node"]
      - send (section_without_visual_node, best_visual_node_if_any, IMAGE_SRC, ALT_TEXT) to Gemini
      - Gemini returns EXACTLY one <section class="section">...</section> HTML block
      - append/write to poster_outline_path

    dag_path can be:
      - a directory that contains dag.json
      - or a direct path to dag.json
    """
    # Resolve dag.json path
    if os.path.isdir(dag_path):
        dag_json_path = os.path.join(dag_path, "dag.json")
    else:
        dag_json_path = dag_path

    dag_obj = _ensure_dag_schema(_load_json(dag_json_path))
    nodes: List[Dict[str, Any]] = dag_obj.get("nodes", [])
    if not nodes:
        raise ValueError("dag.json has empty 'nodes'.")

    # Root node is the first node by your spec
    root = nodes[0]
    root_edges = root.get("edge", [])
    if not isinstance(root_edges, list) or not root_edges:
        raise ValueError("Root node has no valid 'edge' list of section names.")

    # Build lookup: name -> node (first occurrence)
    name2node: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        if isinstance(n, dict) and "name" in n:
            name2node.setdefault(str(n["name"]), n)

    # Prepare OpenAI client
    if client is None:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        client = OpenAI(api_key=api_key, base_url=base_url)

    # Output file init
    out_dir = os.path.dirname(os.path.abspath(poster_outline_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    write_mode = "w" if overwrite else "a"
    with open(poster_outline_path, write_mode, encoding="utf-8") as f_out:
        # Iterate sections in the order of root.edge
        for sec_name in root_edges:
            if sec_name not in name2node:
                # Skip silently or raise; choose raise to make debugging clear
                raise KeyError(f"Section node not found by name from root.edge: {sec_name}")

            section_node = name2node[sec_name]
            if not isinstance(section_node, dict):
                raise ValueError(f"Invalid section node for name={sec_name}")

            # Find all visual nodes referenced by this section
            visual_refs = section_node.get("visual_node", [])
            best_visual_node: Optional[Dict[str, Any]] = None
            best_area = -1
            best_image_src: Optional[str] = None

            if isinstance(visual_refs, list) and len(visual_refs) > 0:
                for ref in visual_refs:
                    ref_str = _strip_md_image(str(ref))
                    # Visual node name in dag may contain leading/trailing whitespace/newlines
                    # So we try exact match first, then fallback to normalized match.
                    cand = name2node.get(ref_str)

                    if cand is None:
                        # normalized matching fallback
                        # (rare case: dag stored "name": "\n![](images/xxx.jpg)\n")
                        for k, v in name2node.items():
                            if isinstance(k, str) and k.strip() == ref_str.strip():
                                cand = v
                                break

                    if cand is None or not isinstance(cand, dict):
                        continue

                    area = _resolution_area(cand.get("resolution"))
                    if area > best_area:
                        best_area = area
                        best_visual_node = cand
                        best_image_src = _extract_image_src_from_md(str(cand.get("name", "")))

                # If best_image_src still None, try extracting from the ref itself
                if best_visual_node is not None and not best_image_src:
                    for ref in visual_refs:
                        tmp = _extract_image_src_from_md(str(ref))
                        if tmp:
                            best_image_src = tmp
                            break

            # Build the section JSON WITHOUT visual_node attribute (per your requirement)
            section_wo_visual = _remove_key_deep(section_node, "visual_node")
            # Also sanitize section title
            section_wo_visual["name"] = _safe_section_title(str(section_wo_visual.get("name", "")))

            # Compose ALT_TEXT (prefer caption if present)
            alt_text = None
            if best_visual_node is not None:
                cap = best_visual_node.get("caption")
                if isinstance(cap, str) and cap.strip():
                    alt_text = cap.strip()
            if not alt_text:
                alt_text = "Figure"

            # Compose prompt input fields
            section_json_str = json.dumps(section_wo_visual, ensure_ascii=False, indent=2)

            if best_visual_node is not None:
                visual_json_str = json.dumps(best_visual_node, ensure_ascii=False, indent=2)
                image_src = best_image_src or ""
                # Ensure the src looks like "images/xxx"
                if image_src and not image_src.startswith("images/") and "images/" in image_src:
                    # keep as-is; user might give relative subpaths
                    pass
                payload = poster_outline_prompt.format(
                    SECTION_JSON=section_json_str,
                    HAS_VISUAL="true",
                    VISUAL_JSON=visual_json_str,
                    IMAGE_SRC=image_src,
                    ALT_TEXT=alt_text,
                )
            else:
                payload = poster_outline_prompt.format(
                    SECTION_JSON=section_json_str,
                    HAS_VISUAL="false",
                    VISUAL_JSON="",
                    IMAGE_SRC="",
                    ALT_TEXT="",
                )

            # Call Gemini
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": payload}],
            )

            if not hasattr(resp, "choices") or not resp.choices:
                raise RuntimeError("Gemini returned empty choices.")

            html_block = resp.choices[0].message.content
            if not isinstance(html_block, str) or not html_block.strip():
                raise RuntimeError("Gemini returned empty content.")

            # Append to output file, keep a blank line between sections
            f_out.write(html_block.strip())
            f_out.write("\n\n")


# ========== Modify poster_outline.txt ==========
from pathlib import Path
def modify_poster_outline(
    poster_outline_path: str,
    poster_paper_name: str,
    modified_poster_outline_path: str
):
    """
    功能：
    1. 找到 section-bar 内容等于 poster_paper_name 的 section (忽略大小写)：
       - 将其 section-bar 改为 "Introduction"
       - 将该 section 移动到文件最前面
    2. 对其余 section：
       - 删除 section-bar 标题前的数字序号
    3. 只保留处理后的前 6 个 section
    4. 将最终结果保存到 modified_poster_outline_path
    """

    text = Path(poster_outline_path).read_text(encoding="utf-8")

    # ===== 1. 提取所有 section 块 =====
    section_pattern = re.compile(
        r"<section class=\"section\">.*?</section>",
        re.DOTALL
    )
    sections = section_pattern.findall(text)

    intro_section = None
    other_sections = []
    
    # 预处理目标名称：去除首尾空格并转小写，用于后续比较
    target_name_normalized = poster_paper_name.strip().lower()

    for sec in sections:
        # 提取 section-bar 内容
        m = re.search(
            r"<div class=\"section-bar\" contenteditable=\"true\">(.*?)</div>",
            sec,
            re.DOTALL
        )
        if not m:
            continue

        # 获取原始内容用于后续替换，同时获取用于比较的归一化字符串
        original_title = m.group(1).strip()
        current_title_normalized = original_title.lower()

        # ===== 2. 处理 paper title 对应的 section (修改点：忽略大小写) =====
        if current_title_normalized == target_name_normalized:
            # 改名为 Introduction
            sec = re.sub(
                r"<div class=\"section-bar\" contenteditable=\"true\">.*?</div>",
                '<div class="section-bar" contenteditable="true">Introduction</div>',
                sec,
                count=1,
                flags=re.DOTALL
            )
            intro_section = sec
        else:
            # ===== 3. 删除其余 section 标题前的数字序号 =====
            # 例如： "2 Contextual Auction Design" -> "Contextual Auction Design"
            new_title = re.sub(r"^\s*\d+(\.\d+)*\s*", "", original_title)
            
            # 仅替换标题部分
            sec = sec.replace(original_title, new_title, 1)
            other_sections.append(sec)

    # ===== 4. 重新组合内容（Introduction 在最前） =====
    final_sections = []
    if intro_section is not None:
        final_sections.append(intro_section)
    final_sections.extend(other_sections)

    # ===== 5. (修改点) 只保留前 6 个 section =====
    final_sections = final_sections[:6]
    # ===== 5.5 清洗 section-bar：确保以字母单词开头 =====
    cleaned_sections = []

    for sec in final_sections:
        def _clean_title(match):
            title = match.group(1)
            # 去掉开头所有非字母字符（直到第一个字母）
            cleaned_title = re.sub(r"^[^A-Za-z]+", "", title)
            return f'<div class="section-bar" contenteditable="true">{cleaned_title}</div>'
        sec = re.sub(
            r'<div class="section-bar" contenteditable="true">(.*?)</div>',
            _clean_title,
            sec,
            count=1,
            flags=re.DOTALL
        )
        cleaned_sections.append(sec)

    final_sections = cleaned_sections
    final_text = "\n\n".join(final_sections)

    # ===== 6. 保存结果 =====
    Path(modified_poster_outline_path).write_text(final_text, encoding="utf-8")








# ========== Build final poster HTML from outline.txt + template ==========
import os
import re
import shutil

def build_poster_from_outline(
    poster_outline_path: str,
    poster_template_path: str,
    poster_path: str,
) -> str:
    """
    输入:
      - poster_outline_path: 一个 .txt 文件路径，内容为要插入的 HTML 片段（若干 <section> ... </section> 等）
      - poster_template_path: poster 模板路径（目录 或 具体文件）。函数会在此路径下定位 poster_template.html
      - poster_path: 输出 HTML 保存路径（完整文件路径，如 /xxx/my_poster.html）

    行为:
      1) 定位 poster_template.html（不修改原文件）
      2) 复制一份到 poster_path
      3) 在复制后的 HTML 中，找到:
            <main class="main">
              <div class="flow" id="flow">
                ...这里...
              </div>
            </main>
         并将 outline txt 的内容插入到 <div class="flow" id="flow"> 与其 </div> 之间
      4) 做基础稳健性处理：换行规范化、缩进对齐、避免破坏标签结构

    返回:
      - poster_path（便于上层链式调用）
    """
    # ---------- 基础检查 ----------
    if not os.path.isfile(poster_outline_path):
        raise FileNotFoundError(f"poster_outline_path not found: {poster_outline_path}")

    if not poster_path.lower().endswith(".html"):
        raise ValueError(f"poster_path must be an .html file path, got: {poster_path}")

    os.makedirs(os.path.dirname(os.path.abspath(poster_path)), exist_ok=True)

    # ---------- 定位模板文件 poster_template.html ----------
    template_file = None
    if os.path.isdir(poster_template_path):
        candidate = os.path.join(poster_template_path, "poster_template.html")
        if os.path.isfile(candidate):
            template_file = candidate
        else:
            # 兜底：递归搜索同名文件（防止模板目录层级不固定）
            for root, _, files in os.walk(poster_template_path):
                if "poster_template.html" in files:
                    template_file = os.path.join(root, "poster_template.html")
                    break
    else:
        # poster_template_path 可能直接就是某个文件
        if os.path.isfile(poster_template_path) and os.path.basename(poster_template_path) == "poster_template.html":
            template_file = poster_template_path
        elif os.path.isfile(poster_template_path):
            # 兜底：如果用户传的是某个 html 文件，也允许用它作为模板
            template_file = poster_template_path

    if template_file is None:
        raise FileNotFoundError(
            f"Cannot locate poster_template.html under: {poster_template_path}"
        )

    # ---------- 读取 outline 内容，并做换行规范化 ----------
    with open(poster_outline_path, "r", encoding="utf-8") as f:
        outline_raw = f.read()

    # 统一换行到 '\n'，并去掉 BOM
    outline_raw = outline_raw.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n").strip()

    # 如果 outline 为空，允许插入空内容（但仍会保持结构）
    # 给 outline 末尾补一个换行，避免与 </div> 直接黏连
    if outline_raw:
        outline_raw += "\n"

    # ---------- 复制模板到输出路径（不修改原模板） ----------
    shutil.copyfile(template_file, poster_path)

    # ---------- 读取复制后的 html ----------
    with open(poster_path, "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")

    # ---------- 在 <div class="flow" id="flow"> ... </div> 内插入 ----------
    # 说明：
    # - 使用非贪婪匹配，尽量锁定 main/flow 这个区域
    # - 捕获 div 起始标签、原内部内容、div 结束标签
    pattern = re.compile(
        r'(<main\s+class="main"\s*>\s*'
        r'<div\s+class="flow"\s+id="flow"\s*>\s*)'
        r'(.*?)'
        r'(\s*</div>\s*</main>)',
        flags=re.DOTALL | re.IGNORECASE,
    )

    m = pattern.search(html)
    if not m:
        # 再做一次更宽松的匹配（只要求 flow div，不强依赖 main 结构）
        pattern2 = re.compile(
            r'(<div\s+class="flow"\s+id="flow"\s*>\s*)'
            r'(.*?)'
            r'(\s*</div>)',
            flags=re.DOTALL | re.IGNORECASE,
        )
        m2 = pattern2.search(html)
        if not m2:
            raise ValueError(
                'Cannot find target insertion block: <div class="flow" id="flow"> ... </div>'
            )

        prefix, _, suffix = m2.group(1), m2.group(2), m2.group(3)
        base_indent = _infer_indent_from_prefix(prefix, html, m2.start(1))
        outline_formatted = _indent_block(outline_raw, base_indent + "  ")  # 默认在 div 内再缩进 2 空格
        new_block = prefix + "\n" + outline_formatted + suffix
        html = html[: m2.start()] + new_block + html[m2.end():]
    else:
        prefix, _, suffix = m.group(1), m.group(2), m.group(3)
        base_indent = _infer_indent_from_prefix(prefix, html, m.start(1))
        outline_formatted = _indent_block(outline_raw, base_indent + "  ")
        new_block = prefix + "\n" + outline_formatted + suffix
        html = html[: m.start()] + new_block + html[m.end():]

    # ---------- 轻量格式稳健性：清理过多空行 ----------
    html = _collapse_blank_lines(html)

    # ---------- 写回输出 ----------
    with open(poster_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)

    return poster_path


def _infer_indent_from_prefix(prefix: str, full_html: str, prefix_start_idx: int) -> str:
    """
    推断插入区域的基础缩进（用于让插入块的空格更“正确”）。
    策略：取 prefix_start_idx 所在行的前导空白作为 base indent。
    """
    line_start = full_html.rfind("\n", 0, prefix_start_idx) + 1
    line = full_html[line_start:prefix_start_idx]
    m = re.match(r"[ \t]*", line)
    return m.group(0) if m else ""


def _indent_block(text: str, indent: str) -> str:
    """
    将一段多行文本整体缩进到指定 indent。
    - 空行保持为空行（不强塞空格），避免出现“看起来很多空格”的脏格式
    """
    if not text:
        return ""
    lines = text.split("\n")
    out = []
    for ln in lines:
        if ln.strip() == "":
            out.append("")
        else:
            out.append(indent + ln)
    return "\n".join(out) + ("\n" if not text.endswith("\n") else "")


def _collapse_blank_lines(html: str, max_blank: int = 2) -> str:
    """
    将连续空行压缩到最多 max_blank 行，避免插入后产生大量空白。
    """
    # 先把只含空白的行变成真正空行
    html = re.sub(r"[ \t]+\n", "\n", html)
    # 压缩空行：\n\n\n... -> 最多 max_blank+1 个 \n（表示 max_blank 个空行）
    html = re.sub(r"\n{"+str(max_blank+2)+r",}", "\n" * (max_blank + 1), html)
    return html





# ======================================以下为Web所用的函数======================================
# ========== 修改 poster.html 中的 title 和 authors ==========
def modify_title_and_author(dag_path: str, poster_path: str) -> None:
    if not os.path.exists(dag_path):
        raise FileNotFoundError(f"dag.json not found: {dag_path}")
    if not os.path.exists(poster_path):
        raise FileNotFoundError(f"poster.html not found: {poster_path}")

    with open(dag_path, "r", encoding="utf-8") as f:
        dag: Dict[str, Any] = json.load(f)

    nodes = dag.get("nodes")
    if not isinstance(nodes, list) or len(nodes) == 0:
        raise ValueError("Invalid dag.json: missing or empty 'nodes' list")

    first = nodes[0]
    if not isinstance(first, dict):
        raise ValueError("Invalid dag.json: first node is not an object")

    title = str(first.get("name", "")).strip()
    authors = str(first.get("content", "")).strip()

    if not title:
        raise ValueError("Invalid dag.json: first node 'name' (title) is empty")
    if not authors:
        raise ValueError("Invalid dag.json: first node 'content' (authors) is empty")

    with open(poster_path, "r", encoding="utf-8") as f:
        html = f.read()

    title_pattern = re.compile(
        r'(<h1\s+class="title"\s*>)(.*?)(</h1\s*>)',
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not title_pattern.search(html):
        raise ValueError('Cannot find <h1 class="title">...</h1> in poster.html')

    html = title_pattern.sub(lambda m: m.group(1) + title + m.group(3), html, count=1)

    authors_pattern = re.compile(
        r'(<div\s+class="authors"\s*>)(.*?)(</div\s*>)',
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not authors_pattern.search(html):
        raise ValueError('Cannot find <div class="authors">...</div> in poster.html')

    html = authors_pattern.sub(lambda m: m.group(1) + authors + m.group(3), html, count=1)

    with open(poster_path, "w", encoding="utf-8") as f:
        f.write(html)


# ========== Extract basic information for Web Generation ==========
def extract_basic_information(
    auto_path: str,
    extract_basic_information_prompt: str,
    model: str = "gpt-4o",
    max_lines: int = 50,
    output_filename: str = "basic_information.txt",
) -> str:
    """
    在 auto_path 目录中：
    1) 找到不以 '_cleaned.md' 结尾的 md 文件（通常 auto 下有两个 md：一个 cleaned，一个原始）。
    2) 读取该 md 的前 max_lines 行。
    3) 调用 GPT 提取 Title/Author/Institution/Github。
    4) 将 GPT 输出写入 auto_path/basic_information.txt（若不存在则创建）。
    
    返回：写入的 txt 的绝对路径（str）
    """
    auto_dir = Path(auto_path).expanduser().resolve()
    if not auto_dir.exists() or not auto_dir.is_dir():
        raise FileNotFoundError(f"auto_path not found or not a directory: {auto_dir}")

    # 1) 找到目标 md（不以 _cleaned.md 结尾）
    md_files = sorted([p for p in auto_dir.glob("*.md") if p.is_file()])

    # 优先选择：不是 *_cleaned.md 的 md
    candidates = [p for p in md_files if not p.name.endswith("_cleaned.md")]
    if not candidates:
        # 兜底：如果没有候选（不排除用户目录结构异常），尝试使用任意 md
        if not md_files:
            raise FileNotFoundError(f"No .md files found under: {auto_dir}")
        target_md = md_files[0]
    else:
        # 如果有多个候选，尽量避开一些常见的中间产物命名（可按需扩展）
        # 这里简单取第一个即可；你也可以改成按文件大小/修改时间排序
        target_md = candidates[0]

    # 2) 读取前 max_lines 行
    with target_md.open("r", encoding="utf-8", errors="replace") as f:
        lines = []
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
    md_head = "\n".join(lines).strip()

    # 3) 调用 GPT
    # 说明：你只要求函数输入为 auto_path + prompt，这里默认从环境变量读取 OPENAI_API_KEY
    # 若你使用自定义 base_url，可在 OpenAI(...) 里传入 base_url=...
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))

    # 将“前50行内容”作为用户输入，prompt 作为 system（更稳）
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": extract_basic_information_prompt},
            {
                "role": "user",
                "content": (
                    "MD_HEAD (first lines):\n"
                    "```md\n"
                    f"{md_head}\n"
                    "```"
                ),
            },
        ],
    )

    content = (resp.choices[0].message.content or "").strip()

    # 4) 写入 basic_information.txt（不存在则创建）
    out_path = auto_dir / output_filename
    out_path.write_text(content + "\n", encoding="utf-8")

    return str(out_path)
from html import escape

def initialize_web(basic_information_path: str, web_template_path: str, auto_path: str) -> str:
    """
    Initialize a paper web page (web.html) from a template by injecting:
      - Title
      - Authors (dynamic count)
      - Institution
      - Github/Project link (href replacement)

    Returns:
      Output html path.
    """

    def _read_basic_info(txt_path: str) -> dict:
        p = Path(txt_path)
        if not p.exists():
            raise FileNotFoundError(f"basic_information_path not found: {txt_path}")

        text = p.read_text(encoding="utf-8", errors="ignore")
        info = {"Title": "", "Author": "", "Institution": "", "Github": ""}

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(Title|Author|Institution|Github)\s*:\s*(.*)$", line, flags=re.I)
            if m:
                key = m.group(1).capitalize()
                val = m.group(2).strip()
                info[key] = val

        return info

    def _split_authors(author_line: str):
        if not author_line:
            return []
        parts = [a.strip() for a in author_line.split(",")]
        return [p for p in parts if p]

    # ---- Load ----
    info = _read_basic_info(basic_information_path)

    tpl_path = Path(web_template_path)
    if not tpl_path.exists():
        raise FileNotFoundError(f"web_template_path not found: {web_template_path}")
    html = tpl_path.read_text(encoding="utf-8", errors="ignore")

    # ---- Prepare injected strings ----
    title = escape(info.get("Title", "") or "")
    institution = escape(info.get("Institution", "") or "")
    github = (info.get("Github", "") or "").strip()

    # ---- Authors (MODIFIED LOGIC) ----
    authors = _split_authors(info.get("Author", "") or "")
    authors_joined = ", ".join(authors)
    # Requested final structure:
    # <span>authors: <a href="/user/123">Alice Smith, Bob Lee, Charlie Zhang</a></span>
    authors_html = (
        f'                    <span>authors: '
        f'<a href="/user/123">{escape(authors_joined)}</a></span>'
    )

    # ---- 1) Replace Title (use function replacement to avoid \1 + digits ambiguity) ----
    h1_pattern = re.compile(
        r'(<h1\s+class="font-display\s+text-3xl\s+md:text-4xl\s+lg:text-5xl\s+font-extrabold\s+tracking-tight\s+text-slate-800\s+max-w-4xl\s+mx-auto"\s*>\s*)(.*?)(\s*</h1>)',
        flags=re.S,
    )

    def _repl_h1(m: re.Match) -> str:
        return m.group(1) + title + m.group(3)

    html, n_h1 = h1_pattern.subn(_repl_h1, html, count=1)

    # ---- 2) Replace Authors block ----
    authors_div_pattern = re.compile(
        r'(<div\s+class="flex\s+flex-wrap\s+justify-center\s+items-center\s+gap-x-8\s+gap-y-2"\s*>\s*)(.*?)(\s*</div>)',
        flags=re.S,
    )

    def _repl_auth(m: re.Match) -> str:
        return m.group(1) + authors_html + m.group(3)

    html, n_auth = authors_div_pattern.subn(_repl_auth, html, count=1)

    # ---- 3) Replace Institution span ----
    inst_pattern = re.compile(
        r'(<div\s+class="mt-2\s+text-sm\s+text-slate-600"\s*>\s*<span>)(.*?)(</span>\s*</div>)',
        flags=re.S,
    )

    def _repl_inst(m: re.Match) -> str:
        return m.group(1) + institution + m.group(3)

    html, n_inst = inst_pattern.subn(_repl_inst, html, count=1)

    # ---- 4) Replace Github/Project link href in the github button anchor ----
    # Your example Github is a project page (not github.com), so we should still replace the href.
    if github:
        github_escaped = escape(github, quote=True)

        # Prefer the anchor containing the github icon, if present.
        github_anchor_pattern = re.compile(
            r'(<a\s+[^>]*href=")(https?://[^"]+)(")([^>]*>\s*<i\s+class="fab\s+fa-github\b)',
            flags=re.S,
        )

        def _repl_gh(m: re.Match) -> str:
            return m.group(1) + github_escaped + m.group(3) + m.group(4)

        html, n_gh = github_anchor_pattern.subn(_repl_gh, html, count=1)

        # Fallback: replace first href inside the header block if icon pattern not found
        if n_gh == 0:
            # limit scope to reduce accidental replacement
            header_block = re.search(r"(<header\b.*?</header>)", html, flags=re.S)
            if header_block:
                block = header_block.group(1)
                href_pattern = re.compile(r'(<a\s+[^>]*href=")(https?://[^"]+)(")', flags=re.S)
                new_block, _ = href_pattern.subn(
                    lambda m: m.group(1) + github_escaped + m.group(3),
                    block,
                    count=1
                )
                html = html.replace(block, new_block, 1)

    # ---- Guardrails ----
    if n_h1 == 0:
        raise ValueError("Failed to locate target <h1 ...> block for title replacement in template.")
    if n_auth == 0:
        raise ValueError("Failed to locate authors <div class=\"flex flex-wrap ...\"> block in template.")
    if n_inst == 0:
        raise ValueError("Failed to locate institution <div class=\"mt-2 text-sm text-slate-600\"> block in template.")

    # ---- Write output ----
    out_dir = Path(auto_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")

    return str(out_path)




# ========== Inject sections into web.html ==========
import os
import re
from html import unescape
from typing import List, Dict, Optional
def inject_sections_into_web(outline_modified_path: str, web_path: str) -> None:
    """
    Read outline_modified_path (txt that contains repeated <section class="section"> ... </section> blocks),
    extract section title (section-bar), paragraphs (<p>...</p>), and first image (<img ...> if any),
    then inject:
      1) nav links into web.html under:
         <div class="flex flex-wrap justify-center items-center gap-6 md:gap-8 h-16">
      2) rendered sections into web.html under:
         <div class="space-y-24 py-20">

    This function edits web_path in-place.
    """

    # -------------------------
    # Helpers
    # -------------------------
    def slugify_id(title: str) -> str:
        # Convert to a safe HTML id: keep letters, numbers, '-', '_', ':', '.'
        # Replace whitespace with '-'; strip leading/trailing separators.
        t = title.strip()
        t = re.sub(r"\s+", "-", t)
        t = re.sub(r"[^A-Za-z0-9\-\_\:\.]", "", t)
        return t or "Section"

    def extract_sections(raw: str) -> List[str]:
        # Non-greedy capture of each section block
        pattern = re.compile(r'<section\s+class="section"\s*>.*?</section>', re.DOTALL | re.IGNORECASE)
        return pattern.findall(raw)

    def extract_title(section_html: str) -> Optional[str]:
        # Match: <div class="section-bar" contenteditable="true">Introduction</div>
        m = re.search(
            r'<div\s+class="section-bar"\s+contenteditable="true"\s*>(.*?)</div>',
            section_html,
            flags=re.DOTALL | re.IGNORECASE
        )
        if not m:
            return None
        title = unescape(m.group(1))
        # remove inner tags if any
        title = re.sub(r"<.*?>", "", title, flags=re.DOTALL).strip()
        return title if title else None

    def extract_paragraphs(section_html: str) -> List[str]:
        # capture full <p ...>...</p> blocks; keep them as html for insertion
        ps = re.findall(r"<p\b[^>]*>.*?</p>", section_html, flags=re.DOTALL | re.IGNORECASE)
        # clean trailing spaces/newlines
        return [p.strip() for p in ps if p.strip()]

    def extract_first_img_tag(section_html: str) -> Optional[str]:
        # capture first <img ...> (self-closing or not)
        m = re.search(r"<img\b[^>]*?>", section_html, flags=re.DOTALL | re.IGNORECASE)
        return m.group(0).strip() if m else None

    def build_nav_link(section_id: str, title: str) -> str:
        return (
            f'                <a href="#{section_id}" '
            f'class="nav-link text-slate-600 font-semibold hover:text-primary-500 '
            f'transition-colors text-sm md:text-base">{title}</a>\n'
        )

    def build_rendered_section(section_id: str, title: str, paragraphs: List[str], img_tag: Optional[str]) -> str:
        # If there are multiple <p>, keep them all in the prose block.
        # If no <p>, still render an empty prose container (safe).
        para_html = "\n                ".join(paragraphs) if paragraphs else "<p></p>"

        img_block = ""
        if img_tag:
            img_block = f"""
            <div class="drag-slider max-w-4xl mx-auto rounded-xl overflow-hidden">
                {img_tag}
            </div>""".rstrip()

        return f"""
            <section id="{section_id}" class="scroll-mt-24 reveal">
                <h2 class="font-display text-2xl md:text-3xl font-bold text-slate-800 mb-6 text-center">{title}</h2>
                <div class="prose max-w-none mx-auto text-slate-600 text-base md:text-lg mb-10 text-center max-w-3xl">
                {para_html}
                </div>{img_block}
            </section>
""".rstrip() + "\n"

    def insert_after_open_tag(html: str, open_tag_regex: str, insertion: str) -> str:
        """
        Insert `insertion` right after the *opening* tag matched by open_tag_regex.
        open_tag_regex should match the full opening tag, e.g.:
          r'<div class="...h-16">'
        """
        m = re.search(open_tag_regex, html, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Cannot find insertion anchor by regex: {open_tag_regex}")
        idx = m.end()
        return html[:idx] + "\n" + insertion + html[idx:]

    # -------------------------
    # Read outline sections
    # -------------------------
    if not os.path.exists(outline_modified_path):
        raise FileNotFoundError(f"outline_modified_path not found: {outline_modified_path}")
    if not os.path.exists(web_path):
        raise FileNotFoundError(f"web_path not found: {web_path}")

    with open(outline_modified_path, "r", encoding="utf-8") as f:
        outline_raw = f.read()

    section_blocks = extract_sections(outline_raw)

    parsed: List[Dict] = []
    for blk in section_blocks:
        title = extract_title(blk)
        if not title:
            # skip malformed section
            continue
        sid = slugify_id(title)
        paragraphs = extract_paragraphs(blk)
        img_tag = extract_first_img_tag(blk)
        parsed.append({"title": title, "id": sid, "paragraphs": paragraphs, "img": img_tag})

    # Nothing to inject
    if not parsed:
        return

    # -------------------------
    # Read web html
    # -------------------------
    with open(web_path, "r", encoding="utf-8") as f:
        web_html = f.read()

    # -------------------------
    # Build nav links and rendered sections
    # -------------------------
    nav_insertion = "".join(build_nav_link(p["id"], p["title"]) for p in parsed)
    sections_insertion = "".join(
        build_rendered_section(p["id"], p["title"], p["paragraphs"], p["img"]) for p in parsed
    )

    # -------------------------
    # Insert into NAV
    # Anchor (your exact target):
    # <div class="flex flex-wrap justify-center items-center gap-6 md:gap-8 h-16">
    # -------------------------
    nav_anchor_regex = r'<div\s+class="flex\s+flex-wrap\s+justify-center\s+items-center\s+gap-6\s+md:gap-8\s+h-16"\s*>'
    web_html = insert_after_open_tag(web_html, nav_anchor_regex, nav_insertion)

    # -------------------------
    # Insert into MAIN content
    # Anchor (your exact target):
    # <div class="space-y-24 py-20">
    # -------------------------
    main_anchor_regex = r'<div\s+class="space-y-24\s+py-20"\s*>'
    web_html = insert_after_open_tag(web_html, main_anchor_regex, sections_insertion)

    # -------------------------
    # Write back
    # -------------------------
    with open(web_path, "w", encoding="utf-8") as f:
        f.write(web_html)




# ========= Generate Future Work and References via LLM and inject into web.html ==========
import html
def generate_futurework_and_reference(
    auto_path: str,
    futurework_and_reference_prompt: str,
    web_path: str,
    model: str = "gpt-4o-mini",
    api_key_env: str = "OPENAI_API_KEY",
    max_md_chars: int = 120_000,
) -> None:
    """
    1) Under auto_path, find the non-*_cleaned.md Markdown file (auto_path usually contains two md files).
    2) Send that MD to LLM with futurework_and_reference_prompt (use {MD_TEXT} placeholder).
       LLM must return a JSON object: {"future_work": "...", "references": ["...", "...", "..."]}.
    3) Read web_path HTML and fill:
       - <section id="Future Work"> ... <p> ... </p> ... </section>
       - <section id="Reference">  ... <p> ... </p> ... </section>
       Put future_work into the Future Work <p>, and the 3 references into Reference <p> (joined by <br/>).
    4) Edit web_path in-place.

    Proxy support:
    - Reads OPENAI_API_KEY from env (api_key_env)
    - Reads OPENAI_BASE_URL from env (optional). If set, will use it as base_url.
    """

    # -------------------------
    # Helper: find target md
    # -------------------------
    auto_dir = Path(auto_path)
    if not auto_dir.exists():
        raise FileNotFoundError(f"auto_path does not exist: {auto_path}")

    md_files = sorted([p for p in auto_dir.glob("*.md") if p.is_file()])
    candidate_md = [p for p in md_files if not p.name.endswith("_cleaned.md") and not p.name.endswith("cleaned.md")]
    if not candidate_md:
        raise FileNotFoundError(
            f"Cannot find non-cleaned .md under: {auto_path}. Found: {[p.name for p in md_files]}"
        )
    md_path = candidate_md[0]

    md_text = md_path.read_text(encoding="utf-8", errors="ignore")
    if len(md_text) > max_md_chars:
        md_text = md_text[:max_md_chars]

    # -------------------------
    # Call LLM (OpenAI SDK v1) with optional proxy base_url
    # -------------------------
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise EnvironmentError(f"Missing API key env var: {api_key_env}")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()  # optional for third-party proxy

    prompt = futurework_and_reference_prompt.replace("{MD_TEXT}", md_text)

    try:
        from openai import OpenAI

        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        llm_text = resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e

    # -------------------------
    # Parse LLM JSON
    # -------------------------
    def _extract_json(text: str) -> str:
        # Prefer fenced ```json ... ```
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        # Otherwise, try first {...} block
        m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()

    raw_json = _extract_json(llm_text)

    try:
        data = json.loads(raw_json)
    except Exception as e:
        raise ValueError(f"Failed to parse LLM output as JSON.\nRaw:\n{llm_text}") from e

    future_work = str(data.get("future_work", "")).strip()
    references = data.get("references", [])

    if not isinstance(references, list):
        references = []

    # Normalize refs: keep first 3 non-empty strings
    refs_clean: List[str] = []
    for r in references:
        s = str(r).strip()
        if s:
            refs_clean.append(s)
        if len(refs_clean) >= 3:
            break

    # If LLM returned fewer than 3, pad with empty lines (do not invent)
    while len(refs_clean) < 3:
        refs_clean.append("")

    # HTML-escape to avoid breaking the page if MD contains angle brackets
    future_work_html = html.escape(future_work, quote=False)
    refs_html = "<br/>\n".join([html.escape(x, quote=False) for x in refs_clean if x is not None])

    # -------------------------
    # Read & update HTML
    # -------------------------
    web_file = Path(web_path)
    if not web_file.exists():
        raise FileNotFoundError(f"web_path does not exist: {web_path}")

    html_text = web_file.read_text(encoding="utf-8", errors="ignore")

    # Replace the <p>...</p> inside Future Work section
    fw_pattern = re.compile(
        r'(<section\s+id="Future Work"[^>]*>.*?<div[^>]*>.*?<p>\s*)(.*?)(\s*</p>)',
        flags=re.DOTALL
    )
    html_text, n_fw = fw_pattern.subn(rf"\1{future_work_html}\3", html_text, count=1)

    # Replace the <p>...</p> inside Reference section
    ref_pattern = re.compile(
        r'(<section\s+id="Reference"[^>]*>.*?<div[^>]*>.*?<p>\s*)(.*?)(\s*</p>)',
        flags=re.DOTALL
    )
    html_text, n_ref = ref_pattern.subn(rf"\1{refs_html}\3", html_text, count=1)

    if n_fw == 0:
        raise ValueError('Cannot locate the target Future Work block: <section id="Future Work" ...><p>...</p>')
    if n_ref == 0:
        raise ValueError('Cannot locate the target Reference block: <section id="Reference" ...><p>...</p>')

    web_file.write_text(html_text, encoding="utf-8")



# # ========== Screenshot HTML to PNG ==========
import os
import time
from pathlib import Path
from urllib.parse import urlparse

def html_to_png_screenshot(
    html_path_or_url: str,
    output_dir: str,
    filename: str = "screenshot.png",
    width: int = 1440,
    height: int = 900,
    scroll_step: int = 800,
    scroll_delay_ms: int = 120,
    post_scroll_wait_ms: int = 800,
):
    """
    Take a full-page screenshot of an HTML page (local file or URL),
    with auto-scrolling to trigger lazy-loaded content.

    Args:
        html_path_or_url: Local HTML path or http/https URL
        output_dir: Directory to save screenshot
        filename: Output image name (default: screenshot.png)
        width/height: Viewport size
        scroll_step: Pixels per scroll step
        scroll_delay_ms: Delay between scroll steps (ms)
        post_scroll_wait_ms: Extra wait after scrolling (ms)

    Returns:
        Absolute path of the saved screenshot
    """
    from playwright.sync_api import sync_playwright

    # ---------- output path ----------
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename

    # ---------- resolve URL ----------
    target = html_path_or_url.strip()
    if target.startswith("http://") or target.startswith("https://"):
        url = target
    else:
        html_file = Path(target).expanduser().resolve()
        if not html_file.exists():
            raise FileNotFoundError(f"HTML file not found: {html_file}")
        url = html_file.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": height})

        # 1️⃣ 加载页面（避免只等 networkidle 导致某些 JS 未执行）
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_load_state("networkidle")

        # 2️⃣ 禁用动画/过渡，避免滚动过程中高度变化
        page.add_style_tag(content="""
            *, *::before, *::after {
                animation: none !important;
                transition: none !important;
                scroll-behavior: auto !important;
            }
            html, body {
                background: white !important;
            }
        """)

        # 3️⃣ 自动滚动到页面底部，触发懒加载
        page.evaluate(
            """async ({step, delay}) => {
                const sleep = (ms) => new Promise(r => setTimeout(r, ms));
                const getHeight = () =>
                    Math.max(
                        document.body.scrollHeight,
                        document.documentElement.scrollHeight
                    );

                let lastHeight = -1;
                let currentHeight = getHeight();
                let y = 0;

                while (currentHeight !== lastHeight) {
                    lastHeight = currentHeight;
                    while (y < currentHeight) {
                        window.scrollTo(0, y);
                        y += step;
                        await sleep(delay);
                    }
                    await sleep(delay);
                    currentHeight = getHeight();
                }

                window.scrollTo(0, 0);
            }""",
            {"step": scroll_step, "delay": scroll_delay_ms},
        )

        # 4️⃣ 滚动后额外等待，确保图片/字体完成渲染
        if post_scroll_wait_ms > 0:
            page.wait_for_timeout(post_scroll_wait_ms)

        # 5️⃣ full-page 截图
        page.screenshot(path=str(out_path), full_page=True)

        browser.close()

    return str(out_path)


import os
import re
from bs4 import BeautifulSoup

def inject_contributions_section_into_web(markdown_path, web_path):
    """
    从Markdown文件中读取Contribution部分，并插入到HTML文件的指定位置。
    插入位置：<body> -> <main> -> <div class="space-y-24 py-20"> 的最前面。
    """
    
    # 1. 读取 Markdown 原文
    if not os.path.exists(markdown_path):
        print(f"Error: Markdown file not found at {markdown_path}")
        return
    
    with open(markdown_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 2. 提取 Contribution 部分 (Rule-based Regex extraction)
    # 匹配规则：以 ## Contribution (或 Contributions) 开头，直到遇到下一个 ## 标题或文件结束
    # re.IGNORECASE 忽略大小写
    pattern = r'##\s*Contributions?[\s\S]*?(?=\n##|\Z)'
    match = re.search(pattern, md_content, re.IGNORECASE)

    if not match:
        print("Warning: No 'Contribution' section found in Markdown file.")
        contribution_text = "No contribution content extracted."
    else:
        # 获取匹配到的完整文本
        raw_section = match.group(0)
        # 去掉第一行标题 (例如 "## Contribution")，保留正文
        contribution_text = re.sub(r'^##.*\n', '', raw_section, flags=re.MULTILINE).strip()

    # 3. 将 Markdown 文本转换为简单的 HTML 段落格式
    # 根据换行符分割段落，并用 <p> 标签包裹
    # 也可以使用 'import markdown'; html_content = markdown.markdown(contribution_text)
    paragraphs = contribution_text.split('\n\n')
    html_paragraphs = ""
    for p in paragraphs:
        clean_p = p.strip()
        if clean_p:
            html_paragraphs += f"<p>{clean_p}</p>\n"

    # 4. 构建符合目标 HTML 风格的 Section 结构
    # 样式参考了原文件中的 Introduction 和 Method 部分
    new_section_html = f"""
    <section id="Contribution" class="scroll-mt-24 reveal">
        <h2 class="font-display text-2xl md:text-3xl font-bold text-slate-800 mb-6 text-center">Contribution</h2>
        <div class="prose max-w-none mx-auto text-slate-600 text-base md:text-lg mb-10 text-center max-w-3xl">
            {html_paragraphs}
        </div>
    </section>
    """

    # 5. 读取并解析 HTML 文件
    if not os.path.exists(web_path):
        print(f"Error: HTML file not found at {web_path}")
        return

    with open(web_path, 'r', encoding='utf-8') as f:
        html_source = f.read()

    soup = BeautifulSoup(html_source, 'html.parser')

    # 6. 定位插入点
    # 目标：<body> -> <main> -> <div class="space-y-24 py-20">
    main_tag = soup.find('main')
    if main_tag:
        # 寻找 main 下面负责布局的 div
        container_div = main_tag.find('div', class_='space-y-24 py-20')
        
        if container_div:
            # 将新构建的 HTML 字符串转换为 BeautifulSoup Tag 对象
            new_tag = BeautifulSoup(new_section_html, 'html.parser')
            # BeautifulSoup 解析片段时可能会包含 html/body 标签，我们只取 section
            section_tag = new_tag.find('section')
            
            if section_tag:
                # 插入到 container_div 的第一个位置 (index 0)
                container_div.insert(0, section_tag)
                print("Successfully injected Contribution section.")
            else:
                 print("Error: Failed to create section tag.")
        else:
            print("Error: Target container <div class='space-y-24 py-20'> not found inside <main>.")
    else:
        print("Error: <main> tag not found in HTML.")

    # 7. 保存修改后的 HTML
    with open(web_path, 'w', encoding='utf-8') as f:
        # 使用 prettify() 格式化输出，或者直接 str(soup)
        f.write(str(soup))

# ========================= 以下为PR所用的函数 ==============================
def initialize_pr_markdown(
    basic_information_path: str,
    auto_path: str,
    pr_template_path: str
) -> str:
    """
    basic_information_path: txt 文件路径（包含 Title/Author/Institution/Github）
    auto_path: 输出目录
    pr_template_path: PR markdown 模板路径
    输出：auto_path/markdown.md
    """

    info_txt_path = Path(basic_information_path)
    auto_dir = Path(auto_path)
    template_md = Path(pr_template_path)

    if not info_txt_path.exists():
        raise FileNotFoundError(f"basic_information_path not found: {basic_information_path}")
    if info_txt_path.suffix.lower() != ".txt":
        raise ValueError(f"basic_information_path must be a .txt file, got: {basic_information_path}")
    if not template_md.exists():
        raise FileNotFoundError(f"pr_template_path not found: {pr_template_path}")

    auto_dir.mkdir(parents=True, exist_ok=True)

    # -------- 1) 读取并解析 txt --------
    txt = info_txt_path.read_text(encoding="utf-8", errors="ignore")

    def _extract_value(key: str) -> str:
        pattern = re.compile(
            rf"^\s*{re.escape(key)}\s*:\s*(.*?)\s*$",
            re.IGNORECASE | re.MULTILINE
        )
        m = pattern.search(txt)
        return m.group(1).strip() if m else ""

    title = _extract_value("Title")
    authors = _extract_value("Author") or _extract_value("Authors")
    institution = _extract_value("Institution")
    github = _extract_value("Github") or _extract_value("GitHub") or _extract_value("Direct Link")

    # -------- 2) 复制模板到 auto_path/markdown.md --------
    out_md_path = auto_dir / "markdown.md"
    shutil.copyfile(template_md, out_md_path)

    md = out_md_path.read_text(encoding="utf-8", errors="ignore")

    # 关键：统一换行，避免 \r\n 导致整行匹配失败
    md = md.replace("\r\n", "\n").replace("\r", "\n")

    # -------- 3) 按行替换（更鲁棒）--------
    def _fill_line_by_anchor(md_text: str, anchor_label: str, value: str) -> str:
        """
        anchor_label: 例如 "Authors"，用于匹配 **Authors**
        将包含 **anchor_label** 的那一行，改写为：
          <该行中 **anchor_label** 及其之前部分>: <value>
        例如：
          ✍️ **Authors**      -> ✍️ **Authors:** xxx
          🏛️**Institution**  -> 🏛️**Institution:** yyy
        """
        if not value:
            return md_text

        anchor = f"**{anchor_label}**"
        lines = md_text.split("\n")
        replaced = False

        for i, line in enumerate(lines):
            if anchor in line:
                # 保留锚点之前的所有内容 + 锚点本身
                left = line.split(anchor, 1)[0] + anchor
                lines[i] = f"{left}: {value}"
                replaced = True
                break

        return "\n".join(lines) if replaced else md_text

    md = _fill_line_by_anchor(md, "Authors", authors)
    md = _fill_line_by_anchor(md, "Direct Link", github)
    md = _fill_line_by_anchor(md, "Paper Title", title)
    md = _fill_line_by_anchor(md, "Institution", institution)

    out_md_path.write_text(md, encoding="utf-8")

    return str(out_md_path)



# 生成pr的内容部分
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests

# def generate_pr_from_dag(
#     dag_path: str,
#     pr_path: str,
#     generate_pr_prompt: str,
#     model: str = "gpt-4o-mini",
#     timeout: int = 120,
#     debug: bool = True,
#     max_retries: int = 5,
#     backoff_base: float = 1.5,
#     max_content_chars: int = 12000,
#     max_visuals: int = 20,
# ) -> None:
#     """
#     Generate a PR markdown from dag.json by iterating section nodes and querying an LLM.
#     Reads OPENAI_API_KEY and OPENAI_BASE_URL from environment variables.

#     Robustness additions:
#       - Truncate content and visual_node to reduce payload size
#       - requests.Session for keep-alive
#       - retries with exponential backoff for ConnectionError/RemoteDisconnected

#     PR template matching:
#       - Supports emoji/prefix before **Label**:
#         e.g. '🚀 **Core Methods**:' is recognized and replaced.
#       - Inserts appended content within the block (after label line, before next label).
#     """

#     # -------------------------
#     # 0) Env: API key & base url
#     # -------------------------
#     api_key = os.getenv("OPENAI_API_KEY", "").strip()
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY is not set")

#     base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").strip()
#     if not base_url:
#         base_url = "https://api.openai.com"
#     base_url = base_url.rstrip("/")
#     if base_url.endswith("/v1"):
#         base_url = base_url[:-3]
#     base_url = base_url.rstrip("/")

#     # -------------------------
#     # 1) Load DAG
#     # -------------------------
#     dag_obj = json.loads(Path(dag_path).read_text(encoding="utf-8"))
#     if isinstance(dag_obj, list):
#         nodes = dag_obj
#     elif isinstance(dag_obj, dict) and isinstance(dag_obj.get("nodes"), list):
#         nodes = dag_obj["nodes"]
#     else:
#         raise ValueError("Unsupported dag.json format (expect list or dict with 'nodes').")

#     if not nodes:
#         return

#     root = nodes[0]
#     root_edges = root.get("edge", [])
#     if not isinstance(root_edges, list):
#         return

#     name2node: Dict[str, Dict[str, Any]] = {}
#     for n in nodes:
#         nm = n.get("name")
#         if isinstance(nm, str) and nm:
#             name2node[nm] = n

#     section_nodes: List[Dict[str, Any]] = []
#     for sec_name in root_edges:
#         if sec_name in name2node:
#             section_nodes.append(name2node[sec_name])

#     if not section_nodes:
#         return

#     # -------------------------
#     # 2) Load PR markdown
#     # -------------------------
#     pr_file = Path(pr_path)
#     pr_text = pr_file.read_text(encoding="utf-8")

#     KNOWN_LABELS = ["Key Question", "Brilliant Idea", "Core Methods", "Core Results", "Significance/Impact"]

#     filled_key_question = False
#     filled_brilliant_idea = False
#     filled_significance = False
#     core_methods_inlined = False
#     core_results_inlined = False

#     # -------------------------
#     # 3) LLM call (OpenAI-compatible) with retries
#     # -------------------------
#     session = requests.Session()

#     def chat_complete(prompt: str, payload_meta: str = "") -> str:
#         url = f"{base_url}/v1/chat/completions"
#         headers = {
#             "Authorization": f"Bearer {api_key}",
#             "Content-Type": "application/json",
#             "Connection": "keep-alive",
#         }
#         payload = {
#             "model": model,
#             "messages": [
#                 {"role": "system", "content": "You are a precise scientific writing assistant."},
#                 {"role": "user", "content": prompt},
#             ],
#             "temperature": 0.4,
#         }

#         last_err: Optional[Exception] = None
#         for attempt in range(1, max_retries + 1):
#             try:
#                 resp = session.post(url, headers=headers, json=payload, timeout=timeout)
#                 if resp.status_code >= 400:
#                     raise RuntimeError(f"LLM request failed ({resp.status_code}): {resp.text[:2000]}")
#                 data = resp.json()
#                 return data["choices"][0]["message"]["content"].strip()
#             except (
#                 requests.exceptions.ConnectionError,
#                 requests.exceptions.Timeout,
#                 requests.exceptions.ChunkedEncodingError,
#             ) as e:
#                 last_err = e
#                 sleep_s = backoff_base ** (attempt - 1)
#                 time.sleep(sleep_s)
#             except Exception:
#                 raise

#         raise RuntimeError(f"LLM request failed after {max_retries} retries. Last error: {repr(last_err)}")

#     # -------------------------
#     # 4) Parse LLM output
#     # -------------------------
#     def parse_llm_output(text: str) -> Dict[str, Any]:
#         out: Dict[str, Any] = {
#             "type": "unknown",
#             "key_question": None,
#             "brilliant_idea": None,
#             "core_methods": None,
#             "core_results": None,
#             "significance": None,
#             "image": None,
#         }

#         img = re.search(r'!\[\]\(([^)]+)\)', text)
#         if img:
#             out["image"] = f"![]({img.group(1).strip()})"

#         def grab(label: str) -> Optional[str]:
#             m = re.search(
#                 rf"(?is)(?:^{re.escape(label)}\s*[:：]\s*)(.*?)(?=^\w[\w /-]*\s*[:：]|\Z)",
#                 text,
#                 flags=re.MULTILINE,
#             )
#             return m.group(1).strip() if m else None

#         out["key_question"] = grab("Key Question")
#         out["brilliant_idea"] = grab("Brilliant Idea")
#         out["core_methods"] = grab("Core Methods")
#         out["core_results"] = grab("Core Results")
#         out["significance"] = grab("Significance/Impact")

#         if out["key_question"] or out["brilliant_idea"]:
#             out["type"] = "intro"
#         elif out["core_methods"]:
#             out["type"] = "methods"
#         elif out["core_results"]:
#             out["type"] = "results"
#         elif out["significance"]:
#             out["type"] = "impact"

#         return out

#     # -------------------------
#     # 5) PR template helpers (emoji-safe + block insertion)
#     # -------------------------
#     def _find_header_line(md: str, label: str) -> Optional[re.Match]:
#         return re.search(rf"(?mi)^(?P<line>.*\*\*{re.escape(label)}\*\*.*)$", md)

#     def has_header(md: str, label: str) -> bool:
#         return _find_header_line(md, label) is not None

#     def set_inline(md: str, label: str, text: str) -> str:
#         m = _find_header_line(md, label)
#         if not m:
#             return md

#         line = m.group("line")
#         token_pat = re.compile(rf"\*\*{re.escape(label)}\*\*", re.I)
#         tm = token_pat.search(line)
#         if not tm:
#             return md

#         prefix = line[:tm.end()]  # includes **Label**
#         new_line = f"{prefix}: {text}".rstrip()

#         start, end = m.start("line"), m.end("line")
#         return md[:start] + new_line + md[end:]

#     def _next_header_pos(md: str, start_pos: int) -> int:
#         tail = md[start_pos:]
#         next_pos = None
#         for lab in KNOWN_LABELS:
#             mm = re.search(rf"(?mi)^\s*.*\*\*{re.escape(lab)}\*\*.*$", tail)
#             if mm:
#                 cand = start_pos + mm.start()
#                 if next_pos is None or cand < next_pos:
#                     next_pos = cand
#         return next_pos if next_pos is not None else len(md)

#     def insert_under_label_block(md: str, label: str, insertion_lines: List[str]) -> str:
#         m = _find_header_line(md, label)
#         if not m:
#             return md

#         insertion = "\n".join([ln for ln in insertion_lines if ln and ln.strip()]).rstrip()
#         if not insertion:
#             return md

#         hdr_end = m.end("line")
#         block_end = _next_header_pos(md, hdr_end)

#         before = md[:hdr_end]
#         middle = md[hdr_end:block_end]
#         after = md[block_end:]

#         if not before.endswith("\n"):
#             before += "\n"
#         if not middle.startswith("\n"):
#             middle = "\n" + middle

#         new_middle = "\n" + insertion + "\n" + middle.lstrip("\n")
#         return before + new_middle + after

#     # -------------------------
#     # 6) Main loop over sections
#     # -------------------------
#     for idx, sec in enumerate(section_nodes):
#         # Build node object for LLM, but truncate to reduce payload size
#         content = sec.get("content", "")
#         if not isinstance(content, str):
#             content = ""

#         visuals = sec.get("visual_node", [])
#         if not isinstance(visuals, list):
#             visuals = []

#         # Normalize visuals to strings if possible
#         visuals_norm: List[str] = []
#         for v in visuals:
#             if isinstance(v, str):
#                 visuals_norm.append(v)
#             elif isinstance(v, dict):
#                 for k in ("path", "src", "url", "md", "markdown"):
#                     if k in v and isinstance(v[k], str):
#                         visuals_norm.append(v[k])
#                         break

#         if len(content) > max_content_chars:
#             content = content[:max_content_chars] + "\n...(truncated)"

#         if len(visuals_norm) > max_visuals:
#             visuals_norm = visuals_norm[:max_visuals]

#         node_obj = {
#             "name": sec.get("name", ""),
#             "content": content,
#             "visual_node": visuals_norm,
#         }

#         # Build prompt
#         prompt = generate_pr_prompt.replace("{NODE_JSON}", json.dumps(node_obj, ensure_ascii=False))

#         # Call LLM with retry
#         llm_text = chat_complete(prompt, payload_meta=f"(section_idx={idx}, section_name={node_obj['name']})")

#         parsed = parse_llm_output(llm_text)

#         # Apply writing rules
#         if parsed["type"] == "intro":
#             kq = parsed.get("key_question")
#             bi = parsed.get("brilliant_idea")
#             img = parsed.get("image")

#             if kq and not filled_key_question:
#                 if has_header(pr_text, "Key Question"):
#                     pr_text = set_inline(pr_text, "Key Question", kq)
#                     filled_key_question = True

#             if bi and not filled_brilliant_idea:
#                 if has_header(pr_text, "Brilliant Idea"):
#                     pr_text = set_inline(pr_text, "Brilliant Idea", bi)
#                     if img:
#                         pr_text = insert_under_label_block(pr_text, "Brilliant Idea", [img])
#                     filled_brilliant_idea = True

#         elif parsed["type"] == "methods":
#             cm = parsed.get("core_methods")
#             img = parsed.get("image")

#             if not cm:
#                 continue
#             if not has_header(pr_text, "Core Methods"):
#                 continue

#             if not core_methods_inlined:
#                 pr_text = set_inline(pr_text, "Core Methods", cm)
#                 core_methods_inlined = True
#                 if img:
#                     pr_text = insert_under_label_block(pr_text, "Core Methods", [img])
#             else:
#                 lines = [cm] + ([img] if img else [])
#                 pr_text = insert_under_label_block(pr_text, "Core Methods", lines)

#         elif parsed["type"] == "results":
#             cr = parsed.get("core_results")
#             img = parsed.get("image")

#             if not cr:
#                 continue
#             if not has_header(pr_text, "Core Results"):
#                 continue

#             if not core_results_inlined:
#                 pr_text = set_inline(pr_text, "Core Results", cr)
#                 core_results_inlined = True
#                 if img:
#                     pr_text = insert_under_label_block(pr_text, "Core Results", [img])
#             else:
#                 lines = [cr] + ([img] if img else [])
#                 pr_text = insert_under_label_block(pr_text, "Core Results", lines)

#         elif parsed["type"] == "impact":
#             si = parsed.get("significance")
#             if not si:
#                 continue
#             if filled_significance:
#                 continue
#             if has_header(pr_text, "Significance/Impact"):
#                 pr_text = set_inline(pr_text, "Significance/Impact", si)
#                 filled_significance = True

#         else:
#             pass

#     # -------------------------
#     # 7) Save
#     # -------------------------
#     pr_file.write_text(pr_text, encoding="utf-8")

def generate_pr_from_dag(
    dag_path: str,
    pr_path: str,
    generate_pr_prompt: str,
    model: str = "gpt-4o-mini",
    timeout: int = 120,
    debug: bool = True,
    max_retries: int = 5,
    backoff_base: float = 1.5,
    max_content_chars: int = 12000,
    max_visuals: int = 20,
) -> None:
    """
    Generate a PR markdown from dag.json by iterating section nodes and querying an LLM.
    Reads OPENAI_API_KEY and OPENAI_BASE_URL from environment variables.

    Robustness:
      - Truncate content and visual_node to reduce payload size
      - requests.Session keep-alive
      - retries with exponential backoff on connection errors
      - debug prints for payload and writes

    PR template matching:
      - Supports emoji/prefix before **Label**: e.g. '🚀 **Core Methods**:'
      - IMPORTANT: Appends new content at the END of the label block (before next header),
        so repeated (text,image) pairs become:
          text1
          img1
          text2
          img2
    """

    def log(msg: str) -> None:
        if debug:
            print(msg)

    # -------------------------
    # 0) Env: API key & base url
    # -------------------------
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").strip()
    if not base_url:
        base_url = "https://api.openai.com"
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    base_url = base_url.rstrip("/")

    log(f"[INFO] Using base_url = {base_url}")
    log(f"[INFO] Using model = {model}")

    # -------------------------
    # 1) Load DAG
    # -------------------------
    dag_obj = json.loads(Path(dag_path).read_text(encoding="utf-8"))
    if isinstance(dag_obj, list):
        nodes = dag_obj
    elif isinstance(dag_obj, dict) and isinstance(dag_obj.get("nodes"), list):
        nodes = dag_obj["nodes"]
    else:
        raise ValueError("Unsupported dag.json format (expect list or dict with 'nodes').")

    if not nodes:
        log("[WARN] dag.json has no nodes; exiting.")
        return

    root = nodes[0]
    root_edges = root.get("edge", [])
    if not isinstance(root_edges, list):
        log("[WARN] Root node edge is not a list; exiting.")
        return

    log(f"[INFO] Root node name = {root.get('name')}")
    log(f"[INFO] Root edges (section names) = {root_edges}")

    name2node: Dict[str, Dict[str, Any]] = {}
    for n in nodes:
        nm = n.get("name")
        if isinstance(nm, str) and nm:
            name2node[nm] = n

    section_nodes: List[Dict[str, Any]] = []
    for sec_name in root_edges:
        if sec_name in name2node:
            section_nodes.append(name2node[sec_name])
        else:
            log(f"[WARN] Section node '{sec_name}' not found in DAG; skipped.")

    log(f"[INFO] Resolved {len(section_nodes)} section nodes.")
    if not section_nodes:
        log("[WARN] No section nodes to process; exiting.")
        return

    # -------------------------
    # 2) Load PR markdown
    # -------------------------
    pr_file = Path(pr_path)
    pr_text = pr_file.read_text(encoding="utf-8")

    KNOWN_LABELS = ["Key Question", "Brilliant Idea", "Core Methods", "Core Results", "Significance/Impact"]

    if debug:
        log("[INFO] PR template header scan:")
        for lab in KNOWN_LABELS:
            found = bool(re.search(rf"(?mi)^.*\*\*{re.escape(lab)}\*\*.*$", pr_text))
            log(f"  - {lab}: found={found}")

    filled_key_question = False
    filled_brilliant_idea = False
    filled_significance = False
    core_methods_inlined = False
    core_results_inlined = False

    # -------------------------
    # 3) LLM call (OpenAI-compatible) with retries
    # -------------------------
    session = requests.Session()

    def chat_complete(prompt: str, payload_meta: str = "") -> str:
        url = f"{base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise scientific writing assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.4,
        }

        if debug:
            try:
                approx = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                log(f"[HTTP] POST {url} approx_payload_bytes={approx} {payload_meta}")
            except Exception:
                pass

        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = session.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code >= 400:
                    raise RuntimeError(f"LLM request failed ({resp.status_code}): {resp.text[:2000]}")
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.ChunkedEncodingError) as e:
                last_err = e
                sleep_s = backoff_base ** (attempt - 1)
                log(f"[HTTP-RETRY] attempt {attempt}/{max_retries} error: {repr(e)}; sleep {sleep_s:.2f}s {payload_meta}")
                time.sleep(sleep_s)
            except Exception:
                raise

        raise RuntimeError(f"LLM request failed after {max_retries} retries. Last error: {repr(last_err)}")

    # -------------------------
    # 4) Parse LLM output
    # -------------------------
    def parse_llm_output(text: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "type": "unknown",
            "key_question": None,
            "brilliant_idea": None,
            "core_methods": None,
            "core_results": None,
            "significance": None,
            "image": None,
        }

        img = re.search(r'!\[\]\(([^)]+)\)', text)
        if img:
            out["image"] = f"![]({img.group(1).strip()})"

        def grab(label: str) -> Optional[str]:
            m = re.search(
                rf"(?is)(?:^{re.escape(label)}\s*[:：]\s*)(.*?)(?=^\w[\w /-]*\s*[:：]|\Z)",
                text,
                flags=re.MULTILINE,
            )
            return m.group(1).strip() if m else None

        out["key_question"] = grab("Key Question")
        out["brilliant_idea"] = grab("Brilliant Idea")
        out["core_methods"] = grab("Core Methods")
        out["core_results"] = grab("Core Results")
        out["significance"] = grab("Significance/Impact")

        if out["key_question"] or out["brilliant_idea"]:
            out["type"] = "intro"
        elif out["core_methods"]:
            out["type"] = "methods"
        elif out["core_results"]:
            out["type"] = "results"
        elif out["significance"]:
            out["type"] = "impact"

        return out

    # -------------------------
    # 5) PR template helpers (emoji-safe + APPEND inside block)
    # -------------------------
    def _find_header_line(md: str, label: str) -> Optional[re.Match]:
        # match line containing **label** with possible emoji/prefix
        return re.search(rf"(?mi)^(?P<line>.*\*\*{re.escape(label)}\*\*.*)$", md)

    def has_header(md: str, label: str) -> bool:
        found = _find_header_line(md, label) is not None
        log(f"[CHECK] Header '{label}' found = {found}")
        return found

    def set_inline(md: str, label: str, text: str) -> str:
        m = _find_header_line(md, label)
        if not m:
            log(f"[SKIP] set_inline: header '{label}' not found")
            return md

        line = m.group("line")
        token_pat = re.compile(rf"\*\*{re.escape(label)}\*\*", re.I)
        tm = token_pat.search(line)
        if not tm:
            log(f"[SKIP] set_inline: token '**{label}**' not found in line")
            return md

        prefix = line[:tm.end()]  # includes **Label**
        new_line = f"{prefix}: {text}".rstrip()

        start, end = m.start("line"), m.end("line")
        return md[:start] + new_line + md[end:]

    def _next_header_pos(md: str, start_pos: int) -> int:
        tail = md[start_pos:]
        next_pos = None
        for lab in KNOWN_LABELS:
            mm = re.search(rf"(?mi)^\s*.*\*\*{re.escape(lab)}\*\*.*$", tail)
            if mm:
                cand = start_pos + mm.start()
                if next_pos is None or cand < next_pos:
                    next_pos = cand
        return next_pos if next_pos is not None else len(md)

    def append_to_label_block(md: str, label: str, insertion_lines: List[str]) -> str:
        """
        APPEND insertion to the END of the label block (before the next header),
        preserving order across multiple appends:
          text1,img1 then text2,img2 ...
        """
        m = _find_header_line(md, label)
        if not m:
            log(f"[SKIP] append_to_label_block: header '{label}' not found")
            return md

        insertion = "\n".join([ln for ln in insertion_lines if ln and ln.strip()]).rstrip()
        if not insertion:
            log(f"[SKIP] append_to_label_block: empty insertion for '{label}'")
            return md

        hdr_end = m.end("line")
        block_end = _next_header_pos(md, hdr_end)

        before = md[:hdr_end]
        middle = md[hdr_end:block_end]
        after = md[block_end:]

        # Normalize spacing inside the block so append is clean.
        # Ensure middle starts with newline
        if not before.endswith("\n"):
            before += "\n"
        if middle and not middle.startswith("\n"):
            middle = "\n" + middle

        # Ensure the block (middle) ends with at least one blank line before appending
        middle_stripped_right = middle.rstrip("\n")
        if middle_stripped_right.strip() == "":
            # block currently empty/whitespace
            new_middle = "\n" + insertion + "\n"
        else:
            # ensure exactly one blank line separation
            new_middle = middle_stripped_right + "\n\n" + insertion + "\n"

        return before + new_middle + after

    # -------------------------
    # 6) Main loop over sections
    # -------------------------
    for idx, sec in enumerate(section_nodes):
        log("\n" + "=" * 90)
        log(f"[SECTION {idx}] name = {sec.get('name')}")

        # Truncate payload to reduce disconnect risk
        content = sec.get("content", "")
        if not isinstance(content, str):
            content = ""

        visuals = sec.get("visual_node", [])
        if not isinstance(visuals, list):
            visuals = []

        visuals_norm: List[str] = []
        for v in visuals:
            if isinstance(v, str):
                visuals_norm.append(v)
            elif isinstance(v, dict):
                for k in ("path", "src", "url", "md", "markdown"):
                    if k in v and isinstance(v[k], str):
                        visuals_norm.append(v[k])
                        break

        if len(content) > max_content_chars:
            content = content[:max_content_chars] + "\n...(truncated)"

        if len(visuals_norm) > max_visuals:
            visuals_norm = visuals_norm[:max_visuals]

        node_obj = {
            "name": sec.get("name", ""),
            "content": content,
            "visual_node": visuals_norm,
        }

        if debug:
            log(f"[PAYLOAD] content_chars={len(node_obj['content'])} visuals={len(node_obj['visual_node'])}")
            preview = dict(node_obj)
            if len(preview["content"]) > 800:
                preview["content"] = preview["content"][:800] + "...(truncated)"
            log("[SEND TO LLM] Node payload preview:")
            print(json.dumps(preview, indent=2, ensure_ascii=False))

        prompt = generate_pr_prompt.replace("{NODE_JSON}", json.dumps(node_obj, ensure_ascii=False))
        llm_text = chat_complete(prompt, payload_meta=f"(section_idx={idx}, section_name={node_obj['name']})")

        log("\n[LLM RAW OUTPUT]")
        print(llm_text)

        parsed = parse_llm_output(llm_text)
        log("\n[PARSED OUTPUT]")
        print(json.dumps(parsed, indent=2, ensure_ascii=False))

        # Apply writing rules
        if parsed["type"] == "intro":
            log("[TYPE] Introduction-like")
            kq = parsed.get("key_question")
            bi = parsed.get("brilliant_idea")
            img = parsed.get("image")

            if kq and not filled_key_question:
                if has_header(pr_text, "Key Question"):
                    pr_text = set_inline(pr_text, "Key Question", kq)
                    filled_key_question = True
                    log("[WRITE] Key Question filled (first time).")
                else:
                    log("[MISS] PR has no Key Question header.")
            else:
                log("[SKIP] Key Question ignored (empty or already filled).")

            if bi and not filled_brilliant_idea:
                if has_header(pr_text, "Brilliant Idea"):
                    pr_text = set_inline(pr_text, "Brilliant Idea", bi)
                    if img:
                        # For Brilliant Idea, just append image (it should appear right after idea line content)
                        pr_text = append_to_label_block(pr_text, "Brilliant Idea", [img])
                    filled_brilliant_idea = True
                    log("[WRITE] Brilliant Idea filled (first time).")
                else:
                    log("[MISS] PR has no Brilliant Idea header.")
            else:
                log("[SKIP] Brilliant Idea ignored (empty or already filled).")

        elif parsed["type"] == "methods":
            log("[TYPE] Methods-like")
            cm = parsed.get("core_methods")
            img = parsed.get("image")

            if not cm:
                log("[SKIP] Core Methods empty.")
                continue
            if not has_header(pr_text, "Core Methods"):
                log("[MISS] PR has no Core Methods header.")
                continue

            if not core_methods_inlined:
                pr_text = set_inline(pr_text, "Core Methods", cm)
                core_methods_inlined = True
                log("[WRITE] Core Methods inlined (first time).")
                if img:
                    pr_text = append_to_label_block(pr_text, "Core Methods", [img])
                    log("[WRITE] Core Methods image appended.")
            else:
                # IMPORTANT: ensure order is text then its image for each append
                lines = [cm] + ([img] if img else [])
                pr_text = append_to_label_block(pr_text, "Core Methods", lines)
                log("[WRITE] Core Methods appended as (text, image) pair.")

        elif parsed["type"] == "results":
            log("[TYPE] Results-like")
            cr = parsed.get("core_results")
            img = parsed.get("image")

            if not cr:
                log("[SKIP] Core Results empty.")
                continue
            if not has_header(pr_text, "Core Results"):
                log("[MISS] PR has no Core Results header.")
                continue

            if not core_results_inlined:
                pr_text = set_inline(pr_text, "Core Results", cr)
                core_results_inlined = True
                log("[WRITE] Core Results inlined (first time).")
                if img:
                    pr_text = append_to_label_block(pr_text, "Core Results", [img])
                    log("[WRITE] Core Results image appended.")
            else:
                lines = [cr] + ([img] if img else [])
                pr_text = append_to_label_block(pr_text, "Core Results", lines)
                log("[WRITE] Core Results appended as (text, image) pair.")

        elif parsed["type"] == "impact":
            log("[TYPE] Impact-like")
            si = parsed.get("significance")
            if not si:
                log("[SKIP] Significance/Impact empty.")
                continue
            if filled_significance:
                log("[SKIP] Significance/Impact ignored (already filled).")
                continue
            if has_header(pr_text, "Significance/Impact"):
                pr_text = set_inline(pr_text, "Significance/Impact", si)
                filled_significance = True
                log("[WRITE] Significance/Impact filled (first time).")
            else:
                log("[MISS] PR has no Significance/Impact header.")

        else:
            log("[WARN] Unknown section type; ignored.")

    # -------------------------
    # 7) Save
    # -------------------------
    pr_file.write_text(pr_text, encoding="utf-8")
    log("\n[SAVED] PR markdown updated in-place.")


# ================================增加标题和hashtag=================================
def add_title_and_hashtag(pr_path: str, add_title_and_hashtag_prompt: str, model: str = "gpt-4o-mini") -> None:
    """
    1) Read markdown from pr_path.
    2) Send it to LLM using add_title_and_hashtag_prompt (expects {MD_TEXT} placeholder).
    3) Parse LLM output:
        Title: ...
        Specific Tag: #A #B #C
        Community Tag: #X
    4) Update:
       - First line: from "#" to "# {Title}" (IMPORTANT: keep '# ' with a space)
       - Update "Specific:" line by locating "Specific:" in pr_path, and writing the 3 tags to the right side
       - Community line: replace the single tag after "Community:" with the LLM community tag
    5) Write back to pr_path in-place.
    """

    # -------------------------
    # 0) Read markdown
    # -------------------------
    pr_file = Path(pr_path)
    if not pr_file.exists():
        raise FileNotFoundError(f"pr_path not found: {pr_path}")

    md_text = pr_file.read_text(encoding="utf-8")

    # -------------------------
    # 1) Call LLM
    # -------------------------
    prompt = add_title_and_hashtag_prompt.replace("{MD_TEXT}", md_text)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Missing environment variable: OPENAI_API_KEY")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()  # optional (third-party proxy)
    try:
        from openai import OpenAI
    except Exception as e:
        raise ImportError("openai package is required. Install with: pip install openai") from e

    client = OpenAI(api_key=api_key, base_url=base_url or None)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise scientific social media copywriter."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.6,
    )

    llm_out = (resp.choices[0].message.content or "").strip()
    if not llm_out:
        raise ValueError("LLM returned empty content.")

    # -------------------------
    # 2) Parse LLM output
    # -------------------------
    title, specific_tags, community_tag = _parse_title_and_tags(llm_out)

    # -------------------------
    # 3) Update first line: "# {Title}"
    # -------------------------
    lines = md_text.splitlines(True)  # keep line endings
    if not lines:
        raise ValueError("Markdown file is empty.")

    first_line = lines[0].rstrip("\r\n")
    if re.fullmatch(r"#\s*", first_line):
        lines[0] = f"# 🔥{title}😯{_line_ending(lines[0])}"
    else:
        if first_line.startswith("#"):
            lines[0] = re.sub(r"^#\s*.*$", f"# 🔥{title}😯", first_line) + _line_ending(lines[0])
        else:
            lines.insert(0, f"# 🔥{title}😯\n")

    updated = "".join(lines)

    # -------------------------
    # 4) Replace "Specific:" line (locate by "Specific:", not by the whole Hashtag block)
    # -------------------------
    # Examples supported:
    #   Specific: #Tag1 #Tag2 #Tag3;
    #   Specific: ;
    #   Specific: (anything...) ;
    #
    # We preserve the trailing semicolon if it exists; otherwise we don't force it.
    def _replace_specific_line(match: re.Match) -> str:
        prefix = match.group(1)  # "Specific:" + spaces
        tail = match.group(2) or ""  # includes possible ';' and rest-of-line
        # Keep existing semicolon if present in tail, else keep tail as-is after tags
        # If tail starts with ';' or contains ';' at end, we keep one ';' at end.
        has_semicolon = ";" in tail
        end = ";" if has_semicolon else ""
        return f"{prefix}{specific_tags[0]} {specific_tags[1]} {specific_tags[2]}{end}"

    # Match a whole line beginning with "Specific:"
    updated, n1 = re.subn(
        r"(?mi)^(Specific:\s*)(.*)$",
        lambda m: _replace_specific_line(m),
        updated,
        count=1,
    )

    # -------------------------
    # 5) Replace "Community:" line (same as before)
    # -------------------------
    updated, n2 = re.subn(
        r"(Community:\s*)(#[^\s;]+)",
        lambda m: f"{m.group(1)}{community_tag}",
        updated,
        count=1,
        flags=re.IGNORECASE,
    )

    if n1 == 0:
        raise ValueError("Could not find a line starting with 'Specific:' to replace.")
    if n2 == 0:
        raise ValueError("Could not find the 'Community: #Tag1' pattern to replace.")

    # -------------------------
    # 6) Write back
    # -------------------------
    pr_file.write_text(updated, encoding="utf-8")


def _parse_title_and_tags(llm_out: str) -> Tuple[str, List[str], str]:
    """
    Parse:
      Title: ...
      Specific Tag: #A #B #C
      Community Tag: #X
    """
    def pick_line(prefix: str) -> str:
        m = re.search(rf"^{re.escape(prefix)}\s*(.+)$", llm_out, flags=re.MULTILINE)
        if not m:
            raise ValueError(f"LLM output missing line: '{prefix} ...'")
        return m.group(1).strip()

    title = pick_line("Title:")
    spec = pick_line("Specific Tag:")
    comm = pick_line("Community Tag:")

    spec_tags = re.findall(r"#[A-Za-z0-9_]+", spec)
    comm_tags = re.findall(r"#[A-Za-z0-9_]+", comm)

    if len(spec_tags) != 3:
        raise ValueError(f"Expected exactly 3 specific tags, got {len(spec_tags)}. Raw: {spec}")
    if len(comm_tags) != 1:
        raise ValueError(f"Expected exactly 1 community tag, got {len(comm_tags)}. Raw: {comm}")

    title = title.strip().strip('"').strip("'")
    if not title:
        raise ValueError("Parsed title is empty.")

    return title, spec_tags, comm_tags[0]


def _line_ending(original_line: str) -> str:
    if original_line.endswith("\r\n"):
        return "\r\n"
    if original_line.endswith("\n"):
        return "\n"
    return "\n"




# ===========================增加机构的tag===============================
def add_institution_tag(pr_path: str) -> None:
    """
    读取 markdown 中 🏛️**Institution**: 后的所有机构名（鲁棒分割），然后：
    1) 将所有机构名拼成一行，写入 Strategic Mentions：# 后面，例如：
       Strategic Mentions：# NVIDIA, Tel Aviv University
    2) 在 Strategic Mentions 这一行的下一行追加一行：
       @NVIDIA, Tel Aviv University
    原地修改文件。
    """

    md_path = Path(pr_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {pr_path}")

    text = md_path.read_text(encoding="utf-8")

    # -------------------------------------------------
    # 1) 提取 Institution 行内容
    # -------------------------------------------------
    institution_pattern = re.compile(r"🏛️\s*\*\*Institution\*\*\s*:\s*(.+)")
    m = institution_pattern.search(text)
    if not m:
        raise ValueError("Institution section not found in markdown.")

    institution_raw = m.group(1).strip()

    # -------------------------------------------------
    # 2) 鲁棒切分：提取所有机构名
    #    支持中英文分号/逗号/顿号/竖线/斜杠/换行/制表符等
    # -------------------------------------------------
    split_pattern = re.compile(r"\s*(?:;|；|,|，|、|\||｜|/|\n|\t)\s*")
    parts = [p.strip() for p in split_pattern.split(institution_raw) if p.strip()]
    if not parts:
        raise ValueError("Institution content is empty after parsing.")

    # 去除常见尾部标点噪声，并保持原顺序去重
    seen = set()
    institutions = []
    for p in parts:
        p = p.strip(" .。")
        if p and p not in seen:
            institutions.append(p)
            seen.add(p)

    if not institutions:
        raise ValueError("No valid institution names parsed.")

    institutions_str = ", ".join(institutions)

    # -------------------------------------------------
    # 3) 替换 Strategic Mentions：# 行，并在其下一行插入 @...
    #    仅处理第一次出现
    # -------------------------------------------------
    strategic_line_pattern = re.compile(r"^(Strategic Mentions：\s*#).*$", re.MULTILINE)

    def repl(mm: re.Match) -> str:
        prefix = mm.group(1)
        return f"{prefix} {institutions_str}\n@{institutions_str}"

    new_text, n = strategic_line_pattern.subn(repl, text, count=1)
    if n == 0:
        raise ValueError("Strategic Mentions section not found in markdown.")

    md_path.write_text(new_text, encoding="utf-8")


# =======================删除重复图片引用======================
def dedup_consecutive_markdown_images(md_path: str, inplace: bool = True) -> Tuple[str, int]:
    """
    去除 Markdown 中“连续出现”的相同图片引用（按 src 判重），只保留一个。
    连续的定义：两张图片之间只要是空白字符（空格/Tab/换行）也视为连续；
              若中间出现任何非空白内容（文字、代码、列表符号等），则不算连续。

    支持：
      ![](path)
      ![alt](path)
      ![alt](path "title")

    返回：
      (new_text, removed_count)
    """

    p = Path(md_path)
    text = p.read_text(encoding="utf-8")

    img_pat = re.compile(
        r'!\[(?P<alt>[^\]]*)\]\((?P<src>[^)\s]+)(?:\s+"[^"]*")?\)',
        flags=re.MULTILINE
    )

    parts = []
    last = 0
    for m in img_pat.finditer(text):
        if m.start() > last:
            parts.append(("text", text[last:m.start()]))
        parts.append(("img", m.group(0), m.group("src")))
        last = m.end()
    if last < len(text):
        parts.append(("text", text[last:]))

    removed = 0
    new_parts = []

    prev_img_src = None
    # 只有遇到“非空白文本”才会真正打断连续性
    saw_non_whitespace_since_prev_img = True  # 初始视为打断状态

    for part in parts:
        if part[0] == "img":
            _, raw_img, src = part

            # 连续重复判定：上一张图存在 + 中间没有非空白内容 + src 相同
            if prev_img_src is not None and (not saw_non_whitespace_since_prev_img) and prev_img_src == src:
                removed += 1
                # 删除该图片（不追加）
                continue

            # 保留该图片
            new_parts.append(raw_img)
            prev_img_src = src
            saw_non_whitespace_since_prev_img = False

        else:
            _, raw_text = part
            new_parts.append(raw_text)

            # 只有出现非空白，才视为打断连续图片序列
            if raw_text.strip() != "":
                saw_non_whitespace_since_prev_img = True

    new_text = "".join(new_parts)

    if inplace and removed > 0:
        p.write_text(new_text, encoding="utf-8")

    return new_text, removed


def refinement_pr(pr_path: str, pr_refine_path: str, prompts: dict, model: str):
    """
    提取Markdown中的特定章节，使用LLM根据传入的prompts指令进行优化，并重组文件。
    严格保留Markdown原有结构、图片引用以及未被选中的尾部内容（如Hashtags）。

    Args:
        pr_path (str): 原始Markdown文件路径
        pr_refine_path (str): 输出Markdown文件路径
        prompts (dict): 修改指令字典
        model (str): 模型名称
    """
    
    # 1. 检查环境变量并初始化客户端
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        raise ValueError("环境变量 OPENAI_API_KEY 未设置")
        
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 2. 读取原始文件
    if not os.path.exists(pr_path):
        raise FileNotFoundError(f"文件未找到: {pr_path}")
        
    with open(pr_path, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # 3. 定义部分标题映射 (需要修改的5个核心部分)
    section_headers = {
        "Key Question": r"🔍 \*\*Key Question\*\*",
        "Brilliant Idea": r"💡 \*\*Brilliant Idea\*\*",
        "Core Methods": r"🚀 \*\*Core Methods\*\*",
        "Core Results": r"📊 \*\*Core Results\*\*",
        "Significance/Impact": r"🧠 \*\*Significance/Impact\*\*"
    }
    
    # === 新增逻辑：定义结束边界标记 ===
    # 这是不需要被发送给LLM修改的部分的开始标记
    footer_pattern = r"🏷️\s*\*\*Hashtag\*\*" 

    # 4. 定位 核心标题 位置
    matches = []
    for key, pattern in section_headers.items():
        # 使用 finditer 以防同一个标题出现多次（虽然一般只会由于误操作出现，取第一个即可）
        found = list(re.finditer(pattern, original_content))
        if found:
            # 只取第一个匹配项
            match = found[0]
            matches.append({
                "key": key,
                "header_start": match.start(),
                "header_end": match.end(),
                "header_text": match.group()
            })
    
    # 按在文中出现的顺序排序
    matches.sort(key=lambda x: x["header_start"])
    
    if not matches:
        print("未检测到目标章节，直接复制文件。")
        with open(pr_refine_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        return

    # === 关键修正：定位 Footer (Hashtag) 位置 ===
    # 搜索整个文档中的 Hashtag 标记
    footer_match = re.search(footer_pattern, original_content)
    
    # 如果找到了 Hashtag，结束位置就是 Hashtag 的开始；否则（容错）设为文件末尾
    if footer_match:
        global_content_end_limit = footer_match.start()
    else:
        # 如果文件中没有 Hashtag 标签，回退到读取到文件末尾
        print("Warning: 未检测到 '🏷️ **Hashtag**' 标记，最后一个章节将读取至文件末尾。")
        global_content_end_limit = len(original_content)

    # 5. 精确计算每个章节的“内容”范围 (Content Range)
    content_ranges = {} 
    
    for i, match in enumerate(matches):
        key = match["key"]
        content_start = match["header_end"]
        
        if i < len(matches) - 1:
            # 如果不是最后一个 Header，内容结束于下一个 Header 的开始
            content_end = matches[i+1]["header_start"]
        else:
            # === 关键修正：处理最后一个章节 (Significance/Impact) ===
            # 内容结束于 Hashtag 标记的开始位置
            # 注意：我们要确保结束位置不早于开始位置（防止文件结构错乱）
            content_end = max(content_start, global_content_end_limit)
        
        content_ranges[key] = {
            "start": content_start,
            "end": content_end,
            "text": original_content[content_start:content_end].strip()
        }

    # 6. 构建 LLM 请求
    extracted_data = {k: v["text"] for k, v in content_ranges.items()}
    
    # 调试打印，确保提取正确
    # print("Extracted keys:", extracted_data.keys()) 
    
    system_prompt = (
        "You are an expert academic editor. Your task is to refine the content of specific sections of a paper summary based on user instructions.\n"
        "Input Format: JSON object {Section Name: Content}.\n"
        "Output Format: JSON object {Section Name: Refined Content}.\n"
        "CRITICAL RULES:\n"
        "1. **KEYS**: Keep the JSON keys EXACTLY the same as the input. Do NOT modify or translate the keys.\n"
        "2. **PURE BODY TEXT**: The output value must be pure body text (paragraphs or bullet points). \n"
        "   - **ABSOLUTELY NO HEADERS**: Do NOT use Markdown headers (e.g., #, ##, ###).\n"
        "   - **NO SUB-TITLES**: Do NOT create bolded sub-titles or independent lines that act as headings.\n"
        "   - **NO REPETITION**: Do NOT repeat the Section Name (Key) at the beginning.\n"
        "3. **IMAGES**: Do NOT remove, reorder, or modify any markdown image links (e.g., ![](img/...)). Keep them exactly where they are relative to the text.\n"
        "4. **JSON ONLY**: Output pure JSON string. No markdown formatting (```json).\n"
        "5. **FORMAT**: Use bolding (**text**) ONLY for emphasizing keywords within sentences, NOT for structure."
    )
    
    user_message = f"""
    [Refinement Instructions]
    {json.dumps(prompts, ensure_ascii=False)}

    [Content to Refine]
    {json.dumps(extracted_data, ensure_ascii=False)}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.2 
        )
        llm_output = response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"LLM API 调用失败: {e}")
        return

    # 7. 清洗 LLM 返回的 JSON
    try:
        cleaned_output = llm_output.replace("```json", "").replace("```", "").strip()
        refined_data = json.loads(cleaned_output)
    except json.JSONDecodeError:
        print("解析 LLM 返回的 JSON 失败。Raw output:", llm_output)
        return

    # 8. 重组文件
    new_file_parts = []
    current_idx = 0
    
    sorted_keys = [m["key"] for m in matches]
    
    for key in sorted_keys:
        range_info = content_ranges[key]
        c_start = range_info["start"]
        c_end = range_info["end"]
        
        # 1. 拼接未修改部分 (上一个节点结束 到 当前节点内容开始)
        pre_content = original_content[current_idx:c_start]
        new_file_parts.append(pre_content)
        
        # 2. 拼接新内容
        if key in refined_data:
            new_text = refined_data[key]
            
            # 简单格式处理：确保换行
            if new_file_parts[-1] and not new_file_parts[-1].endswith('\n'):
                new_text = "\n" + new_text
            new_text = "\n" + new_text.strip() + "\n"
            
            new_file_parts.append(new_text)
        else:
            new_file_parts.append(original_content[c_start:c_end])

        # 强制再换行
        # if not new_file_parts[-1].endswith('\n\n'):
        new_file_parts.append('\n')
            
        # 3. 更新游标
        current_idx = c_end

    # 9. 添加文件剩余的所有内容
    # 这里的 current_idx 现在等于 global_content_end_limit (即 Hashtag 的位置)
    # 所以这里会把 Hashtag 以及之后所有的内容原封不动地接在后面
    new_file_parts.append(original_content[current_idx:])
    
    final_markdown = "".join(new_file_parts)
    
    # 10. 保存结果
    os.makedirs(os.path.dirname(os.path.abspath(pr_refine_path)), exist_ok=True)
    
    with open(pr_refine_path, 'w', encoding='utf-8') as f:
        f.write(final_markdown)
        
    print(f"文件优化完成，已保存至: {pr_refine_path}")






# ========== 主流程 ==========
def main():
    # 输入总文件夹路径（包含多个论文子文件夹）
    root_folder = "mineru_outputs"  # ⚠️ 修改为你的实际路径
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
        outline = outline_initialize_with_gpt(dag_json_path=dag_path,outline_initialize_prompt=outline_initialize_prompt,model="gpt-4o")
        
        # ===  生成完整ouline ===
        outline_data=generate_complete_outline(selected_node_path,outline_path,generate_complete_outline_prompt,model="gpt-4o")
        
        # ===  配模板 ===
        arrange_template_with_gpt(outline_path,arrange_template_prompt,model="gpt-4o")
        
        # ===  生成最终的PPT ===
        ppt_template_path="./ppt_template_lzt"
        generate_ppt_with_gemini_prompt = {"role":"system","content":"You are given (1) a slide node (JSON) and (2) an HTML slide template. Your task: revise the HTML template to produce the final slide HTML using the node content. Node fields: text (slide textual content), figure (list of images to display, each has name, caption, resolution), formula (list of formula images to display, each has name, caption, resolution), template (template filename). IMPORTANT RULES: 1) Only modify places in the HTML that are marked by comments like <!-- You need to revise the following parts. X: ... -->. 2) For 'Subjects' sections: replace the placeholder title with ONE concise summary sentence for this slide. 3) For 'Image' sections (<img ...>): replace src with the relative path extracted from node.figure/formula[i].name; the node image name may be markdown like '![](images/abc.jpg)', use only 'images/abc.jpg'. 4) For 'Text' sections: replace the placeholder text with the node.text content, formatted cleanly in HTML; keep it readable and you may use <p>, <ul><li>, <br/> appropriately. 5) If the template expects more images/text blocks than provided by the node, leave the missing positions unchanged and do not invent content. 6) If the node provides more images than the template has slots, fill slots in order and ignore the rest. 7) Preserve all other HTML, CSS, and structure exactly. OUTPUT FORMAT: Return ONLY the revised HTML as plain text. Do NOT wrap it in markdown fences. Do NOT add explanations."}
        
        # ===下面的函数是用来调用智增增api的，当官方api充足时，可以替换掉这个函数（把zzz删除）===
        # generate_ppt_with_gemini(outline_path,ppt_template_path,generate_ppt_with_gemini_prompt)
        # generate_ppt_with_gemini_zzz(outline_path,ppt_template_path,generate_ppt_with_gemini_prompt)
        generate_ppt_with_gemini_xj(outline_path,ppt_template_path,generate_ppt_with_gemini_prompt,model="gpt-4o")



        # ===  Refiner  ===                
        # 已在P2S/refinement实现
        from refinement import refinement

        # 1. 加载提示词
        prompts = [commenter_prompt, reviser_prompt]

        # 2. 定义路径
        input_index = auto_path
        outline_path = os.path.join(input_index, "outline.json")
        output_index = os.path.join(auto_path, "final")
        output_index_images = os.path.join(output_index, "images") # 保存图片的子目录，用于显示refinement后的html中的图片

        # 确保输出目录存在
        os.makedirs(output_index, exist_ok=True)

        # 将图片复制到final/images目录下
        import shutil
        source_images_dir = os.path.join(auto_path, "images")
        if os.path.exists(source_images_dir):
            shutil.copytree(source_images_dir, output_index_images, dirs_exist_ok=True)
            print(f"📁 Copied images to: {output_index_images}")

        # 3. 加载大纲数据
        with open(outline_path, 'r', encoding='utf-8') as f:
            outline_data = json.load(f)
            if isinstance(outline_data, list):
                # 将列表转换为以索引(字符串)为 Key 的字典
                # 假设 list[0] 对应文件 0_ppt.html 或 1_ppt.html，这里保持原始索引
                outline_full = {str(i): item for i, item in enumerate(outline_data)}
            else:
                outline_full = outline_data

        # ================= 核心修改逻辑开始 =================

        print(f"🚀 开始扫描目录: {input_index}")
        
        # 4.1 先过滤出所有符合 "数字_ppt.html" 格式的文件
        target_files = []
        for f in os.listdir(input_index):
            # 严格匹配：数字开头 + _ppt.html 结尾
            if re.search(r'^\d+_ppt\.html$', f):
                target_files.append(f)
        
        # 4.2 定义排序 Key：直接提取开头的数字
        def get_file_number(filename):
            # 因为上一步已经过滤过了，这里可以直接提取
            return int(filename.split('_')[0])

        # 4.3 执行排序 (这步是关键，确保 2 在 10 前面)
        sorted_files = sorted(target_files, key=get_file_number)

        # Debug: 打印前几个文件确认顺序
        print(f"👀 排序后文件列表前5个: {sorted_files[:5]}")
        
        # 5. 遍历排序后的列表
        for file_name in sorted_files:            
            # 直接提取序号 (之前已经验证过格式了)
            num = str(get_file_number(file_name))
            
            # 获取当前 html 对应的 outline
            outline = outline_full.get(int(num)-1)
            
            # 【容错逻辑】处理索引偏移 (例如文件是 1_ppt，但列表是从 0 开始)
            # 如果 outline 为空，且 num-1 存在，则尝试自动回退
            if outline is None and str(int(num)-1) in outline_full:
                print(f"ℹ️ 尝试修正索引: 文件 {num} -> 使用大纲 {int(num)-1}")
                outline = outline_full.get(str(int(num)-1))

            if outline is None:
                print(f"⚠️ 跳过 {file_name}: 在 outline.json 中找不到序号 {num} 或 {int(num)-1}")
                continue

            # 构建路径
            html_file_path = os.path.join(input_index, file_name)
            html_file_path_refine = os.path.join(output_index, file_name)

            print(f"📝 [顺序处理中] 正在优化: {file_name} (对应大纲 Key: {num})")
            
            # 6. 调用优化函数
            try:
                refinement(input_path=html_file_path, output_path=html_file_path_refine, prompts=prompts, outline=outline)
            except Exception as e:
                print(f"❌ 处理 {file_name} 时出错: {e}")

        print(f"✅ 所有文件处理完成，结果保存在: {output_index}")


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
        outline_modified_path=os.path.join(auto_path, "poster_outline_modified.txt")
        print (f"✅ Poster outline modified.")
        

        print(f"🖼️ Building poster HTML at: {poster_path}")
        build_poster_from_outline(poster_outline_path=poster_outline_path_modified,poster_template_path="./poster_template.html",poster_path=poster_path,)
        print (f"✅ Poster HTML built.")
        

        print(f"🖊️ Modifying title and authors in poster HTML...")
        modify_title_and_author(dag_path=dag_path,poster_path=poster_path)
        print (f"✅ Title and authors modified.")


        from refinement import refinement_poster_with_gemini
        from refinement import refinement_poster_with_gpt
        poster_final_index = os.path.join(auto_path, "final")
        os.makedirs(poster_final_index, exist_ok=True)

        poster_final_output_path = os.path.join(poster_final_index, "poster_final.html")
        print(f"🖊️ Refining poster HTML with Gemini...")
        
        
        # refinement_poster_with_gemini(input_html_path=poster_path, prompts=poster_refinement_prompt,markdown_path=md_path,output_html_path=poster_final_output_path,model="gemini-3-pro-preview")
        refinement_poster_with_gpt(input_html_path=poster_path, prompts=poster_refinement_prompt,markdown_path=md_path,output_html_path=poster_final_output_path,model="gemini-3-pro-preview")
        
        print (f"✅ Poster HTML refined. Final poster at: {poster_final_output_path}")





        # =============================  PR部分  ================================
        pr_template_path="./pr_template.md"
        basic_information_path = extract_basic_information(auto_path=auto_path,extract_basic_information_prompt=extract_basic_information_prompt,model="gpt-4o-2024-11-20")
        initialize_pr_markdown(basic_information_path=basic_information_path,auto_path=auto_path,pr_template_path=pr_template_path)

        pr_path=os.path.join(auto_path, "markdown.md")


        generate_pr_from_dag(dag_path=dag_path, pr_path=pr_path, generate_pr_prompt=generate_pr_prompt, model="gpt-4o-2024-11-20")
        print(f"📝 PR generated at: {pr_path}")
        
        add_title_and_hashtag(pr_path=pr_path, add_title_and_hashtag_prompt=add_title_and_hashtag_prompt, model="gpt-4o-2024-11-20")
        add_institution_tag(pr_path=pr_path)
        


        dedup_consecutive_markdown_images(pr_path, inplace=True)

        print(f"✅ PR markdown post-processed.")

        print(f"🖊️ Refining PR markdown with LLM...")
        pr_refine_path=os.path.join(auto_path, "markdown_refined.md")
        refinement_pr(pr_path=pr_path, pr_refine_path=pr_refine_path, prompts=pr_refinement_prompt, model="gpt-4o-2024-11-20")
        print (f"✅ PR markdown refined.")











        # 在auto目录下创建success.txt作为标志
        success_file_path = os.path.join(auto_path, "success.txt")
        open(success_file_path, "w").close()



        print(f"✅ Finished processing: {subdir}\n{'-' * 80}")

    print("\n🎉 All papers processed successfully!")


if __name__ == "__main__":
    main()