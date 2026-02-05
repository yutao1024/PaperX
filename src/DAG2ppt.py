import json
import os
import re
from typing import Optional
from openai import OpenAI
from google import genai
from typing import Any, Dict, List, Optional, Union


# ==========  生成selected_nodes.json ==========
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