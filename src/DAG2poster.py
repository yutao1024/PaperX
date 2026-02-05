import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional
from openai import OpenAI
from typing import Any, Dict, List, Optional


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
