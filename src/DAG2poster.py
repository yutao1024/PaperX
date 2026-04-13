from __future__ import annotations
import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional
from openai import OpenAI
from google import genai
from typing import Any, Dict, List, Optional
import traceback
import shutil
from pathlib import Path
from bs4 import BeautifulSoup


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
    model: str = "gemini-2.5-pro",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    client: Optional[Any] = None, # Type relaxed to accept both clients
    overwrite: bool = True,
    config: dict = None
) -> None:
    """
    Read dag.json from dag_path, iterate root->section nodes, and for each section:
      - choose the largest-resolution visual node referenced by section["visual_node"]
      - send (section_without_visual_node, best_visual_node_if_any, IMAGE_SRC, ALT_TEXT) to LLM
      - LLM returns EXACTLY one <section class="section">...</section> HTML block
      - append/write to poster_outline_path

    Supports both OpenAI and Google GenAI (Gemini) clients.
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

    # Determine Model Type
    is_gemini = "gemini" in model.lower()

    # Prepare Client if not provided
    if client is None:
        api_keys_config = config.get("api_keys", {}) if config else {}
        
        if is_gemini:
            # Setup Google GenAI Client
            api_key = api_key or api_keys_config.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
                      
            client = genai.Client(api_key=api_key)
        else:
            # Setup OpenAI Client
            api_key = api_key or api_keys_config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            
            client = OpenAI(api_key=api_key)

    # Output file init
    out_dir = os.path.dirname(os.path.abspath(poster_outline_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    write_mode = "w" if overwrite else "a"
    with open(poster_outline_path, write_mode, encoding="utf-8") as f_out:
        # Iterate sections in the order of root.edge
        for sec_name in root_edges:
            if sec_name not in name2node:
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
                    cand = name2node.get(ref_str)

                    if cand is None:
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

                if best_visual_node is not None and not best_image_src:
                    for ref in visual_refs:
                        tmp = _extract_image_src_from_md(str(ref))
                        if tmp:
                            best_image_src = tmp
                            break

            # Build the section JSON WITHOUT visual_node attribute
            section_wo_visual = _remove_key_deep(section_node, "visual_node")
            section_wo_visual["name"] = _safe_section_title(str(section_wo_visual.get("name", "")))

            # Compose ALT_TEXT
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
                if image_src and not image_src.startswith("images/") and "images/" in image_src:
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

            # Call API based on model type
            html_block = ""
            if is_gemini:
                # Gemini New API Call
                # Note: config is passed via helper if needed, usually default generation config is fine
                resp = client.models.generate_content(
                    model=model,
                    contents=payload
                )
                if resp.text:
                    html_block = resp.text
                else:
                    raise RuntimeError("Gemini returned empty content.")
            else:
                # OpenAI API Call
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": payload}],
                )
                if not hasattr(resp, "choices") or not resp.choices:
                    raise RuntimeError("OpenAI returned empty choices.")
                
                html_block = resp.choices[0].message.content

            if not isinstance(html_block, str) or not html_block.strip():
                raise RuntimeError("LLM returned empty content string.")

            # Append to output file
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


def inject_img_section_to_poster(
    figure_path: str,
    auto_path: str,
    poster_path: str,
    target_filename: str = "expore_our_work_in_detail.jpg",
) -> str:
    """
    1) 将 figure_path 指向的图片复制到 auto_path/images/ 下（文件名固定为 target_filename）
    2) 读取 poster_path 对应的 HTML，定位 <main class="main"> 内部的
       <div class="flow" id="flow">（若存在），把 img-section 插入到该 flow div 的末尾，
       从而保证新增块出现在 `</div> </main>` 的 </div>（flow 的闭合）之前。
       若找不到 flow div，则退化为插入到 main 的末尾。

    返回：写回后的 poster_path（绝对路径）
    """
    auto_dir = Path(auto_path).expanduser().resolve()
    poster_file = Path(poster_path).expanduser().resolve()
    src_figure = Path(figure_path).expanduser().resolve()

    if not src_figure.exists() or not src_figure.is_file():
        raise FileNotFoundError(f"figure_path not found or not a file: {src_figure}")
    if not auto_dir.exists() or not auto_dir.is_dir():
        raise FileNotFoundError(f"auto_path not found or not a directory: {auto_dir}")
    if not poster_file.exists() or not poster_file.is_file():
        raise FileNotFoundError(f"poster_path not found or not a file: {poster_file}")

    # 1) copy image into auto/images/
    images_dir = auto_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dst_figure = images_dir / target_filename
    shutil.copy2(src_figure, dst_figure)

    # 2) edit poster html
    html_text = poster_file.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_text, "html.parser")

    main_tag = soup.find("main", class_="main")
    if main_tag is None:
        raise ValueError(f'Cannot find <main class="main"> in poster: {poster_file}')

    # Prefer inserting into the flow div so the new block sits right before </div> </main>
    flow_tag = main_tag.find("div", attrs={"class": "flow", "id": "flow"})

    # Avoid duplicate insertion
    target_src = f"images/{target_filename}"
    existing_img = main_tag.find("img", attrs={"src": target_src})
    if existing_img is None:
        new_div = soup.new_tag("div", attrs={"class": "img-section"})
        new_img = soup.new_tag(
            "img",
            attrs={"src": target_src, "alt": "", "class": "figure"},
        )
        new_div.append(new_img)

        if flow_tag is not None:
            # Insert before </div> of flow
            flow_tag.append(new_div)
        else:
            # Fallback: append to main
            main_tag.append(new_div)

    # Write back
    poster_file.write_text(str(soup), encoding="utf-8")

    return str(poster_file)


# =================================优化逻辑性==================================
def _parse_sections(html_text: str) -> List[dict]:
    """
    解析每个 <section class="section"> ... </section> 块，提取：
    - section_block: 原始块文本
    - title: section-bar 内标题
    - first_p_inner: 第一个 <p>...</p> 的 inner 文本（可为空）
    - p_span: 第一个 <p>...</p> 的 (start,end) 在 section_block 内的 span（包含<p>..</p>）
    """
    section_pat = re.compile(
        r'(<section\s+class="section"\s*>.*?</section>)',
        re.DOTALL | re.IGNORECASE,
    )
    sections = []
    for m in section_pat.finditer(html_text):
        block = m.group(1)

        # 标题
        title_m = re.search(
            r'<div\s+class="section-bar"[^>]*>(.*?)</div>',
            block,
            re.DOTALL | re.IGNORECASE,
        )
        title = title_m.group(1).strip() if title_m else ""

        # 只处理第一个 <p>...</p>
        p_m = re.search(r'(<p\b[^>]*>)(.*?)(</p>)', block, re.DOTALL | re.IGNORECASE)
        if p_m:
            p_open, p_inner, p_close = p_m.group(1), p_m.group(2), p_m.group(3)
            p_span = (p_m.start(0), p_m.end(0))
            first_p_inner = p_inner
        else:
            p_span = None
            first_p_inner = ""

        sections.append(
            {
                "section_block": block,
                "title": title,
                "first_p_inner": first_p_inner,
                "p_span": p_span,
                "match_span_in_full": (m.start(1), m.end(1)),  # span in full html_text
            }
        )
    return sections


def _extract_json_array(text: str) -> List[str]:
    """
    从模型输出中提取 JSON 数组（允许模型带少量前后缀文本，但最终必须能抽到一个 [...]）。
    """
    text = text.strip()
    # 直接就是JSON数组
    if text.startswith("["):
        try:
            arr = json.loads(text)
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return arr
        except Exception:
            pass

    # 尝试抽取第一个 [...] 段
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        raise ValueError("LLM output does not contain a JSON array.")
    arr_str = m.group(0)
    arr = json.loads(arr_str)
    if not (isinstance(arr, list) and all(isinstance(x, str) for x in arr)):
        raise ValueError("Extracted JSON is not a list of strings.")
    return arr

def modified_poster_logic(
    poster_outline_path_modified: str,
    modified_poster_logic_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    config: dict = None
) -> str:
    """
    读取 poster_outline_path_modified(txt) 的 HTML-like 内容；
    把全文发给 LLM，让其“只输出”从第一个到倒数第二个小节需要追加的衔接句（JSON 数组，顺序一致）；
    然后将这些衔接句依次追加到每个小节的第一个 <p>...</p> 的末尾（在 </p> 前），不改变任何原格式/内容；
    最后覆盖写回原 txt，并返回该 txt 的绝对路径。
    """
    txt_path = Path(poster_outline_path_modified).expanduser().resolve()
    if not txt_path.exists() or not txt_path.is_file():
        raise FileNotFoundError(f"txt not found: {txt_path}")

    html_text = txt_path.read_text(encoding="utf-8")

    # 注意：_parse_sections 和 _extract_json_array 需在上下文环境中定义
    sections = _parse_sections(html_text)
    if len(sections) < 2:
        return str(txt_path)

    # 需要加衔接句的小节数量：从第1到倒数第2 => len(sections)-1
    expected_n = len(sections) - 1

    # 确定模型名称
    model_name = model or os.getenv("OPENAI_MODEL") or "gpt-4o"
    is_gemini = "gemini" in model_name.lower()
    
    # 获取 API 配置
    api_keys_config = config.get("api_keys", {}) if config else {}
    
    out_text = ""

    if is_gemini:
        # --- Gemini Client Setup ---
        api_key = api_keys_config.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")
        
        client = genai.Client(api_key=api_key)
        
        # Call Gemini
        # 将 system prompt 放入配置中，user content 放入 contents
        resp = client.models.generate_content(
            model=model_name,
            contents=html_text,
            config={
                "system_instruction": modified_poster_logic_prompt,
                "temperature": temperature,
            }
        )
        if not resp.text:
             raise RuntimeError("Gemini returned empty text.")
        out_text = resp.text

    else:
        # --- OpenAI Client Setup ---
        api_key = api_keys_config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

        client = OpenAI(api_key=api_key)

        # Call OpenAI
        messages = [
            {"role": "system", "content": modified_poster_logic_prompt},
            {"role": "user", "content": html_text},
        ]
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )
        out_text = resp.choices[0].message.content or ""

    # 解析返回的 JSON
    transitions = _extract_json_array(out_text)

    if len(transitions) != expected_n:
        # 容错：如果 LLM 偶尔多生成或少生成，可根据实际情况决定是报错还是截断/填充
        # 这里保持原逻辑报错
        raise ValueError(
            f"Transition count mismatch: expected {expected_n}, got {len(transitions)}"
        )

    # 逐个 section 进行插入：只改 <p>...</p> inner 的末尾（</p> 前）
    new_html_parts = []
    cursor = 0

    for i, sec in enumerate(sections):
        full_start, full_end = sec["match_span_in_full"]
        # 先拼接 section 之前的内容（保持原样）
        new_html_parts.append(html_text[cursor:full_start])

        block = sec["section_block"]
        p_span = sec["p_span"]

        if i <= len(sections) - 2:  # 第1到倒数第2个 section
            trans = transitions[i].strip()
            if trans and p_span:
                # 在该 section 的第一个 </p> 前插入
                p_start, p_end = p_span
                p_block = block[p_start:p_end]

                close_idx = p_block.lower().rfind("</p>")
                if close_idx == -1:
                    new_block = block
                else:
                    insert = (" " + trans) if not p_block[:close_idx].endswith((" ", "\n", "\t")) else trans
                    new_p_block = p_block[:close_idx] + insert + p_block[close_idx:]
                    new_block = block[:p_start] + new_p_block + block[p_end:]
            else:
                new_block = block
        else:
            # 最后一个 section 不加衔接句
            new_block = block

        new_html_parts.append(new_block)
        cursor = full_end

    # 追加尾部
    new_html_parts.append(html_text[cursor:])
    new_html_text = "".join(new_html_parts)

    txt_path.write_text(new_html_text, encoding="utf-8")
    return str(txt_path)
