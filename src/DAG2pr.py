import json
import time
import shutil
from pathlib import Path
from typing import Optional


# ========== Extract basic information for PR Generation ==========
def extract_basic_information(
    dag_path: str,
    extract_basic_information_prompt: str,
    model: str,
    auto_path: str,
    output_filename: str = "basic_information.txt",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    读取 dag.json 的第一个 node（root），将其完整信息发给 LLM，提取并输出：
    Title / Author / Institution / Github
    然后把 LLM 输出写入 auto_path/basic_information.txt（不存在则创建）。

    返回：写入的 txt 的绝对路径（str）

    依赖：
      pip install openai
    环境变量（默认）：
      OPENAI_API_KEY（必需，除非手动传 api_key）
      OPENAI_BASE_URL（可选，用于代理/兼容服务）
    """
    dag_file = Path(dag_path).expanduser().resolve()
    auto_dir = Path(auto_path).expanduser().resolve()
    out_path = auto_dir / output_filename

    if not dag_file.exists() or not dag_file.is_file():
        raise FileNotFoundError(f"dag_path not found or not a file: {dag_file}")

    auto_dir.mkdir(parents=True, exist_ok=True)

    # 1) load dag.json
    with dag_file.open("r", encoding="utf-8") as f:
        dag = json.load(f)

    nodes = dag.get("nodes", [])
    if not isinstance(nodes, list) or len(nodes) == 0 or not isinstance(nodes[0], dict):
        raise ValueError("Invalid dag.json: missing or invalid 'nodes[0]' root node.")

    root_node = nodes[0]

    # 2) build LLM input (send the entire root node)
    #    Keep it explicit and machine-readable for the LLM.
    root_payload = {
        "name": root_node.get("name", ""),
        "content": root_node.get("content", ""),
        "github": root_node.get("github", ""),
        "edge": root_node.get("edge", []),
        "level": root_node.get("level", 0),
        "visual_node": root_node.get("visual_node", []),
    }

    user_message = (
        "ROOT_NODE_JSON:\n"
        + json.dumps(root_payload, ensure_ascii=False, indent=2)
    )

    # 3) call LLM
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key,  # if None, openai SDK will read OPENAI_API_KEY
        base_url=base_url,  # if None, SDK will use default; can set OPENAI_BASE_URL instead
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": extract_basic_information_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    llm_text = (resp.choices[0].message.content or "").strip()

    if not llm_text:
        raise RuntimeError("LLM returned empty content for basic information extraction.")

    # 4) write to auto/basic_information.txt
    #    Ensure trailing newline for cleanliness.
    with out_path.open("w", encoding="utf-8") as f:
        f.write(llm_text.rstrip() + "\n")

    return str(out_path)


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