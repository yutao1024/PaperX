# -*- coding: utf-8 -*-
"""
Optional In-Context Learning (ICL) / few-shot support for PaperX LLM calls.
Minimal change: config-driven, prepend few-shot examples to prompts when enabled.
"""
import json
import os
from typing import Any, Dict, List, Optional


def get_instruction_with_icl(
    prompt_name: str,
    base_prompt: str,
    config: Optional[dict] = None,
) -> str:
    """
    If ICL is enabled in config and examples exist for this prompt,
    prepend few-shot examples to base_prompt. Otherwise return base_prompt unchanged.

    Config shape (optional):
      icl:
        enabled: true
        max_examples_per_prompt: 2
        examples_file: "icl_examples.json"   # or path relative to cwd

    examples_file JSON shape:
      {
        "section_dag_generation_prompt": [
          {"input": "=== SECTION NAME ===\\n1 Introduction\\n\\n=== SECTION MARKDOWN (FULL) ===\\n...", "output": "{\\"nodes\\": [...]}"},
          ...
        ],
        "outline_initialize_prompt": [...],
        ...
      }

    For outline / single-message prompts, "input" = full user message content,
    "output" = expected model response (e.g. JSON string).
    """
    if not config or not isinstance(config.get("icl"), dict):
        return base_prompt

    icl_cfg = config["icl"]
    if not icl_cfg.get("enabled", False):
        return base_prompt

    max_examples = icl_cfg.get("max_examples_per_prompt", 2)
    examples_file = icl_cfg.get("examples_file")
    if not examples_file or not os.path.isfile(examples_file):
        return base_prompt

    try:
        with open(examples_file, "r", encoding="utf-8") as f:
            all_examples: Dict[str, List[Dict[str, str]]] = json.load(f)
    except (json.JSONDecodeError, OSError):
        return base_prompt

    examples = all_examples.get(prompt_name)
    if not examples or not isinstance(examples, list):
        return base_prompt

    # Build few-shot block
    parts = [base_prompt]
    for i, ex in enumerate(examples[:max_examples]):
        if not isinstance(ex, dict):
            continue
        inp = ex.get("input") or ex.get("Input")
        out = ex.get("output") or ex.get("Output")
        if inp is None and out is None:
            continue
        inp = inp if inp is not None else ""
        out = out if out is not None else ""
        parts.append(f"\n--- Few-shot Example {i + 1} ---\nInput:\n{inp}\n\nOutput:\n{out}")

    if len(parts) == 1:
        return base_prompt

    parts.append("\n--- End of examples. Now process the following input. ---\n")
    return "".join(parts)
