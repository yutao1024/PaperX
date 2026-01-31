import json

class Reviser:
    def __init__(self, client):
        self.client = client

    def get_revision_plan(self, structure_map, vlm_feedback, html_snippet):
        """
        根据 VLM 的反馈和当前的 HTML 结构，决定调用什么工具。
        """
        
        system_prompt = f"""
        You are a Code Refinement Agent for HTML Slides.
        Your goal is to fix visual issues identified by the Visual Reviewer.
        
        You have access to the following HTML Structure (The "Scene Graph"):
        {json.dumps(structure_map, indent=2)}
        
        Available Tools:
        1. adjust_layout_ratio(container_id, child_ratios): 
           - Use this to resize columns/rows. 'child_ratios' is a list of floats (e.g. [2, 1]).
           - Check 'containers' in the structure map for valid IDs and current ratios.
           
        2. modify_typography(element_id, font_size_pt, line_height):
           - Use this to shrink text size if layout adjustment isn't enough.
           - font_size_pt (float), line_height (float).
           
        3. rewrite_text(element_id, new_content):
           - Use this ONLY as a last resort if text is way too long.
           - Extract the text ID from the structure map.

        Input Context:
        - Visual Issues: {vlm_feedback['issues']}
        - Suggestion: {vlm_feedback['suggestion']}
        
        Output Format:
        Return a JSON object with a single key "tool_calls" which is a list of actions.
        Example:
        {{
            "tool_calls": [
                {{
                    "tool": "adjust_layout_ratio",
                    "params": {{ "container_id": "root_flex", "child_ratios": [1.5, 1] }}
                }}
            ]
        }}
        """

        # 模拟 LLM 调用
        # response = client.chat.completions.create(model="gpt-4", messages=[{"role": "system", "content": system_prompt}])
        # return json.loads(response)
        pass