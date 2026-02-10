# 伪代码，需接入实际 API (如 OpenAI GPT-4o)
class Commenter:
    def __init__(self, client):
        self.client = client

    def review_slide(self, image_path):
        """
        调用 VLM 分析渲染后的 PPT 图片
        """
        system_prompt = """
        You are a Presentation Design Expert. Analyze the slide screenshot for visual layout issues.
        Focus strictly on:
        1. Text Overflow: Is text going out of its container or the slide boundary?
        2. Balance: Is one side too heavy or too empty?
        3. Spacing: Are images too small or text too crowded?
        
        Output valid JSON only:
        {
            "status": "PASS" or "NEEDS_REVISION",
            "issues": ["list of specific issues..."],
            "suggestion": "Natural language suggestion for the reviser."
        }
        """
        
        # 这里模拟 VLM 调用
        # response = client.chat.completions.create(model="gpt-4o", messages=..., image=...)
        # return json.loads(response)
        pass