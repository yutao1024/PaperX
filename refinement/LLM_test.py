from openai import OpenAI 
from google import genai
from google.genai import types
import os

client = genai.Client(api_key=os.getenv("OPENAI_API_KEY"), http_options={"base_url": "https://open.xiaojingai.com/"})

def LLM_test():
    # 注意：Google SDK 使用 models.generate_content 而不是 chat.completions
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents="Introduce yourself."
    )
    print(response.text)

if __name__ == "__main__":
    LLM_test()