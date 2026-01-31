import base64
import os
import re
import json
import time
import PIL.Image
import shutil
from PIL import Image
from pathlib import Path
from openai import OpenAI 
from google import genai
from google.genai import types
from .html_revise import HTMLMapper, apply_html_modifications, HTMLModificationError
from playwright.sync_api import sync_playwright # 用于截图


class VLMCommenter:
    def __init__(self, api_key, prompt, provider="openai", model_name=None, base_url=None):
        """
        :param api_key: API Key
        :param prompt: 提示词文本
        :param provider: "openai" 或 "gemini"
        :param model_name: 指定模型名称 (可选)
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.prompt_text = prompt

        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
            self.model = model_name if model_name else "gpt-4o"
        elif self.provider == "gemini":
            self.client = genai.Client(api_key=api_key,
                                       http_options={
                            "base_url": base_url
                        })
            self.model = model_name if model_name else "gemini-1.5-flash"
        else:
            raise ValueError("Unsupported provider. Choose 'openai' or 'gemini'.")

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def evaluate_slide(self, image_path, outline, pre_comments):
        """
        输入：截图路径
        输出：诊断文本 string
        """
        prompt_text = self.prompt_text
        # critic_his_intro = "Note: Previous comments might be outdated or incorrect. If the current image looks acceptable, ignore previous requests and mark as PASS."
        full_prompt = f"{prompt_text}\n \
        *****previous comments******\
        \n{pre_comments} \
        *****begin of the outline*****\
        \n{outline} \
        *****end of the outline*****\
        *****the following is the image,not the outline*****"

        if not full_prompt:
             return "Error: Commenter prompt is empty."

        if self.provider == "openai":
            base64_image = self._encode_image(image_path)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": [
                            {"type": "text", "text": full_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]}
                    ],
                    max_tokens=300
                )
                return response.choices[0].message.content
            except Exception as e:
                return f"Error using OpenAI VLM: {e}"

        elif self.provider == "gemini":       
            try:
                img = PIL.Image.open(image_path)
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[full_prompt, img]
                )
                return response.text
            except Exception as e:
                return f"Error using Gemini VLM (google-genai): {e}"


class LLMReviser:
    def __init__(self, api_key, prompt, provider="openai", model_name=None, base_url=None):
        """
        :param api_key: API Key
        :param prompt: 提示词文本
        :param provider: "openai" 或 "gemini"
        :param model_name: 指定模型名称
        """
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = prompt

        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key)
            self.model = model_name if model_name else "gpt-4"
        elif self.provider == "gemini":
            self.client = genai.Client(api_key=api_key, http_options={
                            "base_url": base_url
                        })
            self.model = model_name if model_name else "gemini-1.5-pro"
        else:
            raise ValueError("Unsupported provider. Choose 'openai' or 'gemini'.")

    def generate_revision_plan(self, current_structure_json, vlm_critique):
        """
        输入：HTML 结构 JSON 和 VLM 评价
        输出：修改后的 JSON
        """
        
        if "PASS" in vlm_critique.upper() and len(vlm_critique) < 10:
            return None

        prompt_system = self.system_prompt
        if not prompt_system:
             print("Error: Reviser prompt is empty.")
             return None
        
        user_content = f"""
        --- CURRENT STRUCTURE JSON ---
        {json.dumps(current_structure_json, indent=2)}
        
        --- VISUAL CRITIQUE ---
        {vlm_critique}
        
        --- INSTRUCTION ---
        Generate the modification JSON based on the system instructions.
        """

        if self.provider == "openai":
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt_system},
                        {"role": "user", "content": user_content}
                    ],
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                print(f"OpenAI Error: {e}")
                return None

        elif self.provider == "gemini":
            try:
                # 拼接 System Prompt 和 User Content 
                full_prompt = f"{prompt_system}\n\n{user_content}"
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                
                text_response = response.text
                
                # 清洗可能存在的 Markdown 标记 (即使指定了 JSON mime type，有些模型仍可能加 ```json)
                if text_response.startswith("```"):
                    text_response = text_response.strip("`").replace("json", "").strip()
                
                return json.loads(text_response)
            except json.JSONDecodeError:
                print(f"Gemini returned invalid JSON: {response.text}")
                return None
            except Exception as e:
                print(f"Gemini Error (google-genai): {e}")
                return None
        


def refinement(input_path, output_path, prompts, outline, max_iterations=3):
    """
    自动修复闭环：截图 -> 诊断 -> 修改 -> 循环
    """
    # api_key = os.getenv("GOOGLE_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY")
    commenter_prompt = prompts[0]
    reviser_prompt = prompts[1]
    vlm = VLMCommenter(api_key, commenter_prompt, provider='openai', model_name="gpt-4o", base_url="https://open.xiaojingai.com/v1")
    reviser = LLMReviser(api_key, reviser_prompt, provider='openai', model_name="gpt-4o", base_url="https://open.xiaojingai.com/v1")

    current_input = input_path
    critic_his = ""

    for i in range(max_iterations):
        print(f"\n=== Iteration {i+1} ===")
        
        # 1. 渲染并截图 (这里用伪代码表示，实际可用 Selenium/Playwright)
        screenshot_path = f"{Path(output_path).parent}/{Path(current_input).stem}_{i+1}.png"  # 临时截图路径
        take_screenshot(current_input, screenshot_path) 
        print(f"Screenshot taken: {screenshot_path}")

        # 2. VLM 视觉诊断
        critique = vlm.evaluate_slide(screenshot_path, outline, critic_his)
        critic_his = critic_his + f"this is the {i}th comment: {critique}" 
        print(f"VLM Critique: {critique}")
        
        if "PASS" in critique:
            print("Layout looks good! Stopping loop.")
            
            if os.path.abspath(current_input) != os.path.abspath(output_path):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                shutil.copy2(current_input, output_path)
                print(f"Final result saved to: {output_path}")
            else:
                print(f"Result is already at output path: {output_path}")

            break

        # 3. 读取当前 HTML 结构
        mapper = HTMLMapper(current_input)
        current_tree = mapper.get_structure_tree()

        # 4. LLM 生成修改方案
        modification_json = reviser.generate_revision_plan(current_tree, critique)
        
        if not modification_json:
            print("Reviser suggested no changes. Stopping.")
            break
            
        print(f"Proposed Changes: {json.dumps(modification_json, indent=2)}")

        # 5. 执行修改
        try:
            # 调用 reviser，如果有严重错误，它现在会抛出 HTMLModificationError
            apply_html_modifications(current_input, output_path, modification_json)
            print("Modifications applied to HTML.")
            
        except (HTMLModificationError, Exception) as e:
            # 捕获自定义错误 或 其他意外错误
            print(f"❌ Error applying modifications at iteration {i+1}: {e}")
            
            # ====== 添加另存副本逻辑 ======
            try:
                # 构造备份文件名，例如: original_filename_error_iter1.html
                # 注意：current_input 可能是中间生成的临时文件，我们需要保留它以便复现问题
                
                # 获取 input_path (最原始文件名) 的基础名称，方便辨识
                base_name = Path(input_path).stem
                
                # 定义错误副本路径
                error_backup_path = f"{Path(output_path).parent}/{base_name}_FAILED_iter{i+1}.html"
                
                # 将导致出错的那个 HTML 文件 (current_input) 复制出来
                shutil.copy2(current_input, error_backup_path)
                
                print(f"⚠️  已自动保存出错前的 HTML 副本: {error_backup_path}")
                print(f"    你可以打开此文件，并使用控制台打印的 JSON 尝试复现问题。")

            except Exception as copy_err:
                print(f"❌ 尝试保存错误副本时失败: {copy_err}")
            # ============================
            
            # 出错后必须中断循环，因为无法生成下一步的 output_path
            break
        current_input = output_path
        
        # 等待一会，防止并发读写问题
        time.sleep(1)
    
    # 优化结束后输出最终截图
    final_screenshot_path = f"{Path(output_path).parent}/{Path(current_input).stem}_final.png"
    take_screenshot(current_input, final_screenshot_path)
    print(f"\n📷 Final screenshot saved: {final_screenshot_path}")

def take_screenshot(html_path, output_path):
    """简单的截图工具函数示例 (Playwright)"""
    if not os.path.exists(html_path):
        print(f"错误：文件不存在于 {html_path}")
    abs_path = Path(os.path.abspath(html_path)).as_uri()
    
    with sync_playwright() as p:
        # 1. 显式设置 device_scale_factor=1
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={'width': 960, 'height': 540},
            device_scale_factor=1
        )
        page = context.new_page()
        
        # 2. 访问页面
        page.goto(abs_path, wait_until="networkidle") # 确保图片和字体加载完成
        
        # 3. 截取特定元素而非全屏，这样最保险
        # 你的根 div id 是 slide1，或者直接截取 svg
        element = page.locator(".slideImage") 
        element.screenshot(path=output_path)
        
        browser.close()

def take_screenshot_poster(html_path, output_path):
    """适配 .poster/#flow 的 HTML 海报截图（Playwright, 同步）"""
    if not os.path.exists(html_path):
        raise FileNotFoundError(f"文件不存在: {html_path}")

    abs_uri = Path(os.path.abspath(html_path)).as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-dev-shm-usage"])
        context = browser.new_context(
            # 你的 CSS 固定 --poster-width/height=1400x900
            viewport={"width": 1400, "height": 900},
            device_scale_factor=1
        )
        page = context.new_page()

        # 可选：打开调试信息（需要就取消注释）
        # page.on("console", lambda msg: print("[console]", msg.type, msg.text))
        # page.on("pageerror", lambda err: print("[pageerror]", err))
        # page.on("requestfailed", lambda req: print("[requestfailed]", req.url, req.failure))

        # 1) 先 DOMReady，避免 networkidle 卡死
        page.goto(abs_uri, wait_until="domcontentloaded")

        # 2) 确保关键容器存在
        page.wait_for_selector(".poster", state="attached", timeout=30000)
        page.wait_for_selector("#flow", state="attached", timeout=30000)

        # 3) 等字体就绪（你的脚本里 fit 受字体/排版影响很大）
        try:
            page.evaluate("() => document.fonts ? document.fonts.ready : Promise.resolve()")
        except Exception:
            pass

        # 4) 等 flow 内所有图片加载完成（没有图片也会立即返回）
        page.evaluate(r"""
        () => {
          const flow = document.getElementById("flow");
          if (!flow) return Promise.resolve();
          const imgs = Array.from(flow.querySelectorAll("img"));
          if (imgs.length === 0) return Promise.resolve();
          return Promise.all(imgs.map(img => {
            if (img.complete) return Promise.resolve();
            return new Promise(res => {
              img.addEventListener("load", res, { once: true });
              img.addEventListener("error", res, { once: true });
            });
          }));
        }
        """)

        # 5) 等你的 fit() 执行并让布局稳定：等几帧 + scrollWidth 不再变化
        page.evaluate(r"""
        () => new Promise((resolve) => {
          const flow = document.getElementById("flow");
          if (!flow) return resolve();

          let last = -1;
          let stableCount = 0;

          function tick() {
            const cur = flow.scrollWidth;  // multi-column 溢出判据
            if (cur === last) stableCount += 1;
            else stableCount = 0;

            last = cur;

            // 连续若干帧稳定，就认为 fit/重排结束
            if (stableCount >= 10) return resolve();
            requestAnimationFrame(tick);
          }

          // 给 load 事件/fit 一点点启动时间
          setTimeout(() => requestAnimationFrame(tick), 50);
        })
        """)

        # 6) 截图：截 .poster（不截 stage 背景）
        poster = page.locator(".poster").first
        poster.screenshot(path=output_path, timeout=60000)

        browser.close()


def load_prompt(prompt_path="prompt.json", prompt_name="poster_prompt"):
    with open(prompt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(prompt_name, "")


# import os
# import re
# import json
# # 假设 load_prompt 和 refinement 已经定义好了
# # from your_module import load_prompt, refinement 

# if __name__ == "__main__":
#     # 1. 加载提示词
#     prompts_name = ["commenter_prompt", "reviser_prompt"]
#     prompts = []
#     for i in prompts_name:
#         prompts.append(load_prompt(prompt_path="prompt.json", prompt_name=i))

#     # 2. 定义路径
#     input_index = "/home/yutao/agent/test_html"
#     outline_path = os.path.join(input_index, "outline.json")
#     output_index = "/home/yutao/agent/test_html_refined"

#     # 确保输出目录存在
#     os.makedirs(output_index, exist_ok=True)

#     # 3. 加载大纲数据
#     with open(outline_path, 'r', encoding='utf-8') as f:
#         outline_data = json.load(f)
#         if isinstance(outline_data, list):
#             # 将列表转换为以索引(字符串)为 Key 的字典
#             # 假设 list[0] 对应文件 0_ppt.html 或 1_ppt.html，这里保持原始索引
#             outline_full = {str(i): item for i, item in enumerate(outline_data)}
#         else:
#             outline_full = outline_data

#     # ================= 核心修改逻辑开始 =================

#     print(f"🚀 开始扫描目录: {input_index}")
    
#     # 4.1 先过滤出所有符合 "数字_ppt.html" 格式的文件
#     target_files = []
#     for f in os.listdir(input_index):
#         # 严格匹配：数字开头 + _ppt.html 结尾
#         if re.search(r'^\d+_ppt\.html$', f):
#             target_files.append(f)
    
#     # 4.2 定义排序 Key：直接提取开头的数字
#     def get_file_number(filename):
#         # 因为上一步已经过滤过了，这里可以直接提取
#         return int(filename.split('_')[0])

#     # 4.3 执行排序 (这步是关键，确保 2 在 10 前面)
#     sorted_files = sorted(target_files, key=get_file_number)

#     # Debug: 打印前几个文件确认顺序
#     print(f"👀 排序后文件列表前5个: {sorted_files[:5]}")
    
#     # 5. 遍历排序后的列表
#     for file_name in sorted_files:
#         # 直接提取序号 (之前已经验证过格式了)
#         num = str(get_file_number(file_name))

#         if num < '5':
#             continue
        
#         # 获取当前 html 对应的 outline
#         outline = outline_full.get(int(num)-1)
        
#         # 【容错逻辑】处理索引偏移 (例如文件是 1_ppt，但列表是从 0 开始)
#         # 如果 outline 为空，且 num-1 存在，则尝试自动回退
#         if outline is None and str(int(num)-1) in outline_full:
#              print(f"ℹ️ 尝试修正索引: 文件 {num} -> 使用大纲 {int(num)-1}")
#              outline = outline_full.get(str(int(num)-1))

#         if outline is None:
#             print(f"⚠️ 跳过 {file_name}: 在 outline.json 中找不到序号 {num} 或 {int(num)-1}")
#             continue

#         # 构建路径
#         html_file_path = os.path.join(input_index, file_name)
#         html_file_path_refine = os.path.join(output_index, file_name)

#         print(f"📝 [顺序处理中] 正在优化: {file_name} (对应大纲 Key: {num})")
        
#         # 6. 调用优化函数
#         try:
#             refinement(
#                 input_path=html_file_path, 
#                 output_path=html_file_path_refine, 
#                 prompts=prompts, 
#                 outline=outline
#             )
#         except Exception as e:
#             print(f"❌ 处理 {file_name} 时出错: {e}")


#     print(f"✅ 所有文件处理完成，结果保存在: {output_index}")


def refinement_poster_with_gemini(input_html_path, prompts, markdown_path, output_html_path, model):
    # 配置api_key与refinement模型
    api_key = os.getenv("OPENAI_API_KEY")
    client = genai.Client(api_key=api_key, http_options={
                            "base_url": "https://open.xiaojingai.com/"
                        })

    auto_path = Path(input_html_path).parent
    final_index = os.path.join(auto_path, "final")
    final_index_image = os.path.join(final_index, "images")
    os.makedirs(final_index, exist_ok=True)
    
    # 确保 final 目录下有 images 文件夹，供 HTML 引用
    source_images_dir = os.path.join(auto_path, "images")
    if os.path.exists(source_images_dir):
        # 如果目标文件夹不存在，则复制；如果存在则跳过或覆盖
        if not os.path.exists(final_index_image):
            shutil.copytree(source_images_dir, final_index_image, dirs_exist_ok=True)
            print(f"📁 Images copied to: {final_index_image}")
    
    # 获取原文markdown与html内容
    with open(markdown_path, 'r', encoding='utf-8') as f:
        origin_md = f.read()

    with open(input_html_path, 'r', encoding='utf-8') as f:
        current_html = f.read()
    
    # 截图逻辑
    screenshot_name = Path(input_html_path).stem + ".png"
    screenshot_path = os.path.join(final_index, screenshot_name)
    
    print(f"📸 Taking screenshot of {input_html_path}...")
    take_screenshot_poster(input_html_path, screenshot_path)
    
    if not os.path.exists(screenshot_path):
        raise FileNotFoundError("Screenshot failed to generate.")

    # 调用MLLM进行refinement
    with open(screenshot_path, "rb") as f:
        image_bytes = f.read()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    print("🤖 Sending to Vision Model for refinement...")
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_text(text=prompts),
                types.Part.from_text(text=f"--- ORIGINAL MARKDOWN ---\n{origin_md}"),
                types.Part.from_text(text=f"--- CURRENT HTML ---\n{current_html}"),
                image_part,
            ],
        )
        
        # --- 7. 解析与保存结果 ---
        generated_text = response.text
        
        # 简单的 Markdown 代码块清洗逻辑
        if "```html" in generated_text:
            final_html = generated_text.split("```html")[1].split("```")[0].strip()
        elif "```" in generated_text:
            final_html = generated_text.split("```")[1].strip()
        else:
            final_html = generated_text

        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
            
        print(f"✅ Refined poster saved to: {output_html_path}")

        final_screenshot_name = Path(input_html_path).stem + "_final" + ".png"
        final_screenshot_path = os.path.join(final_index, final_screenshot_name)

        print(f"📸 Taking final poster screenshot of {output_html_path}...")
        take_screenshot_poster(output_html_path, final_screenshot_path)

    except Exception as e:
        print(f"❌ Error during AI generation: {e}")



# 假设 take_screenshot_poster 已经在其他地方定义
# from your_module import take_screenshot_poster 

def refinement_poster_with_gpt(input_html_path, prompts, markdown_path, output_html_path, model):
    # --- 1. 配置 Client ---
    api_key = os.getenv("OPENAI_API_KEY")
    
    # 注意：使用 OpenAI SDK 时，base_url 通常需要以 /v1 结尾（取决于服务商配置）
    # 如果你的服务商不需要 /v1，请去掉它
    base_url = "https://open.xiaojingai.com/v1" 
    
    client = OpenAI(
        api_key=api_key, 
        base_url=base_url
    )

    # --- 2. 路径与文件准备 (保持原逻辑) ---
    auto_path = Path(input_html_path).parent
    final_index = os.path.join(auto_path, "final")
    final_index_image = os.path.join(final_index, "images")
    os.makedirs(final_index, exist_ok=True)
    
    # 确保 final 目录下有 images 文件夹
    source_images_dir = os.path.join(auto_path, "images")
    if os.path.exists(source_images_dir):
        if not os.path.exists(final_index_image):
            shutil.copytree(source_images_dir, final_index_image, dirs_exist_ok=True)
            print(f"📁 Images copied to: {final_index_image}")
    
    # 获取原文内容
    with open(markdown_path, 'r', encoding='utf-8') as f:
        origin_md = f.read()

    with open(input_html_path, 'r', encoding='utf-8') as f:
        current_html = f.read()
    
    # --- 3. 截图逻辑 (保持原逻辑) ---
    screenshot_name = Path(input_html_path).stem + ".png"
    screenshot_path = os.path.join(final_index, screenshot_name)
    
    print(f"📸 Taking screenshot of {input_html_path}...")
    # 请确保 take_screenshot_poster 函数可用
    take_screenshot_poster(input_html_path, screenshot_path) 
    
    if not os.path.exists(screenshot_path):
        raise FileNotFoundError("Screenshot failed to generate.")

    # --- 4. 图片转 Base64 (OpenAI 特有步骤) ---
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(screenshot_path)

    print(f"🤖 Sending to Vision Model ({model}) for refinement...")
    
    try:
        # --- 5. 构建 OpenAI 消息格式 ---
        # OpenAI 视觉模型要求 content 为列表，包含 type: text 和 type: image_url
        messages = [
            {
                "role": "system",
                "content": "You are an expert web designer and developer specialized in HTML and CSS."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompts
                    },
                    {
                        "type": "text",
                        "text": f"--- ORIGINAL MARKDOWN ---\n{origin_md}"
                    },
                    {
                        "type": "text",
                        "text": f"--- CURRENT HTML ---\n{current_html}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high" # 可选 high/low/auto
                        }
                    }
                ]
            }
        ]

        # --- 6. 调用 API ---
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # max_tokens=4096, # 根据需要调整
            # temperature=0.7
        )
        
        # --- 7. 解析与保存结果 ---
        generated_text = response.choices[0].message.content
        
        # 简单的 Markdown 代码块清洗逻辑
        if "```html" in generated_text:
            final_html = generated_text.split("```html")[1].split("```")[0].strip()
        elif "```" in generated_text:
            # 兼容有些模型可能只写了 ``` 而没写 html
            final_html = generated_text.split("```")[1].strip()
        else:
            final_html = generated_text

        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
            
        print(f"✅ Refined poster saved to: {output_html_path}")

        # --- 8. 生成最终截图 ---
        final_screenshot_name = Path(input_html_path).stem + "_final" + ".png"
        final_screenshot_path = os.path.join(final_index, final_screenshot_name)

        print(f"📸 Taking final poster screenshot of {output_html_path}...")
        take_screenshot_poster(output_html_path, final_screenshot_path)

    except Exception as e:
        print(f"❌ Error during AI generation: {e}")



# if __name__ == "__main__":
#     input_path = "/home/yutao/agent/P2S/Paper2Poster_Benchmark/output_gemini/01/Active Learning with Table Language Models/auto/poster.html"
#     output_path = "/home/yutao/agent/screenshot2.png"
#     take_screenshot_poster(input_path, output_path)