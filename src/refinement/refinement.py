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
from playwright.sync_api import sync_playwright


class VLMCommenter:
    def __init__(self, api_key, prompt, provider="openai", model_name=None):
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
            self.client = genai.Client(api_key=api_key)
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
    def __init__(self, api_key, prompt, provider="openai", model_name=None):
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
            self.client = genai.Client(api_key=api_key)
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


def refine_one_slide(input_path, output_path, prompts, outline, max_iterations, model, config):
    """
    自动修复闭环：截图 -> 诊断 -> 修改 -> 循环
    """
    is_gemini = "gemini" in model.lower()

    if is_gemini:
        api_key = config['api_keys'].get('gemini_api_key')
    else:
        api_key = config['api_keys'].get('openai_api_key')

    commenter_prompt = prompts[0]
    reviser_prompt = prompts[1]

    platform = "gemini" if "gemini" in model.lower() else "openai"

    vlm = VLMCommenter(api_key, commenter_prompt, provider=platform, model_name=model)
    reviser = LLMReviser(api_key, reviser_prompt, provider=platform, model_name=model)

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
                # 获取 input_path (最原始文件名)
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
            
            # 出错后中断循环
            break
        current_input = output_path
        
        # 等待一会，防止并发读写问题
        time.sleep(1)
    
    # 优化结束后输出最终截图
    final_screenshot_path = f"{Path(output_path).parent}/{Path(current_input).stem}_final.png"
    take_screenshot(current_input, final_screenshot_path)
    print(f"\n📷 Final screenshot saved: {final_screenshot_path}")


def refinement_ppt(input_index, prompts, max_iterations=3, model="gpt-4o", config=None):
    # 1. 定义路径
    outline_path = os.path.join(input_index, "outline.json")
    output_index = os.path.join(input_index, "final")
    output_index_images = os.path.join(output_index, "images") # 保存图片的子目录，用于显示refinement后的html中的图片

    # 确保输出目录存在
    os.makedirs(output_index, exist_ok=True)

    # 将图片复制到final/images目录下
    import shutil
    source_images_dir = os.path.join(input_index, "images")
    if os.path.exists(source_images_dir):
        shutil.copytree(source_images_dir, output_index_images, dirs_exist_ok=True)
        print(f"📁 Copied images to: {output_index_images}")

    # 2. 加载大纲数据
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
    
    # 3.1 先过滤出所有符合 "数字_ppt.html" 格式的文件
    target_files = []
    for f in os.listdir(input_index):
        # 严格匹配：数字开头 + _ppt.html 结尾
        if re.search(r'^\d+_ppt\.html$', f):
            target_files.append(f)
    
    # 3.2 定义排序 Key：直接提取开头的数字
    def get_file_number(filename):
        # 因为上一步已经过滤过了，这里可以直接提取
        return int(filename.split('_')[0])

    # 3.3 执行排序 (这步是关键，确保 2 在 10 前面)
    sorted_files = sorted(target_files, key=get_file_number)

    # Debug: 打印前几个文件确认顺序
    print(f"👀 排序后文件列表前5个: {sorted_files[:5]}")
    
    # 4. 遍历排序后的列表
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
            refine_one_slide(
                input_path=html_file_path, 
                output_path=html_file_path_refine, 
                prompts=prompts, 
                outline=outline, 
                max_iterations=max_iterations,
                model=model,
                config=config
            )
        except Exception as e:
            print(f"❌ 处理 {file_name} 时出错: {e}")

    print(f"✅ 所有文件处理完成，结果保存在: {output_index}")

def refinement_poster(input_html_path, prompts, output_html_path, model, config=None):
    # ---------------- 0. 配置准备 ----------------
    if config is None:
        config = {}
    
    api_keys_conf = config.get('api_keys', {})
    
    # 判别平台
    is_gemini = "gemini" in model.lower()
    
    # ---------------- 1. 路径与文件准备 ----------------
    auto_path = Path(input_html_path).parent
    final_index = os.path.join(auto_path, "final")
    final_index_image = os.path.join(final_index, "images")
    os.makedirs(final_index, exist_ok=True)
    
    # 复制图片文件夹
    source_images_dir = os.path.join(auto_path, "images")
    if os.path.exists(source_images_dir):
        if not os.path.exists(final_index_image):
            shutil.copytree(source_images_dir, final_index_image, dirs_exist_ok=True)
            print(f"📁 Images copied to: {final_index_image}")
    
    with open(input_html_path, 'r', encoding='utf-8') as f:
        current_html = f.read()
    
    # ---------------- 2. 截图逻辑 (保持不变) ----------------
    screenshot_name = Path(input_html_path).stem + ".png"
    screenshot_path = os.path.join(final_index, screenshot_name)
    
    print(f"📸 Taking screenshot of {input_html_path}...")
    # 假设 take_screenshot_poster 是外部定义的函数
    take_screenshot_poster(input_html_path, screenshot_path)
    
    if not os.path.exists(screenshot_path):
        raise FileNotFoundError("Screenshot failed to generate.")

    # 读取截图数据
    with open(screenshot_path, "rb") as f:
        image_bytes = f.read()

    generated_text = ""

    # ---------------- 3. 调用 LLM ----------------
    print(f"🤖 Sending to Vision Model ({model}) on {'Gemini' if is_gemini else 'OpenAI'}...")
    
    try:
        if is_gemini:
            # === Gemini Client Setup ===
            api_key = api_keys_conf.get('gemini_api_key') or os.getenv("GOOGLE_API_KEY")
            
            client = genai.Client(api_key=api_key)
            
            # 构造 Gemini 所需的 Contents
            # 新版 SDK (google.genai) 推荐的构造方式
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_text(text=prompts),
                    types.Part.from_text(text=f"--- CURRENT HTML ---\n{current_html}"),
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ]
            )
            
            if response.text:
                generated_text = response.text
            else:
                raise RuntimeError("Gemini returned empty text.")

        else:
            # === OpenAI Client Setup ===
            api_key = api_keys_conf.get('openai_api_key') or os.getenv("OPENAI_API_KEY")
            
            client = OpenAI(api_key=api_key)

            # OpenAI 需要 Base64 编码的图片
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert web designer and code refiner."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": f"{prompts}\n\n--- CURRENT HTML ---\n{current_html}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096 
            )
            generated_text = response.choices[0].message.content

        # ---------------- 4. 解析与保存结果 ----------------
        # 清洗 Markdown 代码块标记
        if "```html" in generated_text:
            final_html = generated_text.split("```html")[1].split("```")[0].strip()
        elif "```" in generated_text:
            final_html = generated_text.split("```")[1].strip()
        else:
            final_html = generated_text

        with open(output_html_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
            
        print(f"✅ Refined poster saved to: {output_html_path}")

        # 生成最终截图
        final_screenshot_name = Path(input_html_path).stem + "_final" + ".png"
        final_screenshot_path = os.path.join(final_index, final_screenshot_name)

        print(f"📸 Taking final poster screenshot of {output_html_path}...")
        take_screenshot_poster(output_html_path, final_screenshot_path)

    except Exception as e:
        print(f"❌ Error during AI generation: {e}")

def refinement_pr(pr_path: str, pr_refine_path: str, prompts: dict, model: str, config: dict):
    """
    提取Markdown中的特定章节，使用LLM根据传入的prompts指令进行优化，并重组文件。
    严格保留Markdown原有结构、图片引用以及未被选中的尾部内容（如Hashtags）。
    """
    
    # 1. (修改) 获取配置，不再依赖环境变量，API Key在调用前再具体提取
    if config is None:
        config = {}
    api_keys = config.get('api_keys', {})

    # 2. 读取原始文件
    if not os.path.exists(pr_path):
        raise FileNotFoundError(f"文件未找到: {pr_path}")
        
    with open(pr_path, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # 3. 定义部分标题映射
    section_headers = {
        "Key Question": r"🔍 \*\*Key Question\*\*",
        "Brilliant Idea": r"💡 \*\*Brilliant Idea\*\*",
        "Core Methods": r"🚀 \*\*Core Methods\*\*",
        "Core Results": r"📊 \*\*Core Results\*\*",
        "Significance/Impact": r"🧠 \*\*Significance/Impact\*\*"
    }
    
    footer_pattern = r"🏷️\s*\*\*Hashtag\*\*" 

    # 4. 定位 核心标题 位置
    matches = []
    for key, pattern in section_headers.items():
        found = list(re.finditer(pattern, original_content))
        if found:
            match = found[0]
            matches.append({
                "key": key,
                "header_start": match.start(),
                "header_end": match.end(),
                "header_text": match.group()
            })
    
    matches.sort(key=lambda x: x["header_start"])
    
    if not matches:
        print("未检测到目标章节，直接复制文件。")
        with open(pr_refine_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        return

    # 定位 Footer (Hashtag) 位置
    footer_match = re.search(footer_pattern, original_content)
    if footer_match:
        global_content_end_limit = footer_match.start()
    else:
        print("Warning: 未检测到 '🏷️ **Hashtag**' 标记，最后一个章节将读取至文件末尾。")
        global_content_end_limit = len(original_content)

    # 5. 精确计算每个章节的“内容”范围
    content_ranges = {} 
    for i, match in enumerate(matches):
        key = match["key"]
        content_start = match["header_end"]
        if i < len(matches) - 1:
            content_end = matches[i+1]["header_start"]
        else:
            content_end = max(content_start, global_content_end_limit)
        
        content_ranges[key] = {
            "start": content_start,
            "end": content_end,
            "text": original_content[content_start:content_end].strip()
        }

    # 6. 构建 LLM 请求
    extracted_data = {k: v["text"] for k, v in content_ranges.items()}
    
    system_prompt = (
        "You are an expert academic editor. Your task is to refine the content of specific sections of a paper summary based on user instructions.\n"
        "Input Format: JSON object {Section Name: Content}.\n"
        "Output Format: JSON object {Section Name: Refined Content}.\n"
        "CRITICAL RULES:\n"
        "1. **KEYS**: Keep the JSON keys EXACTLY the same as the input.\n"
        "2. **PURE BODY TEXT**: The output value must be pure body text. No Headers.\n"
        "3. **IMAGES**: Do NOT remove or modify markdown image links.\n"
        "4. **JSON ONLY**: Output pure JSON string.\n"
        "5. **FORMAT**: Use bolding ONLY for emphasis."
    )
    
    user_message = f"""
    [Refinement Instructions]
    {json.dumps(prompts, ensure_ascii=False)}

    [Content to Refine]
    {json.dumps(extracted_data, ensure_ascii=False)}
    """

    # === (修改) 核心：根据模型类型分流调用 ===
    llm_output = ""
    try:
        if "gemini" in model.lower():
            # --- Google Gemini (New SDK) ---
            api_key = api_keys.get("gemini_api_key", "").strip()
            
            if not api_key:
                raise ValueError("Missing config['api_keys']['gemini_api_key']")
            
            from google import genai
            from google.genai import types
            
            # 配置客户端            
            client = genai.Client(api_key=api_key)
            
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    response_mime_type="application/json" # 强制 JSON 模式，提高稳定性
                )
            )
            llm_output = response.text
            
        else:
            # --- OpenAI (Original) ---
            api_key = api_keys.get("openai_api_key", "").strip()
            
            if not api_key:
                 # 兼容性回退：如果config里没有，尝试读环境变量
                api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("Missing config['api_keys']['openai_api_key']")
                
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
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
        # 移除可能存在的 markdown 代码块标记
        cleaned_output = llm_output.replace("```json", "").replace("```", "").strip()
        refined_data = json.loads(cleaned_output)
    except json.JSONDecodeError:
        print("解析 LLM 返回的 JSON 失败。Raw output:", llm_output)
        return

    # 8. 重组文件
    new_file_parts = []
    current_idx = 0
    
    # 按照原文件中的出现顺序处理
    sorted_matches = sorted(matches, key=lambda x: x["header_start"])
    
    for item in sorted_matches:
        key = item["key"]
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

        new_file_parts.append('\n')
            
        # 3. 更新游标
        current_idx = c_end

    # 9. 添加文件剩余的所有内容
    new_file_parts.append(original_content[current_idx:])
    
    final_markdown = "".join(new_file_parts)
    
    # 10. 保存结果
    os.makedirs(os.path.dirname(os.path.abspath(pr_refine_path)), exist_ok=True)
    
    with open(pr_refine_path, 'w', encoding='utf-8') as f:
        f.write(final_markdown)
        
    print(f"文件优化完成，已保存至: {pr_refine_path}")

