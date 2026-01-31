# PaperX
## Requirements
### 创建新环境
conda create -n PaperX python=3.10
### 安装mineru:
pip install --upgrade pip  
pip install uv  
uv pip install -U "mineru[all]"
### 安装所需python软件包：
pip install openai  
pip install pillow  
pip install -U google-genai  
python -m pip install -U beautifulsoup4  
pip install playwright  
playwright install  
### 在papers文件夹下放置需要处理的论文pdf
papers/  
 -paper1.pdf  
 -paper2.pdf  
 -...
## Launch Application
### 设置环境变量
export MINERU_FORMULA_ENABLE=false  
export MINERU_TABLE_ENABLE=false  
export OPENAI_API_KEY="<你的api key>"  
export OPENAI_BASE_URL="<你的代理url>"  
### 解析pdf：
mineru -p papers -o mineru_outputs
### 运行:
python main.py
## Results
1. ppt:
auto/final/<ppt编号>_ppt_final.png
2. poster:
auto/final/poster_final.png
3. pr:
auto/markdown_refinement.md
## 常见问题
1. mineru报错/失败：
可以使用https://deepwiki.com/opendatalab/MinerU询问相关问题  
2. aki key无法读入：
目前只能稳定支持使用代理url的api调用方式，并且代理需要包含gpt-4o和gemini-3-pro-preview两个模型（推荐小镜ai）













 
   
