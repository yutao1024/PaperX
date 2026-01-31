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














 
   
