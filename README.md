<div align="center">
 
<img width="705" height="257.85" alt="label" src="https://github.com/user-attachments/assets/3793a506-2b92-4b35-a285-fe788c7130a0" />


# The Next Step in Multimodal Academic Generation
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-yutao1024%2FPaperX-24292F?style=flat-square&logo=github&logoColor=white)](https://github.com/yutao1024/PaperX)
[![arXiv](https://img.shields.io/badge/arXiv-2602.03866-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](http://arxiv.org/abs/2602.03866)

✨ **Focus on Multimodal Academic Presentation Generation: from paper PDFs to PPTs, Posters, and PRs** ✨  

| 🧩 **Unified Framework** &nbsp;|&nbsp; 💰 **Low-Cost Efficiency** &nbsp;|&nbsp; 🌈 **Modern & Trendy Design** &nbsp;|&nbsp; 🔀 **Multi-Modal Support** |

</div>

## 📑 Table of Contents
<font size=7><div align='center' >  

[[🔥 News](#-news)]  [[🌈 Overview](#-overview)]   [[🧩 Demo](#-demo)]  [[🚀 Quick Start](#-quick-start)]  [[📸 Showcase](#-showcase)] 

</div></font>

## 🔥 News

**[2026/02/05]** 📄 PaperX is now available on arXiv: https://arxiv.org/abs/2602.03866

## 🌈 Overview    

![pipeline](https://github.com/user-attachments/assets/d38c0a63-89f2-4a37-8384-2b363aecf2fa)

## 🧩 Demo

🔜 The demo is coming soon.

## 🚀 Quick Start

### Installation

#### Clone this repository and navigate to the folder:
```bash
git clone https://github.com/yutao1024/PaperX.git
cd PaperX
```
#### Create New Environment
To ensure reproducibility and avoid dependency conflicts, we strongly recommend creating a dedicated Python environment for this project before installation.
```
conda create -n PaperX python=3.10
```
#### Install Mineru
This project relies on **Mineru** for document parsing and structured content extraction.  
Please make sure Mineru is correctly installed before running the pipeline.
```
pip install --upgrade pip  
pip install uv  
uv pip install -U "mineru[all]"
```
If you have any questions about using MinerU, please visit:
https://deepwiki.com/opendatalab/MinerU
#### Install dependencies
After setting up the environment and installing Mineru, install the required Python dependencies for this project.  
All dependencies are listed in `requirements.txt`. You can install them using the following command:

```bash
pip install -r requirements.txt
```
Alternatively, you can install the dependencies individually using the following commands.
```
pip install openai  
pip install pillow  
pip install -U google-genai  
python -m pip install -U beautifulsoup4  
pip install playwright  
playwright install
```

#### Place the PDF papers to be processed in the papers directory
```
papers/  
 -paper1.pdf  
 -paper2.pdf  
 -...
```
### Launch Application
#### Set up environment variables
Before running the project, please configure the required environment variables as shown below
```
export MINERU_FORMULA_ENABLE=false  
export MINERU_TABLE_ENABLE=false  
export OPENAI_API_KEY="<Your api key>"  
export OPENAI_BASE_URL="<Your url>"
```
#### Parse PDF files
Use the following command to parse PDF files using **Mineru** and convert them into structured outputs for DAG construction.
```
mineru -p papers -o mineru_outputs --sourcelocal -b pipeline
```
#### Run the program
```
python main.py
```
### Results
You can find the resulting PPT in the following directory:
```
auto/final/<ppt_number>_ppt_final.png
```
You can find the resulting Poster in the following directory:
```
auto/final/poster_final.png
```
You can find the resulting PR in the following directory:
```
auto/markdown_refinement.md
```
### Evaluation
#### PPT
Move the final slides for evaluation:  
```bash
cd PaperX/evaluation/PPTAgent/  
python move_ppt.py  
```
Run the PPTAgent-eval metrix:  
```bash
# For single paper evaluation
python run_benchmark.py /path/to/specific_paper_folder
# For batch papers evaluation
python run_benchmark.py /path/to/root_folder --batch
```
Run the existing metrix:  
```bash
# For single paper evaluation
python run_exist_metrix.py /path/to/specific_paper_folder --token "Your HuggingFace Token"
# For batch papers evaluation
python run_exist_metrix.py /path/to/root_folder --batch --token "Your HuggingFace Token"
```
Aggregate all evaluation results:  
```bash
python calculate_avg.py
```
For reference to the original **Evaluation** of PPT, please click [here](https://github.com/icip-cas/PPTAgent).  
#### Poster  
Clone the Paper2Poster repository to the desired location
```bash
cd evaluation
git clone https://github.com/Paper2Poster/Paper2Poster.git
cd Paper2Poster
```
Download Paper2Poster evaluation dataset via:
```bash
python -m PosterAgent.create_dataset
```
Create a folder named PaperX_generated_posters and copy the Paper2Poster-data directory into it.
For each subdirectory inside Paper2Poster-data, keep only the paper.pdf file and delete all other files.
```bash
# Create the target folder
mkdir -p PaperX_generated_posters

# Copy Paper2Poster-data into it
cp -r Paper2Poster-data PaperX_generated_posters/

# For each subfolder, keep only paper.pdf and remove all other files
find PaperX_generated_posters/Paper2Poster-data -type f ! -name "paper.pdf" -delete

```

Use src/transfer_poster.py to move the generated poster results to Paper2Poster/PaperX_generated_posters/Paper2Poster-data
```bash
python transfer_poster.py
```

Start evaluation
```bash
# Terminal 1:
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 7000 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.95

# Terminal 2:
python src/run_paper2poster_benchmark.py --dir_a Paper2Poster/PaperX_generated_posters/Paper2Poster-data --dir_b Paper2Poster/eval_results
```



#### PR  
Move the final PRs for evaluation:  
```bash
cd PaperX/evaluation/AutoPR/  
python move_pr.py
```
Edit the .env file with your API credentials:  
```
# Main API Base URL for text and vision models (e.g., OpenAI, Qwen, etc.)
OPENAI_API_BASE="https://api.openai.com/v1"
# Your API Key
OPENAI_API_KEY="sk-..."
```
Download the PRBench dataset from Hugging Face Hub. 
```bash
python download_and_reconstruct_prbench.py \
    --repo-id yzweak/PRBench \
    --subset core \ # or "full"
    --output-dir eval
```
Run PR evaluation:  
```
chmod +x scripts/run_eval.sh  
./scripts/run_eval.sh
```
Calculate and View Metrics:  
```
chmod +x scripts/calc_results.sh  
./scripts/calc_results.sh  
```
For reference to the original **Evaluation** of PR, please click [here]([https://github.com/icip-cas/PPTAgent](https://github.com/LightChen233/AutoPR)).  

## 📸 Showcase
### PPTs

<div align="center">
 
<img width="12975" height="3994" alt="PPT_case1" src="https://github.com/user-attachments/assets/18a81666-e365-4d53-9d0f-cb3d50446d48" />
<img width="12975" height="3994" alt="PPT_case2" src="https://github.com/user-attachments/assets/143ef459-aad5-454d-a40b-0ee76184306f" />
<img width="12975" height="3994" alt="PPT_case3" src="https://github.com/user-attachments/assets/21e62b9b-2305-4953-b385-3a0ae29119c9" />

</div>

### Posters
<div align="center">
 
<img width="9756" height="3369" alt="Poster_case" src="https://github.com/user-attachments/assets/f609bc44-546c-46b9-ad99-aaddfd335f2b" />

</div>

### PRs

<div align="center">

 <img width="13563" height="6581" alt="PR_case" src="https://github.com/user-attachments/assets/d3f5f216-2ac0-4ae3-8a6f-268d6995f390" />

</div>

### PaperX with Nano Banana








 
   
