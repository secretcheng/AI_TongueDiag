# 🚀 Tongue Multi-Modal Large Model API Service

> 🌟 Welcome to the Multi-Modal Large Model API Service! This guide will help you get started with our powerful API for integrating Tongue-Multi-Modal LLM. Dive in to explore how to set up your Python environment, install necessary packages, and use the API with example code.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AI-Powered](https://img.shields.io/badge/AI-Powered-red.svg)]()

</div>

## 📚 Overview

This module is designed for developers to quickly deploy multi-modal large language models locally and leverage API services for intelligent tongue image analysis.

---

## 📦 Installation

### 1. Source Code Installation (Using Conda)

Due to variations in GPU configurations, this guide provides specific installation commands instead of a `requirements.txt` file.

#### 1.1 Python Environment

```bash
# Create a new conda environment
conda create -n aitongue python=3.10 -y
conda activate aitongue

# Install PyTorch according to your machine's CUDA version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --default-timeout=100

# Install LLaMA Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed]"

# Install qwen-vl-utils
pip install qwen-vl-utils[decord]
```

#### 1.2 Model Checkpoint Download
This model checkpoint: https://huggingface.co/secretcheng/qwen2vl_tongue_test is recommended, it is fine-tuned on Private Tongue image data.
Also you can use the original Qwen2-VL model: https://huggingface.co/Qwen/Qwen-VL-7B-Instruct. However, the format may not be consistent with the output, and the output content is slightly different.

#### 1.3 API Service Setup
> **Use LLaMA Factory**  

Configure the `qwen2-vl-inference.yaml` file. A template is provided below:

```yaml
model_name_or_path: xxx  # Provide the path to your qwen2-vl model
template: qwen2_vl
infer_backend: huggingface  

temperature: 0.01
top_p: 0.1
top_k: 20
max_new_tokens: 8192
cutoff_len: 12800
```

Start the API service with the following command:

```bash
API_PORT=8000 llamafactory-cli api qwen2-vl-inference.yaml
```

> **Use FastAPI+transformers**

```bash
python main.py
```

> **Use vllm**  

`qwen2-vl-inference.yaml` has infer_backend argument, change it to vllm  
or use this command to start the API service:
```bash
python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-VL-7B-Instruct --model xxx --gpu_memory_utilization 0.90 --max_model_len 12800 --host localhost --port 8000
```


### 2. Docker Installation

Docker installation is coming soon. Stay tuned!

---

## 🚀 Usage
Each framework starts with a slightly different API call, as described in `test_api.ipynb`. 
The following examples use the **LLaMA Factory** framework as an example.
### 1. Using cURL

```bash
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    "model": "xxx",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "描述这个舌象图片"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "xxx.jpg"
            }
          }
        ]
      }
    ],
  }"

```

### 2. Using Python(Recommended)
we also provide an example of how to use the API with Python. See details in `test_api.ipynb`
```python
import os
from openai import OpenAI

instruction_prompt = """
请根据给定的舌象图片及其标签数据，生成一段描述舌象特征的文字。描述应包含以下必填项：
- 舌质颜色
- 舌体大小（胖、胖瘦适中、瘦）
- 舌面湿润或干燥 (湿润、偏润、润燥适中、偏燥、干燥)
- 舌边有无齿痕 （有齿痕、无齿痕）
- 舌面有无裂纹、有无点刺 （有裂纹、无裂纹、有点刺、无点刺，两种症状都需要描述）
- 舌体老嫩 （嫩、偏嫩、老嫩适中、偏老、老）
- 舌苔覆盖范围 （全苔、舌根部、舌中部、舌尖部、舌左部、舌右部、无偏苔）
- 舌苔表现形式

舌苔的表现形式必须按照以下规则描述：
- 舌苔颜色 + 厚薄 + 腻腐（如：苔黄厚腻）

"""

client = OpenAI(
        api_key="{}".format(os.environ.get("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8000)),
    )
messages = []
messages.append(
    {
        "role": "system",
        "content": instruction_prompt,
    }
)
messages.append(
    {
        "role": "user",
        "content": [
            {"type": "text", "text": instruction_prompt},
            {"type": "text", "text": "描述这个舌象图片"},
            {
                "type": "image_url",
                "image_url": {"url": "/home/sc/LLM_Learning/AI_TongueDiag/data/images/0003-1.jpg"},
            },
        ],
    }
)
result = client.chat.completions.create(messages=messages, model="Qwen2-VL-7B-Instruct")
messages.append(result.choices[0].message)
print(result.choices[0].message.content)
```
输出样例如下:
```markdown
根据您输入的图片，模型分析的舌象结果是：
  -舌质颜色：淡红；
  -舌体大小：胖瘦适中；
  -舌面湿润程度：润燥适中；
  -舌边：无齿痕；
  -舌面：无裂纹、无点刺；
  -舌质老嫩程度：老嫩适中；
  -舌苔主要覆盖情况：无偏苔；
  -舌苔表现形式：苔偏白厚薄适中无腻腐
```

---

## 🛠️ Contributing

We welcome contributions to enhance the API service! Feel free to submit issues or pull requests on [GitHub](https://github.com/your-repo-link).

## 📜 License

This project is licensed under the MIT License. 

---

Enjoy exploring the capabilities of the Tongue Multi-Modal Large Model API Service! 🌟





<div align="center">
Built with ❤️ for advancing Traditional Chinese Medicine

```
     _    ___   ___              _____  _____ __  __ 
    / \  |_ _| |  ___|__  _ __ |_   _|/ ____|  \/  |
   / _ \  | |  | |_ / _ \| '__|  | | | |    | \  / |
  / ___ \ | |  |  _| (_) | |     | | | |    | |\/| |
 /_/   \_\___| |_|  \___/|_|     |_|  \____|_|  |_|
```
</div>

