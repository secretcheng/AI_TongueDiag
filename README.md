# 🔍 AI Tongue Diagnosis System

> 🌟 An innovative artificial intelligence system that combines traditional Chinese medicine (TCM) tongue diagnosis with modern deep learning technology.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AI](https://img.shields.io/badge/AI-Powered-orange.svg)]()

</div>

## 🎯 Overview

This project aims to modernize and standardize traditional tongue diagnosis methods using artificial intelligence. By leveraging advanced computer vision(in the future) and machine learning techniques, the system can analyze tongue images to assist TCM practitioners in making more accurate diagnoses.

## ✨ News

<!-- - 🎨 **Automated Tongue Analysis**: Detect and analyze tongue characteristics including color, coating, shape, and texture
- ⚡ **Real-time Processing**: Instant analysis of tongue images through our advanced AI model
- 📊 **Diagnostic Assistance**: Provides detailed reports based on TCM principles
- 💾 **Data Management**: Secure storage and management of patient tongue diagnosis records -->

- 🎉 **2024.12.15**: A Chinese medical book on tongue diagnosis was used to fine-tune the Qwen2.5-7B-Insturct model, and a preliminary attempt was made to build a Chinese medicine tongue diagnosis assistant.

## 🛠️ Technical Stack(Updating)

- 🤖 LLM: Qwen2.5-7B-Insturct
- 🗄️ Tools: LLaMA-Factory, llamaindex
<!-- - 👁️ Computer Vision: OpenCV
- 🐍 Backend: Python
- 🎨 Frontend: React/Vue.js
- 🗄️ Database: MongoDB/PostgreSQL -->

## 📦 Installation

```bash
# Install LLAMA Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"

# Install dependencies
pip install -r requirements.txt

# Install the ollama
curl -fsSL https://ollama.com/install.sh | sh
## Start the server
ollama serve
## Pull the model
ollama pull qwen2.5-7b:latest

# Run the Process Code
run Dataset_Generate.ipynb and process_data.ipynb

# Run the Fine-tuning
llamafactory-cli train qwen2.5-7b-instruct-tongue-lora-sft.yaml

```

## 📝 Plan

1. 📋 Add more data to generate more instruct data
2. 📸 Add multimodal modules
3. ⏳ Add the RAG module
4. 📈 Release the weights of model(https://www.modelscope.cn/models/secretcheng/AI_Tongue/files)


## 🤝 Contributing

<!-- We welcome contributions! Please feel free to submit pull requests. -->

```
🌟 Star this repo if you find it helpful!
```

## 📄 License

This project is licensed under the Apache-2.0  License - see the LICENSE file for details.

## 📬 Contact

For any queries or suggestions, please open an issue in the repository.

---
<div align="center">
Built with ❤️ for advancing Traditional Chinese Medicine

```
     _    ___   
    / \  |_ _|  
   / _ \  | |   
  / ___ \ | |   
 /_/   \_\___|  
```
</div>
