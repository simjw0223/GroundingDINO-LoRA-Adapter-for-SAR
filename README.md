<img width="1024" height="1536" alt="image" src="https://github.com/user-attachments/assets/5624baa6-fed7-4fea-8039-701a69cbb1cf" />


---


# 🛰️ GroundingDINO + LoRA Adapter for SAR Ship Detection

This repository provides **LoRA-based adapter weights** for specializing the original  
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) model for  
**SAR (Synthetic Aperture Radar) ship detection**.

The goal of this project is to **retain Grounding DINO’s open-set, text-prompt-guided detection capability**  
while adapting it to the unique characteristics of SAR images such as **speckle noise, low contrast,  
and grayscale representation**.

---

## 🔍 Key Features

- 🚀 Lightweight and parameter-efficient fine-tuning with **LoRA**
- 📡 Focused on **ship detection in SAR imagery**
- 🧩 Fully compatible with the **original Grounding DINO inference pipeline**
- 🔍 Supports **natural language detection prompts** (e.g., _"ship"_, _"vessel"_)

---

---

## 📌 Model Information

| Item            | Description                                 |
|-----------------|-----------------------------------------------|
| Base model      | GroundingDINO (`<modify: Swin-T / Swin-B>`)  |
| Fine-tuning     | LoRA (Low-Rank Adaptation)                   |
| Domain          | SAR (Synthetic Aperture Radar)               |
| Task            | Ship / Vessel Detection                      |
| Dataset         | `<modify: Private / Public / Custom>`        |

---

## 🔗 Upstream Reference

This project builds upon the official Grounding DINO repository:

📎 https://github.com/IDEA-Research/GroundingDINO

**Paper:**  
> Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection  
> https://arxiv.org/abs/2303.05499

---

## 🛠️ Installation

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

conda create -n groundingdino python=3.10 -y
conda activate groundingdino
pip install -r requirements.txt
