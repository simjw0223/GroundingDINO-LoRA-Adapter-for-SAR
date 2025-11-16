

<p align="center">
  <img width="256" height="512" alt="removed_bg" src="https://github.com/user-attachments/assets/02018f3c-6ed7-4a00-9116-dba61df0d5db" /
</p>

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

## 🎯 Improvements Achieved Through LoRA Fine-tuning

The comparison below shows how LoRA fine-tuning improves ship detection performance
on SAR imagery compared to the original pre-trained GroundingDINO checkpoint.
<p align="center">
  <img width="1578" height="769" alt="image" src="https://github.com/user-attachments/assets/de45a1a8-2099-4a11-a042-17f40a8cc67c" />
 /
</p>

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
