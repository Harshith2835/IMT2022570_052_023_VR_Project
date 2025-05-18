# AIM825 Course Project: Visual Question Answering with Amazon Berkeley Objects Dataset

## Project Overview
This repository contains the implementation of a Visual Question Answering (VQA) system developed as part of the AIM825 course at IIIT Hyderabad. The project leverages the Amazon Berkeley Objects (ABO) dataset to create a high-quality VQA dataset and evaluates three vision-language models—BLIP, BLIP-2, and ViLT—using baseline and fine-tuned (LoRA) setups. The system achieves up to 67.35% accuracy with fine-tuned BLIP, demonstrating robust visual-semantic alignment.

### Team Members:
- Mupparapu Koushik (IMT2022570)
- Udayagiri Narayana Srimanth (IMT2022052)
- Ananthula Harshith Reddy (IMT2022023)

## Objectives
- **Data Curation:** Create a VQA dataset with 24,312 QA pairs (19,464 train, 4,848 validation) using the ABO small variant (147,702 listings, 398,212 images).
- **Baseline Evaluation:** Assess pre-trained BLIP, BLIP-2, and ViLT models on the dataset without fine-tuning.
- **Fine-Tuning:** Improve model performance using Low-Rank Adaptation (LoRA) within Kaggle's 2x16GB GPU constraints.
- **Evaluation:** Analyze model performance using Accuracy, F1 Score, BERTScore, and WUP Score, with detailed error analysis by question type.

## Methodology

### 1. Data Curation
- **Merging:** Combined product metadata (listings_0.json) and image metadata (images.csv) into cleaned_vqa_metadata_with_images.json using merging_final.py.
- **QA Generation:** Generated 3 diverse QA pairs per image (descriptive, counting, color, function, reasoning) using Gemini 2.0 API with prompt_final.py.
- **Dataset:** Produced vqa_training_data_complete.json with 24,312 QA pairs across 8,104 images, ensuring single-word answers.

### 2. Model Choices
- **BLIP:** 387M parameters, optimized for VQA (Salesforce/blip-vqa-base).
- **BLIP-2:** 2.7B parameters, robust zero-shot performance (Salesforce/blip2-opt-2.7b).
- **ViLT:** 118M parameters, lightweight VQA model (dandelin/vilt-b32-finetuned-vqa).

### 3. Baseline Evaluation
- **Dataset Split:** 80:20 (19,464 train, 4,848 validation).
- **Metrics:** Accuracy, F1 Score, BERTScore, WUP Score.
- **Results:**
  - BLIP: 52.10% accuracy, strong in color/yes-no questions.
  - BLIP-2: 48.74% accuracy, struggles with counting.
  - ViLT: 36.08% accuracy, weak in complex queries.

### 4. Fine-Tuning with LoRA
- **Setup:** Rank=16, alpha=16 (BLIP, BLIP-2), 32 (ViLT), targeting attention layers.
- **Optimizations:** KV Cache, mixed precision (FP16 for BLIP-2).
- **Results:**
  - BLIP: 67.35% accuracy (+15.25%), excels in yes/no (85.44%).
  - BLIP-2: 54.52% accuracy (+5.78%), improved counting (52.28%).
  - ViLT: 40.99% accuracy (+4.91%), limited gains.

### 5. Evaluation and Error Analysis
- **Metrics:** Quantified performance improvements and question-type-specific gains (e.g., BLIP's 73.30% in color questions).
- **Error Analysis:** Identified weaknesses in counting (BLIP: 50% error rate) and complex queries (ViLT: 14.43% in OTHER).
- **Visualizations:** F1 histograms, question-type boxplots (metric_distributions.png).

## Repository Structure
The repository contains the following files:

| File | Description |
|------|-------------|
| blip_baseline_final.ipynb | Baseline evaluation of BLIP model. |
| blip_lora_final.ipynb | Fine-tuning BLIP with LoRA and evaluation. |
| blip2_baseline_final.ipynb | Baseline evaluation of BLIP-2 model. |
| blip2_lora_final.ipynb | Fine-tuning BLIP-2 with LoRA and evaluation. |
| vilt_baseline_final.ipynb | Baseline evaluation of ViLT model. |
| vilt_lora_final.ipynb | Fine-tuning ViLT with LoRA and evaluation. |
| merging_final.py | Merges ABO product and image metadata. |
| prompt_final.py | Generates QA pairs using Gemini 2.0 API. |
| conversion.py | Converts JSON dataset to CSV (vqa_dataset.csv). |
| vqa_training_data_complete.json | VQA dataset with 24,312 QA pairs. |
| vqa_training_data_complete.csv | CSV version of the dataset. |
| Project_Report.pdf | Detailed project report (as submitted). |

## Installation and Setup

### Prerequisites
- **Hardware:** Kaggle notebook with 2x16GB GPUs (or equivalent).
- **Software:**
  - Python 3.8+
  - Libraries: torch, transformers, google-generativeai, pandas, numpy, tqdm, requests, base64
  - Gemini 2.0 API key
- **Dataset:** ABO small variant (download from Amazon Berkeley Objects).

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Harshith2835/IMT2022570_052_023_VR_Project.git
cd IMT2022570_052_023_VR_Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Gemini API key in prompt_final.py:
```bash
export GOOGLE_API_KEY='your-api-key'
```

## Usage

### Data Curation:
1. Run merging_final.py to merge metadata:
```bash
python merging_final.py
```

2. Run prompt_final.py to generate QA pairs:
```bash
python prompt_final.py --start_index 3078 --end_index 4500
```

### Baseline Evaluation:
- Execute Jupyter notebooks (blip_baseline_final.ipynb, blip2_baseline_final.ipynb, vilt_baseline_final.ipynb).

### Fine-Tuning and Evaluation:
- Run LoRA fine-tuning notebooks (blip_lora_final.ipynb, blip2_lora_final.ipynb, vilt_lora_final.ipynb).

### Inference:
- Use inference.py (available in deliverables) for predictions with fine-tuned BLIP:
```bash
python inference.py --model blip_lora --input vqa_dataset.csv
```

## Results

- **Dataset:** 24,312 QA pairs, diverse across descriptive, counting, color, function, and reasoning questions.
- **Fine-Tuned Performance:**
  - BLIP (LoRA): 67.35% accuracy, 97.45% BERTScore.
  - BLIP-2 (LoRA): 54.52% accuracy, improved counting (+25.35%).
  - ViLT (LoRA): 40.99% accuracy, limited gains in complex queries.
- **Optimizations:** KV Cache, mixed precision (FP16), LoRA reduced memory demands.

## Future Work

- **Dataset Expansion:** Increase to 50,000 QA pairs with more counting and functional questions.
- **Model Enhancements:** Explore larger models (e.g., BLIP-2 OPT-6.7B) or ensembles.
- **Advanced Metrics:** Implement Attention-Based Correctness (ABC) for visual grounding.
- **Hyperparameter Tuning:** Broader LoRA search (e.g., r=64, varying learning rates).
