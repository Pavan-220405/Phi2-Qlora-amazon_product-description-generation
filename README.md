### Download and open the ipynb file locally, or else open the PDF
# Instruction Fine-Tuning Phi-2 using QLoRA for Amazon Product Description,Name Generation 

This repository demonstrates how to fine-tune the Phi-2 language model on custom Amazon product data using QLoRA (Quantized Low-Rank Adaptation). The goal is to efficiently adapt a large language model to generate product names and descriptions while minimizing GPU memory usage.

---

## Project Overview

Product title and description generation is a practical text generation task in e-commerce. In this project, the Phi-2 model is instruction fine-tuned on structured product data using QLoRA, enabling efficient training by combining low-bit quantization with LoRA adapters. The notebook focuses on building a scalable and memory-efficient fine-tuning pipeline rather than full model retraining.

---

## Dataset

- Custom dataset containing:
  - Amazon product names
  - Corresponding product descriptions
- Data is formatted into instruction-style or prompt–completion pairs suitable for causal language modeling
- Used for supervised fine-tuning of Phi-2

---

## Model Used

### Phi-2
- Lightweight large language model developed by Microsoft
- Decoder-only transformer architecture
- Suitable for instruction tuning and text generation tasks

---

## Fine-Tuning Approach

### QLoRA (Quantized Low-Rank Adaptation)

- Base model weights loaded in low-bit precision using `bitsandbytes`
- Trainable LoRA adapters injected into selected transformer layers (Wqkv,fc1,fc2)
- Enables fine-tuning large models on limited GPU memory
- Original model weights remain frozen

---

## Workflow

1. **Environment Setup**
   - PyTorch
   - Hugging Face Transformers, Datasets, and PEFT
   - BitsAndBytes for low-bit quantization
   - Accelerate for efficient training

2. **Data Preparation**
   - Loading custom Amazon product dataset
   - Formatting data into prompt–target text pairs
   - Tokenization for causal language modeling

3. **Model Loading**
   - Phi-2 loaded with quantization configuration
   - QLoRA adapters configured using PEFT

4. **Fine-Tuning**
   - Supervised fine-tuning using Hugging Face Trainer
   - Training only LoRA parameters
   - Efficient memory usage suitable for Colab GPUs

5. **Evaluation**
   - Monitoring training and validation loss
   - Qualitative inspection of generated product text

6. **Model Saving**
   - Saving LoRA adapters
   - Option to merge adapters with base model for inference

---

## Results

- The fine-tuned model learns domain-specific patterns in Amazon product data
- Generates coherent product names and descriptions aligned with training format
- Demonstrates effective adaptation using QLoRA with minimal GPU memory overhead

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face PEFT
- BitsAndBytes
- Accelerate
- Google Colab (GPU)

---
