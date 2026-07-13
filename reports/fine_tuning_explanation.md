# Fine-Tuning Explanation

## 1) Why full fine-tuning is expensive
Full fine-tuning updates all model parameters, which increases GPU memory usage for weights, gradients, and optimizer states. It also increases training time and storage for checkpoints, making iteration costly.

## 2) What LoRA does
LoRA freezes the original model weights and learns small low-rank adapter matrices. This reduces the number of trainable parameters while preserving most base-model capability.

## 3) What QLoRA does
QLoRA combines low-rank adapters with 4-bit quantized base weights. The base model stays memory-efficient while adapters are trained in higher precision.

## 4) Why QLoRA is useful on limited GPU
It significantly reduces VRAM requirements, enabling practical fine-tuning on smaller GPUs without full-parameter updates.

## 5) What is non-instruction fine-tuning?
Training on raw domain text without instruction/answer formatting to adapt the model's internal language and concepts to a target domain.

## 6) What is instruction fine-tuning?
Training on instruction-response pairs so the model learns to follow user prompts and produce structured, helpful answers.

## 7) What is DPO?
Direct Preference Optimization trains a model to prefer chosen responses over rejected responses directly from preference pairs, improving response quality without full RLHF complexity.

## 8) Difference between SFT and DPO
SFT teaches the model to imitate labeled target responses. DPO teaches relative preference between two candidate responses, improving alignment quality and style after SFT.

## 9) Values used in this project
Primary SFT/LoRA settings (Stage 2 notebook):
- rank (r): 16
- alpha: 16
- dropout: 0.0
- learning rate: 2e-4
- batch size: 2

Primary DPO settings (Stage 3 notebook):
- learning rate: 5e-6
- batch size: 1
- beta: 0.1

Stage 1 non-instruction configuration (existing notebook) also uses LoRA-style defaults around r=16, alpha=16, dropout=0.0 and learning rate 2e-4.

Note: these values are sourced from current notebook configs and should be updated if you change training cells.
