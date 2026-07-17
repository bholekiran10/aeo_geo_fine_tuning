# Domain AI Assistant Fine-Tuning

## 1) Project Title
Domain AI Assistant Fine-Tuning for AEO/GEO

## 2) Domain Selected
AEO (Answer Engine Optimization) and GEO (Generative Engine Optimization)

## 3) Business Problem
Build an internal AI assistant that gives accurate, actionable, and domain-specific answers for AEO/GEO use cases, improving over a generic base model.

## 4) Dataset Details
- Non-instruction corpus: `data/aeo_geo_fine_tuning_corpus.pdf`
- Instruction dataset: `data/instruction_dataset.jsonl` with `instruction/response`
- Preference dataset: `data/preference_dataset.jsonl` with `prompt/chosen/rejected`

## 5) Base Model Used
- Primary target: `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`
- Open fallback: `Qwen/Qwen2.5-0.5B-Instruct`

## 6) Non-Instruction Fine-Tuning Approach
- Read and clean domain corpus text
- Chunk for causal LM training
- Run Stage 1 adaptation with LoRA/QLoRA-style setup

## 7) Instruction Fine-Tuning Approach
- Load instruction-response JSONL
- Format assistant-style training examples
- Run SFT with LoRA adapters and save Stage 2 artifact

## 8) DPO Alignment Approach
- Load Stage 2 SFT model
- Load preference pairs (prompt/chosen/rejected)
- Run DPO training and save Stage 3 aligned artifact

## 9) LoRA / QLoRA Configuration
Typical values used:
- `r`: 16
- `alpha`: 16
- `dropout`: 0.0
- SFT `learning_rate`: 2e-4
- SFT `batch_size`: 2
- DPO `learning_rate`: 5e-6
- DPO `batch_size`: 1
- DPO `beta`: 0.1

## 10) Training Screenshots or Logs
- Notebook output cells contain training traces.
- Run/checkpoint folders:
  - `models/stage1_runs`
  - `models/stage2_runs`
  - `models/stage3_runs`

## 11) Before vs After Output Comparison
- Base vs SFT: `reports/sft_model_comparison.md`
- Base vs SFT vs DPO: `reports/final_evaluation.md`

## 12) Final Observations
- Domain adaptation + SFT increases domain specificity.
- DPO alignment improves preference-aligned answer quality and response style.

## 13) Challenges Faced
- Environment/version drift across notebook runtimes
- 4-bit quantization compatibility differences
- Fallback behavior when specific checkpoints are unavailable

## 14) Future Improvements
- Add automated evaluation metrics and regression checks
- Expand preference data for edge-case prompts
- Add deployment API and production monitoring
- Add stronger grounding/citation checks

## Quick Inference Usage
Programmatic example:

```python
from src.inference import generate_answer

question = "How necessary is to have a precise questions and answers like a faq to have in blogs or on websites?"
answer = generate_answer(question)
print(answer)
```

CLI example:

```bash
python src/inference.py --question "How necessary is to have a precise questions and answers like a faq to have in blogs or on websites?"
```
