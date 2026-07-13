import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from google.colab import drive
drive.mount('/content/drive')
ROOT = Path.cwd().parent if Path.cwd().name == "src" else Path.cwd()
#ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

DPO_MODEL_DIR = MODELS_DIR / "stage3_dpo_adapter"
SFT_MODEL_DIR = MODELS_DIR / "stage2_instruction_adapter"
FALLBACK_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


def _load_model_and_tokenizer():
    candidates = []
    if DPO_MODEL_DIR.exists():
        candidates.append(str(DPO_MODEL_DIR))
    if SFT_MODEL_DIR.exists():
        candidates.append(str(SFT_MODEL_DIR))
    candidates.extend(FALLBACK_MODELS)

    errors = []
    for model_name in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            return model, tokenizer, model_name
        except Exception as exc:
            errors.append(f"{model_name}: {exc}")

    detail = "\n".join(errors)
    raise RuntimeError(
        "Unable to load any model candidate. Checked DPO/SFT folders and open fallbacks.\n"
        f"Details:\n{detail}"
    )


MODEL, TOKENIZER, MODEL_SOURCE = _load_model_and_tokenizer()


def generate_answer(question: str, max_new_tokens: int = 220) -> str:
    prompt = (
        "You are an internal enterprise AEO and GEO assistant. "
        "Answer clearly, safely, and practically.\n"
        f"Question: {question}\n"
        "Answer:"
    )
    inputs = TOKENIZER(prompt, return_tensors="pt")
    try:
        device = MODEL.device
    except Exception:
        device = next(MODEL.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=TOKENIZER.eos_token_id,
            pad_token_id=TOKENIZER.pad_token_id,
        )

    text = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in text:
        return text.split("Answer:", 1)[-1].strip()
    return text.strip()


def _main():
    parser = argparse.ArgumentParser(description="Run AEO/GEO assistant inference")
    parser.add_argument("--question", type=str, required=False, help="Question to ask the model")
    parser.add_argument("--max_new_tokens", type=int, default=220)
    args = parser.parse_args()

    question = args.question or "How necessary is to have a precise questions and answers like a faq to have in blogs or on websites?"
    answer = generate_answer(question, max_new_tokens=args.max_new_tokens)

    print(f"Model source: {MODEL_SOURCE}")
    print(f"Question: {question}")
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    _main()
