"""
Evaluation script for the math model.

Tests the model on GSM8K and MATH benchmarks to measure accuracy.
"""

import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


SYSTEM_PROMPT = """You are a precise and helpful math tutor. Given a math problem, provide a clear, step-by-step solution. Show your reasoning at each step, then give the final answer on a line starting with 'The answer is:'."""


def extract_answer(text):
    """Extract the final numerical answer from model output."""
    # Look for "The answer is:" pattern
    patterns = [
        r"The answer is:\s*(.+?)(?:\n|$)",
        r"####\s*(.+?)(?:\n|$)",
        r"answer:\s*(.+?)(?:\n|$)",
        r"=\s*(.+?)(?:\n|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Extract just the number
            num_match = re.search(r"[-+]?\d*\.?\d+", answer)
            if num_match:
                return num_match.group()
    return None


def normalize_number(s):
    """Normalize a number string for comparison."""
    if s is None:
        return None
    try:
        return str(float(s.replace(",", "")))
    except (ValueError, AttributeError):
        return s.strip().lower()


def load_model(base_model, adapter_path):
    """Load model with LoRA adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def evaluate_gsm8k(model, tokenizer, num_samples=200):
    """Evaluate on GSM8K test set."""
    print("Loading GSM8K test set...")
    dataset = load_dataset("openai/gsm8k", split="test")

    if num_samples and len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))

    correct = 0
    total = 0
    errors = 0

    for i, example in enumerate(dataset):
        question = example["question"]
        gold_answer = example["answer"].split("####")[-1].strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_answer = extract_answer(response)

        total += 1
        if normalize_number(pred_answer) == normalize_number(gold_answer):
            correct += 1
        else:
            errors += 1

        if (i + 1) % 20 == 0:
            acc = correct / total * 100
            print(f"  [{i + 1}/{len(dataset)}] Accuracy: {acc:.1f}% ({correct}/{total})")

    accuracy = correct / total * 100
    print(f"\nGSM8K Results:")
    print(f"  Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"  Errors (no parse): {errors}")
    return accuracy


def evaluate_base_vs_finetuned(base_model_name, adapter_path, num_samples=100):
    """Compare base model vs fine-tuned model."""
    print("Loading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model.eval()

    print("\n--- Base Model ---")
    base_acc = evaluate_gsm8k(base_model, tokenizer, num_samples)

    print("\nLoading fine-tuned model...")
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    ft_model.eval()

    print("\n--- Fine-tuned Model ---")
    ft_acc = evaluate_gsm8k(ft_model, tokenizer, num_samples)

    print(f"\n{'='*40}")
    print(f"Improvement: {ft_acc - base_acc:+.1f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate math model")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--adapter", default="outputs/math-model/final")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--compare", action="store_true", help="Compare base vs fine-tuned")
    args = parser.parse_args()

    if args.compare:
        evaluate_base_vs_finetuned(args.base_model, args.adapter, args.samples)
    else:
        model, tokenizer = load_model(args.base_model, args.adapter)
        evaluate_gsm8k(model, tokenizer, args.samples)
