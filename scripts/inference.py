"""
Inference script for the fine-tuned math model.

Loads the base model + LoRA adapter and runs interactive math solving.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


SYSTEM_PROMPT = """You are a precise and helpful math tutor. Given a math problem, provide a clear, step-by-step solution. Show your reasoning at each step, then give the final answer on a line starting with 'The answer is:'."""


def load_model(base_model="Qwen/Qwen2.5-Math-1.5B", adapter_path="outputs/math-model/final"):
    """Load the base model with LoRA adapter."""
    print(f"Loading base model: {base_model}")

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
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        padding_side="left",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapter
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def solve_problem(model, tokenizer, problem, max_new_tokens=1024):
    """Solve a math problem using the model."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def interactive_mode(model, tokenizer):
    """Run interactive math solving session."""
    print("\n" + "=" * 60)
    print("  Math AI Solver - Interactive Mode")
    print("=" * 60)
    print("Type a math problem and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            problem = input("Problem: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not problem:
            continue
        if problem.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nSolving...\n")
        solution = solve_problem(model, tokenizer, problem)
        print(solution)
        print("\n" + "-" * 60 + "\n")


def run_benchmark(model, tokenizer):
    """Run a quick benchmark on sample problems."""
    test_problems = [
        "Solve for x: 2x + 5 = 17",
        "What is the derivative of x^3 + 2x^2 - 5x + 1?",
        "Find the integral of 3x^2 + 4x - 7",
        "If a triangle has sides 3, 4, and 5, what is its area?",
        "Solve the system of equations: 2x + y = 7, x - y = 2",
        "What is the sum of the first 100 natural numbers?",
        "Find the limit of (sin x)/x as x approaches 0",
        "What is the probability of getting exactly 3 heads in 5 coin flips?",
    ]

    print("\n" + "=" * 60)
    print("  Running Benchmark")
    print("=" * 60 + "\n")

    for i, problem in enumerate(test_problems, 1):
        print(f"[{i}/{len(test_problems)}] {problem}")
        solution = solve_problem(model, tokenizer, problem)
        print(f"\n{solution}\n")
        print("-" * 40)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Math AI Solver")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Math-1.5B", help="Base model name")
    parser.add_argument("--adapter", default="outputs/math-model/final", help="Path to LoRA adapter")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark instead of interactive mode")
    parser.add_argument("--problem", type=str, help="Solve a single problem")
    args = parser.parse_args()

    model, tokenizer = load_model(args.base_model, args.adapter)

    if args.problem:
        solution = solve_problem(model, tokenizer, args.problem)
        print(solution)
    elif args.benchmark:
        run_benchmark(model, tokenizer)
    else:
        interactive_mode(model, tokenizer)
