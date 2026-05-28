"""
Space LLM Evaluation
Compute perplexity on held-out data and generate sample outputs.
"""

import os
import json
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from config import ModelConfig, GenerateConfig, load_config
from model.transformer import SpaceLLM
from generate import load_model, load_tokenizer, generate_text


def compute_perplexity(model, data_path: str, seq_len: int, batch_size: int = 32, device=None):
    """Compute perplexity on validation set."""
    if device is None:
        device = next(model.parameters()).device

    meta_path = Path(data_path) / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    val_data = np.memmap(
        Path(data_path) / "val.bin",
        dtype=np.uint16,
        mode="r",
        shape=meta["val_tokens"],
    )

    # Create batches
    n_batches = min(100, (len(val_data) - seq_len - 1) // (batch_size * seq_len))
    total_loss = 0.0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size * seq_len
            chunks = []
            targets = []
            for b in range(batch_size):
                s = start + b * seq_len
                chunk = val_data[s : s + seq_len + 1].astype(np.int64)
                chunks.append(chunk[:-1])
                targets.append(chunk[1:])

            x = torch.tensor(np.stack(chunks), dtype=torch.long, device=device)
            y = torch.tensor(np.stack(targets), dtype=torch.long, device=device)

            _, loss, _ = model(x, targets=y)
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return avg_loss, perplexity


SPACE_QUESTIONS = [
    "What is the largest planet in our solar system?",
    "How far is the Sun from Earth?",
    "What causes a solar eclipse?",
    "What is a black hole?",
    "How do stars produce energy?",
    "What is the Milky Way galaxy?",
    "What is dark matter?",
    "How was the universe created?",
    "What is the International Space Station?",
    "What is the James Webb Space Telescope?",
    "Tell me about Mars exploration.",
    "What are gravitational waves?",
    "What is the Hertzsprung-Russell diagram?",
    "How do rockets work?",
    "What is the cosmic microwave background?",
    "What is an exoplanet?",
    "How does nuclear fusion power the Sun?",
    "What happens when a star dies?",
    "What is the Drake equation?",
    "What is the Kuiper Belt?",
]


def run_sample_generations(model, tokenizer, device=None):
    """Generate sample outputs for qualitative evaluation."""
    if device is None:
        device = next(model.parameters()).device

    gen_config = GenerateConfig(
        temperature=0.7,
        top_k=40,
        top_p=0.85,
        max_new_tokens=150,
    )

    print("\n" + "=" * 60)
    print("SAMPLE GENERATIONS")
    print("=" * 60)

    results = []
    for i, question in enumerate(SPACE_QUESTIONS[:10]):
        prompt = f"Question: {question}\nAnswer:"
        response = generate_text(model, tokenizer, prompt, gen_config, device)

        print(f"\n[{i+1}] {question}")
        print(f"    {response}")
        results.append({"question": question, "answer": response})

    return results


def evaluate(
    checkpoint_dir: str = "checkpoints",
    data_dir: str = "data/tokenized",
    tokenizer_path: str = "data/tokenizer/space_tokenizer.model",
):
    """Run full evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, model_config = load_model(checkpoint_dir, device)
    tokenizer = load_tokenizer(tokenizer_path)

    # Perplexity
    print("\nComputing perplexity on validation set...")
    val_loss, perplexity = compute_perplexity(
        model, data_dir, model_config.max_seq_len, device=device
    )
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

    # Sample generations
    samples = run_sample_generations(model, tokenizer, device)

    # Save results
    results = {
        "val_loss": val_loss,
        "perplexity": perplexity,
        "num_params": model.get_num_params(),
        "samples": samples,
    }

    results_path = os.path.join(checkpoint_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Space LLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoints")
    parser.add_argument("--data", type=str, default="data/tokenized")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/space_tokenizer.model")
    args = parser.parse_args()

    evaluate(args.checkpoint, args.data, args.tokenizer)
