"""
Space LLM Text Generation
Load a trained model and generate space-related text.
"""

import os
import json
import torch
import sentencepiece as spm
from pathlib import Path
from config import ModelConfig, GenerateConfig, load_config
from model.transformer import SpaceLLM


def load_model(checkpoint_dir: str, device: torch.device = None):
    """Load model from checkpoint directory."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join(checkpoint_dir, "model_config.json")
    model_config = load_config(ModelConfig, config_path)

    model = SpaceLLM(model_config).to(device)

    ckpt_path = os.path.join(checkpoint_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(checkpoint_dir, "final.pt")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded model from {ckpt_path}")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Device: {device}")

    return model, model_config


def load_tokenizer(tokenizer_path: str):
    """Load SentencePiece tokenizer."""
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    print(f"Tokenizer vocab size: {sp.get_piece_size()}")
    return sp


def generate_text(
    model: SpaceLLM,
    tokenizer,
    prompt: str,
    gen_config: GenerateConfig = None,
    device: torch.device = None,
) -> str:
    """Generate text from a prompt."""
    if gen_config is None:
        gen_config = GenerateConfig()
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt
    input_ids = tokenizer.encode(prompt, out_type=int)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=gen_config.max_new_tokens,
            temperature=gen_config.temperature,
            top_k=gen_config.top_k,
            top_p=gen_config.top_p,
            repetition_penalty=gen_config.repetition_penalty,
        )

    # Decode
    generated_ids = output[0].cpu().tolist()
    full_text = tokenizer.decode(generated_ids)

    return full_text


def interactive_mode(model, tokenizer, gen_config=None):
    """Interactive text generation mode."""
    if gen_config is None:
        gen_config = GenerateConfig()

    print("\n" + "=" * 60)
    print("SPACE LLM - Interactive Mode")
    print("Type your prompt and press Enter. Type 'quit' to exit.")
    print("Commands: /temp <value>, /topk <value>, /topp <value>, /max <value>")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()
            if cmd == "/temp" and len(parts) > 1:
                gen_config.temperature = float(parts[1])
                print(f"Temperature set to {gen_config.temperature}")
            elif cmd == "/topk" and len(parts) > 1:
                gen_config.top_k = int(parts[1])
                print(f"Top-k set to {gen_config.top_k}")
            elif cmd == "/topp" and len(parts) > 1:
                gen_config.top_p = float(parts[1])
                print(f"Top-p set to {gen_config.top_p}")
            elif cmd == "/max" and len(parts) > 1:
                gen_config.max_new_tokens = int(parts[1])
                print(f"Max new tokens set to {gen_config.max_new_tokens}")
            else:
                print("Unknown command. Try /temp, /topk, /topp, /max")
            continue

        if not prompt:
            continue

        response = generate_text(model, tokenizer, prompt, gen_config)
        print(f"\nSpace LLM: {response}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Space LLM Text Generation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer/space_tokenizer.model", help="Tokenizer path")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt (if not interactive)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    gen_config = GenerateConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens,
    )

    model, _ = load_model(args.checkpoint)
    tokenizer = load_tokenizer(args.tokenizer)

    if args.interactive or args.prompt is None:
        interactive_mode(model, tokenizer, gen_config)
    else:
        response = generate_text(model, tokenizer, args.prompt, gen_config)
        print(response)


if __name__ == "__main__":
    main()
