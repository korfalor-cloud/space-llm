"""
Main entry point for the Math AI project.

Usage:
    python run.py prepare    - Download and prepare training data
    python run.py train      - Fine-tune the model
    python run.py solve "What is 2+2?"  - Solve a math problem
    python run.py interactive - Interactive math solver
    python run.py evaluate   - Evaluate on benchmarks
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Math AI Model")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Prepare data
    subparsers.add_parser("prepare", help="Download and prepare training data")

    # Train
    subparsers.add_parser("train", help="Fine-tune the model")

    # Solve
    solve_parser = subparsers.add_parser("solve", help="Solve a math problem")
    solve_parser.add_argument("problem", type=str, help="The math problem to solve")
    solve_parser.add_argument("--base-model", default="Qwen/Qwen2.5-Math-1.5B")
    solve_parser.add_argument("--adapter", default="outputs/math-model/final")

    # Interactive
    interactive_parser = subparsers.add_parser("interactive", help="Interactive math solver")
    interactive_parser.add_argument("--base-model", default="Qwen/Qwen2.5-Math-1.5B")
    interactive_parser.add_argument("--adapter", default="outputs/math-model/final")

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on benchmarks")
    eval_parser.add_argument("--base-model", default="Qwen/Qwen2.5-Math-1.5B")
    eval_parser.add_argument("--adapter", default="outputs/math-model/final")
    eval_parser.add_argument("--samples", type=int, default=200)
    eval_parser.add_argument("--compare", action="store_true")

    args = parser.parse_args()

    if args.command == "prepare":
        from scripts.prepare_data import prepare_datasets
        prepare_datasets()

    elif args.command == "train":
        from scripts.train import main as train_main
        train_main()

    elif args.command == "solve":
        from scripts.inference import load_model, solve_problem
        model, tokenizer = load_model(args.base_model, args.adapter)
        solution = solve_problem(model, tokenizer, args.problem)
        print(solution)

    elif args.command == "interactive":
        from scripts.inference import load_model, interactive_mode
        model, tokenizer = load_model(args.base_model, args.adapter)
        interactive_mode(model, tokenizer)

    elif args.command == "evaluate":
        from scripts.evaluate import evaluate_gsm8k, evaluate_base_vs_finetuned, load_model
        if args.compare:
            evaluate_base_vs_finetuned(args.base_model, args.adapter, args.samples)
        else:
            model, tokenizer = load_model(args.base_model, args.adapter)
            evaluate_gsm8k(model, tokenizer, args.samples)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
