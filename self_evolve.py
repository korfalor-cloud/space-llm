"""
Space LLM Self-Evolution Engine
The model evaluates itself, proposes mutations, retrains, and selects the best variant.
Runs autonomously for N generations.
"""

import os
import json
import copy
import math
import time
import random
import shutil
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from config import ModelConfig, TrainConfig, EvolveConfig, GenerateConfig, save_config, load_config
from model.transformer import SpaceLLM
from train import train, estimate_loss, TokenDataset, ValDataset, get_lr
from evaluate import compute_perplexity, run_sample_generations
from generate import load_model, load_tokenizer, generate_text
from torch.utils.data import DataLoader


EVOLUTION_DIR = Path("evolution")
EVOLUTION_DIR.mkdir(exist_ok=True)


class Mutation:
    """Represents a single mutation to the model or training config."""

    MUTATION_TYPES = [
        "adjust_d_model",
        "adjust_n_layers",
        "adjust_n_heads",
        "adjust_d_ff",
        "adjust_dropout",
        "adjust_lr",
        "adjust_activation",
        "adjust_warmup",
        "adjust_batch_size",
        "adjust_weight_decay",
        "adjust_seq_len",
    ]

    def __init__(self, mutation_type: str, param: str, old_value, new_value, description: str):
        self.type = mutation_type
        self.param = param
        self.old_value = old_value
        self.new_value = new_value
        self.description = description
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "type": self.type,
            "param": self.param,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "description": self.description,
            "timestamp": self.timestamp,
        }


class EvolutionEngine:
    """Self-evolution engine that mutates, trains, and selects model variants."""

    def __init__(self, evolve_config: EvolveConfig = None):
        self.evolve_config = evolve_config or EvolveConfig()
        self.evolution_log = []
        self.generation = 0
        self.best_score = float("inf")
        self.best_config = None

        # Load existing log if available
        log_path = Path(self.evolve_config.evolve_log)
        if log_path.exists():
            with open(log_path) as f:
                self.evolution_log = json.load(f)
            if self.evolution_log:
                self.generation = max(e.get("generation", 0) for e in self.evolution_log)
                best_entry = min(self.evolution_log, key=lambda x: x.get("score", float("inf")))
                self.best_score = best_entry.get("score", float("inf"))

    def mutate_model_config(self, config: ModelConfig) -> tuple:
        """Apply random mutations to model architecture config."""
        mutations = []
        new_config = copy.deepcopy(config)

        possible_mutations = []

        if self.evolve_config.mutate_architecture:
            possible_mutations.extend([
                ("adjust_d_model", "d_model", new_config.d_model,
                 random.choice([128, 192, 224, 256, 320, 384, 512]),
                 "Change model dimension"),
                ("adjust_n_layers", "n_layers", new_config.n_layers,
                 random.choice([4, 5, 6, 7, 8]),
                 "Change number of layers"),
                ("adjust_n_heads", "n_heads", new_config.n_heads,
                 random.choice([4, 8, 16]),
                 "Change number of attention heads"),
                ("adjust_d_ff", "d_ff", new_config.d_ff,
                 random.choice([512, 768, 1024, 1536, 2048]),
                 "Change feed-forward dimension"),
                ("adjust_dropout", "dropout", new_config.dropout,
                 round(random.uniform(0.0, 0.3), 2),
                 "Adjust dropout rate"),
                ("adjust_activation", "activation", new_config.activation,
                 random.choice(["swiglu", "gelu", "relu"]),
                 "Change activation function"),
            ])

        if self.evolve_config.mutate_hyperparams:
            possible_mutations.extend([
                ("adjust_seq_len", "max_seq_len", new_config.max_seq_len,
                 random.choice([256, 384, 512, 768, 1024]),
                 "Change sequence length"),
            ])

        # Apply random subset of mutations
        n_mutations = max(1, int(len(possible_mutations) * self.evolve_config.mutation_rate))
        selected = random.sample(possible_mutations, min(n_mutations, len(possible_mutations)))

        for mut_type, param, old_val, new_val, desc in selected:
            # Ensure d_model is divisible by n_heads
            if param == "d_model":
                while new_val % new_config.n_heads != 0:
                    new_val = random.choice([128, 192, 224, 256, 320, 384, 512])
            if param == "n_heads":
                while new_config.d_model % new_val != 0:
                    new_val = random.choice([4, 8, 16])

            setattr(new_config, param, new_val)
            mutations.append(Mutation(mut_type, param, old_val, new_val, desc))

        return new_config, mutations

    def mutate_train_config(self, config: TrainConfig) -> tuple:
        """Apply random mutations to training config."""
        mutations = []
        new_config = copy.deepcopy(config)

        possible_mutations = [
            ("adjust_lr", "learning_rate", new_config.learning_rate,
             random.choice([1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3]),
             "Change learning rate"),
            ("adjust_warmup", "warmup_steps", new_config.warmup_steps,
             random.choice([500, 1000, 2000, 3000]),
             "Change warmup steps"),
            ("adjust_batch_size", "batch_size", new_config.batch_size,
             random.choice([8, 16, 32, 64]),
             "Change batch size"),
            ("adjust_weight_decay", "weight_decay", new_config.weight_decay,
             round(random.uniform(0.01, 0.3), 2),
             "Change weight decay"),
        ]

        if self.evolve_config.mutate_training:
            n_mutations = max(1, int(len(possible_mutations) * self.evolve_config.mutation_rate))
            selected = random.sample(possible_mutations, min(n_mutations, len(possible_mutations)))

            for mut_type, param, old_val, new_val, desc in selected:
                setattr(new_config, param, new_val)
                mutations.append(Mutation(mut_type, param, old_val, new_val, desc))

        return new_config, mutations

    def generate_mutation_prompt(self, model, tokenizer, current_config: ModelConfig, current_score: float) -> str:
        """Use the model itself to suggest mutations."""
        prompt = f"""You are a neural architecture search agent. The current Space LLM has:
- Parameters: {current_config.d_model}d, {current_config.n_layers} layers, {current_config.n_heads} heads
- Feed-forward: {current_config.d_ff}, Activation: {current_config.activation}
- Dropout: {current_config.dropout}, Seq length: {current_config.max_seq_len}
- Validation loss: {current_score:.4f}

Suggest one improvement to make the model better at understanding space science. Be specific.

Suggestion:"""

        try:
            gen_config = GenerateConfig(temperature=0.9, max_new_tokens=100)
            response = generate_text(model, tokenizer, prompt, gen_config)
            suggestion = response[len(prompt):].strip()
            return suggestion
        except Exception:
            return "Random mutation"

    def evaluate_model(self, model, data_dir: str, seq_len: int, device) -> dict:
        """Evaluate a model and return metrics."""
        val_loss, perplexity = compute_perplexity(model, data_dir, seq_len, device=device)

        return {
            "val_loss": val_loss,
            "perplexity": perplexity,
            "score": val_loss,  # Lower is better
        }

    def evolve_generation(self, parent_model_config: ModelConfig, parent_train_config: TrainConfig) -> dict:
        """Run one generation of evolution: mutate, train, evaluate, select."""
        self.generation += 1
        gen_dir = EVOLUTION_DIR / f"gen_{self.generation}"
        gen_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"EVOLUTION GENERATION {self.generation}")
        print(f"{'='*60}")

        # Generate mutations
        candidates = []
        for i in range(self.evolve_config.mutations_per_gen):
            mut_model_config, model_mutations = self.mutate_model_config(parent_model_config)
            mut_train_config, train_mutations = self.mutate_train_config(parent_train_config)

            # Update checkpoint dir for this mutation
            mut_train_config.checkpoint_dir = str(gen_dir / f"mutant_{i}")

            # Count params
            try:
                temp_model = SpaceLLM(mut_model_config)
                n_params = temp_model.get_num_params()
                del temp_model

                # Skip if model is too large for Colab
                if n_params > 50_000_000:
                    print(f"  Mutant {i}: {n_params:,} params - TOO LARGE, skipping")
                    continue

                candidates.append({
                    "id": i,
                    "model_config": mut_model_config,
                    "train_config": mut_train_config,
                    "model_mutations": model_mutations,
                    "train_mutations": train_mutations,
                    "n_params": n_params,
                })
                print(f"  Mutant {i}: {n_params:,} params, {len(model_mutations) + len(train_mutations)} mutations")
                for m in model_mutations + train_mutations:
                    print(f"    - {m.description}: {m.old_value} -> {m.new_value}")

            except Exception as e:
                print(f"  Mutant {i}: Error - {e}")

        if not candidates:
            print("  No valid mutations generated!")
            return {"generation": self.generation, "status": "no_valid_mutations"}

        # Use model to suggest best mutation (if model exists)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            parent_model, _ = load_model(parent_train_config.checkpoint_dir, device)
            tokenizer_path = "data/tokenizer/space_tokenizer.model"
            if os.path.exists(tokenizer_path):
                tokenizer = load_tokenizer(tokenizer_path)
                suggestion = self.generate_mutation_prompt(
                    parent_model, tokenizer, parent_model_config, self.best_score
                )
                print(f"\n  Model suggestion: {suggestion}")
            del parent_model
        except Exception:
            pass

        # Train and evaluate each candidate
        results = []
        for candidate in candidates:
            print(f"\n--- Training Mutant {candidate['id']} ({candidate['n_params']:,} params) ---")

            try:
                # Short training for evaluation
                short_train_config = copy.deepcopy(candidate["train_config"])
                short_train_config.max_steps = self.evolve_config.train_steps_per_gen
                short_train_config.log_every = 500
                short_train_config.eval_every = 1000
                short_train_config.save_every = 2000

                model = train(
                    model_config=candidate["model_config"],
                    train_config=short_train_config,
                )

                # Evaluate
                metrics = self.evaluate_model(
                    model,
                    short_train_config.data_dir,
                    candidate["model_config"].max_seq_len,
                    device,
                )

                result = {
                    "candidate_id": candidate["id"],
                    "model_config": candidate["model_config"].to_dict(),
                    "train_config": {k: v for k, v in candidate["train_config"].to_dict().items()
                                    if k != "checkpoint_dir"},
                    "n_params": candidate["n_params"],
                    "model_mutations": [m.to_dict() for m in candidate["model_mutations"]],
                    "train_mutations": [m.to_dict() for m in candidate["train_mutations"]],
                    "metrics": metrics,
                }
                results.append(result)

                print(f"  Mutant {candidate['id']}: val_loss={metrics['val_loss']:.4f}, "
                      f"perplexity={metrics['perplexity']:.2f}")

                # Clean up GPU memory
                del model
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Mutant {candidate['id']}: Training failed - {e}")
                results.append({
                    "candidate_id": candidate["id"],
                    "error": str(e),
                })

        # Selection: find best mutant
        valid_results = [r for r in results if "metrics" in r]
        if not valid_results:
            print("\n  No valid results!")
            return {"generation": self.generation, "status": "no_valid_results"}

        best_result = min(valid_results, key=lambda x: x["metrics"]["score"])
        best_score = best_result["metrics"]["score"]

        # Decide whether to adopt the mutant
        adopted = best_score < self.best_score
        if adopted:
            self.best_score = best_score
            self.best_config = best_result["model_config"]
            print(f"\n  NEW BEST! Score: {best_score:.4f} (improved from {self.best_score:.4f})")

            # Copy best checkpoint to main checkpoints dir
            best_ckpt_src = EVOLUTION_DIR / f"gen_{self.generation}" / f"mutant_{best_result['candidate_id']}"
            if best_ckpt_src.exists():
                main_ckpt = Path("checkpoints")
                main_ckpt.mkdir(exist_ok=True)
                for f in best_ckpt_src.glob("*.pt"):
                    shutil.copy2(f, main_ckpt / f.name)
                save_config(
                    ModelConfig.from_dict(best_result["model_config"]),
                    str(main_ckpt / "model_config.json"),
                )
        else:
            print(f"\n  No improvement. Best score remains {self.best_score:.4f}")

        # Log generation
        gen_log = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "parent_score": self.best_score,
            "n_candidates": len(candidates),
            "n_valid": len(valid_results),
            "best_mutant": best_result["candidate_id"],
            "best_score": best_score,
            "adopted": adopted,
            "results": results,
        }
        self.evolution_log.append(gen_log)

        # Save log
        with open(self.evolve_config.evolve_log, "w") as f:
            json.dump(self.evolution_log, f, indent=2)

        return gen_log

    def run(self, initial_model_config: ModelConfig = None, initial_train_config: TrainConfig = None):
        """Run the full evolution loop."""
        if initial_model_config is None:
            initial_model_config = ModelConfig()
        if initial_train_config is None:
            initial_train_config = TrainConfig()

        print(f"\n{'='*60}")
        print("SPACE LLM SELF-EVOLUTION ENGINE")
        print(f"Generations: {self.evolve_config.generations}")
        print(f"Mutations per gen: {self.evolve_config.mutations_per_gen}")
        print(f"Train steps per gen: {self.evolve_config.train_steps_per_gen}")
        print(f"{'='*60}")

        current_model_config = initial_model_config
        current_train_config = initial_train_config

        # Initial training if no checkpoint exists
        if not os.path.exists(os.path.join(current_train_config.checkpoint_dir, "best.pt")):
            print("\n[Initial] Training base model...")
            train(current_model_config, current_train_config)

        # Evaluate base model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model, _ = load_model(current_train_config.checkpoint_dir, device)
            metrics = self.evaluate_model(
                model,
                current_train_config.data_dir,
                current_model_config.max_seq_len,
                device,
            )
            self.best_score = metrics["score"]
            self.best_config = current_model_config.to_dict()
            print(f"\n[Initial] Base model score: {self.best_score:.4f}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[Initial] Could not evaluate base model: {e}")

        # Evolution loop
        for gen in range(self.evolve_config.generations):
            print(f"\n{'='*60}")
            print(f"Starting generation {gen + 1}/{self.evolve_config.generations}")
            print(f"{'='*60}")

            result = self.evolve_generation(current_model_config, current_train_config)

            # Update current config if a better model was found
            if result.get("adopted") and self.best_config:
                current_model_config = ModelConfig.from_dict(self.best_config)
                print(f"\n  Adopted new config: d={current_model_config.d_model}, "
                      f"layers={current_model_config.n_layers}, "
                      f"heads={current_model_config.n_heads}")

            # Print summary
            print(f"\n  Generation {gen + 1} complete. Best score: {self.best_score:.4f}")

        # Final summary
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        print(f"Total generations: {self.evolve_config.generations}")
        print(f"Best score: {self.best_score:.4f}")
        if self.best_config:
            print(f"Best config: d={self.best_config.get('d_model')}, "
                  f"layers={self.best_config.get('n_layers')}, "
                  f"heads={self.best_config.get('n_heads')}")
        print(f"Evolution log: {self.evolve_config.evolve_log}")

        return self.evolution_log


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Space LLM Self-Evolution")
    parser.add_argument("--generations", type=int, default=5, help="Number of evolution generations")
    parser.add_argument("--mutations-per-gen", type=int, default=3, help="Mutations per generation")
    parser.add_argument("--train-steps", type=int, default=5000, help="Training steps per mutation")
    parser.add_argument("--mutation-rate", type=float, default=0.3, help="Mutation rate (0-1)")
    args = parser.parse_args()

    evolve_config = EvolveConfig(
        generations=args.generations,
        mutations_per_gen=args.mutations_per_gen,
        train_steps_per_gen=args.train_steps,
        mutation_rate=args.mutation_rate,
    )

    engine = EvolutionEngine(evolve_config)
    engine.run()


if __name__ == "__main__":
    main()
