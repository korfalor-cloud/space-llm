"""
Data preparation script for the complete math AI model.

Downloads and formats ALL available math instruction datasets into a
unified conversation format for fine-tuning.

Datasets included:
  - MetaMathQA (395K) - augmented GSM8K+MATH
  - MathInstruct (262K) - 13 datasets, CoT+PoT
  - Orca-Math (200K) - GPT-4 solutions
  - NuminaMath-CoT (860K) - competition + K-12
  - OpenMathInstruct-2 (500K subset) - Llama3.1-405B synthetic
  - OpenR1-Math (220K) - DeepSeek R1 traces
  - AceReason-Math (50K) - challenging problems
  - GSM8K (7.5K) - grade school math
  - MATH (7.5K) - competition math
  - MathQA (30K) - word problems
  - Camel-AI Math (50K) - 25 topics
  - Big-Math-RL-Verified (251K) - verifiable solutions
  - DAPO-Math (100K subset) - ByteDance+Tsinghua
  - AQuA-RAT (100K) - algebraic reasoning
  - Lila - diverse aggregation
"""

import os
import json
import yaml
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, DatasetDict

SYSTEM_PROMPT = """You are a precise and helpful math tutor. Given a math problem, provide a clear, step-by-step solution. Show your reasoning at each step, then give the final answer on a line starting with 'The answer is:'."""


def make_conversation(question, answer):
    """Create a standard conversation format."""
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": str(question).strip()},
            {"role": "assistant", "content": str(answer).strip()},
        ]
    }


# ============================================================
# Dataset Format Functions
# ============================================================

def format_meta_math(example):
    """MetaMathQA: query -> response"""
    return make_conversation(example["query"], example["response"])


def format_math_instruct(example):
    """MathInstruct: instruction -> output"""
    return make_conversation(example["instruction"], example["output"])


def format_orca_math(example):
    """Orca-Math: question -> solution"""
    return make_conversation(example["question"], example["solution"])


def format_numina_math(example):
    """NuminaMath-CoT: problem -> solution (messages format)"""
    # NuminaMath uses messages format
    if "messages" in example:
        messages = example["messages"]
        question = None
        answer = None
        for msg in messages:
            if msg["role"] == "user":
                question = msg["content"]
            elif msg["role"] == "assistant":
                answer = msg["content"]
        if question and answer:
            return make_conversation(question, answer)

    # Fallback: try problem/solution fields
    question = example.get("problem", example.get("question", ""))
    answer = example.get("solution", example.get("answer", ""))
    return make_conversation(question, answer)


def format_open_math_instruct(example):
    """OpenMathInstruct-2: problem -> generated_solution"""
    question = example.get("problem", example.get("question", ""))
    answer = example.get("generated_solution", example.get("solution", ""))
    return make_conversation(question, answer)


def format_open_r1_math(example):
    """OpenR1-Math-220k: problem -> solution (messages format)"""
    if "messages" in example:
        messages = example["messages"]
        question = None
        answer = None
        for msg in messages:
            if msg["role"] == "user":
                question = msg["content"]
            elif msg["role"] == "assistant":
                answer = msg["content"]
        if question and answer:
            return make_conversation(question, answer)

    question = example.get("problem", example.get("question", ""))
    answer = example.get("solution", example.get("answer", ""))
    return make_conversation(question, answer)


def format_ace_reason(example):
    """AceReason-Math: problem -> answer"""
    question = example.get("problem", example.get("question", ""))
    answer = example.get("solution", example.get("answer", ""))
    return make_conversation(question, answer)


def format_gsm8k(example):
    """GSM8K: question -> answer (with #### separator)"""
    question = example["question"]
    answer = example["answer"]
    if "####" in answer:
        reasoning, final = answer.split("####", 1)
        answer_text = f"{reasoning.strip()}\n\nThe answer is: {final.strip()}"
    else:
        answer_text = answer
    return make_conversation(question, answer_text)


def format_hendrycks_math(example):
    """Hendrycks MATH: problem -> solution"""
    return make_conversation(example["problem"], example["solution"])


def format_math_qa(example):
    """MathQA: Question -> Rationale + Correct Answer"""
    question = example["Question"]
    rationale = example.get("Rationale", "")
    answer = example.get("Correct", example.get("correct", ""))
    if rationale:
        answer_text = f"{rationale}\n\nThe answer is: {answer}"
    else:
        answer_text = f"The answer is: {answer}"
    return make_conversation(question, answer_text)


def format_camel_math(example):
    """Camel-AI Math: problem -> solution"""
    question = example.get("problem", example.get("question", ""))
    answer = example.get("solution", example.get("answer", ""))
    return make_conversation(question, answer)


def format_big_math(example):
    """Big-Math-RL-Verified: problem -> answer"""
    question = example.get("problem", example.get("question", ""))
    answer = example.get("solution", example.get("answer", ""))
    return make_conversation(question, answer)


def format_dapo_math(example):
    """DAPO-Math: prompt -> ground truth"""
    question = example.get("prompt", example.get("problem", example.get("question", "")))
    answer = example.get("ground_truth", example.get("solution", example.get("answer", "")))
    return make_conversation(question, answer)


def format_aqua_rat(example):
    """AQuA-RAT: question -> rationale + correct answer"""
    question = example.get("question", "")
    options = example.get("options", "")
    rationale = example.get("rationale", "")
    answer = example.get("correct", "")

    # Include options in the question if present
    if options:
        question_text = f"{question}\n\nOptions: {options}"
    else:
        question_text = question

    if rationale:
        answer_text = f"{rationale}\n\nThe answer is: {answer}"
    else:
        answer_text = f"The answer is: {answer}"
    return make_conversation(question_text, answer_text)


def format_lila(example):
    """Lila: diverse format - try common field names"""
    question = example.get("input", example.get("question", example.get("problem", "")))
    answer = example.get("output", example.get("answer", example.get("solution", "")))
    return make_conversation(question, answer)


# ============================================================
# Format function registry
# ============================================================

FORMAT_FUNCTIONS = {
    "meta_math": format_meta_math,
    "math_instruct": format_math_instruct,
    "orca_math": format_orca_math,
    "numina_math": format_numina_math,
    "open_math_instruct": format_open_math_instruct,
    "open_r1_math": format_open_r1_math,
    "ace_reason": format_ace_reason,
    "gsm8k": format_gsm8k,
    "hendrycks_math": format_hendrycks_math,
    "math_qa": format_math_qa,
    "camel_math": format_camel_math,
    "big_math": format_big_math,
    "dapo_math": format_dapo_math,
    "aqua_rat": format_aqua_rat,
    "lila": format_lila,
}


# ============================================================
# Main pipeline
# ============================================================

def load_and_format_dataset(dataset_name, split, format_fn, max_samples=None):
    """Load a dataset, format it, and optionally limit samples."""
    print(f"  Loading {dataset_name} ({split})...")
    try:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"    ERROR loading: {e}")
        return None

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    # Get column names before formatting
    original_columns = ds.column_names

    try:
        ds = ds.map(format_fn, remove_columns=original_columns, num_proc=4)
    except Exception as e:
        print(f"    ERROR formatting: {e}")
        return None

    # Verify the output has the right structure
    if "conversations" not in ds.column_names:
        print(f"    ERROR: format function did not produce 'conversations' column")
        return None

    print(f"    -> {len(ds)} examples loaded")
    return ds


def prepare_datasets(config_path="configs/training_config.yaml"):
    """Main function to prepare all training datasets."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_config = config["data"]
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    all_datasets = []
    total_loaded = 0
    failed_datasets = []

    print("=" * 60)
    print("  Math AI - Complete Data Preparation")
    print("=" * 60)

    for ds_info in data_config["datasets"]:
        name = ds_info["name"]
        split = ds_info["split"]
        format_name = ds_info["format"]
        max_samples = ds_info.get("max_samples")
        description = ds_info.get("description", "")

        print(f"\n[{len(all_datasets) + 1}] {description}")
        print(f"    Dataset: {name}")

        format_fn = FORMAT_FUNCTIONS.get(format_name)
        if format_fn is None:
            print(f"    WARNING: No formatter for '{format_name}', skipping")
            failed_datasets.append(name)
            continue

        ds = load_and_format_dataset(name, split, format_fn, max_samples)
        if ds is None:
            failed_datasets.append(name)
            continue

        all_datasets.append(ds)
        total_loaded += len(ds)

    print("\n" + "=" * 60)
    print(f"  Successfully loaded {len(all_datasets)} datasets")
    print(f"  Total examples: {total_loaded:,}")
    if failed_datasets:
        print(f"  Failed: {', '.join(failed_datasets)}")
    print("=" * 60)

    if not all_datasets:
        raise RuntimeError("No datasets were successfully loaded!")

    # Combine all datasets
    print("\nCombining and shuffling...")
    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=42)
    print(f"Combined total: {len(combined):,} examples")

    # Apply total sample limit if configured
    max_total = data_config.get("max_total_samples")
    if max_total and len(combined) > max_total:
        combined = combined.select(range(max_total))
        print(f"Limited to: {max_total:,} examples")

    # Split into train/val
    val_split = data_config.get("val_split", 0.01)
    split_ds = combined.train_test_split(test_size=val_split, seed=42)

    dataset_dict = DatasetDict({
        "train": split_ds["train"],
        "validation": split_ds["test"],
    })

    # Save to disk
    save_path = output_dir / "math_dataset"
    dataset_dict.save_to_disk(str(save_path))
    print(f"\nSaved to {save_path}")
    print(f"  Train: {len(dataset_dict['train']):,} examples")
    print(f"  Validation: {len(dataset_dict['validation']):,} examples")

    # Save samples for inspection
    samples = [dataset_dict["train"][i] for i in range(min(10, len(dataset_dict["train"])))]
    with open(output_dir / "sample_examples.json", "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"  Samples: {output_dir / 'sample_examples.json'}")

    # Save dataset stats
    stats = {
        "total_examples": len(combined),
        "train_examples": len(dataset_dict["train"]),
        "val_examples": len(dataset_dict["validation"]),
        "datasets_loaded": len(all_datasets),
        "datasets_failed": failed_datasets,
        "dataset_sizes": {ds_info["name"]: len(ds) for ds, ds_info in zip(all_datasets, data_config["datasets"] if len(data_config["datasets"]) == len(all_datasets) else [{} for _ in all_datasets])},
    }
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {output_dir / 'dataset_stats.json'}")

    return dataset_dict


if __name__ == "__main__":
    prepare_datasets()
