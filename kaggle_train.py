"""
Kaggle Notebook: Complete Math AI Training
===========================================
GPU: T4 (16GB VRAM)
Runtime: ~6-8 hours
Output: Fine-tuned Qwen2.5-Math-1.5B with LoRA
"""

# ============================================================
# CELL 1: Install Dependencies
# ============================================================
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("Installing dependencies...")
install("transformers>=4.40.0")
install("datasets>=2.18.0")
install("peft>=0.10.0")
install("accelerate>=0.28.0")
install("bitsandbytes>=0.43.0")
install("trl>=0.8.0")
install("scipy>=1.12.0")
install("sentencepiece>=0.2.0")
install("protobuf>=4.25.0")
install("wandb")
print("Done!\n")

# ============================================================
# CELL 2: Imports & Config
# ============================================================
import os
import json
import torch
import yaml
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Config
BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "/kaggle/working/math-model"

SYSTEM_PROMPT = """You are a precise and helpful math tutor. Given a math problem, provide a clear, step-by-step solution. Show your reasoning at each step, then give the final answer on a line starting with 'The answer is:'."""

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================================================
# CELL 3: Dataset Format Functions
# ============================================================

def make_conv(q, a):
    return {"conversations": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": str(q).strip()},
        {"role": "assistant", "content": str(a).strip()},
    ]}

def fmt_meta_math(ex):
    return make_conv(ex["query"], ex["response"])

def fmt_math_instruct(ex):
    return make_conv(ex["instruction"], ex["output"])

def fmt_orca_math(ex):
    return make_conv(ex["question"], ex["solution"])

def fmt_numina_math(ex):
    if "messages" in ex:
        q = a = None
        for m in ex["messages"]:
            if m["role"] == "user": q = m["content"]
            elif m["role"] == "assistant": a = m["content"]
        if q and a: return make_conv(q, a)
    return make_conv(ex.get("problem", ""), ex.get("solution", ""))

def fmt_open_math_instruct(ex):
    return make_conv(ex.get("problem", ""), ex.get("generated_solution", ""))

def fmt_open_r1_math(ex):
    if "messages" in ex:
        q = a = None
        for m in ex["messages"]:
            if m["role"] == "user": q = m["content"]
            elif m["role"] == "assistant": a = m["content"]
        if q and a: return make_conv(q, a)
    return make_conv(ex.get("problem", ""), ex.get("solution", ""))

def fmt_ace_reason(ex):
    return make_conv(ex.get("problem", ""), ex.get("solution", ex.get("answer", "")))

def fmt_gsm8k(ex):
    ans = ex["answer"]
    if "####" in ans:
        r, f = ans.split("####", 1)
        ans = f"{r.strip()}\n\nThe answer is: {f.strip()}"
    return make_conv(ex["question"], ans)

def fmt_hendrycks_math(ex):
    return make_conv(ex["problem"], ex["solution"])

def fmt_math_qa(ex):
    q = ex["Question"]
    r = ex.get("Rationale", "")
    c = ex.get("Correct", ex.get("correct", ""))
    a = f"{r}\n\nThe answer is: {c}" if r else f"The answer is: {c}"
    opts = ex.get("options", "")
    if opts: q = f"{q}\n\nOptions: {opts}"
    return make_conv(q, a)

def fmt_camel_math(ex):
    return make_conv(ex.get("problem", ""), ex.get("solution", ""))

def fmt_big_math(ex):
    return make_conv(ex.get("problem", ""), ex.get("solution", ex.get("answer", "")))

def fmt_dapo_math(ex):
    return make_conv(ex.get("prompt", ex.get("problem", "")), ex.get("ground_truth", ex.get("solution", "")))

def fmt_aqua_rat(ex):
    q = ex.get("question", "")
    opts = ex.get("options", "")
    if opts: q = f"{q}\n\nOptions: {opts}"
    r = ex.get("rationale", "")
    c = ex.get("correct", "")
    a = f"{r}\n\nThe answer is: {c}" if r else f"The answer is: {c}"
    return make_conv(q, a)

def fmt_lila(ex):
    return make_conv(ex.get("input", ex.get("question", "")), ex.get("output", ex.get("answer", "")))

# ============================================================
# CELL 4: Download & Prepare All Datasets
# ============================================================

DATASETS_CONFIG = [
    # (name, split, format_fn, max_samples, description)
    ("meta-math/MetaMathQA", "train", fmt_meta_math, None, "MetaMathQA (395K)"),
    ("TIGER-Lab/MathInstruct", "train", fmt_math_instruct, None, "MathInstruct (262K)"),
    ("microsoft/orca-math-word-problems-200k", "train", fmt_orca_math, None, "Orca-Math (200K)"),
    ("AI-MO/NuminaMath-CoT", "train", fmt_numina_math, None, "NuminaMath (860K)"),
    ("nvidia/OpenMathInstruct-2", "train_2M", fmt_open_math_instruct, 500000, "OpenMathInstruct-2 (500K)"),
    ("open-r1/OpenR1-Math-220k", "train", fmt_open_r1_math, None, "OpenR1-Math (220K)"),
    ("nvidia/AceReason-Math", "train", fmt_ace_reason, None, "AceReason-Math (50K)"),
    ("openai/gsm8k", "train", fmt_gsm8k, None, "GSM8K (7.5K)"),
    ("EleutherAI/hendrycks_math", "train", fmt_hendrycks_math, None, "MATH (7.5K)"),
    ("allenai/math_qa", "train", fmt_math_qa, None, "MathQA (30K)"),
    ("camel-ai/math", "train", fmt_camel_math, None, "Camel-AI (50K)"),
    ("SynthLabsAI/Big-Math-RL-Verified", "train", fmt_big_math, None, "Big-Math (251K)"),
    ("BytedTsinghua-SIA/DAPO-Math-17k", "train", fmt_dapo_math, 100000, "DAPO-Math (100K)"),
    ("deepmind/aqua_rat", "train", fmt_aqua_rat, None, "AQuA-RAT (100K)"),
    ("allenai/lila", "train", fmt_lila, None, "Lila"),
]

print("=" * 60)
print("  Downloading & Preparing Math Datasets")
print("=" * 60)

all_datasets = []
total = 0

for name, split, fmt_fn, max_samp, desc in DATASETS_CONFIG:
    print(f"\n[{len(all_datasets)+1}] {desc}")
    print(f"    {name} ({split})...")
    try:
        ds = load_dataset(name, split=split, trust_remote_code=True)
        if max_samp and len(ds) > max_samp:
            ds = ds.select(range(max_samp))
        orig_cols = ds.column_names
        ds = ds.map(fmt_fn, remove_columns=orig_cols, num_proc=4)
        if "conversations" not in ds.column_names:
            print(f"    SKIP: bad format")
            continue
        all_datasets.append(ds)
        total += len(ds)
        print(f"    -> {len(ds):,} examples")
    except Exception as e:
        print(f"    ERROR: {e}")

print(f"\n{'='*60}")
print(f"  Loaded {len(all_datasets)} datasets, {total:,} total examples")
print(f"{'='*60}")

# Combine
print("\nCombining & shuffling...")
combined = concatenate_datasets(all_datasets).shuffle(seed=42)
split_ds = combined.train_test_split(test_size=0.01, seed=42)
dataset = DatasetDict({"train": split_ds["train"], "validation": split_ds["test"]})
print(f"Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,}")

# ============================================================
# CELL 5: Load Model with 4-bit Quantization
# ============================================================

print(f"\nLoading model: {BASE_MODEL}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    padding_side="right",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
print("Model loaded!")

# ============================================================
# CELL 6: Apply LoRA
# ============================================================

print("Applying LoRA...")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

model = get_peft_model(model, peft_config)
trainable, total_params = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

# ============================================================
# CELL 7: Format Dataset to Text
# ============================================================

def to_text(example):
    messages = example["conversations"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

print("Formatting to text...")
dataset = dataset.map(to_text, num_proc=4, remove_columns=dataset["train"].column_names)
print(f"Sample:\n{dataset['train'][0]['text'][:500]}...")

# ============================================================
# CELL 8: Train!
# ============================================================

print("\n" + "=" * 60)
print("  STARTING TRAINING")
print("=" * 60)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2.0e-4,
    weight_decay=0.01,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    save_total_limit=3,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    report_to="none",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    evaluation_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

trainer.train()

# ============================================================
# CELL 9: Save Model
# ============================================================

print("\nSaving model...")
final_path = os.path.join(OUTPUT_DIR, "final")
trainer.save_model(final_path)
tokenizer.save_pretrained(final_path)

# Save config
config_info = {
    "base_model": BASE_MODEL,
    "lora_r": 64,
    "lora_alpha": 128,
    "trainable_params": trainable,
    "total_params": total_params,
    "train_examples": len(dataset["train"]),
    "val_examples": len(dataset["validation"]),
    "epochs": 3,
}
with open(os.path.join(OUTPUT_DIR, "training_info.json"), "w") as f:
    json.dump(config_info, f, indent=2)

print(f"\nModel saved to: {final_path}")
print("Training complete!")
