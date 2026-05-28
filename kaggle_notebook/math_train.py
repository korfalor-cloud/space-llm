"""
Kaggle: Math AI Training - Use Kaggle's default PyTorch
P100 warning is just a warning, might still work
"""
import subprocess, sys, os

# Install ML deps only
for p in ["transformers>=4.40.0","datasets>=2.18.0","peft>=0.10.0","accelerate>=0.28.0","bitsandbytes>=0.43.0","trl>=0.8.0","sentencepiece>=0.2.0","protobuf>=4.25.0"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

import torch
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Quick test - can we use the GPU?
print("\nTesting GPU...")
try:
    t = torch.zeros(10, 10, device='cuda')
    t = t + 1
    print(f"GPU test: PASSED - tensor on {t.device}")
except Exception as e:
    print(f"GPU test: FAILED - {e}")
    print("Trying CPU fallback...")

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

SYS = "You are a math tutor. Show step-by-step solution. End with 'The answer is: X'."

def fmt(e):
    a = e["answer"]
    if "####" in a:
        p = a.split("####",1)
        a = p[0].strip()+"\nThe answer is: "+p[1].strip()
    return {"conversations": [
        {"role":"system","content":SYS},
        {"role":"user","content":e["question"]},
        {"role":"assistant","content":a},
    ]}

print("\nLoading GSM8K...")
ds = load_dataset("openai/gsm8k", "main", split="train")
ds = ds.map(fmt, remove_columns=ds.column_names)
sp = ds.train_test_split(test_size=0.1, seed=42)
print(f"Train: {len(sp['train'])}, Val: {len(sp['test'])}")

print("\nLoading model...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B", quantization_config=bnb, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B", trust_remote_code=True, padding_side="right")
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"], bias="none")
model = get_peft_model(model, lora)
tr, tot = model.get_nb_trainable_parameters()
print(f"Trainable: {tr:,}/{tot:,}")

def to_text(e):
    return {"text": tok.apply_chat_template(e["conversations"], tokenize=False, add_generation_prompt=False)}
ds_train = sp["train"].map(to_text, remove_columns=sp["train"].column_names)
ds_val = sp["test"].map(to_text, remove_columns=sp["test"].column_names)

print("\nTRAINING...")
args = SFTConfig(
    output_dir="/kaggle/working/math-model",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=20,
    save_steps=500,
    save_total_limit=1,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    optim="adamw_torch",
    report_to="none",
    max_seq_length=1024,
    dataset_text_field="text",
    evaluation_strategy="steps",
    eval_steps=500,
)
trainer = SFTTrainer(model=model, args=args, train_dataset=ds_train, eval_dataset=ds_val, processing_class=tok)
trainer.train()

fp = "/kaggle/working/math-model/final"
trainer.save_model(fp)
tok.save_pretrained(fp)
print(f"\nSAVED: {fp}")
print("DONE!")
