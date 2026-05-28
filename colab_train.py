"""
Google Colab: Complete Math AI Training
GPU: T4 (15GB) | Model: Qwen2.5-Math-1.5B + LoRA

INSTRUCTIONS:
1. Go to https://colab.research.google.com
2. Create new notebook
3. Runtime > Change runtime type > T4 GPU
4. Paste this ENTIRE script into one cell
5. Run it

Training takes ~2-3 hours on T4.
"""
# ============================================================
# STEP 1: Install Dependencies
# ============================================================
!pip install -q transformers>=4.40.0 datasets>=2.18.0 peft>=0.10.0 accelerate>=0.28.0 bitsandbytes>=0.43.0 trl>=0.8.0 sentencepiece>=0.2.0 protobuf>=4.25.0

# ============================================================
# STEP 2: Imports
# ============================================================
import os, gc, json, torch
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"
OUTPUT_DIR = "/content/math-model"
SYS = "You are a precise and helpful math tutor. Given a math problem, provide a clear, step-by-step solution. Show your reasoning at each step, then give the final answer on a line starting with 'The answer is:'."

def conv(q, a):
    return {"conversations": [
        {"role":"system","content":SYS},
        {"role":"user","content":str(q).strip()},
        {"role":"assistant","content":str(a).strip()}
    ]}

def safe(fn):
    def w(e):
        try:
            r = fn(e)
            if r and len(r.get("conversations",[]))==3: return r
        except: pass
        return conv("solve","The answer is: 0")
    return w

@safe
def f1(e): return conv(e["query"], e["response"])
@safe
def f2(e): return conv(e["instruction"], e["output"])
@safe
def f3(e): return conv(e["question"], e["solution"])
@safe
def f4(e):
    m=e["messages"]
    return conv(next((x["content"] for x in m if x["role"]=="user"),""), next((x["content"] for x in m if x["role"]=="assistant"),""))
@safe
def f5(e): return conv(e.get("problem",""), e.get("generated_solution",""))
@safe
def f6(e):
    m=e["messages"]
    return conv(next((x["content"] for x in m if x["role"],"user"),""), next((x["content"] for x in m if x["role"],"assistant"),""))
@safe
def f7(e): return conv(e.get("problem",""), e.get("solution", e.get("answer","")))
@safe
def f8(e):
    a=e["answer"]
    if "####" in a:
        p=a.split("####",1)
        a=p[0].strip()+"\nThe answer is: "+p[1].strip()
    return conv(e["question"], a)
@safe
def f9(e): return conv(e["problem"], e["solution"])
@safe
def f10(e):
    q=e["Question"]
    if e.get("options"): q+="\nOptions: "+str(e["options"])
    return conv(q, str(e.get("Rationale",""))+"\nThe answer is: "+str(e.get("Correct","")))
@safe
def f11(e): return conv(e.get("problem",""), e.get("solution",""))
@safe
def f12(e): return conv(e.get("problem",""), e.get("solution", e.get("answer","")))
@safe
def f13(e): return conv(e.get("prompt",e.get("problem","")), e.get("ground_truth",e.get("solution","")))
@safe
def f14(e):
    q=e.get("question","")
    if e.get("options"): q+="\nOptions: "+str(e["options"])
    return conv(q, str(e.get("rationale",""))+"\nThe answer is: "+str(e.get("correct","")))
@safe
def f15(e): return conv(e.get("input",e.get("question","")), e.get("output",e.get("answer","")))

# ============================================================
# STEP 3: Download ALL 15 Datasets
# ============================================================
DS = [
    ("meta-math/MetaMathQA", ["train"], f1, 200000),
    ("TIGER-Lab/MathInstruct", ["train"], f2, 150000),
    ("microsoft/orca-math-word-problems-200k", ["train"], f3, 100000),
    ("AI-MO/NuminaMath-CoT", ["train"], f4, 200000),
    ("nvidia/OpenMathInstruct-2", ["train_2M","train"], f5, 200000),
    ("open-r1/OpenR1-Math-220k", ["train"], f6, 200000),
    ("nvidia/AceReason-Math", ["train"], f7, None),
    ("openai/gsm8k", ["main"], f8, None),
    ("EleutherAI/hendrycks_math", ["train"], f9, None),
    ("allenai/math_qa", ["train"], f10, None),
    ("camel-ai/math", ["train"], f11, None),
    ("SynthLabsAI/Big-Math-RL-Verified", ["train"], f12, 100000),
    ("BytedTsinghua-SIA/DAPO-Math-17k", ["train"], f13, 50000),
    ("deepmind/aqua_rat", ["train"], f14, None),
    ("allenai/lila", ["train"], f15, None),
]

print("="*60)
print("  DOWNLOADING ALL MATH DATASETS")
print("="*60)

all_ds = []
for i,(name,splits,fmt,mx) in enumerate(DS):
    print(f"\n[{i+1}/{len(DS)}] {name}")
    ok = False
    for sp in splits:
        try:
            d = load_dataset(name, split=sp, trust_remote_code=True)
            if mx and len(d)>mx: d=d.select(range(mx))
            d = d.map(fmt, remove_columns=d.column_names, num_proc=2, load_from_cache_file=False)
            d = d.filter(lambda e: len(e.get("conversations",[]))==3, num_proc=2)
            if len(d)>0:
                all_ds.append(d)
                print(f"  OK [{sp}]: {len(d):,}")
                ok = True
                break
        except Exception as ex:
            print(f"  FAIL [{sp}]: {ex}")
    if not ok: print("  SKIPPED")
    gc.collect()

total = sum(len(x) for x in all_ds)
print(f"\n{'='*60}")
print(f"  LOADED {len(all_ds)} DATASETS, {total:,} EXAMPLES")
print(f"{'='*60}")

combined = concatenate_datasets(all_ds).shuffle(seed=42)
del all_ds; gc.collect()
sp = combined.train_test_split(test_size=0.01, seed=42)
ds = DatasetDict({"train":sp["train"],"validation":sp["test"]})
del combined, sp; gc.collect()
print(f"Train: {len(ds['train']):,} | Val: {len(ds['validation']):,}")

# ============================================================
# STEP 4: Load Model
# ============================================================
print(f"\nLoading {BASE_MODEL}...")
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=bnb, device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, padding_side="right")
if tok.pad_token is None: tok.pad_token = tok.eos_token
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
print("Model loaded!")

# ============================================================
# STEP 5: LoRA
# ============================================================
lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none")
model = get_peft_model(model, lora)
tr, tot = model.get_nb_trainable_parameters()
print(f"Trainable: {tr:,}/{tot:,} ({100*tr/tot:.2f}%)")

# ============================================================
# STEP 6: Format & Train
# ============================================================
def to_text(e):
    return {"text": tok.apply_chat_template(e["conversations"], tokenize=False, add_generation_prompt=False)}
ds = ds.map(to_text, num_proc=2, remove_columns=ds["train"].column_names)

print("\n" + "="*60)
print("  TRAINING - 3 EPOCHS")
print("="*60)

args = SFTConfig(
    output_dir=OUTPUT_DIR, num_train_epochs=3,
    per_device_train_batch_size=2, gradient_accumulation_steps=8,
    learning_rate=2.0e-4, weight_decay=0.01, warmup_ratio=0.05,
    lr_scheduler_type="cosine", logging_steps=10,
    save_steps=500, eval_steps=500, save_total_limit=3,
    fp16=False, bf16=True, gradient_checkpointing=True,
    optim="paged_adamw_8bit", max_grad_norm=0.3,
    report_to="none", max_seq_length=2048,
    dataset_text_field="text", evaluation_strategy="steps",
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
)
trainer = SFTTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"], processing_class=tok)
trainer.train()

# ============================================================
# STEP 7: Save
# ============================================================
fp = os.path.join(OUTPUT_DIR, "final")
trainer.save_model(fp)
tok.save_pretrained(fp)
json.dump({"base":BASE_MODEL,"trainable":int(tr),"total":int(tot),"train":len(ds["train"]),"val":len(ds["validation"])}, open(os.path.join(OUTPUT_DIR,"info.json"),"w"), indent=2)
print(f"\nSAVED: {fp}")
print("TRAINING COMPLETE!")
