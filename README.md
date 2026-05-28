# Ashua - Complete Math AI Model

A fine-tuned math problem solver trained on **ALL available open-source math datasets** using LoRA for efficient training on consumer GPUs.

## What This Does

Takes Qwen2.5-Math-1.5B and fine-tunes it on **2.5M+ math problems** from 15 datasets covering:

| Topic | Datasets | Examples |
|-------|----------|----------|
| Arithmetic & Word Problems | GSM8K, Orca-Math, MathQA | ~238K |
| Algebra | MetaMathQA, AQuA-RAT, MathInstruct | ~757K |
| Competition Math | MATH, NuminaMath, OpenR1-Math, DAPO | ~1.5M |
| Geometry, Calculus, Stats | Camel-AI, MathInstruct, Lila | ~312K |
| Synthetic & RL-Verified | OpenMathInstruct-2, Big-Math, AceReason | ~800K |

**Capabilities after training:**
- Step-by-step solutions for K-12 through university math
- Chain-of-thought reasoning
- Covers: arithmetic, algebra, geometry, trigonometry, calculus, linear algebra, probability, statistics, number theory, combinatorics, competition math

## Quick Start (Kaggle)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare training data (downloads ~2.5M examples)
```bash
python run.py prepare
```

### 3. Train the model (several hours on T4)
```bash
python run.py train
```

### 4. Solve problems
```bash
# Single problem
python run.py solve "What is the derivative of x^3 + 2x?"

# Interactive mode
python run.py interactive
```

### 5. Evaluate
```bash
python run.py evaluate --samples 100
```

## All Included Datasets

### Tier 1: Core Instruction Tuning
| Dataset | Source | Size | Topics |
|---------|--------|------|--------|
| MetaMathQA | `meta-math/MetaMathQA` | 395K | Augmented GSM8K+MATH |
| MathInstruct | `TIGER-Lab/MathInstruct` | 262K | 13 datasets, CoT+PoT |
| Orca-Math | `microsoft/orca-math-word-problems-200k` | 200K | GPT-4 solutions |
| NuminaMath-CoT | `AI-MO/NuminaMath-CoT` | 860K | Competition + K-12 |
| OpenMathInstruct-2 | `nvidia/OpenMathInstruct-2` | 500K* | Llama3.1-405B synthetic |

### Tier 2: Reasoning Traces
| Dataset | Source | Size | Topics |
|---------|--------|------|--------|
| OpenR1-Math | `open-r1/OpenR1-Math-220k` | 220K | DeepSeek R1 traces |
| AceReason-Math | `nvidia/AceReason-Math` | 50K | Challenging problems |

### Tier 3: Foundational Benchmarks
| Dataset | Source | Size | Topics |
|---------|--------|------|--------|
| GSM8K | `openai/gsm8k` | 7.5K | Grade school math |
| MATH | `EleutherAI/hendrycks_math` | 7.5K | Competition math |
| MathQA | `allenai/math_qa` | 30K | Word problems |

### Tier 4: Specialized
| Dataset | Source | Size | Topics |
|---------|--------|------|--------|
| Camel-AI Math | `camel-ai/math` | 50K | 25 topics, 625 subtopics |
| Big-Math-RL | `SynthLabsAI/Big-Math-RL-Verified` | 251K | Verifiable solutions |
| DAPO-Math | `BytedTsinghua-SIA/DAPO-Math-17k` | 100K* | ByteDance+Tsinghua |
| AQuA-RAT | `deepmind/aqua_rat` | 100K | Algebraic reasoning |
| Lila | `allenai/lila` | varies | Diverse aggregation |

*Subsampled to fit training time

## Project Structure

```
Ashua/
├── run.py                      # Main entry point
├── requirements.txt            # Python dependencies
├── configs/
│   └── training_config.yaml    # All hyperparameters & dataset list
├── scripts/
│   ├── prepare_data.py         # Downloads & formats all 15 datasets
│   ├── train.py                # LoRA fine-tuning with 4-bit quantization
│   ├── inference.py            # Interactive & single-problem solving
│   └── evaluate.py             # GSM8K benchmark evaluation
├── data/                       # Prepared datasets (generated)
├── models/                     # Saved models
└── outputs/                    # Training checkpoints
```

## Configuration

Edit `configs/training_config.yaml`:
- **Base model**: `model.name` (try `Qwen/Qwen2.5-Math-7B` for better quality)
- **LoRA rank**: `lora.r` (64 = good balance, 128 = more capacity)
- **Batch size**: `training.per_device_train_batch_size`
- **Max samples per dataset**: `data.datasets[].max_samples`
- **Total limit**: `data.max_total_samples` (set to 500000 for faster iteration)

## Tips for Better Results

- **More epochs**: Set `num_train_epochs: 5` for deeper learning
- **Higher LoRA rank**: `r: 128` for more model capacity
- **Larger base model**: `Qwen/Qwen2.5-Math-7B` if you have more VRAM
- **Full dataset**: Remove all `max_samples` limits for maximum coverage
- **Learning rate**: Try `1e-4` for more conservative training
