# %% [markdown]
# # Space LLM - Training on Kaggle P100 GPU
#
# Custom 10M parameter decoder-only transformer trained on space/astronomy data.
#
# **Architecture:** RoPE, SwiGLU, 6 layers, 256d, 8 heads, KV-cache
# **Data:** Wikipedia space articles + arXiv astronomy + comprehensive knowledge base
# **Hardware:** P100 GPU (16GB VRAM)

# %%
# Install dependencies
!pip install sentencepiece -q

# %%
# Clone the repo
!git clone https://github.com/korfalor-cloud/space-llm.git
%cd space-llm

# %%
# Run training (self-contained script - downloads data, trains model, generates samples)
!python kaggle_train.py

# %% [markdown]
# ## After Training: Generate Text

# %%
import torch
import sentencepiece as spm
import sys
sys.path.insert(0, ".")

from kaggle_train import SpaceLLM, ModelConfig
import json

# Load model
device = torch.device("cuda")
with open("checkpoints/model_config.json") as f:
    config = ModelConfig(**json.load(f))

model = SpaceLLM(config).to(device)
ckpt = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load tokenizer
sp = spm.SentencePieceProcessor(model_file="/kaggle/working/data/tokenizer/space_tokenizer.model")

# Generate
def ask(question, max_tokens=200, temp=0.7):
    prompt = f"Question: {question}\nAnswer:"
    ids = sp.encode(prompt, out_type=int)
    out = model.generate(torch.tensor([ids], dtype=torch.long, device=device),
                         max_new_tokens=max_tokens, temperature=temp, top_k=40, top_p=0.85)
    return sp.decode(out[0].cpu().tolist())

# Try some questions!
print(ask("What is a black hole?"))
print("\n" + "="*50 + "\n")
print(ask("Tell me about Mars exploration."))
print("\n" + "="*50 + "\n")
print(ask("What is dark matter?"))
