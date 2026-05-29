"""
Space LLM - Kaggle TPU Training (JAX/Flax)
Custom 10M parameter decoder-only transformer.
Architecture: RoPE, SwiGLU, 6 layers, 256d, 8 heads
Hardware: TPU v3-8 or v5e-8
"""

import subprocess
import sys
import os

# Clone repo
if not os.path.exists("space-llm"):
    subprocess.check_call(["git", "clone", "https://github.com/korfalor-cloud/space-llm.git"])

os.chdir("space-llm")
sys.path.insert(0, ".")

# Run TPU training (installs JAX/TPU dependencies automatically)
from tpu_train import train
train()
