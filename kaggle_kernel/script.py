"""
Space LLM - Continued Training with More Data
Loads existing trained model and trains on expanded dataset.
"""

import subprocess
import sys
import os
import shutil

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece", "-q"])

# Clone repo
if not os.path.exists("space-llm"):
    subprocess.check_call(["git", "clone", "https://github.com/korfalor-cloud/space-llm.git"])

os.chdir("space-llm")
sys.path.insert(0, ".")

# Copy checkpoints from Kaggle input
src = "/kaggle/input/space-llm-checkpoints/checkpoints"
dst = "checkpoints_v2"
if os.path.exists(src):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print(f"Copied checkpoints from {src}")
else:
    print(f"No checkpoints at {src}, training from scratch")

# Run continued training
from tpu_train_continue import train
train()
