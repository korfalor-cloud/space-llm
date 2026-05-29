"""
Space LLM - 100M Token Collection + PyTorch GPU Training
Collects data from ALL sources then trains on T4 GPU.
"""

import subprocess
import sys
import os

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "sentencepiece", "tqdm", "requests", "-q"])

# Clone repo
if not os.path.exists("space-llm"):
    subprocess.check_call(["git", "clone", "https://github.com/korfalor-cloud/space-llm.git"])

os.chdir("space-llm")
sys.path.insert(0, ".")

# Step 1: Collect 100M tokens of data
print("\n" + "="*60)
print("STEP 1: COLLECTING 100M TOKENS OF DATA")
print("="*60)
from collect_100m import main as collect_data
collect_data()

# Step 2: Train with PyTorch on GPU
print("\n" + "="*60)
print("STEP 2: TRAINING MODEL ON GPU")
print("="*60)
from train import train
train()
