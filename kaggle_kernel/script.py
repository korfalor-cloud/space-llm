"""
Space LLM - 100M Token Training Pipeline
Collects data from ALL sources and trains a 10M parameter model.
"""

import subprocess
import sys
import os

# Install all dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "jax[tpu]", "-f",
                       "https://storage.googleapis.com/jax-releases/libtpu_releases.html", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "flax", "optax", "sentencepiece", "datasets", "tqdm", "-q"])

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

# Step 2: Train the model
print("\n" + "="*60)
print("STEP 2: TRAINING MODEL")
print("="*60)
from tpu_train import train
train()
