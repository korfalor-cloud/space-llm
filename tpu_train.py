"""
Space LLM - Kaggle TPU Training Script (JAX/Flax)
Custom 10M parameter decoder-only transformer for TPU.
Architecture: RoPE, SwiGLU, 6 layers, 256d, 8 heads
Optimized for Kaggle TPU v3-8 / v5e-8.
"""

import os
import re
import json
import time
import math
import hashlib
import subprocess
import sys
import requests
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm

# Install JAX with TPU support on Kaggle
def setup_tpu():
    """Setup JAX for Kaggle TPU."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "jax[tpu]", "-f",
                               "https://storage.googleapis.com/jax-releases/libtpu_releases.html",
                               "-q"])
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "flax", "optax", "sentencepiece", "-q"])
    except:
        pass

setup_tpu()

# JAX/Flax imports
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Platform: {jax.platform()}")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    vocab_size: int = 8000  # Adaptive, set after tokenizer training
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    use_rope: bool = True
    norm_eps: float = 1e-5


@dataclass
class TrainConfig:
    batch_size: int = 64  # Larger batch for TPU
    grad_accum_steps: int = 4
    max_steps: int = 20000
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 5000
    checkpoint_dir: str = "/kaggle/working/checkpoints" if os.path.exists("/kaggle") else "checkpoints"


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL (JAX/Flax)
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_rope(dim: int, max_seq_len: int = 2048, base: float = 10000.0):
    """Precompute RoPE sin/cos tables."""
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray, offset: int = 0):
    """Apply rotary position embedding."""
    seq_len = x.shape[-2]
    cos_slice = cos[offset:offset + seq_len]
    sin_slice = sin[offset:offset + seq_len]

    # Reshape for broadcasting: (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
    cos_slice = cos_slice[jnp.newaxis, jnp.newaxis, :, :]
    sin_slice = sin_slice[jnp.newaxis, jnp.newaxis, :, :]

    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1) * sin_slice + x * cos_slice


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x / rms * scale


class SwiGLU(nn.Module):
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x):
        w1 = nn.Dense(self.d_ff, use_bias=False, name="w1")
        w2 = nn.Dense(self.d_model, use_bias=False, name="w2")
        w3 = nn.Dense(self.d_ff, use_bias=False, name="w3")
        return w2(nn.silu(w1(x)) * w3(x))


class MultiHeadAttention(nn.Module):
    d_model: int
    n_heads: int
    max_seq_len: int
    dropout: float = 0.1
    use_rope: bool = True

    @nn.compact
    def __call__(self, x, mask, deterministic=True):
        head_dim = self.d_model // self.n_heads

        q = nn.Dense(self.d_model, use_bias=False, name="q_proj")(x)
        k = nn.Dense(self.d_model, use_bias=False, name="k_proj")(x)
        v = nn.Dense(self.d_model, use_bias=False, name="v_proj")(x)

        # Reshape to (batch, seq, heads, head_dim) then transpose to (batch, heads, seq, head_dim)
        B, T, _ = x.shape
        q = q.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE
        if self.use_rope:
            cos, sin = precompute_rope(head_dim, self.max_seq_len)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        # Attention
        scale = 1.0 / math.sqrt(head_dim)
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = jnp.where(mask == 0, jnp.finfo(jnp.float32).min, attn)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.dropout)(attn, deterministic=deterministic)

        out = jnp.matmul(attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)
        out = nn.Dense(self.d_model, use_bias=False, name="out_proj")(out)
        out = nn.Dropout(rate=self.dropout)(out, deterministic=deterministic)
        return out


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    dropout: float = 0.1
    use_rope: bool = True
    norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x, mask, deterministic=True):
        # Pre-norm attention
        normed = RMSNorm(self.d_model, self.norm_eps, name="ln1")(x)
        attn_out = MultiHeadAttention(
            self.d_model, self.n_heads, self.max_seq_len, self.dropout, self.use_rope, name="attn"
        )(normed, mask, deterministic)
        x = x + attn_out

        # Pre-norm FF
        normed = RMSNorm(self.d_model, self.norm_eps, name="ln2")(x)
        ff_out = SwiGLU(self.d_model, self.d_ff, name="ff")(normed)
        ff_out = nn.Dropout(rate=self.dropout)(ff_out, deterministic=deterministic)
        x = x + ff_out

        return x


class SpaceLLM(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic=True):
        cfg = self.config
        B, T = input_ids.shape

        # Token embedding
        x = nn.Embedding(cfg.vocab_size, cfg.d_model, name="tok_emb")(input_ids)
        x = nn.Dropout(rate=cfg.dropout)(x, deterministic=deterministic)

        # Causal mask: (1, 1, T, T)
        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.float32))
        mask = mask[jnp.newaxis, jnp.newaxis, :, :]

        # Transformer blocks
        for i in range(cfg.n_layers):
            x = TransformerBlock(
                cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.max_seq_len,
                cfg.dropout, cfg.use_rope, cfg.norm_eps, name=f"layer_{i}"
            )(x, mask, deterministic)

        # Final norm + LM head
        x = RMSNorm(cfg.d_model, cfg.norm_eps, name="ln_f")(x)

        # Tied weights with embedding
        embedding = self.variables["params"]["tok_emb"]["embedding"]
        logits = jnp.dot(x, embedding.T)

        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING STATE
# ═══════════════════════════════════════════════════════════════════════════════

class TrainState(train_state.TrainState):
    rng: jnp.ndarray


def create_train_state(rng, config: ModelConfig, learning_rate: float, weight_decay: float):
    """Create training state with AdamW optimizer."""
    model = SpaceLLM(config)
    dummy_input = jnp.zeros((1, config.max_seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input, deterministic=True)["params"]

    # Count parameters
    param_count = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {param_count:,}")

    # AdamW optimizer
    tx = optax.adamw(
        learning_rate=learning_rate,
        b1=0.9,
        b2=0.95,
        weight_decay=weight_decay,
    )

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        rng=rng,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS & TRAINING STEP
# ═══════════════════════════════════════════════════════════════════════════════

def compute_loss(params, apply_fn, input_ids, targets, rng, deterministic=False):
    """Compute cross-entropy loss."""
    logits = apply_fn({"params": params}, input_ids, deterministic=deterministic, rngs={"dropout": rng})
    # Shift for next-token prediction
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    # Flatten
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    # Cross-entropy
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    return loss.mean()


@jax.jit
def train_step(state, input_ids, targets):
    """Single training step with gradient accumulation."""
    rng, new_rng = jax.random.split(state.rng)

    def loss_fn(params):
        return compute_loss(params, state.apply_fn, input_ids, targets, rng)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)

    # Gradient clipping
    grads_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    clip_factor = jnp.minimum(1.0, 1.0 / (grads_norm + 1e-8))
    grads = jax.tree.map(lambda g: g * clip_factor, grads)

    state = state.apply_gradients(grads=grads, rng=new_rng)
    return state, loss, grads_norm


@jax.jit
def eval_step(state, input_ids, targets):
    """Evaluation step."""
    rng = jax.random.PRNGKey(0)
    return compute_loss(state.params, state.apply_fn, input_ids, targets, rng, deterministic=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LEARNING RATE SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════════

def get_lr(step, warmup, max_steps, lr, min_lr):
    if step < warmup:
        return lr * step / warmup
    if step > max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text):
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[http[^\]]*\]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"'''?([^']+)'''?", r"\1", text)
    text = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def download_wikipedia():
    print("[Wikipedia] Downloading space articles via API...")
    texts = []
    session = requests.Session()
    session.headers.update({"User-Agent": "SpaceLLM/1.0 (educational)"})

    terms = [
        "astronomy", "planet", "star", "galaxy", "nebula", "supernova",
        "black hole", "cosmology", "telescope", "spacecraft", "NASA",
        "solar system", "exoplanet", "gravitational wave", "dark matter",
        "dark energy", "big bang", "cosmic microwave background",
        "Milky Way", "Andromeda galaxy", "Hubble Space Telescope",
        "James Webb Space Telescope", "Mars exploration", "Jupiter planet",
        "Saturn planet", "asteroid belt", "Kuiper belt", "comet", "meteor",
        "constellation", "light year", "redshift", "quasar", "pulsar",
        "neutron star", "white dwarf", "stellar evolution", "red giant",
        "nuclear fusion", "heliosphere", "International Space Station",
        "Apollo program", "Space Shuttle", "Voyager program", "Mars rover",
        "rocket propulsion", "orbital mechanics", "astronaut", "spacewalk",
        "Hertzsprung-Russell diagram", "parallax", "Cepheid variable",
        "event horizon", "spacetime", "general relativity",
        "gravitational lensing", "Hawking radiation", "cosmic ray",
        "solar wind", "aurora", "Van Allen belt", "planetary ring",
        "tidal force", "Roche limit", "Lagrange point", "Kepler's laws",
        "escape velocity", "space debris", "solar flare", "sunspot",
    ]

    for term in tqdm(terms, desc="[Wikipedia]"):
        try:
            resp = session.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "list": "search", "srsearch": term,
                "srlimit": 20, "format": "json",
            }, timeout=15)
            if resp.status_code != 200:
                continue
            for r in resp.json().get("query", {}).get("search", []):
                page_resp = session.get("https://en.wikipedia.org/w/api.php", params={
                    "action": "query", "pageids": r["pageid"],
                    "prop": "extracts", "explaintext": True, "format": "json",
                }, timeout=15)
                if page_resp.status_code == 200:
                    for page in page_resp.json().get("query", {}).get("pages", {}).values():
                        extract = page.get("extract", "")
                        if len(extract) > 300:
                            texts.append(f"# {r['title']}\n\n{extract}")
            time.sleep(0.1)
        except:
            continue

    print(f"[Wikipedia] Collected {len(texts)} articles")
    return texts


def download_arxiv():
    print("[arXiv] Downloading astronomy abstracts...")
    texts = []
    for cat in ["astro-ph", "gr-qc"]:
        start = 0
        while start < 300:
            try:
                resp = requests.get("http://export.arxiv.org/api/query", params={
                    "search_query": f"cat:{cat}*", "start": start,
                    "max_results": 50, "sortBy": "submittedDate", "sortOrder": "descending",
                }, timeout=60)
                if resp.status_code != 200:
                    break
                entries = resp.text.split("<entry>")[1:]
                if not entries:
                    break
                for entry in entries:
                    tm = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                    am = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
                    if tm and am:
                        texts.append(clean_text(f"{tm.group(1).strip()}\n\n{am.group(1).strip()}"))
                start += 50
                time.sleep(3)
            except:
                break
    print(f"[arXiv] Collected {len(texts)} abstracts")
    return texts


def generate_space_knowledge():
    """Comprehensive space knowledge base."""
    knowledge = []

    planets = {
        "Mercury": "Mercury is the smallest planet in our solar system and closest to the Sun. It has no atmosphere and experiences extreme temperature variations from minus 180 degrees Celsius at night to 430 degrees Celsius during the day. A year on Mercury lasts only 88 Earth days. Mercury has a large iron core that makes up about 75 percent of its radius. The MESSENGER spacecraft orbited Mercury from 2011 to 2015.",
        "Venus": "Venus is the second planet from the Sun and is often called Earth twin due to similar size and mass. However, Venus has a thick toxic atmosphere composed mainly of carbon dioxide with clouds of sulfuric acid. The surface temperature reaches 465 degrees Celsius, making it the hottest planet in the solar system. Venus rotates backwards compared to most planets.",
        "Earth": "Earth is the third planet from the Sun and the only known planet to support life. It has one natural satellite, the Moon. Earth atmosphere is composed of 78 percent nitrogen and 21 percent oxygen. The planet has a magnetic field that protects it from solar radiation. About 71 percent of Earth surface is covered by water.",
        "Mars": "Mars is the fourth planet from the Sun, known as the Red Planet due to iron oxide on its surface. Mars has the largest volcano in the solar system, Olympus Mons, standing 21.9 km high, and the deepest canyon, Valles Marineris, stretching 4000 km. Mars has two small moons: Phobos and Deimos.",
        "Jupiter": "Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined. It is a gas giant composed mainly of hydrogen and helium. Jupiter Great Red Spot is a storm larger than Earth that has been raging for at least 350 years. Jupiter has at least 95 known moons.",
        "Saturn": "Saturn is the sixth planet from the Sun, famous for its spectacular ring system made of ice and rock particles. Saturn is a gas giant composed mainly of hydrogen and helium. It has at least 146 known moons, including Titan, which has a thick atmosphere and liquid methane lakes.",
        "Uranus": "Uranus is the seventh planet from the Sun and the first discovered using a telescope, found by William Herschel in 1781. It is an ice giant composed mainly of water, methane, and ammonia ices. Uranus rotates on its side with an axial tilt of 98 degrees.",
        "Neptune": "Neptune is the eighth and farthest planet from the Sun. It is an ice giant with the strongest winds in the solar system, reaching speeds of 2100 km per hour. Neptune has 16 known moons, the largest being Triton, which orbits in the opposite direction to Neptune rotation.",
    }
    for planet, desc in planets.items():
        knowledge.extend([f"# {planet}\n\n{desc}", f"Question: Tell me about {planet}.\nAnswer: {desc}", f"What is {planet}? {desc}"])

    stellar = [
        "Stars are massive celestial bodies that produce light and heat through nuclear fusion in their cores. They form from clouds of gas and dust called nebulae. The life cycle of a star depends on its mass. Low-mass stars like our Sun become red giants and then white dwarfs. High-mass stars can explode as supernovae and become neutron stars or black holes.",
        "The Sun is a G-type main-sequence star at the center of our solar system. It contains 99.86 percent of the mass in the solar system. The Sun core temperature reaches 15 million degrees Celsius, where hydrogen atoms fuse into helium. The Sun is approximately 4.6 billion years old and is expected to continue burning for another 5 billion years.",
        "A supernova is a powerful stellar explosion that occurs at the end of a massive star life cycle. Supernovae are so bright they can outshine entire galaxies for weeks. They create and distribute heavy elements like gold, platinum, and uranium into space, seeding future generations of stars and planets.",
        "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once past the event horizon. The first image of a black hole was captured in 2019 by the Event Horizon Telescope, showing the supermassive black hole in galaxy M87. Sagittarius A star is the supermassive black hole at the center of our Milky Way galaxy.",
        "Neutron stars are the collapsed cores of massive stars that have undergone supernova explosions. They are incredibly dense, with a mass of 1.4 to 2 solar masses packed into a sphere only about 20 km in diameter. A teaspoon of neutron star material would weigh about 6 billion tons.",
        "White dwarfs are the remnants of low and medium-mass stars after they exhaust their nuclear fuel. They are about the size of Earth but have a mass similar to the Sun. The Chandrasekhar limit of 1.4 solar masses is the maximum mass a white dwarf can have before it collapses.",
    ]
    for text in stellar:
        knowledge.extend([text, f"Question: {text.split('.')[0]}?\nAnswer: {text}"])

    cosmology = [
        "The Milky Way is the galaxy that contains our solar system. It is a barred spiral galaxy with a diameter of approximately 100,000 light-years and contains between 100 billion and 400 billion stars. The supermassive black hole at its center has a mass of about 4 million times that of our Sun.",
        "Dark matter is a hypothetical form of matter that does not emit or interact with electromagnetic radiation. It is estimated to make up about 27 percent of the total mass-energy content of the universe. Evidence comes from gravitational effects on visible matter.",
        "Dark energy is a hypothetical form of energy that permeates all of space and causes the accelerating expansion of the universe. It makes up about 68 percent of the total mass-energy content of the universe.",
        "The Big Bang theory describes the origin of the universe as an extremely hot and dense state approximately 13.8 billion years ago. Key evidence includes the cosmic microwave background radiation, the abundance of light elements, and the redshift of distant galaxies.",
        "Exoplanets are planets that orbit stars outside our solar system. The Kepler space telescope discovered over 2,600 exoplanets. The James Webb Space Telescope is now characterizing exoplanet atmospheres.",
        "Gravitational waves are ripples in spacetime caused by accelerating massive objects, predicted by Albert Einstein in 1916. They were first directly detected in 2015 by LIGO from the merger of two black holes.",
        "Einstein theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy. It predicts the bending of light by gravity, time dilation near massive objects, and the existence of black holes.",
    ]
    for text in cosmology:
        knowledge.extend([text, f"Question: {text.split('.')[0]}?\nAnswer: {text}"])

    exploration = [
        "The Apollo program was NASA human spaceflight program that landed the first humans on the Moon. Apollo 11 landed on July 20, 1969. Neil Armstrong was the first person to walk on the lunar surface. In total, 12 astronauts walked on the Moon during six Apollo missions.",
        "The International Space Station is a modular space station in low Earth orbit. It has been continuously occupied since November 2000. The station orbits Earth approximately every 90 minutes at an altitude of about 408 km.",
        "The Hubble Space Telescope was launched in 1990 and has made over 1.5 million observations. It has helped determine the age of the universe at 13.8 billion years.",
        "The James Webb Space Telescope is the largest and most powerful space telescope ever built, launched on December 25, 2021. It observes in infrared light and can see the earliest galaxies formed after the Big Bang.",
        "Voyager 1 and Voyager 2 are NASA space probes launched in 1977. Voyager 1 is the most distant human-made object, currently over 24 billion km from Earth. Voyager 1 entered interstellar space in 2012.",
        "The Curiosity rover landed on Mars on August 6, 2012. The Perseverance rover landed on February 18, 2021, and is collecting samples for future return to Earth. The Ingenuity helicopter made the first powered flight on another planet.",
        "SpaceX is revolutionizing space travel with reusable rockets. The Falcon 9 rocket has successfully landed and been reused over 200 times. The Starship vehicle is designed for missions to the Moon and Mars.",
    ]
    for text in exploration:
        knowledge.extend([text, f"Question: {text.split('.')[0]}?\nAnswer: {text}"])

    return knowledge


def prepare_data():
    base_dir = Path("/kaggle/working") if os.path.exists("/kaggle") else Path(".")
    data_dir = base_dir / "data"
    tokenized_dir = data_dir / "tokenized"
    tokenizer_dir = data_dir / "tokenizer"
    for d in [tokenized_dir, tokenizer_dir]:
        d.mkdir(parents=True, exist_ok=True)

    meta_path = tokenized_dir / "meta.json"
    if meta_path.exists():
        print("[Skip] Data already prepared.")
        with open(meta_path) as f:
            return json.load(f)

    all_texts = []
    all_texts.extend(download_wikipedia())
    all_texts.extend(download_arxiv())
    knowledge = generate_space_knowledge()
    all_texts.extend(knowledge)
    print(f"[Knowledge] Generated {len(knowledge)} entries")

    if not all_texts:
        raise RuntimeError("No data collected!")

    # Deduplicate
    seen = set()
    unique = []
    for t in all_texts:
        h = hashlib.md5(t.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(t)
    print(f"[Dedup] {len(unique)} unique documents")

    # Train tokenizer
    import sentencepiece as spm
    corpus_file = tokenizer_dir / "corpus.txt"
    with open(corpus_file, "w") as f:
        for t in unique:
            f.write(t.replace("\n", " ") + "\n")

    total_chars = sum(len(t) for t in unique)
    max_vocab = min(8000, max(1000, total_chars // 10))
    print(f"[Tokenizer] Corpus: {total_chars:,} chars, vocab_size={max_vocab}")

    model_prefix = str(tokenizer_dir / "space_tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(corpus_file), model_prefix=model_prefix,
        vocab_size=max_vocab, model_type="bpe", character_coverage=0.9995,
        num_threads=4, split_digits=True, byte_fallback=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    )

    # Tokenize
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    all_tokens = []
    for t in tqdm(unique, desc="[Tokenize]"):
        tokens = sp.encode(t, out_type=int)
        all_tokens.extend([sp.bos_id()] + tokens + [sp.eos_id()])

    all_tokens = np.array(all_tokens, dtype=np.int32)
    val_size = int(len(all_tokens) * 0.05)
    train_tokens = all_tokens[:-val_size]
    val_tokens = all_tokens[-val_size:]

    np.save(tokenized_dir / "train.npy", train_tokens)
    np.save(tokenized_dir / "val.npy", val_tokens)

    meta = {"vocab_size": sp.get_piece_size(), "train_tokens": len(train_tokens), "val_tokens": len(val_tokens)}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Train: {len(train_tokens):,}, Val: {len(val_tokens):,} tokens")
    return meta


def make_batches(data, seq_len, batch_size):
    """Create batches from token array."""
    n = len(data)
    n_batches = (n - seq_len - 1) // (batch_size * seq_len)
    data = data[:n_batches * batch_size * seq_len + seq_len + 1]

    inputs = []
    targets = []
    for i in range(n_batches * batch_size):
        start = i * seq_len
        chunk = data[start:start + seq_len + 1]
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])

    inputs = np.array(inputs, dtype=np.int32).reshape(batch_size, n_batches, seq_len)
    targets = np.array(targets, dtype=np.int32).reshape(batch_size, n_batches, seq_len)

    # Transpose to (n_batches, batch_size, seq_len) for iteration
    return inputs.transpose(1, 0, 2), targets.transpose(1, 0, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate(state, tokenizer, prompt, max_new_tokens=150, temperature=0.7, top_k=40):
    """Generate text from a prompt."""
    ids = tokenizer.encode(prompt, out_type=int)
    input_ids = jnp.array([ids], dtype=jnp.int32)

    for _ in range(max_new_tokens):
        logits = state.apply_fn({"params": state.params}, input_ids, deterministic=True)
        next_logits = logits[0, -1, :] / temperature

        # Top-k
        if top_k > 0:
            top_k_val = jnp.sort(next_logits)[-top_k]
            next_logits = jnp.where(next_logits < top_k_val, jnp.finfo(jnp.float32).min, next_logits)

        # Sample
        probs = nn.softmax(next_logits)
        rng = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))
        next_token = jax.random.categorical(rng, jnp.log(probs))

        input_ids = jnp.concatenate([input_ids, next_token[jnp.newaxis, jnp.newaxis]], axis=1)

    return tokenizer.decode(input_ids[0].tolist())


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train():
    print(f"JAX devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()}")
    print(f"Platform: {jax.platform()}")

    # Prepare data
    meta = prepare_data()
    import sentencepiece as spm
    base_dir = "/kaggle/working" if os.path.exists("/kaggle") else "."
    tokenizer = spm.SentencePieceProcessor(model_file=f"{base_dir}/data/tokenizer/space_tokenizer.model")

    # Config
    model_config = ModelConfig(vocab_size=meta["vocab_size"])
    train_config = TrainConfig()

    print(f"Vocab size: {model_config.vocab_size}")

    # Create model state
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, model_config, train_config.learning_rate, train_config.weight_decay)

    # Load data
    train_data = np.load(f"{base_dir}/data/tokenized/train.npy")
    val_data = np.load(f"{base_dir}/data/tokenized/val.npy")

    train_inputs, train_targets = make_batches(train_data, model_config.max_seq_len, train_config.batch_size)
    val_inputs, val_targets = make_batches(val_data, model_config.max_seq_len, train_config.batch_size)

    n_train_batches = len(train_inputs)
    n_val_batches = min(20, len(val_inputs))

    print(f"Train batches: {n_train_batches}, Val batches: {n_val_batches}")
    print(f"\n{'='*60}")
    print(f"Training on TPU: {train_config.max_steps} steps")
    print(f"{'='*60}\n")

    # Training loop
    step = 0
    total_loss = 0.0
    best_val_loss = float("inf")
    start_time = time.time()
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    rng = jax.random.PRNGKey(42)

    while step < train_config.max_steps:
        # Shuffle batches
        perm = np.random.permutation(n_train_batches)

        for batch_idx in range(n_train_batches):
            if step >= train_config.max_steps:
                break

            idx = perm[batch_idx]
            input_ids = jnp.array(train_inputs[idx])
            targets = jnp.array(train_targets[idx])

            # Update learning rate
            lr = get_lr(step, train_config.warmup_steps, train_config.max_steps,
                       train_config.learning_rate, train_config.min_lr)
            state = state.replace(opt_state=optax.inject_hyperparams(state.tx.update)(lr, state.opt_state))

            # Train step
            state, loss, grad_norm = train_step(state, input_ids, targets)
            total_loss += loss.item()
            step += 1

            # Logging
            if step % train_config.log_every == 0:
                avg = total_loss / train_config.log_every
                elapsed = time.time() - start_time
                print(f"Step {step:>6d}/{train_config.max_steps} | Loss: {avg:.4f} | LR: {lr:.2e} | Grad: {grad_norm:.2f} | Time: {elapsed:.0f}s")
                total_loss = 0.0

            # Eval
            if step % train_config.eval_every == 0:
                val_losses = []
                for i in range(n_val_batches):
                    vloss = eval_step(state, jnp.array(val_inputs[i]), jnp.array(val_targets[i]))
                    val_losses.append(vloss.item())
                val_loss = np.mean(val_losses)
                ppl = math.exp(val_loss)
                print(f"  [Eval] Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save checkpoint
                    import pickle
                    with open(f"{train_config.checkpoint_dir}/best.pkl", "wb") as f:
                        pickle.dump({"step": step, "params": jax.device_get(state.params), "val_loss": val_loss}, f)
                    print(f"  [Checkpoint] Saved best model")

            # Save periodic checkpoint
            if step % train_config.save_every == 0:
                import pickle
                with open(f"{train_config.checkpoint_dir}/step_{step}.pkl", "wb") as f:
                    pickle.dump({"step": step, "params": jax.device_get(state.params)}, f)

    # Save final
    import pickle
    with open(f"{train_config.checkpoint_dir}/final.pkl", "wb") as f:
        pickle.dump({"step": step, "params": jax.device_get(state.params), "val_loss": best_val_loss}, f)

    # Save config
    with open(f"{train_config.checkpoint_dir}/model_config.json", "w") as f:
        json.dump({"vocab_size": model_config.vocab_size, "d_model": 256, "n_heads": 8,
                    "n_layers": 6, "d_ff": 1024, "max_seq_len": 512}, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/3600:.1f}h | Best val loss: {best_val_loss:.4f} | Perplexity: {math.exp(best_val_loss):.2f}")
    print(f"{'='*60}")

    # Generate samples
    print("\n--- Sample Generations ---")
    for prompt in ["Question: What is a black hole?\nAnswer:", "Question: How far is Mars?\nAnswer:", "The Sun is a star that"]:
        print(f"\nQ: {prompt}")
        print(f"A: {generate(state, tokenizer, prompt)}")


if __name__ == "__main__":
    train()
