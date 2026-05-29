"""
Space LLM - Continued Training with More Data
Loads existing trained model and continues training on expanded dataset.
"""

import os
import re
import json
import time
import math
import hashlib
import subprocess
import sys
import pickle
import requests
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

# Install JAX/Flax on Kaggle
def setup_tpu():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "jax[tpu]", "-f",
                               "https://storage.googleapis.com/jax-releases/libtpu_releases.html", "-q"])
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "flax", "optax", "sentencepiece", "-q"])
    except:
        pass

setup_tpu()

import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

print(f"JAX: {jax.__version__} | Devices: {jax.device_count()} | Backend: {jax.default_backend()}")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    vocab_size: int = 8000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1
    use_rope: bool = True
    norm_eps: float = 1e-5

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainConfig:
    batch_size: int = 16
    max_steps: int = 30000  # More steps for continued training
    learning_rate: float = 1e-4  # Lower LR for fine-tuning
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 5000
    checkpoint_dir: str = "/kaggle/working/checkpoints_v2" if os.path.exists("/kaggle") else "checkpoints_v2"
    resume_from: str = "/kaggle/input/space-llm-checkpoints/checkpoints/best.pkl"  # Kaggle input path


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL (same as before)
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_rope(dim, max_seq_len=2048, base=10000.0):
    inv_freq = 1.0 / (base ** (jnp.arange(0, dim, 2).astype(jnp.float32) / dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x, cos, sin):
    seq_len = x.shape[-2]
    cos_s = cos[:seq_len][jnp.newaxis, jnp.newaxis, :, :]
    sin_s = sin[:seq_len][jnp.newaxis, jnp.newaxis, :, :]
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1) * sin_s + x * cos_s


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", jax.nn.initializers.ones, (self.dim,))
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
        return w2(jax.nn.silu(w1(x)) * w3(x))


class Embedding(nn.Module):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self, x):
        emb = self.param("embedding", jax.nn.initializers.normal(stddev=0.02),
                          (self.num_embeddings, self.features))
        return emb[x]


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

        B, T, _ = x.shape
        q = q.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        if self.use_rope:
            cos, sin = precompute_rope(head_dim, self.max_seq_len)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        scale = 1.0 / math.sqrt(head_dim)
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        attn = jnp.where(mask == 0, jnp.finfo(jnp.float32).min, attn)
        attn = jax.nn.softmax(attn, axis=-1)
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
        normed = RMSNorm(self.d_model, self.norm_eps, name="ln1")(x)
        x = x + MultiHeadAttention(
            self.d_model, self.n_heads, self.max_seq_len, self.dropout, self.use_rope, name="attn"
        )(normed, mask, deterministic)

        normed = RMSNorm(self.d_model, self.norm_eps, name="ln2")(x)
        x = x + SwiGLU(self.d_model, self.d_ff, name="ff")(normed)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        return x


class SpaceLLM(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic=True):
        cfg = self.config
        B, T = input_ids.shape

        x = Embedding(cfg.vocab_size, cfg.d_model, name="tok_emb")(input_ids)
        x = nn.Dropout(rate=cfg.dropout)(x, deterministic=deterministic)

        mask = jnp.tril(jnp.ones((T, T), dtype=jnp.float32))[jnp.newaxis, jnp.newaxis, :, :]

        for i in range(cfg.n_layers):
            x = TransformerBlock(
                cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.max_seq_len,
                cfg.dropout, cfg.use_rope, cfg.norm_eps, name=f"layer_{i}"
            )(x, mask, deterministic)

        x = RMSNorm(cfg.d_model, cfg.norm_eps, name="ln_f")(x)
        embedding = self.variables["params"]["tok_emb"]["embedding"]
        logits = jnp.dot(x, embedding.T)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_loss(params, apply_fn, input_ids, targets, rng):
    logits = apply_fn({"params": params}, input_ids, deterministic=False, rngs={"dropout": rng})
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    ).mean()


def train_step_fn(params, apply_fn, input_ids, targets, rng):
    def loss_fn(p):
        return compute_loss(p, apply_fn, input_ids, targets, rng)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    grads_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    clip_factor = jnp.minimum(1.0, 1.0 / (grads_norm + 1e-8))
    grads = jax.tree.map(lambda g: g * clip_factor, grads)
    return loss, grads, grads_norm


def eval_fn(params, apply_fn, input_ids, targets, rng):
    logits = apply_fn({"params": params}, input_ids, deterministic=True)
    logits = logits[:, :-1, :]
    targets = targets[:, 1:]
    return optax.softmax_cross_entropy_with_integer_labels(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    ).mean()


train_step_jit = jax.jit(train_step_fn, static_argnums=(1,))
eval_jit = jax.jit(eval_fn, static_argnums=(1,))


# ═══════════════════════════════════════════════════════════════════════════════
# EXPANDED DATA PIPELINE
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


def download_wikipedia_expanded():
    """Download many more Wikipedia articles with expanded search terms."""
    print("[Wikipedia] Downloading expanded space articles...")
    texts = []
    session = requests.Session()
    session.headers.update({"User-Agent": "SpaceLLM/2.0 (educational)"})

    # Massively expanded search terms
    terms = [
        # Planets and solar system
        "Mercury planet", "Venus planet", "Earth planet", "Mars planet",
        "Jupiter planet", "Saturn planet", "Uranus planet", "Neptune planet",
        "Pluto dwarf planet", "Ceres dwarf planet", "Eris dwarf planet",
        "asteroid belt", "Kuiper belt", "Oort cloud", "heliosphere",
        "solar wind", "solar flare", "sunspot", "solar cycle",
        "planetary ring", "planetary atmosphere", "planetary geology",
        "tidal force", "Roche limit", "Hill sphere", "Lagrange point",
        "orbital resonance", "orbital mechanics", "escape velocity",

        # Moons
        "Moon", "Europa moon", "Ganymede moon", "Callisto moon",
        "Io moon", "Titan moon", "Enceladus moon", "Triton moon",
        "Phobos moon", "Deimos moon",

        # Stars and stellar physics
        "main sequence star", "red giant", "white dwarf", "neutron star",
        "pulsar", "magnetar", "brown dwarf", "red dwarf", "blue giant",
        "supergiant star", "hypergiant", "Wolf-Rayet star",
        "stellar evolution", "Hertzsprung-Russell diagram",
        "nuclear fusion", "proton-proton chain", "CNO cycle",
        "Chandrasekhar limit", "Tolman-Oppenheimer-Volkoff limit",
        "supernova", "Type Ia supernova", "Type II supernova",
        "supernova remnant", "nebula", "planetary nebula",
        "emission nebula", "reflection nebula", "dark nebula",
        "HII region", "molecular cloud", "Bok globule",
        "stellar classification", "spectral class", "absolute magnitude",
        "apparent magnitude", "parallax", "standard candle",
        "Cepheid variable", "RR Lyrae", "Mira variable",

        # Black holes
        "black hole", "event horizon", "singularity", "Hawking radiation",
        "Penrose process", "Blandford-Znajek", "accretion disk",
        "relativistic jet", "supermassive black hole",
        "stellar black hole", "intermediate black hole",
        "primordial black hole", "black hole information paradox",

        # Galaxies
        "Milky Way", "Andromeda galaxy", "galaxy cluster",
        "galaxy formation", "spiral galaxy", "elliptical galaxy",
        "irregular galaxy", "dwarf galaxy", "active galactic nucleus",
        "quasar", "blazar", "Seyfert galaxy", "LINER galaxy",
        "galactic halo", "galactic bulge", "galactic disk",
        "dark matter halo", "galaxy rotation curve",

        # Cosmology
        "Big Bang", "cosmic microwave background", "cosmic inflation",
        "dark matter", "dark energy", "cosmological constant",
        "Hubble constant", "Hubble law", "redshift",
        "cosmic distance ladder", "standard siren",
        "baryon acoustic oscillation", "cosmic web",
        "large-scale structure", "void cosmology",
        "observable universe", "age of the universe",
        "Friedmann equations", "Lambda-CDM model",

        # Relativity and gravitational physics
        "general relativity", "special relativity", "spacetime",
        "gravitational wave", "LIGO", "gravitational lensing",
        "strong gravitational lensing", "weak gravitational lensing",
        "frame dragging", "Lense-Thirring effect",
        "gravitational time dilation", "Shapiro delay",
        "perihelion precession", "geodetic effect",

        # Telescopes and observatories
        "Hubble Space Telescope", "James Webb Space Telescope",
        "Chandra X-ray Observatory", "Spitzer Space Telescope",
        "Fermi Gamma-ray Space Telescope", "Compton Gamma Ray Observatory",
        "Kepler space telescope", "TESS telescope",
        "Atacama Large Millimeter Array", "Very Large Array",
        "Square Kilometre Array", "Event Horizon Telescope",
        "gravational wave observatory", "neutrino observatory",

        # Space exploration
        "Apollo program", "Space Shuttle", "International Space Station",
        "SpaceX", "Falcon 9", "Starship", "Blue Origin",
        "Voyager program", "Pioneer program", "New Horizons",
        "Cassini-Huygens", "Galileo spacecraft", "Juno spacecraft",
        "Mars rover", "Curiosity rover", "Perseverance rover",
        "Spirit rover", "Opportunity rover", "Ingenuity helicopter",
        "Mars helicopter", "Mars sample return",
        "Artemis program", "Lunar Gateway", "Mars colonization",
        "space colonization", "generation ship", "interstellar travel",
        "Breakthrough Starshot", "Project Daedalus",

        # Astrobiology
        "astrobiology", "extraterrestrial life", "habitable zone",
        "Drake equation", "Fermi paradox", "Great Filter",
    ]

    for term in tqdm(terms, desc="[Wikipedia]"):
        try:
            resp = session.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "list": "search", "srsearch": term,
                "srlimit": 50, "format": "json", "sroffset": 0,
            }, timeout=15)
            if resp.status_code != 200:
                continue

            results = resp.json().get("query", {}).get("search", [])
            for r in results[:30]:  # Limit per term
                page_resp = session.get("https://en.wikipedia.org/w/api.php", params={
                    "action": "query", "pageids": r["pageid"],
                    "prop": "extracts", "explaintext": True, "format": "json",
                }, timeout=15)
                if page_resp.status_code == 200:
                    for page in page_resp.json().get("query", {}).get("pages", {}).values():
                        extract = page.get("extract", "")
                        if len(extract) > 300:
                            texts.append(f"# {r['title']}\n\n{extract}")
            time.sleep(0.05)
        except:
            continue

    print(f"[Wikipedia] Collected {len(texts)} articles")
    return texts


def download_arxiv_expanded():
    """Download more arXiv papers from multiple categories."""
    print("[arXiv] Downloading expanded astronomy papers...")
    texts = []
    categories = [
        "astro-ph", "astro-ph.GA", "astro-ph.CO", "astro-ph.EP",
        "astro-ph.HE", "astro-ph.IM", "astro-ph.SR",
        "gr-qc",  # General Relativity
        "hep-ph",  # High Energy Physics - Phenomenology
        "hep-th",  # High Energy Physics - Theory
    ]

    for cat in categories:
        start = 0
        while start < 500:
            try:
                resp = requests.get("http://export.arxiv.org/api/query", params={
                    "search_query": f"cat:{cat}", "start": start,
                    "max_results": 100, "sortBy": "submittedDate", "sortOrder": "descending",
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
                start += 100
                time.sleep(2)
            except:
                break

    print(f"[arXiv] Collected {len(texts)} abstracts")
    return texts


def generate_comprehensive_knowledge():
    """Generate comprehensive space knowledge base - much larger."""
    knowledge = []

    # Planet descriptions with multiple angles
    planets_data = {
        "Mercury": {
            "basic": "Mercury is the smallest planet in our solar system and closest to the Sun. It has no atmosphere and experiences extreme temperature variations from minus 180 degrees Celsius at night to 430 degrees Celsius during the day.",
            "orbit": "Mercury orbits the Sun at an average distance of 57.9 million kilometers. A year on Mercury lasts only 88 Earth days, but a single day lasts 59 Earth days due to its slow rotation.",
            "structure": "Mercury has a large iron core that makes up about 75 percent of its radius, making it the second densest planet after Earth. The surface is covered in impact craters and ancient lava flows.",
            "exploration": "The MESSENGER spacecraft orbited Mercury from 2011 to 2015, mapping its surface in detail. Mariner 10 flew by Mercury three times in 1974-1975.",
        },
        "Venus": {
            "basic": "Venus is the second planet from the Sun and is often called Earth's twin due to similar size and mass. However, Venus has a thick toxic atmosphere composed mainly of carbon dioxide with clouds of sulfuric acid.",
            "atmosphere": "The surface temperature reaches 465 degrees Celsius, making it the hottest planet in the solar system. The atmospheric pressure is about 90 times that of Earth. Venus rotates backwards compared to most planets.",
            "exploration": "The Soviet Venera program successfully landed several spacecraft on Venus's surface. Magellan mapped Venus using radar in the early 1990s. Venus has no moons and no magnetic field.",
        },
        "Earth": {
            "basic": "Earth is the third planet from the Sun and the only known planet to support life. It has one natural satellite, the Moon.",
            "atmosphere": "Earth's atmosphere is composed of 78 percent nitrogen and 21 percent oxygen. The planet has a magnetic field that protects it from solar radiation.",
            "structure": "About 71 percent of Earth's surface is covered by water. Earth's axial tilt of 23.5 degrees causes seasons. The planet formed approximately 4.5 billion years ago.",
        },
        "Mars": {
            "basic": "Mars is the fourth planet from the Sun, known as the Red Planet due to iron oxide on its surface.",
            "geology": "Mars has the largest volcano in the solar system, Olympus Mons, standing 21.9 km high, and the deepest canyon, Valles Marineris, stretching 4000 km. Mars has two small moons: Phobos and Deimos.",
            "exploration": "The Curiosity and Perseverance rovers are currently exploring Mars. Evidence suggests Mars once had liquid water on its surface. Mars has a thin atmosphere composed mainly of carbon dioxide.",
        },
        "Jupiter": {
            "basic": "Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined. It is a gas giant composed mainly of hydrogen and helium.",
            "storms": "Jupiter's Great Red Spot is a storm larger than Earth that has been raging for at least 350 years. Jupiter has at least 95 known moons, including the four large Galilean moons: Io, Europa, Ganymede, and Callisto.",
            "moons": "Europa is considered one of the most likely places to find extraterrestrial life due to its subsurface ocean. Io is the most volcanically active body in the solar system.",
        },
        "Saturn": {
            "basic": "Saturn is the sixth planet from the Sun, famous for its spectacular ring system made of ice and rock particles. Saturn is a gas giant composed mainly of hydrogen and helium.",
            "rings": "Saturn's rings are divided into several groups separated by gaps called divisions. The rings extend up to 282,000 km from Saturn but are only about 10 meters thick.",
            "moons": "Saturn has at least 146 known moons, including Titan, which has a thick atmosphere and liquid methane lakes. Enceladus has geysers of water ice erupting from its south pole.",
        },
        "Uranus": {
            "basic": "Uranus is the seventh planet from the Sun and the first discovered using a telescope, found by William Herschel in 1781.",
            "structure": "It is an ice giant composed mainly of water, methane, and ammonia ices. Uranus rotates on its side with an axial tilt of 98 degrees, likely caused by a massive collision early in its history.",
            "moons": "Uranus has 27 known moons and a faint ring system. The atmosphere appears blue-green due to methane absorption.",
        },
        "Neptune": {
            "basic": "Neptune is the eighth and farthest planet from the Sun. It is an ice giant with the strongest winds in the solar system, reaching speeds of 2100 km per hour.",
            "moons": "Neptune has 16 known moons, the largest being Triton, which orbits in the opposite direction to Neptune's rotation, suggesting it was captured from the Kuiper Belt.",
            "exploration": "Neptune was visited only once by Voyager 2 in 1989. Its blue color comes from methane in its atmosphere.",
        },
    }

    for planet, sections in planets_data.items():
        for key, text in sections.items():
            knowledge.append(f"# {planet} - {key.title()}\n\n{text}")
            knowledge.append(f"Question: What is {planet}'s {key}?\nAnswer: {text}")
        # Combined description
        combined = " ".join(sections.values())
        knowledge.append(f"# {planet}\n\n{combined}")
        knowledge.append(f"Question: Tell me about {planet}.\nAnswer: {combined}")

    # Stellar physics topics
    stellar_topics = {
        "Stellar Evolution": "Stars are massive celestial bodies that produce light and heat through nuclear fusion in their cores. They form from clouds of gas and dust called nebulae. The life cycle of a star depends on its mass. Low-mass stars like our Sun become red giants and then white dwarfs. High-mass stars can explode as supernovae and become neutron stars or black holes.",
        "The Sun": "The Sun is a G-type main-sequence star at the center of our solar system. It contains 99.86 percent of the mass in the solar system. The Sun's core temperature reaches 15 million degrees Celsius, where hydrogen atoms fuse into helium. The Sun is approximately 4.6 billion years old and is expected to continue burning for another 5 billion years.",
        "Supernovae": "A supernova is a powerful stellar explosion that occurs at the end of a massive star's life cycle. Supernovae are so bright they can outshine entire galaxies for weeks. They create and distribute heavy elements like gold, platinum, and uranium into space, seeding future generations of stars and planets.",
        "Black Holes": "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once past the event horizon. They form when massive stars collapse at the end of their lives. The first image of a black hole was captured in 2019 by the Event Horizon Telescope.",
        "Neutron Stars": "Neutron stars are the collapsed cores of massive stars that have undergone supernova explosions. They are incredibly dense, with a mass of 1.4 to 2 solar masses packed into a sphere only about 20 km in diameter. A teaspoon of neutron star material would weigh about 6 billion tons.",
        "White Dwarfs": "White dwarfs are the remnants of low and medium-mass stars after they exhaust their nuclear fuel. They are about the size of Earth but have a mass similar to the Sun. The Chandrasekhar limit of 1.4 solar masses is the maximum mass a white dwarf can have before it collapses.",
        "Nebulae": "Nebulae are giant clouds of dust and gas in space. Some nebulae are regions where new stars are being formed, while others are the remnants of dying stars. The Orion Nebula is one of the brightest nebulae visible to the naked eye.",
        "Stellar Classification": "Stars are classified by their spectral type, which relates to their surface temperature and color. The classification system goes O, B, A, F, G, K, M from hottest to coolest. Our Sun is a G-type star.",
    }

    for topic, text in stellar_topics.items():
        knowledge.append(f"# {topic}\n\n{text}")
        knowledge.append(f"Question: What is {topic.lower()}?\nAnswer: {text}")
        knowledge.append(text)

    # Cosmology topics
    cosmology_topics = {
        "The Milky Way": "The Milky Way is the galaxy that contains our solar system. It is a barred spiral galaxy with a diameter of approximately 100,000 light-years and contains between 100 billion and 400 billion stars.",
        "Dark Matter": "Dark matter is a hypothetical form of matter that does not emit or interact with electromagnetic radiation. It is estimated to make up about 27 percent of the total mass-energy content of the universe.",
        "Dark Energy": "Dark energy is a hypothetical form of energy that permeates all of space and causes the accelerating expansion of the universe. It makes up about 68 percent of the total mass-energy content of the universe.",
        "The Big Bang": "The Big Bang theory describes the origin of the universe as an extremely hot and dense state approximately 13.8 billion years ago that has been expanding ever since.",
        "Cosmic Microwave Background": "The cosmic microwave background is the thermal radiation left over from the early universe, about 380,000 years after the Big Bang. It has a temperature of approximately 2.725 Kelvin.",
        "Exoplanets": "Exoplanets are planets that orbit stars outside our solar system. The Kepler space telescope discovered over 2,600 exoplanets. The James Webb Space Telescope is now characterizing exoplanet atmospheres.",
        "Gravitational Waves": "Gravitational waves are ripples in spacetime caused by accelerating massive objects, predicted by Albert Einstein in 1916. They were first directly detected in 2015 by LIGO.",
        "General Relativity": "Einstein's theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy. It predicts the bending of light by gravity, time dilation near massive objects, and the existence of black holes.",
    }

    for topic, text in cosmology_topics.items():
        knowledge.append(f"# {topic}\n\n{text}")
        knowledge.append(f"Question: What is {topic.lower()}?\nAnswer: {text}")
        knowledge.append(text)

    # Space exploration
    exploration_topics = {
        "Apollo Program": "The Apollo program was NASA's human spaceflight program that landed the first humans on the Moon. Apollo 11 landed on July 20, 1969. Neil Armstrong was the first person to walk on the lunar surface.",
        "International Space Station": "The International Space Station is a modular space station in low Earth orbit. It has been continuously occupied since November 2000. The station orbits Earth approximately every 90 minutes.",
        "Hubble Space Telescope": "The Hubble Space Telescope was launched in 1990 and has made over 1.5 million observations. It has helped determine the age of the universe at 13.8 billion years.",
        "James Webb Space Telescope": "The James Webb Space Telescope is the largest and most powerful space telescope ever built, launched on December 25, 2021. It observes in infrared light.",
        "Voyager Program": "Voyager 1 and Voyager 2 are NASA space probes launched in 1977. Voyager 1 is the most distant human-made object, currently over 24 billion km from Earth.",
        "Mars Exploration": "The Curiosity rover landed on Mars on August 6, 2012. The Perseverance rover landed on February 18, 2021. The Ingenuity helicopter made the first powered flight on another planet.",
        "SpaceX": "SpaceX is revolutionizing space travel with reusable rockets. The Falcon 9 rocket has successfully landed and been reused over 200 times. The Starship vehicle is designed for missions to the Moon and Mars.",
    }

    for topic, text in exploration_topics.items():
        knowledge.append(f"# {topic}\n\n{text}")
        knowledge.append(f"Question: What is the {topic.lower()}?\nAnswer: {text}")
        knowledge.append(text)

    return knowledge


def prepare_data():
    base_dir = Path("/kaggle/working") if os.path.exists("/kaggle") else Path(".")
    data_dir = base_dir / "data_v2"
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
    all_texts.extend(download_wikipedia_expanded())
    all_texts.extend(download_arxiv_expanded())

    print("[Knowledge] Generating comprehensive knowledge base...")
    knowledge = generate_comprehensive_knowledge()
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
    max_vocab = min(16000, max(8000, total_chars // 50))
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
    n = len(data)
    n_seq = (n - 1) // seq_len
    n_batches = n_seq // batch_size
    if n_batches == 0:
        return np.zeros((0, batch_size, seq_len), dtype=np.int32), np.zeros((0, batch_size, seq_len), dtype=np.int32)

    trimmed = data[:n_batches * batch_size * seq_len + 1]
    inputs = []
    targets = []
    for i in range(n_batches * batch_size):
        start = i * seq_len
        chunk = trimmed[start:start + seq_len + 1]
        inputs.append(chunk[:-1])
        targets.append(chunk[1:])

    inputs = np.array(inputs, dtype=np.int32).reshape(n_batches, batch_size, seq_len)
    targets = np.array(targets, dtype=np.int32).reshape(n_batches, batch_size, seq_len)
    return inputs, targets


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate(params, apply_fn, tokenizer, prompt, max_new_tokens=150, temperature=0.7, top_k=40):
    ids = tokenizer.encode(prompt, out_type=int)
    input_ids = jnp.array([ids], dtype=jnp.int32)

    for _ in range(max_new_tokens):
        logits = apply_fn({"params": params}, input_ids, deterministic=True)
        next_logits = logits[0, -1, :] / temperature

        if top_k > 0:
            top_k_val = jnp.sort(next_logits)[-top_k]
            next_logits = jnp.where(next_logits < top_k_val, jnp.finfo(jnp.float32).min, next_logits)

        probs = jax.nn.softmax(next_logits)
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
    print(f"Backend: {jax.default_backend()}")

    # Prepare expanded data
    meta = prepare_data()
    import sentencepiece as spm
    base_dir = "/kaggle/working" if os.path.exists("/kaggle") else "."
    tokenizer = spm.SentencePieceProcessor(model_file=f"{base_dir}/data_v2/tokenizer/space_tokenizer.model")

    # Config
    train_config = TrainConfig()

    # Load or create model
    model_config = ModelConfig(vocab_size=meta["vocab_size"])
    model = SpaceLLM(model_config)

    # Try to load existing model
    resume_path = train_config.resume_from
    if os.path.exists(resume_path):
        print(f"[Resume] Loading model from {resume_path}")
        with open(resume_path, "rb") as f:
            checkpoint = pickle.load(f)
        params = checkpoint["params"]
        print(f"[Resume] Loaded model from step {checkpoint.get('step', '?')}")
    else:
        print("[New] Training from scratch")
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.zeros((1, model_config.max_seq_len), dtype=jnp.int32)
        params = model.init(rng, dummy_input, deterministic=True)["params"]

    param_count = sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {param_count:,}")

    # Optimizer
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=train_config.learning_rate,
        warmup_steps=200,
        decay_steps=train_config.max_steps,
        end_value=1e-6,
    )
    tx = optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, weight_decay=train_config.weight_decay)
    opt_state = tx.init(params)

    # Load data
    train_data = np.load(f"{base_dir}/data_v2/tokenized/train.npy")
    val_data = np.load(f"{base_dir}/data_v2/tokenized/val.npy")

    train_inputs, train_targets = make_batches(train_data, model_config.max_seq_len, train_config.batch_size)
    val_inputs, val_targets = make_batches(val_data, model_config.max_seq_len, train_config.batch_size)

    n_train_batches = len(train_inputs)
    n_val_batches = min(20, len(val_inputs))

    print(f"Train batches: {n_train_batches}, Val batches: {n_val_batches}")

    if n_train_batches == 0:
        print("[ERROR] No training batches!")
        return

    print(f"\n{'='*60}")
    print(f"Continued Training on TPU: {train_config.max_steps} steps")
    print(f"{'='*60}\n")

    # Training loop
    step = 0
    total_loss = 0.0
    best_val_loss = float("inf")
    start_time = time.time()
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    rng = jax.random.PRNGKey(42)

    while step < train_config.max_steps:
        perm = np.random.permutation(n_train_batches)

        for batch_idx in range(n_train_batches):
            if step >= train_config.max_steps:
                break

            idx = perm[batch_idx]
            input_ids = jnp.array(train_inputs[idx])
            targets = jnp.array(train_targets[idx])

            rng, step_rng = jax.random.split(rng)
            loss, grads, grad_norm = train_step_jit(params, model.apply, input_ids, targets, step_rng)

            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            total_loss += loss.item()
            step += 1

            if step % train_config.log_every == 0:
                avg = total_loss / train_config.log_every
                elapsed = time.time() - start_time
                lr = float(lr_schedule(step))
                print(f"Step {step:>6d}/{train_config.max_steps} | Loss: {avg:.4f} | LR: {lr:.2e} | Grad: {grad_norm:.2f} | Time: {elapsed:.0f}s")
                total_loss = 0.0

            if step % train_config.eval_every == 0 and n_val_batches > 0:
                val_losses = []
                for i in range(n_val_batches):
                    rng, eval_rng = jax.random.split(rng)
                    vloss = eval_jit(params, model.apply, jnp.array(val_inputs[i]), jnp.array(val_targets[i]), eval_rng)
                    val_losses.append(float(vloss))
                val_loss = np.mean(val_losses)
                ppl = math.exp(val_loss)
                print(f"  [Eval] Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    with open(f"{train_config.checkpoint_dir}/best.pkl", "wb") as f:
                        pickle.dump({"step": step, "params": jax.device_get(params), "val_loss": val_loss}, f)
                    print(f"  [Checkpoint] Saved best model")

            if step % train_config.save_every == 0:
                with open(f"{train_config.checkpoint_dir}/step_{step}.pkl", "wb") as f:
                    pickle.dump({"step": step, "params": jax.device_get(params)}, f)

    # Save final
    with open(f"{train_config.checkpoint_dir}/final.pkl", "wb") as f:
        pickle.dump({"step": step, "params": jax.device_get(params), "val_loss": best_val_loss}, f)

    with open(f"{train_config.checkpoint_dir}/model_config.json", "w") as f:
        json.dump(model_config.to_dict(), f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/3600:.1f}h | Best val loss: {best_val_loss:.4f} | Perplexity: {math.exp(best_val_loss):.2f}")
    print(f"{'='*60}")

    # Generate samples
    print("\n--- Sample Generations ---")
    for prompt in ["Question: What is a black hole?\nAnswer:", "Question: How far is Mars?\nAnswer:", "The Sun is a star that"]:
        print(f"\nQ: {prompt}")
        print(f"A: {generate(params, model.apply, tokenizer, prompt)}")


if __name__ == "__main__":
    train()
