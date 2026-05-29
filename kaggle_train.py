"""
Space LLM - Self-Contained Kaggle Training Script
Trains a 10M parameter decoder-only transformer on space data.
GPU: P100 (16GB VRAM)
"""

import os
import re
import json
import time
import math
import hashlib
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    activation: str = "swiglu"
    use_rope: bool = True
    tie_weights: bool = True
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    batch_size: int = 16
    grad_accum_steps: int = 8
    max_steps: int = 30000
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    fp16: bool = True
    save_every: int = 5000
    eval_every: int = 1000
    log_every: int = 100
    checkpoint_dir: str = "/kaggle/working/checkpoints"
    data_dir: str = "/kaggle/working/data/tokenized"


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, offset=0):
        seq_len = x.shape[-2] + offset
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return (
            self.cos_cached[offset:offset + x.shape[-2]].unsqueeze(0).unsqueeze(0),
            self.sin_cached[offset:offset + x.shape[-2]].unsqueeze(0).unsqueeze(0),
        )


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.rope = RoPE(config.head_dim, config.max_seq_len) if config.use_rope else None

    def forward(self, x, mask=None, kv_cache=None, cache_offset=0):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        if self.rope:
            cos, sin = self.rope(q, offset=cache_offset)
            q, k = apply_rope(q, k, cos, sin)
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask[:, :, :T, :k.shape[2]] == 0, float("-inf"))
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = self.resid_drop(self.out_proj(torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)))
        return out, (k, v)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.ff = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x, mask=None, kv_cache=None, cache_offset=0):
        residual = x
        x_norm = self.ln1(x)
        attn_out, new_cache = self.attn(x_norm, mask=mask, kv_cache=kv_cache, cache_offset=cache_offset)
        x = residual + attn_out
        x = x + self.ff(self.ln2(x))
        return x, new_cache


class SpaceLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None, kv_caches=None, cache_offset=0):
        B, T = input_ids.shape
        device = input_ids.device
        if kv_caches is None:
            mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        else:
            total_len = cache_offset + T
            mask = torch.ones(1, 1, T, total_len, device=device)
            mask = torch.tril(mask, diagonal=total_len - T)
        x = self.drop(self.tok_emb(input_ids))
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = layer(x, mask=mask, kv_cache=cache, cache_offset=cache_offset)
            new_caches.append(new_cache)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss, new_caches

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.9):
        self.eval()
        generated = input_ids.clone()
        kv_caches = None
        for _ in range(max_new_tokens):
            if kv_caches is None:
                logits, _, kv_caches = self(generated)
            else:
                logits, _, new_caches = self(generated[:, -1:], kv_caches=kv_caches, cache_offset=generated.shape[1] - 1)
                kv_caches = new_caches
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


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
        "Mercury": "Mercury is the smallest planet in our solar system and closest to the Sun. It has no atmosphere and experiences extreme temperature variations from minus 180 degrees Celsius at night to 430 degrees Celsius during the day. A year on Mercury lasts only 88 Earth days. Mercury has a large iron core that makes up about 75 percent of its radius. The MESSENGER spacecraft orbited Mercury from 2011 to 2015, revealing a world covered in impact craters and ancient lava flows. Mercury has a very thin exosphere composed of atoms blasted off its surface by solar radiation and micrometeoroid impacts.",
        "Venus": "Venus is the second planet from the Sun and is often called Earth twin due to similar size and mass. However, Venus has a thick toxic atmosphere composed mainly of carbon dioxide with clouds of sulfuric acid. The surface temperature reaches 465 degrees Celsius, making it the hottest planet in the solar system. Venus rotates backwards compared to most planets, and a day on Venus is longer than its year. The atmospheric pressure on Venus is about 90 times that of Earth. Venus has no moons and no magnetic field.",
        "Earth": "Earth is the third planet from the Sun and the only known planet to support life. It has one natural satellite, the Moon. Earth atmosphere is composed of 78 percent nitrogen and 21 percent oxygen. The planet has a magnetic field that protects it from solar radiation. About 71 percent of Earth surface is covered by water. Earth axial tilt of 23.5 degrees causes seasons. The planet formed approximately 4.5 billion years ago. Earth core is divided into a solid inner core and a liquid outer core.",
        "Mars": "Mars is the fourth planet from the Sun, known as the Red Planet due to iron oxide on its surface. Mars has the largest volcano in the solar system, Olympus Mons, standing 21.9 km high, and the deepest canyon, Valles Marineris, stretching 4000 km. Mars has two small moons: Phobos and Deimos. The Curiosity and Perseverance rovers are currently exploring Mars. Evidence suggests Mars once had liquid water on its surface. Mars has a thin atmosphere composed mainly of carbon dioxide.",
        "Jupiter": "Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined. It is a gas giant composed mainly of hydrogen and helium. Jupiter Great Red Spot is a storm larger than Earth that has been raging for at least 350 years. Jupiter has at least 95 known moons, including the four large Galilean moons: Io, Europa, Ganymede, and Callisto. Europa is considered one of the most likely places to find extraterrestrial life.",
        "Saturn": "Saturn is the sixth planet from the Sun, famous for its spectacular ring system made of ice and rock particles. Saturn is a gas giant composed mainly of hydrogen and helium. It has at least 146 known moons, including Titan, which has a thick atmosphere and liquid methane lakes. Saturn density is so low that it would float in water. The Cassini spacecraft studied Saturn and its moons for 13 years from 2004 to 2017.",
        "Uranus": "Uranus is the seventh planet from the Sun and the first discovered using a telescope, found by William Herschel in 1781. It is an ice giant composed mainly of water, methane, and ammonia ices. Uranus rotates on its side with an axial tilt of 98 degrees, likely caused by a massive collision early in its history. Uranus has 27 known moons and a faint ring system. The atmosphere appears blue-green due to methane absorption.",
        "Neptune": "Neptune is the eighth and farthest planet from the Sun. It is an ice giant with the strongest winds in the solar system, reaching speeds of 2100 km per hour. Neptune has 16 known moons, the largest being Triton, which orbits in the opposite direction to Neptune rotation, suggesting it was captured from the Kuiper Belt. Neptune was visited only once by Voyager 2 in 1989. Its blue color comes from methane in its atmosphere.",
    }
    for planet, desc in planets.items():
        knowledge.append(f"# {planet}\n\n{desc}")
        knowledge.append(f"Question: Tell me about {planet}.\nAnswer: {desc}")
        knowledge.append(f"What is {planet}? {desc}")
        knowledge.append(f"Describe the planet {planet}. {desc}")
        knowledge.append(f"{planet} is a planet in our solar system. {desc}")

    stellar = [
        "Stars are massive celestial bodies that produce light and heat through nuclear fusion in their cores. They form from clouds of gas and dust called nebulae. The life cycle of a star depends on its mass. Low-mass stars like our Sun become red giants and then white dwarfs. High-mass stars can explode as supernovae and become neutron stars or black holes. Stars are classified by their spectral type, which relates to their surface temperature and color.",
        "The Sun is a G-type main-sequence star at the center of our solar system. It contains 99.86 percent of the mass in the solar system. The Sun core temperature reaches 15 million degrees Celsius, where hydrogen atoms fuse into helium. The Sun is approximately 4.6 billion years old and is expected to continue burning for another 5 billion years. The Sun surface temperature is about 5500 degrees Celsius.",
        "A supernova is a powerful stellar explosion that occurs at the end of a massive star life cycle. There are two main types: Type Ia thermonuclear and Type II core-collapse supernovae. Supernovae are so bright they can outshine entire galaxies for weeks. They create and distribute heavy elements like gold, platinum, and uranium into space. The Crab Nebula is the remnant of a supernova observed in 1054 AD.",
        "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once past the event horizon. They form when massive stars collapse at the end of their lives. The first image of a black hole was captured in 2019 by the Event Horizon Telescope, showing the supermassive black hole in galaxy M87. Sagittarius A star is the supermassive black hole at the center of our Milky Way galaxy.",
        "Neutron stars are the collapsed cores of massive stars that have undergone supernova explosions. They are incredibly dense, with a mass of 1.4 to 2 solar masses packed into a sphere only about 20 km in diameter. A teaspoon of neutron star material would weigh about 6 billion tons. Pulsars are rapidly rotating neutron stars that emit beams of electromagnetic radiation.",
        "White dwarfs are the remnants of low and medium-mass stars after they exhaust their nuclear fuel. They are about the size of Earth but have a mass similar to the Sun. White dwarfs gradually cool and fade over billions of years. The Chandrasekhar limit of 1.4 solar masses is the maximum mass a white dwarf can have before it collapses.",
        "The Hertzsprung-Russell diagram is a scatter graph of stars showing the relationship between their absolute magnitudes or luminosities versus their spectral types or temperatures. It was developed independently by Ejnar Hertzsprung and Henry Norris Russell in the early 20th century. The main sequence runs diagonally from hot luminous stars to cool dim ones.",
        "Nebulae are giant clouds of dust and gas in space. Some nebulae are regions where new stars are being formed, while others are the remnants of dying stars. The Orion Nebula is one of the brightest nebulae visible to the naked eye. The Crab Nebula is a supernova remnant. Planetary nebulae are shells of gas ejected by dying low-mass stars.",
    ]
    for text in stellar:
        knowledge.append(text)
        knowledge.append(f"Question: {text.split('.')[0]}?\nAnswer: {text}")

    cosmology = [
        "The Milky Way is the galaxy that contains our solar system. It is a barred spiral galaxy with a diameter of approximately 100,000 light-years and contains between 100 billion and 400 billion stars. The supermassive black hole at its center, Sagittarius A star, has a mass of about 4 million times that of our Sun. The Milky Way is part of the Local Group of galaxies.",
        "The Andromeda Galaxy is the nearest large galaxy to the Milky Way, located about 2.5 million light-years away. It is approaching the Milky Way at about 110 kilometers per second and the two galaxies are expected to collide in about 4.5 billion years, forming a single elliptical galaxy sometimes called Milkdromeda.",
        "Dark matter is a hypothetical form of matter that does not emit or interact with electromagnetic radiation. It is estimated to make up about 27 percent of the total mass-energy content of the universe. Evidence for dark matter comes from gravitational effects on visible matter, such as the rotation curves of galaxies and gravitational lensing.",
        "Dark energy is a hypothetical form of energy that permeates all of space and causes the accelerating expansion of the universe. It is estimated to make up about 68 percent of the total mass-energy content of the universe. The discovery of the accelerating expansion in 1998 earned Saul Perlmutter, Brian Schmidt, and Adam Riess the Nobel Prize in Physics in 2011.",
        "The Big Bang theory describes the origin of the universe as an extremely hot and dense state approximately 13.8 billion years ago that has been expanding ever since. Key evidence includes the cosmic microwave background radiation, the abundance of light elements, and the redshift of distant galaxies.",
        "The cosmic microwave background is the thermal radiation left over from the early universe, about 380,000 years after the Big Bang. It has a temperature of approximately 2.725 Kelvin. The CMB provides a snapshot of the universe when it was very young and has tiny fluctuations that seeded the formation of galaxies.",
        "Exoplanets are planets that orbit stars outside our solar system. The first confirmed exoplanet discovery was in 1992 around a pulsar. The Kepler space telescope discovered over 2,600 exoplanets during its mission. The James Webb Space Telescope is now characterizing exoplanet atmospheres.",
        "Gravitational waves are ripples in spacetime caused by accelerating massive objects, predicted by Albert Einstein in 1916. They were first directly detected in 2015 by LIGO from the merger of two black holes about 1.3 billion light-years away.",
        "Einstein theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy. It predicts phenomena such as the bending of light by gravity, time dilation near massive objects, and the existence of black holes.",
        "The observable universe is approximately 93 billion light-years in diameter. It contains an estimated 2 trillion galaxies and more stars than grains of sand on Earth. The universe is expanding, with distant galaxies moving away from us faster than nearby ones.",
    ]
    for text in cosmology:
        knowledge.append(text)
        knowledge.append(f"Question: {text.split('.')[0]}?\nAnswer: {text}")

    exploration = [
        "The Apollo program was NASA human spaceflight program that landed the first humans on the Moon. Apollo 11, commanded by Neil Armstrong with pilot Buzz Aldrin, landed on the Moon on July 20, 1969. Armstrong was the first person to walk on the lunar surface. In total, 12 astronauts walked on the Moon during six Apollo missions between 1969 and 1972.",
        "The International Space Station is a modular space station in low Earth orbit. It is the largest artificial object in space and can often be seen with the naked eye. The ISS has been continuously occupied since November 2000. The station orbits Earth approximately every 90 minutes at an altitude of about 408 km.",
        "The Hubble Space Telescope was launched in 1990 and has made over 1.5 million observations. It orbits Earth at an altitude of about 547 km. Hubble has helped determine the age of the universe at 13.8 billion years, discovered that most galaxies have supermassive black holes, and provided evidence for the accelerating expansion of the universe.",
        "The James Webb Space Telescope is the largest and most powerful space telescope ever built, launched on December 25, 2021. It orbits the Sun at the second Lagrange point. JWST observes in infrared light and can see the earliest galaxies formed after the Big Bang. It has a 6.5-meter primary mirror made of 18 hexagonal segments.",
        "Voyager 1 and Voyager 2 are NASA space probes launched in 1977 to study the outer planets. Voyager 1 is the most distant human-made object, currently over 24 billion km from Earth. Both spacecraft carry a Golden Record containing sounds and images of Earth. Voyager 1 entered interstellar space in 2012.",
        "The Curiosity rover landed on Mars on August 6, 2012, in Gale Crater. It has discovered evidence that Mars once had conditions suitable for microbial life. The Perseverance rover landed on February 18, 2021, in Jezero Crater, and is collecting samples for future return to Earth.",
        "SpaceX is revolutionizing space travel with reusable rockets. The Falcon 9 rocket has successfully landed and been reused over 200 times. The Starship vehicle is designed for missions to the Moon and Mars. SpaceX Crew Dragon regularly transports astronauts to the ISS.",
        "The search for extraterrestrial intelligence uses radio telescopes to listen for signals from advanced civilizations. The Drake equation estimates the number of active communicative extraterrestrial civilizations in the Milky Way.",
    ]
    for text in exploration:
        knowledge.append(text)
        knowledge.append(f"Question: {text.split('.')[0]}?\nAnswer: {text}")

    return knowledge


def prepare_data():
    data_dir = Path("/kaggle/working/data")
    tokenized_dir = data_dir / "tokenized"
    tokenizer_dir = data_dir / "tokenizer"
    for d in [tokenized_dir, tokenizer_dir]:
        d.mkdir(parents=True, exist_ok=True)

    meta_path = tokenized_dir / "meta.json"
    if meta_path.exists():
        print("[Skip] Data already prepared.")
        return

    all_texts = []
    all_texts.extend(download_wikipedia())
    all_texts.extend(download_arxiv())

    print("[Knowledge] Generating space knowledge base...")
    knowledge = generate_space_knowledge()
    all_texts.extend(knowledge)
    print(f"[Knowledge] Generated {len(knowledge)} entries")

    if not all_texts:
        print("[ERROR] No data!")
        return

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

    model_prefix = str(tokenizer_dir / "space_tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(corpus_file), model_prefix=model_prefix,
        vocab_size=32000, model_type="bpe", character_coverage=0.9995,
        num_threads=4, split_digits=True, byte_fallback=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    )

    # Tokenize
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    all_tokens = []
    for t in tqdm(unique, desc="[Tokenize]"):
        tokens = sp.encode(t, out_type=int)
        all_tokens.extend([sp.bos_id()] + tokens + [sp.eos_id()])

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    val_size = int(len(all_tokens) * 0.05)
    train_tokens = all_tokens[:-val_size]
    val_tokens = all_tokens[-val_size:]

    np.memmap(tokenized_dir / "train.bin", dtype=np.uint16, mode="w+", shape=len(train_tokens))[:] = train_tokens
    np.memmap(tokenized_dir / "val.bin", dtype=np.uint16, mode="w+", shape=len(val_tokens))[:] = val_tokens

    meta = {"vocab_size": sp.get_piece_size(), "train_tokens": len(train_tokens), "val_tokens": len(val_tokens)}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] Train: {len(train_tokens):,}, Val: {len(val_tokens):,} tokens")


class TokenDataset(Dataset):
    def __init__(self, data_path, seq_len, split="train"):
        with open(Path(data_path) / "meta.json") as f:
            meta = json.load(f)
        fname = f"{split}.bin"
        n_tokens = meta[f"{split}_tokens"]
        self.data = np.memmap(Path(data_path) / fname, dtype=np.uint16, mode="r", shape=n_tokens)
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.data) - self.seq_len - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def get_lr(step, warmup, max_steps, lr, min_lr):
    if step < warmup:
        return lr * step / warmup
    if step > max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, val_loader, device, num_batches=20):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= num_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train():
    model_config = ModelConfig()
    train_config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        print(f"Memory: {mem / 1e9:.1f} GB")

    prepare_data()

    model = SpaceLLM(model_config).to(device)
    print(f"Parameters: {model.get_num_params():,}")

    data_dir = "/kaggle/working/data/tokenized"
    train_loader = DataLoader(TokenDataset(data_dir, model_config.max_seq_len, "train"),
                              batch_size=train_config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(TokenDataset(data_dir, model_config.max_seq_len, "val"),
                            batch_size=train_config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate,
                                   weight_decay=train_config.weight_decay, betas=(0.9, 0.95))
    use_fp16 = train_config.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    autocast_ctx = torch.amp.autocast("cuda", enabled=use_fp16)

    model.train()
    step = 0
    total_loss = 0.0
    best_val_loss = float("inf")
    start_time = time.time()
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: {train_config.max_steps} steps, effective batch {train_config.batch_size * train_config.grad_accum_steps}")
    print(f"{'='*60}\n")

    while step < train_config.max_steps:
        for batch_idx, (x, y) in enumerate(train_loader):
            if step >= train_config.max_steps:
                break
            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                _, loss, _ = model(x, targets=y)
                loss = loss / train_config.grad_accum_steps
            scaler.scale(loss).backward()
            total_loss += loss.item() * train_config.grad_accum_steps

            if (batch_idx + 1) % train_config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
                lr = get_lr(step, train_config.warmup_steps, train_config.max_steps,
                           train_config.learning_rate, train_config.min_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if step % train_config.log_every == 0:
                    avg = total_loss / train_config.log_every
                    print(f"Step {step:>6d}/{train_config.max_steps} | Loss: {avg:.4f} | LR: {lr:.2e} | Time: {time.time()-start_time:.0f}s")
                    total_loss = 0.0

                if step % train_config.eval_every == 0:
                    val_loss = estimate_loss(model, val_loader, device)
                    print(f"  [Eval] Val Loss: {val_loss:.4f} | Perplexity: {math.exp(val_loss):.2f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({"step": step, "model_state_dict": model.state_dict(), "val_loss": val_loss},
                                   f"{train_config.checkpoint_dir}/best.pt")

                if step % train_config.save_every == 0:
                    torch.save({"step": step, "model_state_dict": model.state_dict()},
                               f"{train_config.checkpoint_dir}/step_{step}.pt")

    torch.save({"step": step, "model_state_dict": model.state_dict(), "val_loss": best_val_loss},
               f"{train_config.checkpoint_dir}/final.pt")

    with open(f"{train_config.checkpoint_dir}/model_config.json", "w") as f:
        json.dump({"vocab_size": 32000, "d_model": 256, "n_heads": 8, "n_layers": 6,
                    "d_ff": 1024, "max_seq_len": 512, "dropout": 0.1, "activation": "swiglu",
                    "use_rope": True, "tie_weights": True, "norm_eps": 1e-5}, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/3600:.1f}h | Best val loss: {best_val_loss:.4f} | Perplexity: {math.exp(best_val_loss):.2f}")
    print(f"{'='*60}")

    # Generate samples
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file="/kaggle/working/data/tokenizer/space_tokenizer.model")
    model.eval()
    for prompt in ["Question: What is a black hole?\nAnswer:", "Question: How far is Mars from Earth?\nAnswer:", "The Sun is a star that"]:
        ids = sp.encode(prompt, out_type=int)
        out = model.generate(torch.tensor([ids], dtype=torch.long, device=device), max_new_tokens=150, temperature=0.7, top_k=40)
        print(f"\nQ: {prompt}\nA: {sp.decode(out[0].cpu().tolist())}")


if __name__ == "__main__":
    train()
