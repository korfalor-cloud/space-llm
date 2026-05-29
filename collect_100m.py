"""
Space LLM - 100M Token Data Collection Pipeline
NO datasets library — uses direct APIs only to avoid numpy/pyarrow conflicts.
Target: 100 million tokens of high-quality space content.

Sources:
1. Wikipedia API (filtered for space/astronomy) ~50M tokens
2. arXiv API (astro-ph, gr-qc, hep-ph abstracts + HF parquet) ~30M tokens
3. NASA APIs (APOD, Image Library, NSSDCA) ~5M tokens
4. HuggingFace Hub parquet downloads (physics, astronomy) ~10M tokens
5. Space Q&A generated knowledge base ~5M tokens
"""

import os
import re
import json
import time
import hashlib
import subprocess
import sys
import io
import gzip
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Only install sentencepiece — NO datasets, NO pyarrow
def setup():
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "sentencepiece", "tqdm", "requests", "-q"])

setup()
import sentencepiece as spm

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_TOKENS = 100_000_000
BASE_DIR = Path("/kaggle/working") if os.path.exists("/kaggle") else Path(".")
OUTPUT_DIR = BASE_DIR / "space_corpus"
RAW_DIR = OUTPUT_DIR / "raw"
CLEAN_DIR = OUTPUT_DIR / "clean"
TOKENIZED_DIR = OUTPUT_DIR / "tokenized"

for d in [RAW_DIR, CLEAN_DIR, TOKENIZED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SPACE_KEYWORDS_HIGH = [
    "astronomy", "astrophysics", "cosmology", "telescope", "observatory",
    "black hole", "neutron star", "pulsar", "quasar", "supernova",
    "galaxy", "galactic", "nebula", "stellar", "planetary",
    "exoplanet", "habitable zone", "dark matter", "dark energy",
    "big bang", "cosmic microwave", "gravitational wave",
    "hubble", "james webb", "chandra", "spitzer",
    "nasa", "esa", "jaxa", "spacex",
    "astronaut", "cosmonaut", "spacecraft",
    "apollo", "voyager", "cassini", "curiosity", "perseverance",
    "international space station", "space station",
]

SPACE_KEYWORDS_MED = [
    "planet", "star", "moon", "asteroid", "comet", "meteor",
    "orbit", "solar", "lunar", "mars", "jupiter", "saturn",
    "venus", "mercury", "neptune", "uranus", "pluto",
    "rocket", "launch", "satellite", "probe", "rover",
    "eclipse", "constellation", "light year", "parsec", "redshift",
    "fusion", "relativity", "spacetime", "quantum",
    "gravity", "magnetic field", "radiation", "cosmic",
    "milky way", "andromeda", "universe", "expansion",
    "spectroscop", "magnitude", "luminosity",
]


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text):
    text = re.sub(r'\$[^$]+\$', ' ', text)
    text = re.sub(r'\$\$[^$]+\$\$', ' ', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r"'''?([^']+)'''?", r'\1', text)
    text = re.sub(r'={2,}\s*(.*?)\s*={2,}', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[http[^\]]*\]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +\n', '\n', text)
    lines = [l for l in text.split('\n') if len(l.strip()) > 10 or l.strip() == '']
    return '\n'.join(lines).strip()


def is_space_related(text, threshold="high"):
    text_lower = text.lower()
    kws = SPACE_KEYWORDS_HIGH if threshold == "high" else SPACE_KEYWORDS_MED
    matches = sum(1 for kw in kws if kw in text_lower)
    return matches >= (2 if threshold == "high" else 1)


def quality_filter(text):
    if len(text) < 100:
        return False
    words = text.split()
    if len(words) < 20:
        return False
    special_ratio = sum(1 for c in text if not c.isalnum() and c not in ' \n.,;:!?-') / len(text)
    return special_ratio < 0.3


def save_source(name, texts):
    output_file = CLEAN_DIR / f"{name}.txt"
    total_chars = sum(len(t) for t in texts)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(texts))
    est_tokens = total_chars // 4
    print(f"  [{name}] Saved {len(texts):,} items | {total_chars:,} chars | ~{est_tokens:,} tokens")
    return total_chars


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: WIKIPEDIA API (Target: ~50M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_wikipedia():
    print("\n" + "="*60)
    print("SOURCE 1: WIKIPEDIA API")
    print("="*60)

    output_file = CLEAN_DIR / "wikipedia.txt"
    if output_file.exists() and output_file.stat().st_size > 1_000_000:
        print(f"[Skip] Already collected: {output_file.stat().st_size:,} bytes")
        return

    session = requests.Session()
    session.headers.update({"User-Agent": "SpaceLLM/2.0 (educational project)"})

    # Comprehensive search terms
    search_terms = [
        # Solar system
        "Mercury planet", "Venus planet", "Earth planet", "Mars planet",
        "Jupiter planet", "Saturn planet", "Uranus planet", "Neptune planet",
        "Pluto dwarf planet", "Ceres dwarf planet", "Eris dwarf planet",
        "asteroid belt", "Kuiper belt", "Oort cloud", "heliosphere",
        "solar wind", "solar flare", "sunspot", "solar cycle",
        "planetary ring", "planetary atmosphere", "planetary science",
        "Moon", "Europa moon", "Ganymede moon", "Titan moon", "Enceladus moon",
        "Io moon", "Callisto moon", "Triton moon",

        # Stars
        "main sequence star", "red giant", "white dwarf", "neutron star",
        "pulsar", "magnetar", "brown dwarf", "red dwarf", "blue giant",
        "supergiant", "stellar evolution", "Hertzsprung-Russell diagram",
        "nuclear fusion", "proton-proton chain", "supernova",
        "supernova remnant", "nebula", "planetary nebula",
        "emission nebula", "dark nebula", "HII region",
        "stellar classification", "spectral class", "absolute magnitude",
        "parallax", "standard candle", "Cepheid variable",

        # Black holes
        "black hole", "event horizon", "singularity", "Hawking radiation",
        "accretion disk", "relativistic jet", "supermassive black hole",

        # Galaxies
        "Milky Way", "Andromeda galaxy", "galaxy cluster", "galaxy formation",
        "spiral galaxy", "elliptical galaxy", "dwarf galaxy",
        "active galactic nucleus", "quasar", "blazar",

        # Cosmology
        "Big Bang", "cosmic microwave background", "cosmic inflation",
        "dark matter", "dark energy", "cosmological constant",
        "Hubble constant", "redshift", "cosmic distance ladder",
        "baryon acoustic oscillation", "cosmic web", "observable universe",

        # Relativity
        "general relativity", "special relativity", "spacetime",
        "gravitational wave", "LIGO", "gravitational lensing",
        "frame dragging", "gravitational time dilation",

        # Telescopes
        "Hubble Space Telescope", "James Webb Space Telescope",
        "Chandra X-ray Observatory", "Spitzer Space Telescope",
        "Kepler space telescope", "TESS telescope",
        "Atacama Large Millimeter Array", "Very Large Array",
        "Event Horizon Telescope",

        # Space exploration
        "Apollo program", "Space Shuttle", "International Space Station",
        "SpaceX", "Falcon 9", "Starship", "Blue Origin",
        "Voyager program", "New Horizons", "Cassini-Huygens",
        "Mars rover", "Curiosity rover", "Perseverance rover",
        "Artemis program", "Mars colonization", "space colonization",

        # Astrobiology
        "astrobiology", "extraterrestrial life", "habitable zone",
        "Drake equation", "Fermi paradox",

        # Physics
        "quantum mechanics", "quantum field theory", "particle physics",
        "standard model", "Higgs boson", "string theory",
        "thermodynamics", "electromagnetism", "nuclear physics",
    ]

    articles = []
    total_chars = 0
    seen_ids = set()

    for term in tqdm(search_terms, desc="[Wikipedia]"):
        try:
            # Search for articles
            r = session.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "list": "search", "srsearch": term,
                "srlimit": 50, "format": "json",
            }, timeout=15)
            if r.status_code != 200:
                continue

            results = r.json().get("query", {}).get("search", [])
            for result in results[:30]:
                page_id = result.get("pageid")
                if page_id in seen_ids:
                    continue
                seen_ids.add(page_id)

                # Fetch full article
                pr = session.get("https://en.wikipedia.org/w/api.php", params={
                    "action": "query", "pageids": page_id,
                    "prop": "extracts", "explaintext": True, "format": "json",
                }, timeout=15)
                if pr.status_code != 200:
                    continue

                for page in pr.json().get("query", {}).get("pages", {}).values():
                    title = page.get("title", "")
                    extract = page.get("extract", "")
                    if len(extract) > 300:
                        cleaned = clean_text(extract)
                        if quality_filter(cleaned):
                            formatted = f"# {title}\n\n{cleaned}"
                            articles.append(formatted)
                            total_chars += len(formatted)

            time.sleep(0.1)

            # Progress
            if len(articles) % 500 == 0 and len(articles) > 0:
                print(f"  {len(articles):,} articles | {total_chars:,} chars | ~{total_chars//4:,} tokens")

            # Stop at ~200M chars = ~50M tokens
            if total_chars > 200_000_000:
                break

        except Exception as e:
            continue

    save_source("wikipedia", articles)


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: ARXIV API (Target: ~30M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_arxiv():
    print("\n" + "="*60)
    print("SOURCE 2: ARXIV API")
    print("="*60)

    output_file = CLEAN_DIR / "arxiv.txt"
    if output_file.exists() and output_file.stat().st_size > 1_000_000:
        print(f"[Skip] Already collected: {output_file.stat().st_size:,} bytes")
        return

    categories = [
        "astro-ph", "astro-ph.GA", "astro-ph.CO", "astro-ph.EP",
        "astro-ph.HE", "astro-ph.IM", "astro-ph.SR",
        "gr-qc", "hep-ph", "hep-th", "physics.space-ph",
        "physics.gen-ph", "physics.ed-ph", "physics.pop-ph",
    ]

    papers = []
    total_chars = 0

    for cat in categories:
        print(f"  Category: {cat}")
        start = 0
        while start < 5000:
            try:
                r = requests.get("http://export.arxiv.org/api/query", params={
                    "search_query": f"cat:{cat}", "start": start,
                    "max_results": 100, "sortBy": "submittedDate", "sortOrder": "descending",
                }, timeout=60)
                if r.status_code != 200:
                    break

                entries = r.text.split("<entry>")[1:]
                if not entries:
                    break

                for entry in entries:
                    title_m = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                    abstract_m = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
                    if title_m and abstract_m:
                        title = clean_text(title_m.group(1).strip())
                        abstract = clean_text(abstract_m.group(1).strip())
                        if len(abstract) > 50:
                            formatted = f"# {title}\n\n{abstract}"
                            papers.append(formatted)
                            total_chars += len(formatted)

                start += 100
                time.sleep(3)

            except Exception as e:
                print(f"    Error at {start}: {e}")
                time.sleep(10)
                break

        if total_chars > 120_000_000:  # ~30M tokens
            break

    save_source("arxiv", papers)


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: NASA APIs (Target: ~5M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_nasa():
    print("\n" + "="*60)
    print("SOURCE 3: NASA APIs")
    print("="*60)

    output_file = CLEAN_DIR / "nasa.txt"
    if output_file.exists() and output_file.stat().st_size > 100_000:
        print(f"[Skip] Already collected: {output_file.stat().st_size:,} bytes")
        return

    texts = []
    session = requests.Session()

    # APOD
    print("  [NASA] APOD...")
    try:
        for i in range(10):
            r = session.get("https://api.nasa.gov/planetary/apod",
                          params={"api_key": "DEMO_KEY", "count": 100, "thumbs": True}, timeout=30)
            if r.status_code == 200:
                for item in r.json():
                    title = item.get("title", "")
                    explanation = item.get("explanation", "")
                    if explanation and len(explanation) > 50:
                        texts.append(f"# {title}\n\n{explanation}")
            time.sleep(1)
    except Exception as e:
        print(f"    APOD error: {e}")

    # NASA Image Library
    print("  [NASA] Image Library...")
    try:
        for topic in ["astronomy", "space", "galaxy", "nebula", "planet", "star", "rocket", "satellite",
                       "earth", "moon", "mars", "jupiter", "saturn", "hubble", "telescope"]:
            r = session.get("https://images-api.nasa.gov/search",
                          params={"q": topic, "media_type": "image", "page_size": 100}, timeout=30)
            if r.status_code == 200:
                items = r.json().get("collection", {}).get("items", [])
                for item in items:
                    data = item.get("data", [{}])[0]
                    title = data.get("title", "")
                    desc = data.get("description", "")
                    if desc and len(desc) > 50:
                        texts.append(f"# {title}\n\n{clean_text(desc)}")
            time.sleep(0.5)
    except Exception as e:
        print(f"    Image Library error: {e}")

    # NSSDCA Planetary Fact Sheets
    print("  [NASA] NSSDCA...")
    try:
        for planet in ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]:
            r = session.get(f"https://nssdc.gsfc.nasa.gov/planetary/factsheet/{planet}fact.html", timeout=15)
            if r.status_code == 200:
                text = re.sub(r'<[^>]+>', ' ', r.text)
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 100:
                    texts.append(f"# {planet.title()} Planetary Fact Sheet\n\n{text}")
    except Exception as e:
        print(f"    NSSDCA error: {e}")

    # NASA APOD archive (older entries)
    print("  [NASA] APOD Archive...")
    try:
        for year in range(2015, 2025):
            for month in range(1, 13):
                day = 1
                try:
                    r = session.get("https://api.nasa.gov/planetary/apod",
                                  params={"api_key": "DEMO_KEY",
                                          "date": f"{year}-{month:02d}-{day:02d}",
                                          "thumbs": True}, timeout=15)
                    if r.status_code == 200:
                        item = r.json()
                        title = item.get("title", "")
                        explanation = item.get("explanation", "")
                        if explanation and len(explanation) > 50:
                            texts.append(f"# {title}\n\n{explanation}")
                    time.sleep(0.5)
                except:
                    pass
    except Exception as e:
        print(f"    APOD Archive error: {e}")

    save_source("nasa", texts)


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 4: HuggingFace Parquet Downloads (Target: ~10M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_huggingface_parquet():
    """Download datasets from HuggingFace using direct parquet file downloads."""
    print("\n" + "="*60)
    print("SOURCE 4: HuggingFace Parquet Downloads")
    print("="*60)

    output_file = CLEAN_DIR / "huggingface.txt"
    if output_file.exists() and output_file.stat().st_size > 1_000_000:
        print(f"[Skip] Already collected: {output_file.stat().st_size:,} bytes")
        return

    all_texts = []

    # Download parquet files directly from HuggingFace
    hf_datasets = [
        ("camel-ai/physics", "default/train", "message_1", "message_2"),
        ("mlfoundations-dev/stackexchange_astronomy", "default/train", "instruction", "completion"),
    ]

    for ds_name, config_split, q_key, a_key in hf_datasets:
        print(f"  Downloading {ds_name}...")
        try:
            # Get parquet file URLs
            api_url = f"https://huggingface.co/api/datasets/{ds_name}/parquet/{config_split}"
            r = requests.get(api_url, timeout=30)
            if r.status_code == 200:
                parquet_urls = r.json() if isinstance(r.json(), list) else [r.json()]
                for pq_info in parquet_urls[:3]:  # Limit files
                    pq_url = pq_info if isinstance(pq_info, str) else pq_info.get("url", "")
                    if pq_url:
                        try:
                            pr = requests.get(pq_url, timeout=60)
                            if pr.status_code == 200:
                                import pyarrow.parquet as pq
                                table = pq.read_table(io.BytesIO(pr.content))
                                df = table.to_pandas()
                                for _, row in df.iterrows():
                                    q = str(row.get(q_key, ""))
                                    a = str(row.get(a_key, ""))
                                    if q and a and len(q) > 10 and len(a) > 10:
                                        all_texts.append(f"Question: {q}\nAnswer: {a}")
                        except Exception as e:
                            print(f"    Error: {e}")
        except Exception as e:
            print(f"  {ds_name} error: {e}")

    # If parquet download fails, use API fallback
    if len(all_texts) < 100:
        print("  Parquet download failed, using knowledge base fallback...")
        all_texts.extend(generate_comprehensive_knowledge())

    save_source("huggingface", all_texts)


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 5: COMPREHENSIVE KNOWLEDGE BASE (Target: ~5M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_comprehensive_knowledge():
    """Generate massive space knowledge base."""
    knowledge = []

    # Detailed planet data
    planets = {
        "Mercury": [
            "Mercury is the smallest planet in our solar system and closest to the Sun at 57.9 million kilometers. It has no atmosphere and experiences extreme temperature variations from minus 180 degrees Celsius at night to 430 degrees Celsius during the day.",
            "A year on Mercury lasts only 88 Earth days, but a single Mercury day lasts 59 Earth days due to its slow rotation. Mercury has a large iron core that makes up about 75 percent of its radius, making it the second densest planet after Earth.",
            "The MESSENGER spacecraft orbited Mercury from 2011 to 2015, mapping its surface in detail. Mariner 10 flew by Mercury three times in 1974-1975. Mercury has a very thin exosphere and no magnetic field to speak of.",
        ],
        "Venus": [
            "Venus is the second planet from the Sun and is often called Earth's twin due to similar size and mass. However, Venus has a thick toxic atmosphere composed mainly of carbon dioxide with clouds of sulfuric acid.",
            "The surface temperature on Venus reaches 465 degrees Celsius, making it the hottest planet in the solar system. The atmospheric pressure is about 90 times that of Earth. Venus rotates backwards compared to most planets, and a day on Venus is longer than its year.",
            "The Soviet Venera program successfully landed several spacecraft on Venus's surface. Magellan mapped Venus using radar in the early 1990s. Venus has no moons and no magnetic field.",
        ],
        "Earth": [
            "Earth is the third planet from the Sun and the only known planet to support life. It has one natural satellite, the Moon. Earth's atmosphere is composed of 78 percent nitrogen and 21 percent oxygen.",
            "About 71 percent of Earth's surface is covered by water. Earth's axial tilt of 23.5 degrees causes seasons. The planet formed approximately 4.5 billion years ago. Earth's magnetic field protects it from solar radiation.",
        ],
        "Mars": [
            "Mars is the fourth planet from the Sun, known as the Red Planet due to iron oxide on its surface. Mars has the largest volcano in the solar system, Olympus Mons, standing 21.9 km high.",
            "Mars has the deepest canyon in the solar system, Valles Marineris, stretching 4000 km. Mars has two small moons: Phobos and Deimos. The Curiosity and Perseverance rovers are currently exploring Mars.",
            "Evidence suggests Mars once had liquid water on its surface. Mars has a thin atmosphere composed mainly of carbon dioxide. The planet experiences planet-wide dust storms.",
        ],
        "Jupiter": [
            "Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined. It is a gas giant composed mainly of hydrogen and helium.",
            "Jupiter's Great Red Spot is a storm larger than Earth that has been raging for at least 350 years. Jupiter has at least 95 known moons, including the four large Galilean moons: Io, Europa, Ganymede, and Callisto.",
            "Europa is considered one of the most likely places to find extraterrestrial life due to its subsurface ocean. Io is the most volcanically active body in the solar system. Ganymede is the largest moon in the solar system.",
        ],
        "Saturn": [
            "Saturn is the sixth planet from the Sun, famous for its spectacular ring system made of ice and rock particles. Saturn is a gas giant composed mainly of hydrogen and helium.",
            "Saturn has at least 146 known moons, including Titan, which has a thick atmosphere and liquid methane lakes on its surface. Enceladus has geysers of water ice erupting from its south pole.",
            "Saturn's density is so low that it would float in water if there were a bathtub large enough. The Cassini spacecraft studied Saturn and its moons for 13 years from 2004 to 2017.",
        ],
        "Uranus": [
            "Uranus is the seventh planet from the Sun and the first discovered using a telescope, found by William Herschel in 1781. It is an ice giant composed mainly of water, methane, and ammonia ices.",
            "Uranus rotates on its side with an axial tilt of 98 degrees, likely caused by a massive collision early in its history. Uranus has 27 known moons and a faint ring system.",
        ],
        "Neptune": [
            "Neptune is the eighth and farthest planet from the Sun. It is an ice giant with the strongest winds in the solar system, reaching speeds of 2100 km per hour.",
            "Neptune has 16 known moons, the largest being Triton, which orbits in the opposite direction to Neptune's rotation, suggesting it was captured from the Kuiper Belt. Neptune was visited only once by Voyager 2 in 1989.",
        ],
    }

    for planet, descs in planets.items():
        for desc in descs:
            knowledge.append(f"# {planet}\n\n{desc}")
            knowledge.append(f"Question: Tell me about {planet}.\nAnswer: {desc}")
            knowledge.append(f"What is {planet}? {desc}")

    # Stellar topics
    stellar = [
        ("Stellar Evolution", "Stars are massive celestial bodies that produce light and heat through nuclear fusion in their cores. They form from clouds of gas and dust called nebulae. The life cycle of a star depends on its mass. Low-mass stars like our Sun become red giants and then white dwarfs. High-mass stars can explode as supernovae and become neutron stars or black holes."),
        ("The Sun", "The Sun is a G-type main-sequence star at the center of our solar system. It contains 99.86 percent of the mass in the solar system. The Sun's core temperature reaches 15 million degrees Celsius, where hydrogen atoms fuse into helium. The Sun is approximately 4.6 billion years old and is expected to continue burning for another 5 billion years."),
        ("Supernovae", "A supernova is a powerful stellar explosion that occurs at the end of a massive star's life cycle. Supernovae are so bright they can outshine entire galaxies for weeks. They create and distribute heavy elements like gold, platinum, and uranium into space, seeding future generations of stars and planets."),
        ("Black Holes", "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once past the event horizon. They form when massive stars collapse at the end of their lives. The first image of a black hole was captured in 2019 by the Event Horizon Telescope, showing the supermassive black hole in galaxy M87."),
        ("Neutron Stars", "Neutron stars are the collapsed cores of massive stars that have undergone supernova explosions. They are incredibly dense, with a mass of 1.4 to 2 solar masses packed into a sphere only about 20 km in diameter. A teaspoon of neutron star material would weigh about 6 billion tons."),
        ("White Dwarfs", "White dwarfs are the remnants of low and medium-mass stars after they exhaust their nuclear fuel. They are about the size of Earth but have a mass similar to the Sun. The Chandrasekhar limit of 1.4 solar masses is the maximum mass a white dwarf can have before it collapses."),
        ("Nebulae", "Nebulae are giant clouds of dust and gas in space. Some nebulae are regions where new stars are being formed, while others are the remnants of dying stars. The Orion Nebula is one of the brightest nebulae visible to the naked eye. The Crab Nebula is a supernova remnant."),
        ("Galaxies", "Galaxies are massive systems of stars, gas, dust, and dark matter bound together by gravity. The Milky Way is our home galaxy, containing between 100 billion and 400 billion stars. The Andromeda Galaxy is the nearest large galaxy to us at 2.5 million light-years away."),
        ("Cosmology", "The Big Bang theory describes the origin of the universe as an extremely hot and dense state approximately 13.8 billion years ago. The cosmic microwave background radiation is the afterglow of the Big Bang. Dark matter makes up about 27 percent and dark energy about 68 percent of the universe."),
        ("Gravitational Waves", "Gravitational waves are ripples in spacetime caused by accelerating massive objects, predicted by Albert Einstein in 1916. They were first directly detected in 2015 by LIGO from the merger of two black holes about 1.3 billion light-years away."),
        ("Exoplanets", "Exoplanets are planets that orbit stars outside our solar system. The Kepler space telescope discovered over 2,600 exoplanets. The James Webb Space Telescope is now characterizing exoplanet atmospheres. The habitable zone is the region around a star where liquid water could exist."),
    ]

    for topic, text in stellar:
        knowledge.append(f"# {topic}\n\n{text}")
        knowledge.append(f"Question: What is {topic.lower()}?\nAnswer: {text}")
        knowledge.append(text)  # Plain text version

    # Space exploration
    exploration = [
        ("Apollo Program", "The Apollo program was NASA's human spaceflight program that landed the first humans on the Moon. Apollo 11 landed on July 20, 1969. Neil Armstrong was the first person to walk on the lunar surface, followed by Buzz Aldrin. In total, 12 astronauts walked on the Moon during six Apollo missions between 1969 and 1972."),
        ("International Space Station", "The International Space Station is a modular space station in low Earth orbit. It has been continuously occupied since November 2000. The station orbits Earth approximately every 90 minutes at an altitude of about 408 km. The ISS is a collaboration between NASA, Roscosmos, ESA, JAXA, and CSA."),
        ("Hubble Space Telescope", "The Hubble Space Telescope was launched in 1990 and has made over 1.5 million observations. It orbits Earth at an altitude of about 547 km. Hubble has helped determine the age of the universe at 13.8 billion years and provided evidence for the accelerating expansion of the universe."),
        ("James Webb Space Telescope", "The James Webb Space Telescope is the largest and most powerful space telescope ever built, launched on December 25, 2021. It orbits the Sun at the second Lagrange point. JWST observes in infrared light and can see the earliest galaxies formed after the Big Bang. It has a 6.5-meter primary mirror made of 18 hexagonal segments."),
        ("Voyager Program", "Voyager 1 and Voyager 2 are NASA space probes launched in 1977 to study the outer planets. Voyager 1 is the most distant human-made object, currently over 24 billion km from Earth. Both spacecraft carry a Golden Record containing sounds and images of Earth. Voyager 1 entered interstellar space in 2012."),
        ("Mars Exploration", "The Curiosity rover landed on Mars on August 6, 2012, in Gale Crater. It has discovered evidence that Mars once had conditions suitable for microbial life. The Perseverance rover landed on February 18, 2021, in Jezero Crater. The Ingenuity helicopter made the first powered flight on another planet."),
        ("SpaceX", "SpaceX is revolutionizing space travel with reusable rockets. The Falcon 9 rocket has successfully landed and been reused over 200 times. The Starship vehicle is designed for missions to the Moon and Mars. SpaceX's Crew Dragon spacecraft regularly transports astronauts to the ISS."),
    ]

    for topic, text in exploration:
        knowledge.append(f"# {topic}\n\n{text}")
        knowledge.append(f"Question: What is the {topic.lower()}?\nAnswer: {text}")
        knowledge.append(text)

    return knowledge


def collect_knowledge():
    print("\n" + "="*60)
    print("SOURCE 5: COMPREHENSIVE KNOWLEDGE BASE")
    print("="*60)

    output_file = CLEAN_DIR / "knowledge.txt"
    if output_file.exists() and output_file.stat().st_size > 100_000:
        print(f"[Skip] Already collected: {output_file.stat().st_size:,} bytes")
        return

    texts = generate_comprehensive_knowledge()
    save_source("knowledge", texts)


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINE AND TOKENIZE
# ═══════════════════════════════════════════════════════════════════════════════

def combine_and_tokenize():
    print("\n" + "="*60)
    print("COMBINING AND TOKENIZING")
    print("="*60)

    meta_file = TOKENIZED_DIR / "meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"[Skip] Already tokenized: {meta['total_tokens']:,} tokens")
        return meta

    # Combine all clean text
    all_text = []
    for f in sorted(CLEAN_DIR.glob("*.txt")):
        print(f"  Loading {f.name}...")
        with open(f, encoding='utf-8') as fh:
            text = fh.read()
            all_text.append(text)
            print(f"    {len(text):,} chars")

    combined = '\n\n'.join(all_text)
    total_chars = len(combined)
    print(f"\nTotal combined: {total_chars:,} chars (~{total_chars//4:,} tokens)")

    # Save combined text
    combined_file = OUTPUT_DIR / "combined_corpus.txt"
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write(combined)

    # Train tokenizer
    print("\nTraining SentencePiece tokenizer...")
    vocab_size = min(32000, max(16000, total_chars // 100000))
    model_prefix = str(TOKENIZED_DIR / "space_tokenizer")

    spm.SentencePieceTrainer.train(
        input=str(combined_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        num_threads=4,
        split_digits=True,
        byte_fallback=True,
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        max_sentence_length=4096,
    )

    # Tokenize
    print("\nTokenizing corpus...")
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    all_tokens = []
    for text in tqdm(all_text, desc="[Tokenize]"):
        chunks = [text[i:i+2000] for i in range(0, len(text), 2000)]
        for chunk in chunks:
            tokens = sp.encode(chunk, out_type=int)
            all_tokens.extend([sp.bos_id()] + tokens + [sp.eos_id()])

    all_tokens = np.array(all_tokens, dtype=np.int32)
    total_tokens = len(all_tokens)

    val_size = int(total_tokens * 0.05)
    train_tokens = all_tokens[:-val_size]
    val_tokens = all_tokens[-val_size:]

    np.save(TOKENIZED_DIR / "train.npy", train_tokens)
    np.save(TOKENIZED_DIR / "val.npy", val_tokens)

    meta = {
        "vocab_size": sp.get_piece_size(),
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "total_chars": total_chars,
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n[DONE] Total tokens: {total_tokens:,}")
    print(f"  Train: {len(train_tokens):,}")
    print(f"  Val: {len(val_tokens):,}")
    print(f"  Vocab: {sp.get_piece_size():,}")

    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*60)
    print("SPACE LLM - 100M TOKEN DATA COLLECTION")
    print("="*60)
    print(f"Target: {TARGET_TOKENS:,} tokens")
    print(f"Output: {OUTPUT_DIR}")

    start_time = time.time()

    collect_wikipedia()
    collect_arxiv()
    collect_nasa()
    collect_huggingface_parquet()
    collect_knowledge()

    meta = combine_and_tokenize()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE in {elapsed/3600:.1f} hours")
    print(f"Total tokens: {meta['total_tokens']:,}")
    print(f"Target: {TARGET_TOKENS:,}")
    print(f"Progress: {meta['total_tokens']/TARGET_TOKENS*100:.1f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
