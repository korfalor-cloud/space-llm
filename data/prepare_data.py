"""
Space Data Pipeline
Downloads, cleans, tokenizes, and saves space/astronomy text data.
Sources: Wikipedia (space articles), arXiv (astro-ph), NASA public text.
"""

import os
import re
import json
import time
import hashlib
import requests
import numpy as np
import sentencepiece as spm
from tqdm import tqdm
from pathlib import Path
from typing import List, Iterator

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
TOKENIZED_DIR = DATA_DIR / "tokenized"
TOKENIZER_DIR = DATA_DIR / "tokenizer"
WIKI_DIR = RAW_DIR / "wikipedia"
ARXIV_DIR = RAW_DIR / "arxiv"
NASA_DIR = RAW_DIR / "nasa"

SPACE_KEYWORDS = [
    "astronomy", "astrophysics", "planet", "planetary", "star", "stellar",
    "galaxy", "galactic", "nebula", "supernova", "black hole", "cosmic",
    "universe", "cosmology", "telescope", "observatory", "nasa", "spacecraft",
    "orbit", "orbital", "solar system", "mars", "jupiter", "saturn", "venus",
    "mercury", "neptune", "uranus", "pluto", "moon", "asteroid", "comet",
    "meteor", "exoplanet", "habitable", "light year", "parsec", "redshift",
    "quasar", "pulsar", "neutron star", "white dwarf", "main sequence",
    "hertzsprung", "hr diagram", "spectral class", "magnitude", "luminosity",
    "dark matter", "dark energy", "big bang", "cosmic microwave",
    "gravitational wave", "einstein", "relativity", "spacetime",
    "rocket", "launch", "satellite", "iss", "hubble", "james webb",
    "curiosity", "perseverance", "voyager", "cassini", "new horizons",
    "space exploration", "astronaut", "cosmonaut", "spacewalk",
    "milky way", "andromeda", "celestial", "constellation", "zodiac",
    "eclipse", "solstice", "equinox", "tide", "gravity",
    "ionosphere", "magnetosphere", "heliosphere", "van allen",
    "space station", "rover", "lander", "probe", "mission",
]

for d in [RAW_DIR, WIKI_DIR, ARXIV_DIR, NASA_DIR, TOKENIZED_DIR, TOKENIZER_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r"\{\{.*?\}\}", "", text)  # Remove {{templates}}
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)  # [[link|text]] -> text
    text = re.sub(r"\[http[^\]]*\]", "", text)  # Remove [http...]
    text = re.sub(r"<[^>]+>", "", text)  # Remove HTML tags
    text = re.sub(r"'''?([^']+)'''?", r"\1", text)  # Remove bold/italic
    text = re.sub(r"={2,}\s*(.*?)\s*={2,}", r"\1", text)  # Remove headers
    text = re.sub(r"\n{3,}", "\n\n", text)  # Collapse multiple newlines
    text = re.sub(r"[ \t]+", " ", text)  # Collapse spaces
    text = text.strip()
    return text


def is_space_related(text: str) -> bool:
    """Check if text is related to space/astronomy."""
    text_lower = text.lower()
    matches = sum(1 for kw in SPACE_KEYWORDS if kw in text_lower)
    return matches >= 2


# ─── Wikipedia ────────────────────────────────────────────────────────────────

def download_wikipedia() -> List[str]:
    """Download and filter Wikipedia articles about space using Wikipedia API."""
    print("[Wikipedia] Downloading space-related articles via Wikipedia API...")
    texts = []

    # Use Wikipedia's search + parse API directly (more reliable than HF datasets)
    space_search_terms = [
        "astronomy", "planet", "star", "galaxy", "nebula", "supernova",
        "black hole", "cosmology", "telescope", "spacecraft", "NASA",
        "solar system", "exoplanet", "gravitational wave", "dark matter",
        "dark energy", "big bang", "cosmic microwave background",
        "Milky Way", "Andromeda galaxy", "Hubble Space Telescope",
        "James Webb Space Telescope", "Mars exploration", "Jupiter planet",
        "Saturn planet", "Venus planet", "Mercury planet", "Neptune planet",
        "asteroid belt", "Kuiper belt", "Oort cloud", "comet", "meteor",
        "lunar eclipse", "solar eclipse", "constellation", "light year",
        "redshift", "quasar", "pulsar", "neutron star", "white dwarf",
        "main sequence star", "red giant", "stellar evolution",
        "nuclear fusion", "heliosphere", "magnetosphere", "ionosphere",
        "International Space Station", "Apollo program", "Space Shuttle",
        "Voyager program", "Mars rover", "Curiosity rover", "Perseverance",
        "rocket propulsion", "orbital mechanics", "space suit",
        "astronaut", "cosmonaut", "spacewalk", "space station",
        "Hertzsprung-Russell diagram", "spectral class", "absolute magnitude",
        "parallax", "standard candle", "Cepheid variable",
        "Type Ia supernova", "core collapse", "neutron star",
        "event horizon", "singularity", "spacetime", "general relativity",
        "special relativity", "gravitational lensing", "frame dragging",
        "Hawking radiation", "Penrose process", "Blandford-Znajek",
        "cosmic ray", "solar wind", "aurora", "Van Allen belt",
        "planetary ring", "tidal force", "Roche limit", "Hill sphere",
        "Lagrange point", "gravitational assist", "slingshot effect",
        "Kepler's laws", "Newton's law of gravitation", "escape velocity",
        "Hohmann transfer", "geostationary orbit", "polar orbit",
        "space debris", "Kessler syndrome", "space weather",
        "solar flare", "coronal mass ejection", "sunspot", "solar cycle",
    ]

    session = requests.Session()
    session.headers.update({"User-Agent": "SpaceLLM/1.0 (educational project)"})

    for term in tqdm(space_search_terms, desc="[Wikipedia] Searching"):
        try:
            # Search for articles
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": term,
                "srlimit": 20,
                "format": "json",
            }
            resp = session.get(search_url, params=search_params, timeout=15)
            if resp.status_code != 200:
                continue

            results = resp.json().get("query", {}).get("search", [])
            for result in results:
                title = result.get("title", "")
                page_id = result.get("pageid")

                # Fetch article content
                page_params = {
                    "action": "query",
                    "pageids": page_id,
                    "prop": "extracts",
                    "explaintext": True,
                    "format": "json",
                }
                page_resp = session.get(search_url, params=page_params, timeout=15)
                if page_resp.status_code != 200:
                    continue

                pages = page_resp.json().get("query", {}).get("pages", {})
                for page in pages.values():
                    extract = page.get("extract", "")
                    if len(extract) > 300:
                        texts.append(f"# {title}\n\n{extract}")

            time.sleep(0.1)  # Rate limiting

        except Exception as e:
            continue

    print(f"[Wikipedia] Collected {len(texts)} articles")
    return texts


# ─── arXiv ────────────────────────────────────────────────────────────────────

def download_arxiv() -> List[str]:
    """Download astronomy/astrophysics abstracts from arXiv."""
    print("[arXiv] Downloading astronomy abstracts...")
    texts = []
    categories = ["astro-ph", "gr-qc", "hep-ph"]
    base_url = "http://export.arxiv.org/api/query"

    for cat in categories:
        start = 0
        max_results = 500
        retries = 0
        max_retries = 3

        while start < max_results:
            try:
                params = {
                    "search_query": f"cat:{cat}*",
                    "start": start,
                    "max_results": 50,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
                resp = requests.get(base_url, params=params, timeout=60)
                if resp.status_code != 200:
                    retries += 1
                    if retries >= max_retries:
                        break
                    time.sleep(5)
                    continue

                retries = 0
                content = resp.text
                entries = content.split("<entry>")[1:]
                if not entries:
                    break

                for entry in entries:
                    title_match = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
                    abstract_match = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
                    if title_match and abstract_match:
                        title = title_match.group(1).strip()
                        abstract = abstract_match.group(1).strip()
                        text = f"{title}\n\n{abstract}"
                        text = clean_text(text)
                        if len(text) > 50:
                            texts.append(text)

                start += 50
                time.sleep(3)  # Rate limiting for arXiv API

            except requests.exceptions.Timeout:
                retries += 1
                print(f"[arXiv] Timeout at {start}, retry {retries}/{max_retries}")
                if retries >= max_retries:
                    break
                time.sleep(10)
            except Exception as e:
                print(f"[arXiv] Error at {start}: {e}")
                break

    print(f"[arXiv] Collected {len(texts)} abstracts")
    return texts


# ─── NASA ─────────────────────────────────────────────────────────────────────

def download_nasa() -> List[str]:
    """Download NASA public text data."""
    print("[NASA] Downloading public space text...")
    texts = []

    # NASA APOD (Astronomy Picture of the Day) explanations
    try:
        resp = requests.get(
            "https://api.nasa.gov/planetary/apod",
            params={"api_key": "DEMO_KEY", "count": 200},
            timeout=30,
        )
        if resp.status_code == 200:
            for item in resp.json():
                explanation = item.get("explanation", "")
                title = item.get("title", "")
                if explanation:
                    texts.append(clean_text(f"{title}\n\n{explanation}"))
    except Exception as e:
        print(f"[NASA APOD] Error: {e}")

    # NASA Tech Reports
    try:
        nasa_urls = [
            "https://www.nasa.gov/wp-content/uploads/2023/04/nasa-strategy-for-exoplanet-exploration.pdf",
        ]
        # Use NASA's public text content from their website
        resp = requests.get("https://www.nasa.gov/mission/", timeout=15)
        if resp.status_code == 200:
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.text = []
                    self.skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "nav", "header", "footer"):
                        self.skip = True

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "nav", "header", "footer"):
                        self.skip = False

                def handle_data(self, data):
                    if not self.skip:
                        stripped = data.strip()
                        if stripped:
                            self.text.append(stripped)

            parser = TextExtractor()
            parser.feed(resp.text)
            page_text = " ".join(parser.text)
            if len(page_text) > 500:
                texts.append(clean_text(page_text))
    except Exception as e:
        print(f"[NASA Web] Error: {e}")

    # Generate synthetic space knowledge base
    print("[NASA] Generating synthetic space knowledge base...")
    space_facts = generate_space_knowledge()
    texts.extend(space_facts)

    print(f"[NASA] Collected {len(texts)} items")
    return texts


def generate_space_knowledge() -> List[str]:
    """Generate a comprehensive space knowledge base."""
    knowledge = []

    # Planets of the solar system
    planets = {
        "Mercury": "Mercury is the smallest planet in our solar system and closest to the Sun. It has no atmosphere and experiences extreme temperature variations from -180°C at night to 430°C during the day. A year on Mercury lasts only 88 Earth days. Mercury has a large iron core that makes up about 75% of its radius. The MESSENGER spacecraft orbited Mercury from 2011 to 2015, revealing a world covered in impact craters and ancient lava flows.",
        "Venus": "Venus is the second planet from the Sun and is often called Earth's twin due to similar size and mass. However, Venus has a thick toxic atmosphere composed mainly of carbon dioxide with clouds of sulfuric acid. The surface temperature reaches 465°C, making it the hottest planet in the solar system. Venus rotates backwards compared to most planets, and a day on Venus is longer than its year. The atmospheric pressure on Venus is about 90 times that of Earth.",
        "Earth": "Earth is the third planet from the Sun and the only known planet to support life. It has one natural satellite, the Moon. Earth's atmosphere is composed of 78% nitrogen and 21% oxygen. The planet has a magnetic field that protects it from solar radiation. About 71% of Earth's surface is covered by water. Earth's axial tilt of 23.5 degrees causes seasons. The planet formed approximately 4.5 billion years ago.",
        "Mars": "Mars is the fourth planet from the Sun, known as the Red Planet due to iron oxide on its surface. Mars has the largest volcano in the solar system, Olympus Mons, standing 21.9 km high, and the deepest canyon, Valles Marineris, stretching 4000 km. Mars has two small moons: Phobos and Deimos. The Curiosity and Perseverance rovers are currently exploring Mars. Evidence suggests Mars once had liquid water on its surface.",
        "Jupiter": "Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined. It is a gas giant composed mainly of hydrogen and helium. Jupiter's Great Red Spot is a storm larger than Earth that has been raging for at least 350 years. Jupiter has at least 95 known moons, including the four large Galilean moons: Io, Europa, Ganymede, and Callisto. Europa is considered one of the most likely places to find extraterrestrial life.",
        "Saturn": "Saturn is the sixth planet from the Sun, famous for its spectacular ring system made of ice and rock particles. Saturn is a gas giant composed mainly of hydrogen and helium. It has at least 146 known moons, including Titan, which has a thick atmosphere and liquid methane lakes. Saturn's density is so low that it would float in water if there were a bathtub large enough. The Cassini spacecraft studied Saturn and its moons for 13 years from 2004 to 2017.",
        "Uranus": "Uranus is the seventh planet from the Sun and the first discovered using a telescope, found by William Herschel in 1781. It is an ice giant composed mainly of water, methane, and ammonia ices. Uranus rotates on its side with an axial tilt of 98 degrees, likely caused by a massive collision early in its history. Uranus has 27 known moons and a faint ring system. The atmosphere appears blue-green due to methane absorption.",
        "Neptune": "Neptune is the eighth and farthest planet from the Sun. It is an ice giant with the strongest winds in the solar system, reaching speeds of 2100 km/h. Neptune has 16 known moons, the largest being Triton, which orbits in the opposite direction to Neptune's rotation, suggesting it was captured from the Kuiper Belt. Neptune was visited only once by Voyager 2 in 1989. Its blue color comes from methane in its atmosphere.",
    }

    for planet, desc in planets.items():
        knowledge.append(f"# {planet}\n\n{desc}")
        knowledge.append(f"Question: Tell me about {planet}.\nAnswer: {desc}")
        knowledge.append(f"What is {planet}? {desc}")

    # Stars and stellar evolution
    stellar_topics = [
        "Stars are massive celestial bodies that produce light and heat through nuclear fusion in their cores. They form from clouds of gas and dust called nebulae. The life cycle of a star depends on its mass. Low-mass stars like our Sun become red giants and then white dwarfs. High-mass stars can explode as supernovae and become neutron stars or black holes.",
        "The Sun is a G-type main-sequence star (G2V) at the center of our solar system. It contains 99.86% of the mass in the solar system. The Sun's core temperature reaches 15 million degrees Celsius, where hydrogen atoms fuse into helium. The Sun is approximately 4.6 billion years old and is expected to continue burning for another 5 billion years.",
        "A supernova is a powerful stellar explosion that occurs at the end of a massive star's life cycle. There are two main types: Type Ia (thermonuclear) and Type II (core-collapse). Supernovae are so bright they can outshine entire galaxies for weeks. They create and distribute heavy elements like gold, platinum, and uranium into space, seeding future generations of stars and planets.",
        "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape once past the event horizon. They form when massive stars collapse at the end of their lives. The first image of a black hole was captured in 2019 by the Event Horizon Telescope, showing the supermassive black hole in galaxy M87. Sagittarius A* is the supermassive black hole at the center of our Milky Way galaxy.",
        "Neutron stars are the collapsed cores of massive stars that have undergone supernova explosions. They are incredibly dense, with a mass of 1.4 to 2 solar masses packed into a sphere only about 20 km in diameter. A teaspoon of neutron star material would weigh about 6 billion tons. Pulsars are rapidly rotating neutron stars that emit beams of electromagnetic radiation.",
        "White dwarfs are the remnants of low and medium-mass stars after they exhaust their nuclear fuel. They are about the size of Earth but have a mass similar to the Sun. White dwarfs gradually cool and fade over billions of years. The Chandrasekhar limit of 1.4 solar masses is the maximum mass a white dwarf can have before it collapses.",
        "The Hertzsprung-Russell diagram is a scatter graph of stars showing the relationship between their absolute magnitudes or luminosities versus their spectral types or temperatures. It was developed independently by Ejnar Hertzsprung and Henry Norris Russell in the early 20th century. The main sequence runs diagonally from hot, luminous stars to cool, dim ones.",
    ]

    for text in stellar_topics:
        knowledge.append(text)
        knowledge.append(f"Question: {text.split('.')[0]}?\nAnswer: {text}")

    # Galaxies and cosmology
    cosmology_topics = [
        "The Milky Way is the galaxy that contains our solar system. It is a barred spiral galaxy with a diameter of approximately 100,000 light-years and contains between 100 billion and 400 billion stars. The supermassive black hole at its center, Sagittarius A*, has a mass of about 4 million times that of our Sun. The Milky Way is part of the Local Group of galaxies.",
        "The Andromeda Galaxy (M31) is the nearest large galaxy to the Milky Way, located about 2.5 million light-years away. It is approaching the Milky Way at about 110 kilometers per second and the two galaxies are expected to collide in about 4.5 billion years, forming a single elliptical galaxy sometimes called Milkdromeda.",
        "Dark matter is a hypothetical form of matter that does not emit or interact with electromagnetic radiation. It is estimated to make up about 27% of the total mass-energy content of the universe. Evidence for dark matter comes from gravitational effects on visible matter, such as the rotation curves of galaxies and gravitational lensing.",
        "Dark energy is a hypothetical form of energy that permeates all of space and causes the accelerating expansion of the universe. It is estimated to make up about 68% of the total mass-energy content of the universe. The discovery of the accelerating expansion of the universe in 1998 by Saul Perlmutter, Brian Schmidt, and Adam Riess earned them the Nobel Prize in Physics in 2011.",
        "The Big Bang theory describes the origin of the universe as an extremely hot and dense state approximately 13.8 billion years ago that has been expanding ever since. Key evidence includes the cosmic microwave background radiation, the abundance of light elements, and the redshift of distant galaxies. The cosmic microwave background was discovered in 1965 by Arno Penzias and Robert Wilson.",
        "The cosmic microwave background (CMB) is the thermal radiation left over from the early universe, about 380,000 years after the Big Bang. It was discovered in 1965 and has a temperature of approximately 2.725 Kelvin. The CMB provides a snapshot of the universe when it was very young and has tiny fluctuations that seeded the formation of galaxies and large-scale structure.",
        "Exoplanets are planets that orbit stars outside our solar system. The first confirmed exoplanet discovery was in 1992 around a pulsar. The Kepler space telescope discovered over 2,600 exoplanets during its mission. The James Webb Space Telescope is now characterizing exoplanet atmospheres. The habitable zone, or Goldilocks zone, is the region around a star where liquid water could exist on a planet's surface.",
        "Gravitational waves are ripples in spacetime caused by accelerating massive objects, predicted by Albert Einstein in 1916. They were first directly detected in 2015 by LIGO from the merger of two black holes about 1.3 billion light-years away. This discovery earned Rainer Weiss, Kip Thorne, and Barry Barish the Nobel Prize in Physics in 2017.",
    ]

    for text in cosmology_topics:
        knowledge.append(text)
        knowledge.append(f"Question: {text.split('.')[0]}?\nAnswer: {text}")

    # Space exploration
    exploration_topics = [
        "The Apollo program was NASA's human spaceflight program that landed the first humans on the Moon. Apollo 11, commanded by Neil Armstrong with pilot Buzz Aldrin, landed on the Moon on July 20, 1969. Armstrong was the first person to walk on the lunar surface, followed by Aldrin. In total, 12 astronauts walked on the Moon during six Apollo missions between 1969 and 1972.",
        "The International Space Station (ISS) is a modular space station in low Earth orbit. It is the largest artificial object in space and can often be seen with the naked eye. The ISS serves as a microgravity and space environment research laboratory. It has been continuously occupied since November 2000. The station orbits Earth approximately every 90 minutes at an altitude of about 408 km.",
        "The Hubble Space Telescope was launched in 1990 and has made over 1.5 million observations. It orbits Earth at an altitude of about 547 km. Hubble has helped determine the age of the universe (13.8 billion years), discovered that most galaxies have supermassive black holes at their centers, and provided evidence for the accelerating expansion of the universe.",
        "The James Webb Space Telescope (JWST) is the largest and most powerful space telescope ever built, launched on December 25, 2021. It orbits the Sun at the second Lagrange point (L2), about 1.5 million km from Earth. JWST observes in infrared light and can see the earliest galaxies formed after the Big Bang. It has a 6.5-meter primary mirror made of 18 hexagonal segments.",
        "Voyager 1 and Voyager 2 are NASA space probes launched in 1977 to study the outer planets. Voyager 1 is the most distant human-made object, currently over 24 billion km from Earth. Both spacecraft carry a Golden Record containing sounds and images of Earth. Voyager 1 entered interstellar space in 2012, and Voyager 2 followed in 2018.",
        "The Curiosity rover landed on Mars on August 6, 2012, in Gale Crater. It has discovered evidence that Mars once had conditions suitable for microbial life, including ancient river beds and organic molecules. The Perseverance rover landed on February 18, 2021, in Jezero Crater, and is collecting samples for future return to Earth. The Ingenuity helicopter made the first powered flight on another planet.",
        "SpaceX is revolutionizing space travel with reusable rockets. The Falcon 9 rocket has successfully landed and been reused over 200 times. The Starship vehicle is designed for missions to the Moon and Mars. SpaceX's Crew Dragon spacecraft regularly transports astronauts to the ISS. The company's goal is to make humanity a multi-planetary species by establishing a colony on Mars.",
    ]

    for text in exploration_topics:
        knowledge.append(text)
        knowledge.append(f"Question: {text.split('.')[0]}?\nAnswer: {text}")

    # Physics and space science
    physics_topics = [
        "Einstein's theory of general relativity describes gravity as the curvature of spacetime caused by mass and energy. It predicts phenomena such as the bending of light by gravity (gravitational lensing), time dilation near massive objects (gravitational time dilation), and the existence of black holes. General relativity has been confirmed by numerous experiments, including the observation of gravitational waves.",
        "The electromagnetic spectrum in astronomy includes radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays. Different wavelengths reveal different physical processes and objects. Radio astronomy reveals cold gas and pulsars. Infrared penetrates dust clouds. X-rays reveal hot gas and accretion disks. Gamma rays show the most energetic events in the universe.",
        "Nuclear fusion is the process that powers stars. In the Sun's core, hydrogen nuclei fuse to form helium, releasing enormous amounts of energy according to Einstein's equation E=mc². The proton-proton chain is the dominant fusion process in the Sun. More massive stars use the CNO cycle. Understanding fusion is key to developing clean energy on Earth.",
        "The cosmic distance ladder is a series of methods by which astronomers determine the distances to celestial objects. Nearby distances are measured using parallax. Cepheid variable stars and Type Ia supernovae are used as standard candles for intermediate distances. Redshift is used for the most distant objects. Each method builds upon the previous one, forming a ladder of distance measurements.",
        "Astrobiology is the study of the origin, evolution, and distribution of life in the universe. Key questions include: How did life begin on Earth? Is there life elsewhere in the solar system? Are there habitable exoplanets? Mars, Europa, Enceladus, and Titan are considered the most likely places to find extraterrestrial life in our solar system.",
        "The search for extraterrestrial intelligence (SETI) uses radio telescopes to listen for signals from advanced civilizations. The Drake equation estimates the number of active, communicative extraterrestrial civilizations in the Milky Way galaxy. Projects like Breakthrough Listen are conducting the most comprehensive search for extraterrestrial intelligence ever undertaken.",
    ]

    for text in physics_topics:
        knowledge.append(text)
        knowledge.append(f"Question: {text.split('.')[0]}?\nAnswer: {text}")

    return knowledge


# ─── Tokenizer ────────────────────────────────────────────────────────────────

def train_tokenizer(texts: List[str], vocab_size: int = 32000) -> str:
    """Train SentencePiece BPE tokenizer on the corpus."""
    print(f"[Tokenizer] Training BPE tokenizer with vocab_size={vocab_size}...")
    corpus_file = TOKENIZER_DIR / "corpus.txt"

    with open(corpus_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text.replace("\n", " ") + "\n")

    model_prefix = str(TOKENIZER_DIR / "space_tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        num_threads=4,
        split_digits=True,
        byte_fallback=True,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )
    print(f"[Tokenizer] Saved to {model_prefix}.model")
    return f"{model_prefix}.model"


# ─── Tokenize & Save ─────────────────────────────────────────────────────────

def tokenize_and_save(texts: List[str], tokenizer_path: str, val_ratio: float = 0.05):
    """Tokenize all texts and save as memory-mapped numpy arrays."""
    print("[Tokenize] Tokenizing corpus...")
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    # Tokenize all texts
    all_tokens = []
    for text in tqdm(texts, desc="[Tokenize] Encoding"):
        tokens = sp.encode(text, out_type=int)
        tokens = [sp.bos_id()] + tokens + [sp.eos_id()]
        all_tokens.extend(tokens)

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    total_tokens = len(all_tokens)
    print(f"[Tokenize] Total tokens: {total_tokens:,}")

    # Split into train/val
    val_size = int(total_tokens * val_ratio)
    train_tokens = all_tokens[:-val_size]
    val_tokens = all_tokens[-val_size:]

    # Save as memmap
    train_path = TOKENIZED_DIR / "train.bin"
    val_path = TOKENIZED_DIR / "val.bin"

    train_mm = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=len(train_tokens))
    train_mm[:] = train_tokens[:]
    train_mm.flush()

    val_mm = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=len(val_tokens))
    val_mm[:] = val_tokens[:]
    val_mm.flush()

    # Save metadata
    meta = {
        "vocab_size": sp.get_piece_size(),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "tokenizer": tokenizer_path,
        "dtype": "uint16",
    }
    with open(TOKENIZED_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Tokenize] Train: {len(train_tokens):,} tokens -> {train_path}")
    print(f"[Tokenize] Val: {len(val_tokens):,} tokens -> {val_path}")
    return meta


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("SPACE LLM - Data Preparation Pipeline")
    print("=" * 60)

    # Check if already prepared
    meta_path = TOKENIZED_DIR / "meta.json"
    if meta_path.exists():
        print("[Skip] Data already prepared. Delete data/tokenized/ to re-run.")
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Train tokens: {meta['train_tokens']:,}")
        print(f"  Val tokens: {meta['val_tokens']:,}")
        return

    # Collect texts from all sources
    all_texts = []

    wiki_texts = download_wikipedia()
    all_texts.extend(wiki_texts)

    arxiv_texts = download_arxiv()
    all_texts.extend(arxiv_texts)

    nasa_texts = download_nasa()
    all_texts.extend(nasa_texts)

    if not all_texts:
        print("[ERROR] No data collected! Check your internet connection.")
        return

    # Deduplicate
    print(f"\n[Dedup] Deduplicating {len(all_texts)} documents...")
    seen = set()
    unique_texts = []
    for text in all_texts:
        h = hashlib.md5(text.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_texts.append(text)
    print(f"[Dedup] {len(unique_texts)} unique documents (removed {len(all_texts) - len(unique_texts)})")

    # Train tokenizer
    tokenizer_path = train_tokenizer(unique_texts)

    # Tokenize and save
    meta = tokenize_and_save(unique_texts, tokenizer_path)

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print(f"Total tokens: {meta['train_tokens'] + meta['val_tokens']:,}")
    print(f"Vocab size: {meta['vocab_size']:,}")
    print("=" * 60)


if __name__ == "__main__":
    main()
