"""
Space LLM - 100M Token Data Collection Pipeline
Collects space/astronomy/science text from ALL available sources.
Target: 100 million tokens of high-quality space content.

Sources:
1. Wikipedia (filtered for space/astronomy) ~50M tokens
2. arXiv (astro-ph, gr-qc, hep-ph papers) ~30M tokens
3. NASA APIs (APOD, missions, tech reports) ~5M tokens
4. HuggingFace datasets (physics, astronomy, science) ~10M tokens
5. Stack Exchange (astronomy, physics, space) ~3M tokens
6. OpenWebText (filtered for space content) ~2M tokens
"""

import os
import re
import json
import time
import hashlib
import subprocess
import sys
import requests
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Install dependencies with compatible versions
def setup():
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "datasets>=2.14.0,<3.0.0", "pyarrow>=12.0.0,<15.0.0",
                           "sentencepiece", "tqdm", "requests", "-q"])

setup()

from datasets import load_dataset
import sentencepiece as spm

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

TARGET_TOKENS = 100_000_000
OUTPUT_DIR = Path("/kaggle/working/space_corpus") if os.path.exists("/kaggle") else Path("space_corpus")
RAW_DIR = OUTPUT_DIR / "raw"
CLEAN_DIR = OUTPUT_DIR / "clean"
TOKENIZED_DIR = OUTPUT_DIR / "tokenized"

for d in [RAW_DIR, CLEAN_DIR, TOKENIZED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Space keywords for filtering
SPACE_KEYWORDS = {
    "high": [  # Must match 2+ of these
        "astronomy", "astrophysics", "cosmology", "telescope", "observatory",
        "black hole", "neutron star", "pulsar", "quasar", "supernova",
        "galaxy", "galactic", "nebula", "stellar", "planetary",
        "exoplanet", "habitable zone", "dark matter", "dark energy",
        "big bang", "cosmic microwave", "gravitational wave",
        "hubble", "james webb", "chandra", "spitzer",
        "nasa", "esa", "jaxa", "spacex", "blue origin",
        "astronaut", "cosmonaut", "spacecraft", "spacecraft",
        "apollo", "voyager", "cassini", "curiosity", "perseverance",
        "international space station", "iss", "space station",
    ],
    "medium": [  # Match 1+ of these
        "planet", "star", "moon", "asteroid", "comet", "meteor",
        "orbit", "solar", "lunar", "mars", "jupiter", "saturn",
        "venus", "mercury", "neptune", "uranus", "pluto",
        "rocket", "launch", "satellite", "probe", "rover",
        "eclipse", "solstice", "equinox", "constellation",
        "light year", "parsec", "redshift", "spectroscop",
        "fusion", "fission", "relativity", "spacetime", "quantum",
        "gravity", "magnetic field", "radiation", "cosmic",
        "milky way", "andromeda", "universe", "expansion",
    ]
}

# ═══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text):
    """Clean and normalize text, removing LaTeX, HTML, etc."""
    # Remove LaTeX
    text = re.sub(r'\$[^$]+\$', ' MATH ', text)  # Inline math
    text = re.sub(r'\$\$[^$]+\$\$', ' MATH ', text)  # Display math
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)  # \command{...}
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # \command
    text = re.sub(r'\{[^}]*\}', '', text)  # {braces}

    # Remove HTML
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)

    # Remove wiki markup
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r"'''?([^']+)'''?", r'\1', text)
    text = re.sub(r'={2,}\s*(.*?)\s*={2,}', r'\1', text)

    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[http[^\]]*\]', '', text)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' +\n', '\n', text)

    # Remove very short lines (likely artifacts)
    lines = text.split('\n')
    lines = [l for l in lines if len(l.strip()) > 10 or l.strip() == '']
    text = '\n'.join(lines)

    return text.strip()


def is_space_related(text, threshold="high"):
    """Check if text is related to space/astronomy."""
    text_lower = text.lower()

    if threshold == "high":
        # Must match 2+ high keywords
        matches = sum(1 for kw in SPACE_KEYWORDS["high"] if kw in text_lower)
        return matches >= 2
    else:
        # Match 1+ medium keywords
        matches = sum(1 for kw in SPACE_KEYWORDS["medium"] if kw in text_lower)
        return matches >= 1


def quality_filter(text):
    """Filter out low-quality text."""
    if len(text) < 100:
        return False

    # Check for too much special characters
    special_ratio = sum(1 for c in text if not c.isalnum() and c not in ' \n.,;:!?-') / len(text)
    if special_ratio > 0.3:
        return False

    # Check for reasonable word length
    words = text.split()
    if len(words) < 20:
        return False

    avg_word_len = sum(len(w) for w in words) / len(words)
    if avg_word_len < 3 or avg_word_len > 15:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: WIKIPEDIA (Target: ~50M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_wikipedia():
    """Collect space-related Wikipedia articles."""
    print("\n" + "="*60)
    print("SOURCE 1: WIKIPEDIA")
    print("="*60)

    output_file = CLEAN_DIR / "wikipedia.txt"
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"[Skip] Already collected: {size:,} bytes")
        return

    print("Loading Wikipedia dataset (streaming)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    articles = []
    total_chars = 0
    count = 0

    for article in tqdm(ds, desc="[Wikipedia] Filtering"):
        title = article.get("title", "")
        text = article.get("text", "")

        # Quick filter: check title first
        if not is_space_related(title, "medium") and not is_space_related(text[:500], "high"):
            continue

        cleaned = clean_text(text)
        if not quality_filter(cleaned):
            continue

        # Format as article
        formatted = f"# {title}\n\n{cleaned}"
        articles.append(formatted)
        total_chars += len(formatted)
        count += 1

        # Progress update every 1000 articles
        if count % 1000 == 0:
            est_tokens = total_chars // 4
            print(f"  {count:,} articles | {total_chars:,} chars | ~{est_tokens:,} tokens")

        # Stop when we have enough (~50M tokens = ~200M chars)
        if total_chars > 200_000_000:
            break

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(articles))

    est_tokens = total_chars // 4
    print(f"\n[Wikipedia] Saved {count:,} articles | {total_chars:,} chars | ~{est_tokens:,} tokens")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: ARXIV (Target: ~30M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_arxiv():
    """Collect arXiv astronomy/physics papers."""
    print("\n" + "="*60)
    print("SOURCE 2: ARXIV")
    print("="*60)

    output_file = CLEAN_DIR / "arxiv.txt"
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"[Skip] Already collected: {size:,} bytes")
        return

    # Use arXiv classification dataset (has full papers)
    print("Loading arXiv dataset...")
    ds = load_dataset("ccdv/arxiv-classification", split="train", streaming=True)

    # arXiv categories for space/physics
    space_categories = [
        "astro-ph", "gr-qc", "hep-ph", "hep-th", "physics.space-ph",
        "physics.gen-ph", "physics.ed-ph"
    ]

    papers = []
    total_chars = 0
    count = 0

    for paper in tqdm(ds, desc="[arXiv] Filtering"):
        text = paper.get("text", "")

        # Check if it's a space/physics paper
        if not is_space_related(text[:1000], "high"):
            continue

        # Clean LaTeX and format
        cleaned = clean_text(text)
        if not quality_filter(cleaned):
            continue

        # Truncate very long papers to first 5000 chars
        if len(cleaned) > 5000:
            cleaned = cleaned[:5000] + "\n\n[Paper truncated for training]"

        papers.append(cleaned)
        total_chars += len(cleaned)
        count += 1

        if count % 500 == 0:
            est_tokens = total_chars // 4
            print(f"  {count:,} papers | {total_chars:,} chars | ~{est_tokens:,} tokens")

        # Stop at ~30M tokens = ~120M chars
        if total_chars > 120_000_000:
            break

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(papers))

    est_tokens = total_chars // 4
    print(f"\n[arXiv] Saved {count:,} papers | {total_chars:,} chars | ~{est_tokens:,} tokens")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: NASA APIs (Target: ~5M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_nasa():
    """Collect data from NASA APIs."""
    print("\n" + "="*60)
    print("SOURCE 3: NASA APIs")
    print("="*60)

    output_file = CLEAN_DIR / "nasa.txt"
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"[Skip] Already collected: {size:,} bytes")
        return

    texts = []
    session = requests.Session()

    # 3a. NASA APOD (Astronomy Picture of the Day)
    print("[NASA] Collecting APOD descriptions...")
    try:
        for i in range(10):  # Get 10 batches of 100
            r = session.get("https://api.nasa.gov/planetary/apod",
                          params={"api_key": "DEMO_KEY", "count": 100, "thumbs": True},
                          timeout=30)
            if r.status_code == 200:
                for item in r.json():
                    title = item.get("title", "")
                    explanation = item.get("explanation", "")
                    if explanation and len(explanation) > 50:
                        texts.append(f"# {title}\n\n{explanation}")
            time.sleep(1)
    except Exception as e:
        print(f"  APOD error: {e}")

    # 3b. NASA Image and Video Library
    print("[NASA] Collecting Image Library descriptions...")
    try:
        for topic in ["astronomy", "space", "galaxy", "nebula", "planet", "star", "rocket", "satellite"]:
            r = session.get("https://images-api.nasa.gov/search",
                          params={"q": topic, "media_type": "image", "page_size": 100},
                          timeout=30)
            if r.status_code == 200:
                items = r.json().get("collection", {}).get("items", [])
                for item in items:
                    data = item.get("data", [{}])[0]
                    title = data.get("title", "")
                    desc = data.get("description", "")
                    if desc and len(desc) > 50:
                        texts.append(f"# {title}\n\n{desc}")
            time.sleep(0.5)
    except Exception as e:
        print(f"  Image Library error: {e}")

    # 3c. NASA TechPort (projects)
    print("[NASA] Collecting TechPort projects...")
    try:
        for page in range(5):
            r = session.get("https://techport.nasa.gov/api/projects",
                          params={"page": page, "pageSize": 100},
                          timeout=30)
            if r.status_code == 200:
                projects = r.json().get("projects", [])
                for proj in projects:
                    title = proj.get("title", "")
                    desc = proj.get("description", "")
                    if desc and len(desc) > 50:
                        texts.append(f"# NASA Project: {title}\n\n{desc}")
            time.sleep(1)
    except Exception as e:
        print(f"  TechPort error: {e}")

    # 3d. NASA NSSDCA (planetary fact sheets)
    print("[NASA] Collecting NSSDCA planetary data...")
    try:
        planets = ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
        for planet in planets:
            r = session.get(f"https://nssdc.gsfc.nasa.gov/planetary/factsheet/{planet}fact.html", timeout=15)
            if r.status_code == 200:
                # Extract text from HTML
                text = re.sub(r'<[^>]+>', ' ', r.text)
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 100:
                    texts.append(f"# {planet.title()} Planetary Fact Sheet\n\n{text}")
    except Exception as e:
        print(f"  NSSDCA error: {e}")

    # 3e. NASA Earthdata (earth science)
    print("[NASA] Collecting Earthdata metadata...")
    try:
        r = session.get("https://cmr.earthdata.nasa.gov/search/collections.json",
                       params={"keyword": "space", "page_size": 100},
                       timeout=30)
        if r.status_code == 200:
            refs = r.json().get("feed", {}).get("entry", [])
            for ref in refs:
                title = ref.get("title", "")
                summary = ref.get("summary", "")
                if summary and len(summary) > 50:
                    texts.append(f"# {title}\n\n{summary}")
    except Exception as e:
        print(f"  Earthdata error: {e}")

    # Save
    total_chars = sum(len(t) for t in texts)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(texts))

    est_tokens = total_chars // 4
    print(f"\n[NASA] Saved {len(texts):,} items | {total_chars:,} chars | ~{est_tokens:,} tokens")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 4: HUGGINGFACE DATASETS (Target: ~10M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_huggingface():
    """Collect from HuggingFace datasets."""
    print("\n" + "="*60)
    print("SOURCE 4: HUGGINGFACE DATASETS")
    print("="*60)

    output_file = CLEAN_DIR / "huggingface.txt"
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"[Skip] Already collected: {size:,} bytes")
        return

    all_texts = []
    total_chars = 0

    # 4a. Physics Q&A (camel-ai/physics)
    print("[HF] Collecting Physics Q&A...")
    try:
        ds = load_dataset("camel-ai/physics", split="train", streaming=True)
        for item in tqdm(ds, desc="[Physics Q&A]"):
            msg1 = item.get("message_1", "")
            msg2 = item.get("message_2", "")
            topic = item.get("topic;", "")

            if is_space_related(topic + " " + msg1, "medium"):
                formatted = f"Question: {msg1}\nAnswer: {msg2}"
                all_texts.append(formatted)
                total_chars += len(formatted)

            if total_chars > 10_000_000:  # ~2.5M tokens
                break
    except Exception as e:
        print(f"  Physics Q&A error: {e}")

    # 4b. Astronomy StackExchange
    print("[HF] Collecting Astronomy StackExchange...")
    try:
        ds = load_dataset("mlfoundations-dev/stackexchange_astronomy", split="train", streaming=True)
        for item in tqdm(ds, desc="[Astronomy SE]"):
            instruction = item.get("instruction", "")
            completion = item.get("completion", "")

            if instruction and completion:
                formatted = f"Question: {instruction}\nAnswer: {completion}"
                all_texts.append(formatted)
                total_chars += len(formatted)

            if total_chars > 20_000_000:  # ~5M tokens
                break
    except Exception as e:
        print(f"  Astronomy SE error: {e}")

    # 4c. Physics Reasoning
    print("[HF] Collecting Physics Reasoning...")
    try:
        ds = load_dataset("0xZee/dataset-CoT-Relativity-Astrophysics-Nuclear-Physics-313", split="train", streaming=True)
        for item in tqdm(ds, desc="[Physics Reasoning]"):
            question = item.get("question", "")
            response = item.get("response", "")

            if question and response:
                formatted = f"Question: {question}\nAnswer: {response}"
                all_texts.append(formatted)
                total_chars += len(formatted)
    except Exception as e:
        print(f"  Physics Reasoning error: {e}")

    # 4d. Science Education
    print("[HF] Collecting Science Education...")
    try:
        ds = load_dataset("Josephgflowers/Par-Four-Fineweb-Edu-Fortified-Chemistry-Physics-Astronomy-Math-Reason", split="train", streaming=True)
        for item in tqdm(ds, desc="[Science Edu]"):
            text = item.get("text", "")

            if is_space_related(text[:500], "high"):
                cleaned = clean_text(text)
                if quality_filter(cleaned):
                    all_texts.append(cleaned)
                    total_chars += len(cleaned)

            if total_chars > 30_000_000:  # ~7.5M tokens
                break
    except Exception as e:
        print(f"  Science Edu error: {e}")

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_texts))

    est_tokens = total_chars // 4
    print(f"\n[HuggingFace] Saved {len(all_texts):,} items | {total_chars:,} chars | ~{est_tokens:,} tokens")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 5: OPENWEBTEXT FILTERED (Target: ~2M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_openwebtext():
    """Collect space-related OpenWebText articles."""
    print("\n" + "="*60)
    print("SOURCE 5: OPENWEBTEXT (filtered)")
    print("="*60)

    output_file = CLEAN_DIR / "openwebtext.txt"
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"[Skip] Already collected: {size:,} bytes")
        return

    print("Loading OpenWebText (streaming)...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    texts = []
    total_chars = 0
    count = 0

    for item in tqdm(ds, desc="[OpenWebText] Filtering"):
        text = item.get("text", "")

        if is_space_related(text[:1000], "high"):
            cleaned = clean_text(text)
            if quality_filter(cleaned):
                texts.append(cleaned)
                total_chars += len(cleaned)
                count += 1

        if total_chars > 10_000_000:  # ~2.5M tokens
            break

        # Limit search to avoid hanging
        if count + (total_chars // 1000) > 500000:
            break

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(texts))

    est_tokens = total_chars // 4
    print(f"\n[OpenWebText] Saved {count:,} items | {total_chars:,} chars | ~{est_tokens:,} tokens")


# ═══════════════════════════════════════════════════════════════════════════════
# SOURCE 6: ARXIV API DIRECT (Target: ~5M tokens)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_arxiv_api():
    """Collect arXiv abstracts via API."""
    print("\n" + "="*60)
    print("SOURCE 6: ARXIV API (abstracts)")
    print("="*60)

    output_file = CLEAN_DIR / "arxiv_api.txt"
    if output_file.exists():
        size = output_file.stat().st_size
        print(f"[Skip] Already collected: {size:,} bytes")
        return

    categories = [
        "astro-ph", "astro-ph.GA", "astro-ph.CO", "astro-ph.EP",
        "astro-ph.HE", "astro-ph.IM", "astro-ph.SR",
        "gr-qc", "hep-ph", "hep-th", "physics.space-ph",
    ]

    texts = []
    total_chars = 0

    for cat in categories:
        print(f"[arXiv API] Category: {cat}")
        start = 0
        while start < 2000:
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
                        title = title_m.group(1).strip()
                        abstract = abstract_m.group(1).strip()
                        formatted = f"# {title}\n\n{abstract}"
                        texts.append(formatted)
                        total_chars += len(formatted)

                start += 100
                time.sleep(3)  # Rate limit

            except Exception as e:
                print(f"  Error at {start}: {e}")
                break

        if total_chars > 20_000_000:  # ~5M tokens
            break

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(texts))

    est_tokens = total_chars // 4
    print(f"\n[arXiv API] Saved {len(texts):,} items | {total_chars:,} chars | ~{est_tokens:,} tokens")


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINE AND TOKENIZE
# ═══════════════════════════════════════════════════════════════════════════════

def combine_and_tokenize():
    """Combine all sources and tokenize."""
    print("\n" + "="*60)
    print("COMBINING AND TOKENIZING")
    print("="*60)

    # Check if already done
    meta_file = TOKENIZED_DIR / "meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        print(f"[Skip] Already tokenized: {meta['total_tokens']:,} tokens")
        return meta

    # Combine all clean text files
    all_text = []
    for f in sorted(CLEAN_DIR.glob("*.txt")):
        print(f"  Loading {f.name}...")
        with open(f, encoding='utf-8') as fh:
            text = fh.read()
            all_text.append(text)
            print(f"    {len(text):,} chars")

    combined = '\n\n'.join(all_text)
    total_chars = len(combined)
    print(f"\nTotal combined: {total_chars:,} chars")

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
        # Split into chunks of ~1000 chars for tokenization
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        for chunk in chunks:
            tokens = sp.encode(chunk, out_type=int)
            all_tokens.extend([sp.bos_id()] + tokens + [sp.eos_id()])

    all_tokens = np.array(all_tokens, dtype=np.int32)
    total_tokens = len(all_tokens)

    # Save train/val split
    val_size = int(total_tokens * 0.05)
    train_tokens = all_tokens[:-val_size]
    val_tokens = all_tokens[-val_size:]

    np.save(TOKENIZED_DIR / "train.npy", train_tokens)
    np.save(TOKENIZED_DIR / "val.npy", val_tokens)

    # Save metadata
    meta = {
        "vocab_size": sp.get_piece_size(),
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "total_chars": total_chars,
        "sources": [f.name for f in CLEAN_DIR.glob("*.txt")],
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

    # Collect from all sources
    collect_wikipedia()
    collect_arxiv()
    collect_nasa()
    collect_huggingface()
    collect_openwebtext()
    collect_arxiv_api()

    # Combine and tokenize
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
