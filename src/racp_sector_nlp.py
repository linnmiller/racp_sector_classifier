#!/usr/bin/env python3
"""
RACP Sector Classifier — Chunk 0: Imports & Project Setup
- Imports core libraries
- Defines CONFIG with relative paths
- Sets up logging
- Checks environment and required folders/files
- Does NOT load or process data yet
"""

# ---- Imports ----
from __future__ import annotations
import sys
import json
import logging
from pathlib import Path

# Third-party packages (installed in your venv)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---- Configuration ----
CONFIG = {
    # relative paths (keep these as-is if you used the folder structure we discussed)
    "INPUT_EXCEL_PATH": "RACP Data/2023_RACP.xlsx",
    "PROTOTYPES_JSON_PATH": "prototypes/prototypes.json",
    "NONE_JSON_PATH": "prototypes/prototypes_none.json",

    # column names (adjust later if needed)
    "DESCRIPTION_COLUMN": "Project Description",

    # canonical sector keys (used later)
    "CANONICAL_SECTORS": [
        "agriculture",
        "life_sciences",
        "robotics_tech",
        "manufacturing",
        "energy",
    ],

    # output folder (created if missing)
    "OUTPUT_DIR": "outputs",
    "OUTPUT_EXCEL_PATH": "outputs/racp_sector_classified.xlsx",
}


# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)
log = logging.getLogger("racp_nlp_setup")


# ---- Friendly error helpers ----
class UserConfigError(Exception):
    """Raised when user-configurable inputs (paths, columns, JSON schema) are invalid."""
    pass


def hint_and_exit(hint: str, exc: Exception) -> None:
    """
    Log a short, actionable message and exit cleanly.
    Use this in a top-level try/except around setup or chunk calls.
    """
    logging.error("")  # spacer
    logging.error(f"❌ {hint}")
    logging.error(f"   Details: {exc.__class__.__name__}: {exc}")
    logging.error("   Tip: fix the issue above, then run again.")
    sys.exit(1)


# ---- Setup helpers (no NLP yet) ----
def check_environment() -> None:
    """Print basic environment info to help verify the venv/interpreter."""
    log.info("----- ENVIRONMENT -----")
    log.info(f"Python: {sys.version.split()[0]}")
    log.info(f"Interpreter: {sys.executable}")

    # Show versions of key packages (imported above)
    log.info(f"pandas: {pd.__version__}")

    # sklearn doesn’t expose a single __version__ in a namespace; we can print the class name as a quick check
    log.info(f"sklearn: TfidfVectorizer available -> {TfidfVectorizer}")
    log.info(f"cosine_similarity available -> {cosine_similarity}")
    log.info("-----------------------")


def ensure_directories() -> None:
    """Create required directories if they don’t exist."""
    out_dir = Path(CONFIG["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Ensured output directory: {out_dir.resolve()}")


def check_required_files() -> None:
    """
    Verify that the expected input files exist.
    We’re not opening them yet—just confirming presence to avoid path surprises.
    """
    root = Path.cwd()

    excel_path = root / CONFIG["INPUT_EXCEL_PATH"]
    proto_path = root / CONFIG["PROTOTYPES_JSON_PATH"]
    none_path = root / CONFIG["NONE_JSON_PATH"]

    missing = []
    for p in (excel_path, proto_path, none_path):
        if not p.exists():
            missing.append(str(p))

    log.info("----- REQUIRED FILES -----")
    log.info(f"Excel:      {excel_path} {'(OK)' if excel_path.exists() else '(MISSING)'}")
    log.info(f"Prototypes: {proto_path} {'(OK)' if proto_path.exists() else '(MISSING)'}")
    log.info(f"None JSON:  {none_path} {'(OK)' if none_path.exists() else '(MISSING)'}")
    log.info("--------------------------")

    if missing:
        raise FileNotFoundError(
            "The following required files are missing:\n  - " + "\n  - ".join(missing) +
            "\n\nTip: Check your folder names/paths and ensure files are placed under "
            "data/ and prototypes/ as configured."
        )

# -----------------------------
# Chunk 1: Load & Validate Inputs (Excel + JSON)
# -----------------------------
from typing import Dict, List, Any
from pathlib import Path
import json
import logging
import pandas as pd
import re

# Map common display labels to canonical sector keys used internally
DISPLAY_TO_CANON = {
    "agriculture": "agriculture",
    "Agriculture": "agriculture",

    "life_sciences": "life_sciences",
    "Life Sciences": "life_sciences",
    "LifeSciences": "life_sciences",

    "robotics_tech": "robotics_tech",
    "Robotics and Technology": "robotics_tech",
    "Robotics & Technology": "robotics_tech",
    "Robotics": "robotics_tech",
    "Technology": "robotics_tech",

    "manufacturing": "manufacturing",
    "Manufacturing": "manufacturing",

    "energy": "energy",
    "Energy": "energy",
}

def load_excel(path: str | Path, sheet_name: str | int | None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Excel file not found: {p}")

    df_or_dict = pd.read_excel(p, sheet_name=sheet_name)

    # If multiple sheets were requested (sheet_name=None), pandas returns a dict of DataFrames
    if isinstance(df_or_dict, dict):
        # Use a specific sheet if provided in CONFIG, otherwise first sheet
        if isinstance(sheet_name, (str, int)):
            # If sheet_name is a valid key/index, pandas would have returned a DF, not a dict.
            # So here we just fall back to first available in the dict.
            first_key = next(iter(df_or_dict))
            return df_or_dict[first_key]
        else:
            # No explicit sheet requested; pick the first one deterministically
            first_key = next(iter(df_or_dict))
            return df_or_dict[first_key]

    # Single sheet -> DataFrame
    return df_or_dict


def validate_dataset_columns(df: pd.DataFrame, required_col: str) -> None:
    if required_col not in df.columns:
        raise KeyError(
            f"Required column '{required_col}' not found in Excel.\n"
            f"Available columns (first 30): {list(df.columns)[:30]}"
        )
    # Optional: ensure there is at least one non-empty description
    if df[required_col].dropna().astype(str).str.len().sum() == 0:
        raise ValueError(
            f"Column '{required_col}' appears empty. Please confirm your Excel data."
        )


def load_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


import re  # <-- Add this near the top of your file if not already present

def normalize_sector_prototypes(obj: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Accept various shapes and names; return a dict:
      { 'agriculture': [prototype_phrase, ...], ... }

    Supported input shapes per sector:
      - {"Agriculture": {"prototypes": ["phrase", ...]}}
      - {"Agriculture": ["phrase", ...]}
      - {"Agriculture": "phrase 1, phrase 2; phrase 3"}  (we split on commas/semicolons)

    Unknown top-level keys are ignored with a warning.
    """
    # Initialize with empty lists for all canonical sectors
    out: Dict[str, List[str]] = {s: [] for s in CONFIG["CANONICAL_SECTORS"]}

    for k, v in obj.items():
        canon = DISPLAY_TO_CANON.get(k)
        if canon is None:
            logging.warning(f"Ignoring unknown top-level key in prototypes.json: '{k}'")
            continue

        # Case 1: dict containing 'prototypes' list
        if isinstance(v, dict) and isinstance(v.get("prototypes"), list):
            phrases = [str(x).strip() for x in v["prototypes"] if str(x).strip()]
            out[canon].extend(phrases)
            continue

        # Case 2: list of phrases directly
        if isinstance(v, list):
            phrases = [str(x).strip() for x in v if str(x).strip()]
            out[canon].extend(phrases)
            continue

        # Case 3: plain string of phrases -> split on commas/semicolons (keep multi-word phrases intact)
        if isinstance(v, str):
            parts = [p.strip() for p in re.split(r"[;,]", v) if p.strip()]
            out[canon].extend(parts)
            continue

        logging.warning(f"Unrecognized format for key '{k}' in prototypes.json; skipping.")

    # Deduplicate while preserving order (case-insensitive)
    for s, phrases in out.items():
        seen: set[str] = set()
        deduped: List[str] = []
        for phrase in phrases:
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(phrase)  # keep original casing for readability
        out[s] = deduped

    return out


def validate_sector_prototypes(proto: Dict[str, List[str]]) -> None:
    missing = [s for s in CONFIG["CANONICAL_SECTORS"] if s not in proto]
    if missing:
        raise ValueError(f"Missing sector keys after normalization: {missing}")

    for sector, phrases in proto.items():
        if not isinstance(phrases, list) or len(phrases) == 0:
            raise ValueError(f"Sector '{sector}' has no prototype phrases.")
        bad = [p for p in phrases if not isinstance(p, str) or not p.strip()]
        if bad:
            raise ValueError(f"Sector '{sector}' has invalid/empty phrases (example): {bad[:3]}")


def normalize_none_keywords(obj: Dict[str, Any]) -> List:
    """
    Expected:
      - { "keywords": [ "museum", "library", ... ] }
    Fallbacks:
      - { "Keywords": [ ... ] } or any list under a top-level key
      - "space/comma/semicolon separated string" -> split
    """
    if "keywords" in obj and isinstance(obj["keywords"], list):
        return [str(x).strip() for x in obj["keywords"] if str(x).strip()]

    # Try common variations or any list/string at top level
    for k, v in obj.items():
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            raw = v.replace(";", " ").replace(",", " ")
            return [t.strip() for t in raw.split() if t.strip()]

    raise ValueError("prototypes_none.json must contain a 'keywords' list (or a recognizable list/string of terms).")


def print_input_summary(df: pd.DataFrame,
                        sectors: Dict[str, List[str]],
                        none_keywords: List[str]) -> None:
    logging.info("----- INPUT SUMMARY -----")
    logging.info(f"Rows in Excel: {len(df)}")
    logging.info(f"Text column: '{CONFIG['DESCRIPTION_COLUMN']}'")
    for s in CONFIG["CANONICAL_SECTORS"]:
        logging.info(f"Sector '{s}': {len(sectors.get(s, []))} prototype phrases")
    logging.info(f"'none' keywords: {len(none_keywords)}")
    logging.info("-------------------------")


def run_chunk1() -> tuple[pd.DataFrame, Dict[str, List[str]], List[str]]:
    """
    Orchestrates the input loading and validation step.
    Returns:
      - df (DataFrame)
      - sectors (dict of sector -> list of prototype phrases)
      - none_keywords (list of strings)
    """
    logging.info("Chunk 1: Loading Excel and JSONs; normalizing & validating ...")

    # 1) Load Excel (use CONFIG['SHEET_NAME'] if present; else first sheet)
    df = load_excel(
        CONFIG["INPUT_EXCEL_PATH"],
        sheet_name=CONFIG.get("SHEET_NAME", 0)  # 0 = first sheet
    )

    # 2) Validate required column exists and has data
    validate_dataset_columns(df, CONFIG["DESCRIPTION_COLUMN"])

    # 3) Load sector prototypes JSON and normalize
    raw_prototypes = load_json(CONFIG["PROTOTYPES_JSON_PATH"])
    sectors = normalize_sector_prototypes(raw_prototypes)
    validate_sector_prototypes(sectors)

    # 4) Load 'none' keywords
    raw_none = load_json(CONFIG["NONE_JSON_PATH"])
    none_keywords = normalize_none_keywords(raw_none)
    if len(none_keywords) == 0:
        raise ValueError(
            "No 'none' keywords found in prototypes_none.json. "
            "Add at least a few terms/phrases."
        )

    # 5) Print summary
    print_input_summary(df, sectors, none_keywords)

    logging.info("Chunk 1 complete: Inputs loaded and validated.")

    # 6) IMPORTANT: return the three objects
    return df, sectors, none_keywords

# -----------------------------
# Chunk 2: Text cleaning + TF-IDF fit
# -----------------------------

from typing import Dict, List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Domain stop words (customizable; keeps non-informative words out of vectors)
DOMAIN_STOP_WORDS: set[str] = {
    "project", "grant", "community", "county", "phase", "initiative"
}

def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text: str) -> str:
    """
    Normalize text for vectorization:
      - lowercase
      - remove punctuation/symbols (keep a-z, 0-9, spaces)
      - collapse whitespace
      - remove English + domain stop words
    """
    if text is None:
        return ""
    t = str(text).lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)   # remove punctuation/symbols
    t = _normalize_whitespace(t)
    tokens = t.split()
    stop = ENGLISH_STOP_WORDS.union(DOMAIN_STOP_WORDS)
    tokens = [tok for tok in tokens if tok and tok not in stop]
    return " ".join(tokens)

def clean_list(items: List[str]) -> List:
    """Clean each phrase/keyword; drop empties post-cleaning."""
    out: List[str] = []
    for s in items:
        c = clean_text(s)
        if c:
            out.append(c)
    return out

def build_combined_corpus(
    df: pd.DataFrame,
    cleaned_sector_phrases: Dict[str, List[str]],
    cleaned_none_keywords: List[str]
) -> List:
    """
    Combined corpus used to fit TF-IDF:
      - all cleaned project descriptions
      - all cleaned sector prototype phrases (grouped later by sector)
      - all cleaned 'none' keywords/phrases
    """
    descs = df[CONFIG["DESCRIPTION_COLUMN"]].astype(str).apply(clean_text).tolist()

    combined: List[str] = []
    combined.extend(descs)
    for s in CONFIG["CANONICAL_SECTORS"]:
        combined.extend(cleaned_sector_phrases.get(s, []))
    combined.extend(cleaned_none_keywords)

    return combined

def fit_tfidf(corpus: List[str]) -> TfidfVectorizer:
    """
    Fit TF-IDF vectorizer on the combined corpus.
    Defaults:
      - stop_words='english' (we also removed stop-words in cleaning)
      - ngram_range=(1,2) (unigrams + bigrams)
      - max_df=0.95 (drop terms occurring in >95% of docs)
      - min_df=1 (keep rare terms initially)
      - norm='l2'
    """
    if not corpus or all(len(doc.strip()) == 0 for doc in corpus):
        raise ValueError("Combined corpus is empty after cleaning. Check inputs and stop-word settings.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
        norm="l2"
    )
    vectorizer.fit(corpus)
    return vectorizer

def run_chunk2(
    df: pd.DataFrame,
    sectors: Dict[str, List[str]],
    none_keywords: List[str]
) -> tuple[TfidfVectorizer, Dict[str, List[str]], List[str]]:
    """
    Orchestrates text cleaning and TF-IDF fitting.
    Returns:
      - vectorizer (fitted on combined corpus)
      - cleaned_sector_phrases (dict per sector)
      - cleaned_none_keywords (list)
    Also stores df['clean_description'] for later chunks.
    """
    logging.info("Chunk 2: Cleaning text and fitting TF-IDF ...")

    # Clean prototypes and 'none' keywords
    cleaned_sector_phrases: Dict[str, List[str]] = {
        s: clean_list(sectors.get(s, [])) for s in CONFIG["CANONICAL_SECTORS"]
    }
    cleaned_none_keywords: List[str] = clean_list(none_keywords)

    # Clean descriptions and store in the DataFrame for later use
    df["clean_description"] = df[CONFIG["DESCRIPTION_COLUMN"]].astype(str).apply(clean_text)

    # Build combined corpus and fit TF-IDF
    combined_corpus = build_combined_corpus(df, cleaned_sector_phrases, cleaned_none_keywords)
    vectorizer = fit_tfidf(combined_corpus)

    # Quick summary + preview
    vocab_size = len(vectorizer.vocabulary_)
    logging.info(f"TF-IDF fitted. Vocabulary size={vocab_size}, ngram_range=(1,2), max_df=0.95, min_df=1")
    logging.info(f"Sample cleaned description: {df['clean_description'].head(1).tolist()}")

    logging.info("Chunk 2 complete: vectorizer is ready.")
    return vectorizer, cleaned_sector_phrases, cleaned_none_keywords

# def verify_tfidf(
#     vectorizer: TfidfVectorizer,
#     df: pd.DataFrame,
#     cleaned_sector_phrases: Dict[str, List[str]],
#     cleaned_none_keywords: List[str]
# ) -> None:
#     """
#     Lightweight checks to confirm TF-IDF is fitted and behaves as expected.
#     Prints:
#       - feature count and a small sample of feature names
#       - IDF statistics (min/max/mean)
#       - non-zero vector checks for a few descriptions
#       - top-weighted terms in those descriptions
#       - non-zero vectors for a sample sector phrase and a none keyword
#       - stop-word-only doc should produce zero features
#     """
#     import numpy as np

#     logging.info("===== TF-IDF DIAGNOSTICS =====")

#     # 1) Feature space sanity
#     features = vectorizer.get_feature_names_out()
#     logging.info(f"Feature count: {len(features)}")
#     logging.info(f"First 20 features: {features[:20].tolist()}")

#     # 2) IDF statistics (should be positive; rarer terms have higher IDF)
#     idf = vectorizer.idf_
#     logging.info(f"IDF stats: min={idf.min():.3f}, mean={idf.mean():.3f}, max={idf.max():.3f}")

#     # 3) Transform a few cleaned descriptions; check for non-empty vectors
#     sample_descs = df["clean_description"].head(3).tolist()
#     X = vectorizer.transform(sample_descs)
#     nnz_per_row = [int(X[i].nnz) for i in range(X.shape[0])]
#     logging.info(f"Sample description matrix shape: {X.shape} | nnz per row: {nnz_per_row}")
#     for i in range(X.shape[0]):
#         row = X[i]
#         if row.nnz == 0:
#             logging.warning(f"clean_description[{i}] produced an EMPTY TF-IDF vector; check cleaning/stop-words.")
#             continue
#         # Show top 5 weighted terms for readability
#         indices = row.indices
#         data = row.data
#         top_idx = np.argsort(data)[-5:][::-1]  # indices of top weights
#         top_terms = [(features[indices[j]], float(data[j])) for j in top_idx]
#         logging.info(f"Top terms for clean_description[{i}]: {top_terms}")

#     # 4) Sector phrase sanity: pick the first available phrase from any sector
#     sample_sector_phrase = None
#     for s in CONFIG["CANONICAL_SECTORS"]:
#         if cleaned_sector_phrases.get(s):
#             sample_sector_phrase = cleaned_sector_phrases[s][0]
#             break
#     if sample_sector_phrase:
#         v = vectorizer.transform([sample_sector_phrase])
#         logging.info(f"Sector sample '{sample_sector_phrase}' -> nnz={v.nnz} (should be > 0 if terms are in vocab)")

#     # 5) None keyword sanity
#     if cleaned_none_keywords:
#         v_none = vectorizer.transform([cleaned_none_keywords[0]])
#         logging.info(f"None keyword sample '{cleaned_none_keywords[0]}' -> nnz={v_none.nnz}")

#     # 6) Stop-word-only doc should be empty (due to cleaning + stop_words='english')
#     v_stop = vectorizer.transform(["the and of for is with"])
#     logging.info(f"Stop-word-only doc nnz={v_stop.nnz} (expected 0)")

#     logging.info("===== END TF-IDF DIAGNOSTICS =====")

def main():
    # Setup (from Chunk 0)
    check_environment()
    ensure_directories()
    check_required_files()

    # --- Call Chunk 1 ---
    df, sectors, none_keywords = run_chunk1()

    # --- Call Chunk 2 ---
    vectorizer, cleaned_sector_phrases, cleaned_none_keywords = run_chunk2(df, sectors, none_keywords)


    # # --- TEMP: verify TF-IDF is working ---
    # verify_tfidf(vectorizer, df, cleaned_sector_phrases, cleaned_none_keywords)

    # Stop here until you say to proceed
    logging.info("Diagnostics done. Ready for Chunk 3 when you are.")


if __name__ == "__main__":
    main()