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
import numpy as np

# Third-party packages (installed in your venv)
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---- Configuration ----
CONFIG = {

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
        ngram_range=(1, 2),   # you can try (1, 3) later
        max_df=0.99,          # was 0.95; drop only extremely ubiquitous terms
        min_df=1,             # keep rares initially
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

# -----------------------------
# Chunk 3: Vectorize & Cosine Similarity (scores)
# -----------------------------
from typing import Dict, List

def _vectorize_list(vectorizer: TfidfVectorizer, items: List[str]) -> "scipy.sparse.csr_matrix":
    """
    Transform a list of cleaned strings into TF-IDF vectors.
    Returns a sparse matrix of shape (len(items), n_features).
    If items is empty, returns an empty 0 x n_features matrix (handled upstream).
    """
    if not items:
        # Create an empty matrix with the right number of columns to avoid shape errors
        # We infer n_features from the vectorizer vocabulary
        n_features = len(vectorizer.vocabulary_)
        from scipy.sparse import csr_matrix
        return csr_matrix((0, n_features))
    return vectorizer.transform(items)


def _aggregate_similarity(sim_mat: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Aggregate similarities across prototypes for each project.
    sim_mat shape: (n_projects, n_prototypes)
    Returns: (n_projects,) aggregated scores
    """
    if sim_mat.size == 0:
        # No prototypes for this bucket -> zeros for all projects
        return np.zeros((sim_mat.shape[0],), dtype=float)
    if method == "max":
        return sim_mat.max(axis=1)
    # default: mean
    return sim_mat.mean(axis=1)


def run_chunk3(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    cleaned_sector_phrases: Dict[str, List[str]],
    cleaned_none_keywords: List[str],
    agg_method: str = "mean"  # "mean" | "max"
) -> pd.DataFrame:
    """
    Compute cosine-similarity scores for each project against:
      - each sector's prototype phrases
      - 'none' keywords/phrases

    Returns:
      scores_df: DataFrame indexed like df, with columns:
        ['agriculture', 'life_sciences', 'robotics_tech', 'manufacturing', 'energy', 'none']
    """
    logging.info("Chunk 3: Vectorizing descriptions and prototypes; computing cosine similarities ...")

    # 1) Vectorize project descriptions (cleaned in Chunk 2)
    descs = df["clean_description"].astype(str).tolist()
    project_matrix = vectorizer.transform(descs)  # shape (n_projects, n_features)

    # 2) Vectorize each sector's prototypes
    sector_mats: Dict[str, "scipy.sparse.csr_matrix"] = {}
    for s in CONFIG["CANONICAL_SECTORS"]:
        phrases = cleaned_sector_phrases.get(s, [])
        mat = _vectorize_list(vectorizer, phrases)  # shape (n_prototypes_s, n_features)
        if mat.shape[0] == 0:
            logging.warning(f"Sector '{s}' has no cleaned prototype phrases; scores will be zeros.")
        sector_mats[s] = mat

    # 3) Vectorize 'none' keywords/phrases
    none_mat = _vectorize_list(vectorizer, cleaned_none_keywords)
    if none_mat.shape[0] == 0:
        logging.warning("'none' keywords list is empty after cleaning; 'none' scores will be zeros.")

    # 4) Compute cosine similarity per sector (aggregate across prototypes)
    scores = {}
    for s in CONFIG["CANONICAL_SECTORS"]:
        proto_mat = sector_mats[s]  # (n_proto_s, n_features)
        # cosine(project_matrix, proto_mat) -> (n_projects, n_proto_s)
        try:
            sim = cosine_similarity(project_matrix, proto_mat)
        except ValueError as e:
            # If shapes are incompatible or proto_mat is empty, handle gracefully
            logging.warning(f"Cosine similarity failed for sector '{s}': {e}. Using zeros.")
            sim = np.zeros((project_matrix.shape[0], 0), dtype=float)
        scores[s] = _aggregate_similarity(sim, method=agg_method)

    # 5) Compute 'none' similarity (aggregate)
    try:
        sim_none = cosine_similarity(project_matrix, none_mat)  # (n_projects, n_none)
    except ValueError as e:
        logging.warning(f"Cosine similarity failed for 'none' bucket: {e}. Using zeros.")
        sim_none = np.zeros((project_matrix.shape[0], 0), dtype=float)
    scores["none"] = _aggregate_similarity(sim_none, method=agg_method)

    # 6) Assemble scores into a DataFrame aligned to df.index
    scores_df = pd.DataFrame(scores, index=df.index)

    # 7) Quick preview
    logging.info("Chunk 3 complete: similarity scores computed.")
    logging.info(f"Scores columns: {list(scores_df.columns)}")
    logging.info(f"Sample scores (first row): {scores_df.head(1).to_dict(orient='records')}")

    return scores_df

# -----------------------------
# Chunk 4: Classification + Reasoning + Excel write
# -----------------------------
from typing import Dict, List, Optional
from pathlib import Path

def _format_score(v: float) -> str:
    try:
        return f"{float(v):.3f}"
    except Exception:
        return "NA"

def classify_with_reasoning(
    row_idx: int,
    scores_row: Dict[str, float],
    sector_keys: List[str],
    thresholds: Dict[str, float]
) -> tuple[str, str]:
    """
    Decide sector label for one project and return (label, reasoning_text).
    Rules:
      1) Pick sector with highest similarity.
      2) If best sector score < MIN_POSITIVE_REQUIRED -> 'none'.
      3) Else if none_score >= best_sector_score - NONE_OVERRIDE_DELTA -> 'none'.
      4) Otherwise -> best sector.

    thresholds example:
      {"MIN_POSITIVE_REQUIRED": 0.22, "NONE_OVERRIDE_DELTA": 0.02}
    """
    # Pull scores for all sectors and none
    sector_scores = {s: float(scores_row.get(s, 0.0)) for s in sector_keys}
    none_score = float(scores_row.get("none", 0.0))

    # Determine best sector
    best_sector = max(sector_scores, key=sector_scores.get)
    best_score = sector_scores[best_sector]

    # Threshold checks
    min_pos = float(thresholds.get("MIN_POSITIVE_REQUIRED", 0.8))
    none_delta = float(thresholds.get("NONE_OVERRIDE_DELTA", -0.01))

    # Case A: weak sector evidence -> none
    if best_score < min_pos:
        reason = (
            f"[row={row_idx}] Classified as none because best sector '{best_sector}' "
            f"similarity { _format_score(best_score) } < minimum required { _format_score(min_pos) }."
            f" none={ _format_score(none_score) }; "
            f"sectors: " +
            ", ".join([f"{s}={_format_score(sector_scores[s])}" for s in sector_keys])
        )
        return "none", reason

    # Case B: none tie-break override
    # has to do with how close you want to decide between none and best_score; can set up this threshold to be any gap
    if none_score >= (best_score - none_delta):
        reason = (
            f"[row={row_idx}] Classified as none because none similarity "
            f"{ _format_score(none_score) } >= best sector '{best_sector}' "
            f"{ _format_score(best_score) } - delta { _format_score(none_delta) }."
            f" sectors: " +
            ", ".join([f"{s}={_format_score(sector_scores[s])}" for s in sector_keys])
        )
        return "none", reason

    # Case C: assign best sector
    reason = (
        f"[row={row_idx}] Classified as {best_sector} because {best_sector} "
        f"similarity { _format_score(best_score) } is highest. none={ _format_score(none_score) }; "
        f"other sector sims: " +
        ", ".join([f"{s}={_format_score(sector_scores[s])}" for s in sector_keys if s != best_sector])
    )
    return best_sector, reason
# --- Helper: extract top contributing terms for the winning sector match ---
def _extract_top_terms_for_sector(
    vectorizer: TfidfVectorizer,
    project_text: str,
    sector_phrases_clean: List[str],
    top_k: int = 6
) -> List[tuple[str, float]]:
    """
    For the given project text and the sector's cleaned prototype phrases:
      1) Vectorize project and all sector phrases
      2) Find the best-matching sector phrase by cosine similarity
      3) Compute contributions: product of TF-IDF weights on shared features
      4) Return top_k terms with their contribution weights (descending)

    Notes:
      - Uses cleaned text (you stored df['clean_description'] in Chunk 2)
      - Contributions are approximate but intuitive: proj_w * proto_w for shared features
    """
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Guard: no phrases -> nothing to show
    if not sector_phrases_clean:
        return []

    # Vectorize the single project (clean) and all sector phrases (clean)
    proj_vec = vectorizer.transform([project_text])         # (1, n_features)
    proto_mat = vectorizer.transform(sector_phrases_clean)  # (n_proto, n_features)

    # If either side is empty after vectorization, bail early
    if proj_vec.nnz == 0 or proto_mat.shape[0] == 0:
        return []

    # Pick the best matching prototype by cosine sim
    sims = cosine_similarity(proj_vec, proto_mat)           # (1, n_proto)
    j_best = int(np.argmax(sims[0]))
    best_proto_vec = proto_mat[j_best]                      # (1, n_features)

    # Shared feature indices
    shared_idx = np.intersect1d(proj_vec.indices, best_proto_vec.indices)
    if shared_idx.size == 0:
        return []

    # Contribution = TF-IDF weight product on shared features
    # Get per-feature weights efficiently
    proj_data = {idx: w for idx, w in zip(proj_vec.indices, proj_vec.data)}
    proto_data = {idx: w for idx, w in zip(best_proto_vec.indices, best_proto_vec.data)}

    contrib = []
    features = vectorizer.get_feature_names_out()
    for idx in shared_idx:
        w = proj_data.get(idx, 0.0) * proto_data.get(idx, 0.0)
        if w > 0:
            contrib.append((features[idx], float(w)))

    # Sort by contribution descending and take top_k
    contrib.sort(key=lambda x: x[1], reverse=True)
    return contrib[:top_k]

def run_chunk4(
    df: pd.DataFrame,
    scores_df: pd.DataFrame,
    output_path: str,
    thresholds: Optional[Dict[str, float]] = None,
    *,
    vectorizer: Optional[TfidfVectorizer] = None,
    cleaned_sector_phrases: Optional[Dict[str, List[str]]] = None,
    include_top_terms: bool = True,
    top_k_terms: int = 6
) -> pd.DataFrame:
    """
    Finalize results:
      - Classify each project into a sector (or 'none')
      - Create a human-readable reasoning column with key scores and decision path
      - Optionally include top contributing words for the winning sector
      - Write the Excel to `output_path`
    """
    logging.info("Chunk 4: Classifying projects, generating reasoning, and writing Excel ...")

    thresholds = thresholds or {
        "MIN_POSITIVE_REQUIRED": 0.12,
        "NONE_OVERRIDE_DELTA": -.01,
    }

    sector_keys = CONFIG["CANONICAL_SECTORS"]
    for col in sector_keys + ["none"]:
        if col not in scores_df.columns:
            logging.warning(f"Scores missing column '{col}'; substituting zeros.")
            scores_df[col] = 0.0

    scores_df = scores_df.loc[df.index]
    df_out = df.copy()

    labels: List[str] = []
    reasons: List[str] = []

    # Classify each row
    labels: List[str] = []
    reasons: List[str] = []
    top_terms_col: List[str] = []  # <-- NEW: collect terms per row

    for i in df_out.index:
        scores_row = scores_df.loc[i].to_dict()
        label, reason = classify_with_reasoning(
            row_idx=int(i) if isinstance(i, (int, np.integer)) else i,
            scores_row=scores_row,
            sector_keys=sector_keys,
            thresholds=thresholds
        )

        # Default when no terms are available
        row_top_terms = ""

        # If a sector won and user wants top terms, compute & store them
        if include_top_terms and label != "none" and vectorizer is not None and cleaned_sector_phrases is not None:
            project_clean = str(df_out.loc[i, "clean_description"])
            phrases_clean = cleaned_sector_phrases.get(label, [])
            top_terms = _extract_top_terms_for_sector(
                vectorizer=vectorizer,
                project_text=project_clean,
                sector_phrases_clean=phrases_clean,
                top_k=top_k_terms
            )
            if top_terms:
                # Format the terms (omit numeric weights for readability)
                row_top_terms = ", ".join([t for t, _w in top_terms])
                # Optionally append terms into the reasoning text (keep or remove as you prefer)
                reason = reason + f" Top terms for '{label}': {row_top_terms}."

        labels.append(label)
        reasons.append(reason)
        top_terms_col.append(row_top_terms)

    # Add columns to output
    df_out["sector"] = labels
    df_out["sector_reasoning"] = reasons
    df_out["sector_top_terms"] = top_terms_col  # <-- NEW COLUMN

    counts = df_out["sector"].value_counts(dropna=False).to_dict()
    logging.info(f"Sector counts: {counts}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df_out.to_excel(out_path, index=False)
        logging.info(f"✅ Wrote results to: {out_path.resolve()}")
    except Exception as e:
        logging.error(f"Failed to write Excel to '{out_path}': {e}")
        raise

    logging.info("Chunk 4 complete.")
    return df_out

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

    # --- Call Chunk 3 ---
    
    # After
    scores_df = run_chunk3(df, vectorizer, cleaned_sector_phrases, cleaned_none_keywords, agg_method="max")

    best_sector = scores_df[CONFIG["CANONICAL_SECTORS"]].max(axis=1)
    logging.info(
        f"Best sector score stats: min={best_sector.min():.3f}, "
        f"mean={best_sector.mean():.3f}, median={best_sector.median():.3f}, max={best_sector.max():.3f}"
    )
    logging.info(
        f"'none' score stats: min={scores_df['none'].min():.3f}, "
        f"mean={scores_df['none'].mean():.3f}, median={scores_df['none'].median():.3f}, "
        f"max={scores_df['none'].max():.3f}"
)
    
    # --- Call Chunk 4 ---
    df_final = run_chunk4(
        df=df,
        scores_df=scores_df,
        output_path=CONFIG["OUTPUT_EXCEL_PATH"],
        thresholds={"MIN_POSITIVE_REQUIRED": 0.05, "NONE_OVERRIDE_DELTA": -0.01},  # tune as needed
        vectorizer=vectorizer,
        cleaned_sector_phrases=cleaned_sector_phrases,
        include_top_terms=True,
        top_k_terms=6
    )
    logging.info("Workflow complete. You can open the output Excel now.")


if __name__ == "__main__":
    main()


