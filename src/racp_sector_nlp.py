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
    "DESCRIPTION_COLUMN": "project_description",

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
    Log a concise, actionable message and exit with non-zero status.
    Use this to convert common exceptions into human-readable guidance.
    """
    logging.error("")  # spacer line
    logging.error("❌ " + hint)
    logging.error(f"   Details: {exc.__class__.__name__}: {exc}")
    logging.error("   Tip: Fix the issue above, then run again.")
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


def main():
    try:
        # Setup-only actions (no data processing yet)
        check_environment()
        ensure_directories()
        check_required_files()

    except FileNotFoundError as e:
        # Missing Excel or JSON files
        hint_and_exit(
            "A required input file is missing. Ensure your Excel and JSON files "
            "exist at the paths shown above (data/ and prototypes/), or update CONFIG.",
            e
        )

    except UserConfigError as e:
        # For future: if you raise UserConfigError from validators (e.g., JSON schema)
        hint_and_exit(
            "Your configuration or input schema appears invalid. Check CONFIG values "
            "and JSON structure (e.g., 'prototypes' for sectors, 'keywords' for none).",
            e
        )

    except KeyError as e:
        # For future: missing columns in Excel (after we add Chunk 1)
        hint_and_exit(
            "A required column is missing in the Excel file. Verify CONFIG.DESCRIPTION_COLUMN "
            "matches your sheet's header exactly.",
            e
        )

    except ValueError as e:
        # For future: empty lists, empty corpus, invalid thresholds, etc.
        hint_and_exit(
            "An input value or derived data is invalid (e.g., empty list, empty corpus). "
            "Please review your inputs and JSON content.",
            e
        )

    except Exception as e:
        # Catch-all: unexpected errors. Keep concise, but signal it's unanticipated.
        hint_and_exit(
            "Unexpected error during setup. If this persists, try re-activating the virtual "
            "environment and re-running. You can also share the error text for help.",
            e
        )

    # If we got here, setup succeeded
    log.info("Setup complete ✅  Ready for next chunk (loading & validation) when you say go.")


if __name__ == "__main__":
    main()