"""
load_instruct_dset.py
---------------------
Downloads the SQuAD dataset from HuggingFace, converts each example into
instruction/response pairs, cleans & verifies the data, and writes the final
dataset to data/instruction_dataset.jsonl.

Output format (one JSON object per line):
{
  "instruction": "<question>",
  "response": "<answer>"
}
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # utils/
REPO_ROOT  = SCRIPT_DIR.parent                        # sft/
OUTPUT_DIR = REPO_ROOT / "data"
OUTPUT_FILE = OUTPUT_DIR / "instruction_dataset.jsonl"

# ---------------------------------------------------------------------------
# Constants / thresholds
# ---------------------------------------------------------------------------
MIN_INSTRUCTION_LEN = 10   # characters
MAX_INSTRUCTION_LEN = 512
MIN_RESPONSE_LEN    = 5
MAX_RESPONSE_LEN    = 1024

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _install_if_missing(package: str) -> None:
    """Attempt to pip-install a package if it cannot be imported."""
    import importlib
    try:
        importlib.import_module(package.replace("-", "_"))
    except ImportError:
        log.info("Package '%s' not found – installing …", package)
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space and strip ends."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_valid_example(instruction: str, response: str) -> tuple[bool, str]:
    """
    Return (True, '') when the example passes all quality checks,
    otherwise (False, <reason>).
    """
    # --- instruction checks ---
    if not instruction:
        return False, "empty instruction"
    if len(instruction) < MIN_INSTRUCTION_LEN:
        return False, f"instruction too short ({len(instruction)} chars)"
    if len(instruction) > MAX_INSTRUCTION_LEN:
        return False, f"instruction too long ({len(instruction)} chars)"
    if not instruction.strip().endswith("?"):
        # SQuAD questions should always end with '?'; skip if malformed
        # (allow '?' anywhere as a loose check)
        if "?" not in instruction:
            return False, "instruction does not contain a question mark"

    # --- response checks ---
    if not response:
        return False, "empty response"
    if len(response) < MIN_RESPONSE_LEN:
        return False, f"response too short ({len(response)} chars)"
    if len(response) > MAX_RESPONSE_LEN:
        return False, f"response too long ({len(response)} chars)"

    # --- duplicate / junk patterns ---
    if instruction.lower() == response.lower():
        return False, "instruction and response are identical"

    return True, ""


def build_example(raw_question: str, raw_answer: str) -> dict | None:
    """
    Clean and validate a single question/answer pair.
    Returns a dict on success, None if the example should be discarded.
    """
    instruction = normalize_whitespace(raw_question)
    response    = normalize_whitespace(raw_answer)

    ok, reason = is_valid_example(instruction, response)
    if not ok:
        return None

    return {"instruction": instruction, "response": response}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_and_process() -> None:
    # -- ensure dependencies --
    for pkg in ["datasets"]:
        _install_if_missing(pkg)

    from datasets import load_dataset  # noqa: PLC0415

    # -- download --
    log.info("Downloading rajpurkar/squad from HuggingFace …")
    dataset = load_dataset("rajpurkar/squad")
    log.info("Splits available: %s", list(dataset.keys()))

    # -- prepare output directory --
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -- process all splits --
    total_raw      = 0
    total_kept     = 0
    total_skipped  = 0
    seen_keys: set[str] = set()   # deduplication

    with OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for split_name, split_data in dataset.items():
            log.info("Processing split '%s' (%d rows) …", split_name, len(split_data))

            for row in split_data:
                total_raw += 1

                question = row.get("question", "")

                # SQuAD stores answers as {"text": [...], "answer_start": [...]}
                answers_field = row.get("answers", {})
                answer_texts  = answers_field.get("text", [])

                if not answer_texts:
                    total_skipped += 1
                    continue

                # Use the first (typically the most canonical) answer
                answer = answer_texts[0]

                example = build_example(question, answer)
                if example is None:
                    total_skipped += 1
                    continue

                # -- deduplication on (instruction, response) --
                dedup_key = (example["instruction"].lower(), example["response"].lower())
                if dedup_key in seen_keys:
                    total_skipped += 1
                    continue
                seen_keys.add(dedup_key)

                fout.write(json.dumps(example, ensure_ascii=False) + "\n")
                total_kept += 1

    # -- summary --
    log.info("─" * 60)
    log.info("Total raw examples  : %d", total_raw)
    log.info("Kept (clean)        : %d", total_kept)
    log.info("Skipped / filtered  : %d", total_skipped)
    log.info("Output file         : %s", OUTPUT_FILE)
    log.info("─" * 60)

    # -- spot-check: print first 3 records --
    log.info("Sample records from output file:")
    with OUTPUT_FILE.open("r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if i >= 3:
                break
            record = json.loads(line)
            log.info(
                "  [%d] instruction: %s …\n       response   : %s …",
                i + 1,
                record["instruction"][:80],
                record["response"][:80],
            )

    # -- final verification pass --
    _verify_output_file()


def _verify_output_file() -> None:
    """
    Read back every line in the output file and assert structural integrity.
    Raises RuntimeError if any record is malformed.
    """
    log.info("Running final verification pass …")
    errors = 0
    with OUTPUT_FILE.open("r", encoding="utf-8") as fin:
        for lineno, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                log.error("Line %d: invalid JSON – %s", lineno, exc)
                errors += 1
                continue

            if "instruction" not in record or "response" not in record:
                log.error("Line %d: missing required keys – %s", lineno, list(record.keys()))
                errors += 1
                continue

            if not isinstance(record["instruction"], str) or not isinstance(record["response"], str):
                log.error("Line %d: values must be strings", lineno)
                errors += 1

    if errors:
        raise RuntimeError(f"Verification failed with {errors} error(s). Check the log above.")

    log.info("Verification passed – all records are well-formed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_and_process()
