"""
text_forensics/manual_test.py

Manual Test Interface for the Text Forensics Pipeline.

Runs the real analyze_text() pipeline on user-supplied input. The input
is never generated, modified, or substituted by this script — it is passed
through to analyze_text() exactly as provided.

Usage:
    # Mode 1: paste text directly
    python manual_test.py --text "paste any paragraph here"

    # Mode 2: read from a plain .txt file
    python manual_test.py --file path/to/document.txt

    # Optional: suppress robustness test (faster, for quick signal checks)
    python manual_test.py --text "..." --no-robustness

Output (printed to terminal):
    - Final text score (0-100, higher = more AI-like)
    - Each of the 4 signal sub-scores, labeled
    - Stability flag and paraphrase delta
    - Total time taken
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure text_forensics package is importable when run from any working directory
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJ_ROOT  = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJ_ROOT))


def _print_divider(char: str = "-", width: int = 60) -> None:
    print(char * width)


def _fmt(val: object, decimals: int = 2) -> str:
    """Format a float or None for display."""
    if val is None:
        return "N/A (signal not applicable for this text)"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def run_analysis(text: str) -> None:
    """
    Call analyze_text() on the provided text and print a formatted report.

    Args:
        text: The exact text to analyze — not modified in any way.
    """
    from text_forensics.pipeline import analyze_text

    word_count = len(text.split())
    char_count  = len(text)

    print()
    _print_divider("=")
    print("  Text Forensics Pipeline -- Manual Test")
    _print_divider("=")
    print(f"  Input: {word_count} words, {char_count} characters")
    print(f"  Preview: {text[:120].strip()}{'...' if len(text) > 120 else ''}")
    _print_divider()

    t_start = time.time()

    try:
        result = analyze_text(text)
    except (TypeError, ValueError) as e:
        print(f"\n  [ERROR] Input error: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n  [ERROR] Pipeline error: {type(e).__name__}: {e}\n")
        raise

    elapsed = time.time() - t_start
    sigs = result.get("signals", {})

    print()
    _print_divider("=")
    print("  Text Forensics Pipeline -- Manual Test")
    _print_divider("=")
    print(f"  FINAL TEXT SCORE:  {_fmt(result['text_score'])} / 100")
    print(f"  (Higher = more AI-like. Threshold: >70 likely AI, <50 likely human)")
    print()
    _print_divider()
    print("  SIGNAL BREAKDOWN")
    _print_divider()
    print(f"  {'Curvature (Fast-DetectGPT):':<40} raw={_fmt(sigs.get('curvature_raw'), 4)}   score={_fmt(sigs.get('curvature_score'))}/100")
    print(f"  {'Burstiness (sentence rhythm):':<40} raw={_fmt(sigs.get('burstiness_raw'), 4)}   score={_fmt(sigs.get('burstiness_score'))}/100")
    print(f"  {'Cliche density:':<40} {_fmt(sigs.get('cliche_density_pct'))}%   score={_fmt(sigs.get('cliche_score'))}/100")
    print(f"  {'Lexical entropy (Shannon bits):':<40} {_fmt(sigs.get('entropy'), 3)} bits")
    print(f"  {'Type-token ratio:':<40} {_fmt(sigs.get('ttr'), 4)}")
    print(f"  {'Entropy score:':<40} {_fmt(sigs.get('entropy_score'))}/100")
    print()
    _print_divider()
    print("  ROBUSTNESS CHECK")
    _print_divider()
    print(f"  Stability flag:        {result.get('stability_flag', 'N/A')}")
    print(f"  Paraphrase delta:      {_fmt(result.get('paraphrase_delta'))} points")
    trunc = result.get('compared_on_truncated', False)
    print(f"  Compared on truncated: {'Yes (input >300 words)' if trunc else 'No (full text used)'}")
    print()
    _print_divider()
    print(f"  Total time: {elapsed:.1f}s")
    _print_divider("=")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="manual_test.py",
        description=(
            "Run the Text Forensics Pipeline on your own text. "
            "Input is passed through unmodified to analyze_text()."
        ),
    )

    # --- Input source (mutually exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        metavar="TEXT",
        type=str,
        help="Text to analyze (paste directly as a quoted string).",
    )
    input_group.add_argument(
        "--file",
        metavar="PATH",
        type=str,
        help="Path to a plain .txt file to read and analyze.",
    )

    args = parser.parse_args()

    # --- Load input ---
    if args.text is not None:
        # Direct paste mode
        text = args.text
        if not text or not text.strip():
            print("Error: --text argument is empty. Please provide actual text content.")
            sys.exit(1)

    else:
        # File mode
        file_path = Path(args.file)

        if not file_path.exists():
            print(f"Error: file not found: {file_path}")
            sys.exit(1)

        if not file_path.is_file():
            print(f"Error: path is not a file: {file_path}")
            sys.exit(1)

        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = file_path.read_text(encoding="latin-1")
                print("Note: file read with latin-1 encoding (not UTF-8).")
            except Exception as e:
                print(f"Error: could not read file: {e}")
                sys.exit(1)

        if not text or not text.strip():
            print(f"Error: file is empty or contains only whitespace: {file_path}")
            sys.exit(1)

    # --- Run analysis ---
    run_analysis(text)


if __name__ == "__main__":
    main()
