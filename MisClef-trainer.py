"""MisClef-trainer.py
Hyperparameter optimisation loop for MisClef.

Strategy
--------
1. Baseline   – evaluate the current MisClef.py parameters.
2. Random     – evaluate N_RANDOM random parameter combinations.
3. Hill climb – starting from the best point found so far, adjust one
                parameter at a time (±step) and keep improvements.
                Repeat for N_HILL_ROUNDS full passes over all parameters.
4. Write-back – patch the winning values into MisClef.py and regenerate
                the annotated PDF.

Tunable parameters
------------------
All seven live as module-level constants in MisClef.py so this script can
monkey-patch them in-process without touching any files during the search.

  HOUGH_PARAM1    – Canny high threshold              (int,   default 50)
  HOUGH_PARAM2    – Hough accumulator threshold        (int,   default 9)
  MIN_R_FACTOR    – min-radius / staff-spacing         (float, default 0.27)
  MAX_R_FACTOR    – max-radius / staff-spacing         (float, default 0.58)
  MIN_DIST_FACTOR – min centre distance / sp           (float, default 0.55)
  DENSITY_MIN     – ink-density floor                  (float, default 0.12)
  CIRCULARITY_MIN – circularity floor (4π·A/P²)        (float, default 0.40)

Usage
-----
  python MisClef-trainer.py [--random N] [--rounds N] [--seed N]

Optional flags
  --random N   number of random-search trials  (default 12)
  --rounds N   number of hill-climbing rounds  (default 3)
  --seed   N   random seed                     (default 42)
"""

import argparse
import importlib.util
import random
import re
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import fitz  # PyMuPDF

# ── Bootstrap ─────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import MisClef as mc  # noqa: E402  (sys.path modified above)

# Load MisClef-checker without triggering a second MisClef import.
# Because 'MisClef' is already in sys.modules, the checker's own
#   import MisClef as mc
# will reuse the same object, so attribute patches propagate automatically.
_spec = importlib.util.spec_from_file_location(
    "checker", _HERE / "MisClef-checker.py"
)
checker = importlib.util.module_from_spec(_spec)
sys.modules["checker"] = checker
_spec.loader.exec_module(checker)

# ── Parameter space ───────────────────────────────────────────────────────────
# Format: name → (lo, hi, type, hill_step)

PARAM_SPACE: dict[str, tuple] = {
    "HOUGH_PARAM1":    (30,   70,   int,   5),
    "HOUGH_PARAM2":    (6,    16,   int,   1),
    "MIN_R_FACTOR":    (0.18, 0.36, float, 0.02),
    "MAX_R_FACTOR":    (0.45, 0.72, float, 0.03),
    "MIN_DIST_FACTOR": (0.38, 0.80, float, 0.05),
    "DENSITY_MIN":     (0.05, 0.25, float, 0.02),
    "CIRCULARITY_MIN": (0.25, 0.60, float, 0.05),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_params() -> dict:
    """Read the current values from the mc module."""
    return {k: getattr(mc, k) for k in PARAM_SPACE}


def _clip(value, lo, hi, typ):
    if typ is int:
        return max(lo, min(hi, int(round(value))))
    return max(lo, min(hi, float(value)))


def _sample_random() -> dict:
    params = {}
    for k, (lo, hi, typ, _step) in PARAM_SPACE.items():
        if typ is int:
            params[k] = random.randint(lo, hi)
        else:
            params[k] = round(random.uniform(lo, hi), 3)
    return params


def _apply(params: dict) -> None:
    for k, v in params.items():
        setattr(mc, k, v)


# ── Fixed reference heads (computed once at baseline) ─────────────────────────

def _compute_reference_heads() -> list[list[tuple[float, float, float]]]:
    """
    Detect note heads on every page of the *source* PDF using the current
    (baseline) detection parameters.  The result is a list of per-page head
    lists: [ [(cx, cy, r), …], … ].

    These are computed ONCE before any parameter search begins and are then
    held fixed so that varying HOUGH_* / *_FACTOR params cannot simultaneously
    shrink both predictions and references (which would let degenerate
    low-detection configurations score a trivially perfect F1).
    """
    print("Computing fixed reference note heads from source PDF … ", end="", flush=True)
    src_doc = fitz.open(str(mc.PDF_PATH))
    refs: list[list[tuple[float, float, float]]] = []
    for page in src_doc:
        refs.append(checker.detect_notes_on_page(page))
    src_doc.close()
    total = sum(len(h) for h in refs)
    print(f"{total} heads across {len(refs)} page(s).\n", flush=True)
    return refs


def _evaluate(
    params: dict,
    ref_heads: list[list[tuple[float, float, float]]],
    label: str = "",
) -> tuple[float, int, int, int]:
    """
    Apply *params*, regenerate the annotated PDF, and compare extracted blue
    labels against the pre-computed *ref_heads*.  Returns (f1, tp, fp, fn).
    """
    _apply(params)

    tag = f"[{label}]" if label else ""
    param_str = "  ".join(f"{k}={v}" for k, v in params.items())
    print(f"  {tag:16s}  {param_str}", flush=True)

    # Re-annotate into a temp file so the real output PDF stays unlocked
    tmp = Path(tempfile.mktemp(suffix=".pdf", dir=mc.OUTPUT_PATH.parent))
    try:
        mc.annotate_pdf(mc.PDF_PATH, tmp, mc.KEY_SIG)

        # Compare blue labels against the fixed reference heads
        doc = fitz.open(str(tmp))
        G = [0, 0, 0]   # TP, FP, FN
        for pg_idx, page in enumerate(doc):
            labels = checker.extract_blue_labels(page)
            heads  = ref_heads[pg_idx] if pg_idx < len(ref_heads) else []
            tp, fp, fn = checker._match(labels, heads)
            G[0] += tp;  G[1] += fp;  G[2] += fn
        doc.close()
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)

    _, _, f1 = checker._prf(*G)
    print(f"  {'':16s}  → F1={f1:.4f}  TP={G[0]}  FP={G[1]}  FN={G[2]}\n",
          flush=True)
    return f1, G[0], G[1], G[2]


# ── Write-back ────────────────────────────────────────────────────────────────

# Maps each constant name to the regex that locates its assignment line.
# Matches  NAME   =   <value>  and replaces only the value token.
_PATTERNS = {
    k: re.compile(rf"({re.escape(k)}\s*=\s*)\S+")
    for k in PARAM_SPACE
}


def _write_best_params(best: dict) -> None:
    path = _HERE / "MisClef.py"
    src  = path.read_text(encoding="utf-8")
    for k, v in best.items():
        src = _PATTERNS[k].sub(rf"\g<1>{v}", src)
    path.write_text(src, encoding="utf-8")
    print("MisClef.py updated with best parameters.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n_random: int, n_rounds: int, seed: int) -> None:
    random.seed(seed)

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("=" * 70)
    print("MisClef-trainer  –  hyperparameter optimisation")
    print(f"random_trials={n_random}  hill_rounds={n_rounds}  seed={seed}")
    print("=" * 70 + "\n")

    # Compute fixed reference heads ONCE (before any param changes)
    base_params = _default_params()
    _apply(base_params)
    ref_heads = _compute_reference_heads()

    best_params = base_params
    best_f1, best_tp, best_fp, best_fn = _evaluate(best_params, ref_heads, "baseline")
    print(f"Baseline  F1={best_f1:.4f}  TP={best_tp}  FP={best_fp}  FN={best_fn}\n")

    # ── Phase 1: random search ─────────────────────────────────────────────────
    if n_random > 0:
        print(f"=== Phase 1: random search ({n_random} trials) ===\n")
        for i in range(n_random):
            params = _sample_random()
            f1, tp, fp, fn = _evaluate(params, ref_heads, f"rand {i + 1}/{n_random}")
            if f1 > best_f1:
                best_f1, best_tp, best_fp, best_fn = f1, tp, fp, fn
                best_params = deepcopy(params)
                print(f"  *** New best  F1={best_f1:.4f} ***\n")
        print(f"--- After random search: best F1={best_f1:.4f} ---\n")

    # ── Phase 2: coordinate-wise hill climbing ────────────────────────────────
    if n_rounds > 0:
        print(f"=== Phase 2: hill climbing ({n_rounds} rounds) ===\n")
        for rnd in range(n_rounds):
            improved_this_round = False
            for k, (lo, hi, typ, step) in PARAM_SPACE.items():
                for direction in (+1, -1):
                    candidate = deepcopy(best_params)
                    candidate[k] = _clip(candidate[k] + direction * step, lo, hi, typ)
                    if candidate[k] == best_params[k]:
                        continue   # already at boundary
                    f1, tp, fp, fn = _evaluate(
                        candidate, ref_heads, f"hill r{rnd + 1} {k}{direction:+d}"
                    )
                    if f1 > best_f1:
                        best_f1, best_tp, best_fp, best_fn = f1, tp, fp, fn
                        best_params = deepcopy(candidate)
                        improved_this_round = True
                        print(f"  *** New best  F1={best_f1:.4f} ***\n")
                        break  # move to next parameter after improvement

            if not improved_this_round:
                print(f"  Round {rnd + 1}: no improvement – stopping early.\n")
                break

    # ── Results ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"Best  F1={best_f1:.4f}  TP={best_tp}  FP={best_fp}  FN={best_fn}")
    print("Best parameters:")
    for k, v in best_params.items():
        marker = "  *" if v != _default_params()[k] else ""
        print(f"  {k:<20s} = {v}{marker}")
    print("=" * 70 + "\n")

    # ── Write-back & final PDF ────────────────────────────────────────────────
    _apply(best_params)
    _write_best_params(best_params)

    print("\nGenerating final annotated PDF with best parameters …")
    mc.annotate_pdf(mc.PDF_PATH, mc.OUTPUT_PATH, mc.KEY_SIG)
    print(f"Saved: {mc.OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimise MisClef detection hyperparameters."
    )
    parser.add_argument("--random", type=int, default=12,
                        dest="n_random", metavar="N",
                        help="number of random-search trials (default 12)")
    parser.add_argument("--rounds", type=int, default=3,
                        dest="n_rounds", metavar="N",
                        help="number of hill-climbing rounds (default 3)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default 42)")
    args = parser.parse_args()
    main(args.n_random, args.n_rounds, args.seed)
