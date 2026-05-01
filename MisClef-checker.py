"""MisClef-checker.py
Evaluate annotation quality of a MisClef-annotated PDF by computing
Precision, Recall, and F1-score per page and overall.

How it works
------------
Predictions  – blue note-name labels (e.g. "C", "F#") extracted from the
               PDF text layer by colour (RGB 0, 0, 1  →  fitz integer 255).
Reference    – note-head positions found by re-running the same HoughCircles
               pipeline used in MisClef.py on the rendered page.  Blue labels
               are only ~29/255 in grayscale so they do not materially affect
               note-head detection.

Matching
--------
A prediction at (lx, ly) matches a reference head at (cx, cy, r) when the
Euclidean distance between (lx, ly) and the expected label-insertion point
(cx + r·0.8, cy − r·1.5) is ≤ MATCH_DIST_PT in PDF points.
Greedy nearest-neighbour matching is used (closest pair first).

Metrics
-------
  TP  – reference note heads with a matching label        (correct)
  FP  – labels with no nearby reference head              (over-annotation / low specificity)
  FN  – reference heads with no matching label            (missed notes  / low sensitivity)

  Precision = TP / (TP + FP)    [specificity proxy]
  Recall    = TP / (TP + FN)    [sensitivity proxy]
  F1        = 2·P·R / (P + R)
"""

import re
import sys
from pathlib import Path

import cv2
import fitz
import numpy as np

# ── Import MisClef helpers ────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import MisClef as mc  # noqa: E402  (sys.path modified above)

# ── Configuration ─────────────────────────────────────────────────────────────
ANNOTATED_PATH = mc.OUTPUT_PATH   # default: MisClef's output path
MATCH_DIST_PT  = 12.0             # PDF-point radius for a label–head match
_BLUE_INT      = 255              # fitz integer encoding of colour (0, 0, 1)
_NOTE_RE       = re.compile(r'^[A-G][#b]?$')


# ── Extract blue annotations ──────────────────────────────────────────────────

def extract_blue_labels(page: fitz.Page) -> list[tuple[float, float]]:
    """
    Return list of (x, y) baseline origins (PDF points) for every blue
    note-name label on the page.
    """
    origins: list[tuple[float, float]] = []
    for block in page.get_text("dict", flags=0)["blocks"]:
        if block.get("type") != 0:          # 0 = text block
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span.get("color") != _BLUE_INT:
                    continue
                if not _NOTE_RE.match(span.get("text", "").strip()):
                    continue
                # "origin" is the baseline point that was passed to insert_text()
                pt = span.get("origin")
                if pt is None:
                    bb = span["bbox"]       # (x0, y0, x1, y1)
                    pt = (bb[0], bb[3])     # bottom-left ≈ baseline
                origins.append((float(pt[0]), float(pt[1])))
    return origins


# ── Independent note detection ────────────────────────────────────────────────

def detect_notes_on_page(page: fitz.Page) -> list[tuple[float, float, float]]:
    """
    Re-run the MisClef HoughCircles pipeline on *page* and return
    (cx, cy, r) tuples in PDF-point coordinates.
    The result is independent of the annotation step and serves as the
    pseudo-ground-truth reference for computing F1.
    """
    doc    = page.parent
    pg_idx = page.number
    n_pgs  = len(doc)

    mat  = fitz.Matrix(mc.SCALE, mc.SCALE)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    line_ys = mc.find_staff_lines(binary)
    staves  = mc.group_staves(line_ys)
    if not staves:
        return []

    for i, s in enumerate(staves):
        s['clef'] = 'treble' if (i % 2 == 0) else 'bass'

    fallback_base = mc.CLEF_SKIP_SP + abs(mc.KEY_SIG) * mc.KEYSIG_SP_PER_ACC
    heads_pt: list[tuple[float, float, float]] = []

    for s_idx, stave in enumerate(staves):
        is_first = (pg_idx == 0 and s_idx < 2)

        min_srch = int(fallback_base * stave['sp'])
        if s_idx < 2:
            min_srch += int(mc.TIMESIG_SP * stave['sp'])

        bl_x = mc.find_first_barline_x(binary, stave, min_search_x=min_srch)
        if bl_x is not None:
            skip_x = bl_x
        else:
            skip_sp = (fallback_base
                       + (mc.TIMESIG_SP if s_idx < 2 else 0.0)
                       + (mc.TEMPO_SP   if is_first  else 0.0))
            skip_x = int(skip_sp * stave['sp'])

        top_mgn = int(mc.TEMPO_SP * stave['sp']) if (pg_idx == 0 and s_idx == 0) else 0

        is_last_sys = (pg_idx == n_pgs - 1) and (s_idx >= len(staves) - 2)
        r_clip = None
        if is_last_sys:
            lb = mc.find_last_barline_x(binary, stave)
            if lb is not None:
                r_clip = lb

        heads = mc.detect_heads_in_stave(
            gray, binary, stave, staves,
            left_skip_px=skip_x,
            top_margin_px=top_mgn,
            bot_clip_y=int(stave['lines'][4] + mc.LYRIC_SKIP_SP * stave['sp']),
            right_clip_px=r_clip,
        )
        for cx, cy, r in heads:
            heads_pt.append((cx / mc.SCALE, cy / mc.SCALE, r / mc.SCALE))

    return heads_pt


# ── Matching ──────────────────────────────────────────────────────────────────

def _match(labels: list[tuple[float, float]],
           heads:  list[tuple[float, float, float]],
           thresh: float = MATCH_DIST_PT) -> tuple[int, int, int]:
    """
    Greedy nearest-neighbour match of label origins to expected annotation
    points (cx + r·0.8, cy − r·1.5).
    Returns (TP, FP, FN).
    """
    if not heads:
        return 0, len(labels), 0
    if not labels:
        return 0, 0, len(heads)

    # Expected insertion point for each reference head
    exp  = np.array([(cx + r * 0.8, cy - r * 1.5) for cx, cy, r in heads])  # (H, 2)
    pred = np.array(labels)                                                    # (L, 2)

    diff = pred[:, None, :] - exp[None, :, :]          # (L, H, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))             # (L, H)

    pairs = sorted(
        [(dist[li, hi], li, hi)
         for li in range(len(labels))
         for hi in range(len(heads))],
        key=lambda t: t[0],
    )

    used_l: set[int] = set()
    used_h: set[int] = set()
    for d, li, hi in pairs:
        if d > thresh:
            break
        if li in used_l or hi in used_h:
            continue
        used_l.add(li)
        used_h.add(hi)

    tp = len(used_l)
    return tp, len(labels) - tp, len(heads) - tp


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ── Main ──────────────────────────────────────────────────────────────────────

def check(annotated_path: Path) -> None:
    doc = fitz.open(str(annotated_path))
    n   = len(doc)

    W       = 64
    row_fmt = '{:>4}  {:>6}  {:>5}  {:>4}  {:>4}  {:>4}  {:>6}  {:>6}  {:>6}'

    print(f'\nChecking: {annotated_path.name}  ({n} page{"s" if n != 1 else ""})\n')
    print(row_fmt.format('Page', 'Labels', 'Ref', 'TP', 'FP', 'FN', 'Prec', 'Rec', 'F1'))
    print('─' * W)

    G = [0, 0, 0]   # grand TP, FP, FN

    for page in doc:
        pg     = page.number + 1
        labels = extract_blue_labels(page)
        heads  = detect_notes_on_page(page)
        tp, fp, fn = _match(labels, heads)
        p,  r,  f  = _prf(tp, fp, fn)
        G[0] += tp;  G[1] += fp;  G[2] += fn
        print(row_fmt.format(pg, len(labels), len(heads), tp, fp, fn,
                             f'{p:.3f}', f'{r:.3f}', f'{f:.3f}'))

    print('─' * W)
    P, R, F = _prf(*G)
    print(row_fmt.format('ALL', G[0] + G[1], G[0] + G[2], G[0], G[1], G[2],
                         f'{P:.3f}', f'{R:.3f}', f'{F:.3f}'))
    print(f'\nOverall  Precision={P:.4f}  Recall={R:.4f}  F1={F:.4f}\n')

    doc.close()


if __name__ == '__main__':
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ANNOTATED_PATH
    check(path)
