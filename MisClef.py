"""MisClef.py
Annotates piano sheet music PDF with blue note-name labels (e.g. B, C#) offset to the top-right of each note head.

Detection pipeline (per page, per stave):
  1. Render page at SCALE x to grayscale.
  2. Detect & cluster horizontal staff lines; group into staves of 5.
  3. Remove staff lines from a per-stave ROI.
  4. Erase thin vertical elements (stems, bar lines).
  5. Find contours; filter by size & aspect ratio → note-head candidates.
  6. Map each candidate's Y-position to a diatonic pitch name.
  7. Insert a small blue label into the original PDF page.

Music theory:
  Treble clef – bottom line = E4   (diatonic index 2 in C D E F G A B)
  Bass   clef – bottom line = G2   (diatonic index 4)
  Each half-space up = +1 diatonic step.

Key signature:
  Set KEY_SIG below.  0 = C major / A minor (no sharps/flats).
  Positive integers add sharps (F# C# G# …); negative add flats (Bb Eb Ab …).
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path
import fitz          # PyMuPDF
import numpy as np
import cv2
from tqdm import tqdm

# ─── User configuration ───────────────────────────────────────────────────────

_HERE = Path(__file__).parent

SCALE        = 3.0   # render scale (3× ≈ 216 DPI from 72-pt base)
KEY_SIG      = 0     # default: C major / A minor (no sharps/flats)
FONT_SIZE    = 6.5   # label size in PDF points
BLUE         = (0, 0, 1)  # RGB 0-1 for fitz
CLEF_SKIP_SP      = 4.0   # staff-spacings reserved for the clef symbol at the staff left
KEYSIG_SP_PER_ACC = 1.5   # additional staff-spacings per sharp/flat in the key signature
TIMESIG_SP        = 2.5   # extra staff-spacings for the time signature (4/4 etc.) on the first system
TEMPO_SP          = 2.0   # extra staff-spacings for the tempo mark (♩ = 150 etc.) on the first system
LYRIC_SKIP_SP     = 2.5   # skip detections more than this many staff-spacings below the bottom staff line (lyrics zone)
USE_OEMER         = True  # True → replace HoughCircles with oemer's UNet notehead model

# ─── Detection hyperparameters (tuned by MisClef-trainer.py) ─────────────────

HOUGH_PARAM1    = 50    # Canny high threshold
HOUGH_PARAM2    = 7     # Hough accumulator threshold (lower → more circles found)
MIN_R_FACTOR    = 0.27  # min-radius as fraction of staff spacing
MAX_R_FACTOR    = 0.58  # max-radius as fraction of staff spacing
MIN_DIST_FACTOR = 0.48  # min centre-to-centre distance as fraction of staff spacing
DENSITY_MIN     = 0.10  # minimum ink density in bounding patch
CIRCULARITY_MIN = 0.35  # minimum circularity (4π·area / perimeter²)

# ─── Music theory helpers ─────────────────────────────────────────────────────

_DIATONIC     = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
_CLEF_BASE    = {'treble': 2, 'bass': 4}   # diatonic index of each clef's bottom line
_SHARP_ORDER  = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
_FLAT_ORDER   = ['B', 'E', 'A', 'D', 'G', 'C', 'F']


def key_accidentals(n):
    """Return {note_letter: '#' or 'b'} for key signature n."""
    if n > 0:
        return {note: '#' for note in _SHARP_ORDER[:n]}
    if n < 0:
        return {note: 'b' for note in _FLAT_ORDER[:-n]}
    return {}


def pitch_name(diatonic_pos, clef, accidentals):
    """
    Convert diatonic_pos (0 = bottom staff line, +1 per half-space upward)
    to a note letter, applying key-signature accidentals.
    """
    idx  = (_CLEF_BASE[clef] + diatonic_pos) % 7
    note = _DIATONIC[idx]
    if note in accidentals:
        note += accidentals[note]
    return note


# ─── oemer notehead model ─────────────────────────────────────────────────────

_OEMER_CHECKPOINTS = {
    'seg_net': 'https://github.com/BreezeWhite/oemer/releases/download/checkpoints/2nd_model.onnx',
}


def _ensure_oemer_checkpoints():
    """Download missing oemer ONNX checkpoint files on first use."""
    import urllib.request
    from oemer import MODULE_PATH as _MP
    for folder, url in _OEMER_CHECKPOINTS.items():
        dest = os.path.join(_MP, 'checkpoints', folder, 'model.onnx')
        if not os.path.exists(dest):
            print(f'Downloading oemer {folder} checkpoint (~37 MB) ...', flush=True)
            urllib.request.urlretrieve(url, dest)
            print(f'Saved to {dest}')


def _oemer_notehead_map_for_page(gray_np):
    """
    Run oemer's seg_net UNet on one page (grayscale numpy array) and return a
    binary notehead mask (uint8, 1 where a notehead is predicted) scaled back to
    the same H×W as the input.  Model checkpoints (~40 MB) are downloaded
    automatically from GitHub on the first call.
    """
    _ensure_oemer_checkpoints()
    import onnxruntime as _ort
    _ort.set_default_logger_severity(3)   # suppress CUDA EP fallback warnings (ERROR-only)
    print('ONNX providers:', _ort.get_available_providers(), flush=True)
    from PIL import Image as _PIL
    from oemer import MODULE_PATH as _MP
    from oemer.inference import inference as _infer

    h, w = gray_np.shape
    rgb = np.dstack([gray_np] * 3)          # oemer expects an RGB image

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
        tmp = tf.name
    try:
        _PIL.fromarray(rgb).save(tmp)
        # seg_net outputs: 0=background, 1=stems/rests, 2=noteheads, 3=clefs/keys
        sep, _ = _infer(
            os.path.join(_MP, 'checkpoints/seg_net'),
            tmp,
            manual_th=None,
            use_tf=False,
        )
    finally:
        os.unlink(tmp)

    notehead = (sep == 2).astype(np.uint8)
    if notehead.shape[:2] != (h, w):
        notehead = cv2.resize(notehead, (w, h), interpolation=cv2.INTER_NEAREST)
    return notehead


def detect_heads_in_stave_oemer(notehead_mask, stave, staves,
                                 left_skip_px=0, top_margin_px=0,
                                 bot_clip_y=None, right_clip_px=None):
    """
    Detect note heads from oemer's binary notehead segmentation mask using
    connected-component centroids – one centroid per blob, so neither
    double-printing (two Hough circles per notehead) nor Hough false-negatives
    (missed blobs that aren't perfectly circular) can occur.
    Blobs larger than ~2 single noteheads are split via distance transform.
    Returns list of (cx, cy, radius) in full-image pixel coordinates.
    """
    sp  = stave['sp']
    top = stave['lines'][0]
    bot = stave['lines'][4]
    idx = staves.index(stave)
    h, w = notehead_mask.shape
    default_pad = int(3.5 * sp)

    if idx > 0:
        prev_bot = staves[idx - 1]['lines'][4]
        y0 = max((prev_bot + top) // 2, top - default_pad)
    else:
        y0 = max(0, top - default_pad)

    if idx < len(staves) - 1:
        next_top = staves[idx + 1]['lines'][0]
        y1 = min((bot + next_top) // 2, bot + default_pad)
    else:
        y1 = min(h, bot + default_pad)

    roi = notehead_mask[y0:y1].copy()

    # Apply the same boundary-blanking zones as the Hough path
    if left_skip_px > 0:
        roi[:, :left_skip_px] = 0
    if top_margin_px > 0:
        roi[:top_margin_px, :] = 0
    if bot_clip_y is not None:
        bot_row = max(0, bot_clip_y - y0)
        if bot_row < roi.shape[0]:
            roi[bot_row:, :] = 0
    if right_clip_px is not None and right_clip_px < roi.shape[1]:
        roi[:, right_clip_px:] = 0

    if not np.any(roi):
        return []

    min_r = max(3, int(sp * MIN_R_FACTOR))
    max_r = max(5, int(sp * MAX_R_FACTOR))
    # Area thresholds for a single notehead blob
    min_area    = int(np.pi * min_r ** 2 * 0.35)          # below → noise
    single_area = int(np.pi * max_r ** 2 * 1.8)           # above → likely merged

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        roi, connectivity=8
    )

    heads = []
    for label_id in range(1, n_labels):   # 0 is background
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue   # noise / artefact

        if area <= single_area:
            # Single notehead – use the blob centroid directly
            r    = max(min_r, min(int(round(np.sqrt(area / np.pi))), max_r))
            cx_l = int(round(centroids[label_id][0]))
            cy_l = int(round(centroids[label_id][1]))
            candidates = [(cx_l, cy_l, r)]
        else:
            # Oversized blob – split by distance-transform local maxima
            blob_mask = (labels == label_id).astype(np.uint8)
            dist      = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)
            dmax      = float(dist.max())
            if dmax < min_r * 0.4:
                continue
            _, peaks = cv2.threshold(dist, dmax * 0.5, 255, cv2.THRESH_BINARY)
            peaks = peaks.astype(np.uint8)
            n_pk, _, pk_stats, pk_cents = cv2.connectedComponentsWithStats(
                peaks, connectivity=8
            )
            candidates = []
            for pk_id in range(1, n_pk):
                pk_area = pk_stats[pk_id, cv2.CC_STAT_AREA]
                r_pk    = max(min_r, min(int(round(np.sqrt(pk_area / np.pi))), max_r))
                candidates.append((
                    int(round(pk_cents[pk_id][0])),
                    int(round(pk_cents[pk_id][1])),
                    r_pk,
                ))

        for cx_l, cy_l, r in candidates:
            cx_g = cx_l
            cy_g = cy_l + y0

            # Boundary sanity checks (mirror the blanking above)
            if left_skip_px > 0 and cx_g < left_skip_px:
                continue
            if top_margin_px > 0 and (cy_g - y0) < top_margin_px:
                continue
            if bot_clip_y is not None and cy_g > bot_clip_y:
                continue
            if right_clip_px is not None and cx_g > right_clip_px:
                continue

            heads.append((cx_g, cy_g, r))

    return heads


# ─── Staff-line detection ─────────────────────────────────────────────────────

def find_staff_lines(binary):
    """Return sorted list of Y pixel positions of staff lines (one per line)."""
    h, w = binary.shape
    hk   = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
    hm   = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    sums = np.sum(hm, axis=1)
    rows = np.where(sums > w * 0.25)[0]
    if not len(rows):
        return []
    # cluster adjacent rows into single positions
    groups = []
    cl = [int(rows[0])]
    for r in rows[1:]:
        if r - cl[-1] <= 4:
            cl.append(int(r))
        else:
            groups.append(int(np.mean(cl)))
            cl = [int(r)]
    groups.append(int(np.mean(cl)))
    return groups


def group_staves(line_ys):
    """Group Y positions into staves of 5 consistently-spaced lines."""
    staves = []
    i = 0
    while i <= len(line_ys) - 5:
        g    = line_ys[i:i + 5]
        gaps = [g[j + 1] - g[j] for j in range(4)]
        mu   = float(np.mean(gaps))
        if mu > 2 and max(abs(x - mu) / mu for x in gaps) < 0.45:
            staves.append({'lines': g, 'sp': mu})
            i += 5
        else:
            i += 1
    return staves


def find_first_barline_x(binary, stave, min_search_x=0):
    """
    Return the x pixel position of the first barline after min_search_x.
    A barline is a thin vertical line spanning >= 85% of the staff height
    (top staff line to bottom staff line).  This is used as a dynamic
    left-skip boundary so that clef, key-sig, and time-sig glyphs are
    never treated as note candidates.
    Returns an int x, or None if no barline is detected.
    """
    sp       = stave['sp']
    top      = stave['lines'][0]
    bot      = stave['lines'][4]
    staff_h  = bot - top

    # Extract the staff row band starting from min_search_x
    band = binary[top: bot + 1, min_search_x:]

    # Morphological open with a tall, narrow vertical kernel isolates barlines.
    # Height = 85% of staff; width = 2 px tolerates slight anti-aliasing.
    vk_h = max(3, int(staff_h * 0.85))
    vk   = cv2.getStructuringElement(cv2.MORPH_RECT, (2, vk_h))
    vm   = cv2.morphologyEx(band, cv2.MORPH_OPEN, vk)

    col_sums  = np.sum(vm, axis=0) // 255          # ink rows per column
    threshold = int(staff_h * 0.75)                # 75% of staff height
    hits      = np.where(col_sums >= threshold)[0]
    if not len(hits):
        return None

    # Return the left edge of the first cluster of hit columns
    return min_search_x + int(hits[0])


def find_last_barline_x(binary, stave):
    """
    Return the x pixel position of the leftmost column of the last barline
    on this stave (searches right-to-left).  Used to blank the final
    double/end barline so its thick lines are not detected as note heads.
    Returns an int x, or None if no barline is detected.
    """
    top     = stave['lines'][0]
    bot     = stave['lines'][4]
    staff_h = bot - top

    band = binary[top: bot + 1, :]

    vk_h = max(3, int(staff_h * 0.85))
    vk   = cv2.getStructuringElement(cv2.MORPH_RECT, (2, vk_h))
    vm   = cv2.morphologyEx(band, cv2.MORPH_OPEN, vk)

    col_sums  = np.sum(vm, axis=0) // 255
    threshold = int(staff_h * 0.75)
    hits      = np.where(col_sums >= threshold)[0]
    if not len(hits):
        return None

    # Walk back from the rightmost hit to find the left edge of the last cluster
    last_hit = int(hits[-1])
    x = last_hit
    while x > 0 and col_sums[x - 1] >= threshold:
        x -= 1
    return x


def detect_heads_in_stave(gray, binary, stave, staves, left_skip_px=0, top_margin_px=0, bot_clip_y=None, right_clip_px=None):
    """
    Detect note heads using HoughCircles on a staff-line-cleaned grayscale ROI.
    Works for both filled (quarter/eighth) and hollow (half/whole) note heads
    even when note heads are connected to stems or beams.
    ROI is clamped to the midpoints between neighbouring staves so that
    lyrics / slurs in the gap between treble and bass do not cause false positives.
    left_skip_px:  columns [0, left_skip_px) are blanked before Hough detection
                   so that clef and key-signature symbols are never treated as note candidates.
    top_margin_px: rows [0, top_margin_px) of the ROI are blanked before Hough
                   detection to suppress tempo-mark symbols above the first staff.
    bot_clip_y:    full-image Y pixel coordinate below which all ROI rows are
                   blanked before Hough detection (lyrics zone suppression).
    right_clip_px: full-image X pixel coordinate from which all ROI columns are
                   blanked (end barline suppression).
    Returns list of (cx, cy, radius) in full-image pixel coordinates.
    """
    sp  = stave['sp']
    top = stave['lines'][0]
    bot = stave['lines'][4]
    idx = staves.index(stave)
    h   = gray.shape[0]
    w   = gray.shape[1]
    default_pad = int(3.5 * sp)

    # Clamp upper edge to midpoint of gap above this stave
    if idx > 0:
        prev_bot = staves[idx - 1]['lines'][4]
        y0 = max((prev_bot + top) // 2, top - default_pad)
    else:
        y0 = max(0, top - default_pad)

    # Clamp lower edge to midpoint of gap below this stave
    if idx < len(staves) - 1:
        next_top = staves[idx + 1]['lines'][0]
        y1 = min((bot + next_top) // 2, bot + default_pad)
    else:
        y1 = min(h, bot + default_pad)

    roi_gray = gray[y0:y1].copy()
    roi_bin  = binary[y0:y1].copy()

    # Remove staff lines: detect them, then whiten those pixels in grayscale
    hk          = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
    staff_mask  = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, hk)
    clean_gray  = roi_gray.copy()
    clean_gray[staff_mask > 0] = 255

    # Remove wide horizontal beams (wider than 2 sp, up to 5px thick)
    # so beamed eighth-note groups don't create one big merged blob
    bk         = cv2.getStructuringElement(cv2.MORPH_RECT, (int(sp * 2.2), 5))
    beam_mask  = cv2.morphologyEx(roi_bin, cv2.MORPH_OPEN, bk)
    clean_gray[beam_mask > 0] = 255

    # Blank out the clef / key-signature strip so those symbols are invisible
    # to HoughCircles (more reliable than post-filtering by x coordinate).
    if left_skip_px > 0:
        clean_gray[:, :left_skip_px] = 255

    # Blank out the above-staff margin (e.g. tempo mark ♩=150 on first system).
    if top_margin_px > 0:
        clean_gray[:top_margin_px, :] = 255

    # Blank out the lyrics zone below the staff so round letter glyphs
    # (o, a, e, d …) are invisible to HoughCircles.
    if bot_clip_y is not None:
        bot_row = max(0, bot_clip_y - y0)
        if bot_row < clean_gray.shape[0]:
            clean_gray[bot_row:, :] = 255

    # Blank out the end-barline strip at the right so thick final barlines
    # are invisible to HoughCircles.
    if right_clip_px is not None and right_clip_px < clean_gray.shape[1]:
        clean_gray[:, right_clip_px:] = 255

    # Slight Gaussian blur helps Hough accumulator
    blurred = cv2.GaussianBlur(clean_gray, (5, 5), 1.5)

    min_r    = max(3, int(sp * MIN_R_FACTOR))
    max_r    = max(5, int(sp * MAX_R_FACTOR))
    min_dist = int(sp * MIN_DIST_FACTOR)    # minimum gap between two note centres

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    heads = []
    if circles is None:
        return heads

    for cx, cy, r in np.round(circles[0]).astype(int):
        cy_g = int(cy) + y0
        cx_g = int(cx)

        # Reject detections whose circle edge reaches into the clef / key-sig
        # blanked zone.  Hough accumulator leakage can place a circle centre
        # just outside the blanked strip while its arc still traces a clef or
        # sharp symbol inside it.
        if left_skip_px > 0 and cx_g - int(r) < left_skip_px:
            continue

        # Reject detections whose circle edge reaches into the top margin
        # blanked zone (tempo mark above the first staff).
        if top_margin_px > 0 and cy_g - y0 - int(r) < top_margin_px:
            continue

        # Reject detections whose circle edge reaches into the lyrics zone.
        if bot_clip_y is not None and cy_g + int(r) > bot_clip_y:
            continue

        # Reject detections whose circle edge reaches into the end-barline zone.
        if right_clip_px is not None and cx_g + int(r) > right_clip_px:
            continue

        # Verify ink density in the circle's bounding patch
        y1c = max(0, cy_g - int(r));  y2c = min(binary.shape[0], cy_g + int(r) + 1)
        x1c = max(0, cx_g - int(r));  x2c = min(binary.shape[1], cx_g + int(r) + 1)
        patch   = binary[y1c:y2c, x1c:x2c]
        density = np.sum(patch) / (max(patch.size, 1) * 255)
        if density < DENSITY_MIN:       # skip near-empty regions
            continue

        # Circularity check: rests, clefs, and other non-note symbols are
        # irregular; note heads (filled or hollow) are close to ellipses.
        # circularity = 4π·area / perimeter²  →  1.0 for a perfect circle.
        cnts, _ = cv2.findContours(patch.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            area_c  = cv2.contourArea(largest)
            perim_c = cv2.arcLength(largest, True)
            if perim_c > 0:
                circularity = 4 * np.pi * area_c / (perim_c ** 2)
                if circularity < CIRCULARITY_MIN:   # reject rests / irregular symbols
                    continue

        heads.append((cx_g, cy_g, int(r)))

    return heads


# ─── Pitch mapping ────────────────────────────────────────────────────────────

def y_to_diatonic_pos(cy, stave):
    """
    Convert pixel Y to diatonic position relative to the bottom staff line.
    0 = bottom line, 1 = first space, 2 = second line, …
    Negative values = below the staff (ledger lines below).
    """
    bot     = stave['lines'][4]          # bottom (5th) staff line Y
    half_sp = stave['sp'] / 2.0
    return int(round((bot - cy) / half_sp))


# ─── Main ─────────────────────────────────────────────────────────────────────

def annotate_pdf(pdf_path, output_path, key_sig=0):
    acc = key_accidentals(key_sig)
    doc = fitz.open(pdf_path)

    for page_num, page in (pbar := tqdm(enumerate(doc), total=len(doc), desc='Annotating', unit='page')):

        # Render to grayscale at high resolution
        mat  = fitz.Matrix(SCALE, SCALE)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Detect staves
        line_ys = find_staff_lines(binary)
        staves  = group_staves(line_ys)
        if not staves:
            pbar.write(f'Page {page_num + 1}: no staves detected')
            continue

        if USE_OEMER:
            pbar.set_postfix_str('running oemer…')
            notehead_mask = _oemer_notehead_map_for_page(gray)

        # Piano grand staff: staves come in pairs – top = treble, bottom = bass
        for i, s in enumerate(staves):
            s['clef'] = 'treble' if (i % 2 == 0) else 'bass'

        total = 0
        fallback_base_sp = CLEF_SKIP_SP + abs(key_sig) * KEYSIG_SP_PER_ACC
        for stave_idx, stave in enumerate(staves):
            # Time signature (4/4) only appears on the very first system (page 0, staves 0-1).
            # Clef + key signature appear at the left of every stave on every page.
            is_first_system = (page_num == 0 and stave_idx < 2)
            # Detect the first barline dynamically — it marks the boundary
            # between the header (clef + key sig [+ time sig on page 0]) and
            # the first measure.  For the first stave pair of every page
            # (stave_idx < 2), start the search with an extra TIMESIG_SP margin
            # so that sharp / flat strokes in the key signature are never
            # mistaken for a barline and used as an undersized clef_x_limit.
            min_search = int(fallback_base_sp * stave['sp'])
            if stave_idx < 2:
                min_search += int(TIMESIG_SP * stave['sp'])
            barline_x  = find_first_barline_x(binary, stave, min_search_x=min_search)
            if barline_x is not None:
                clef_x_limit = barline_x
            else:
                # Fallback: constant-based estimate
                left_skip_sp  = fallback_base_sp
                if stave_idx < 2:
                    left_skip_sp += TIMESIG_SP
                if is_first_system:
                    left_skip_sp += TEMPO_SP
                clef_x_limit  = int(left_skip_sp * stave['sp'])
            # Blank rows above the first staff on page 0 to suppress the ♩=150
            # tempo mark, which sits above the top staff line in the ROI margin.
            top_margin_px = int(TEMPO_SP * stave['sp']) if (page_num == 0 and stave_idx == 0) else 0
            # On the last page, detect the final (end) barline and blank
            # everything to its right so the thick double barline is ignored.
            is_last_page = (page_num == len(doc) - 1)
            is_last_system = is_last_page and (stave_idx >= len(staves) - 2)
            right_clip_px = None
            if is_last_system:
                last_bl = find_last_barline_x(binary, stave)
                if last_bl is not None:
                    right_clip_px = last_bl
            if USE_OEMER:
                heads = detect_heads_in_stave_oemer(
                    notehead_mask, stave, staves,
                    left_skip_px=clef_x_limit,
                    top_margin_px=top_margin_px,
                    bot_clip_y=int(stave['lines'][4] + LYRIC_SKIP_SP * stave['sp']),
                    right_clip_px=right_clip_px,
                )
            else:
                heads = detect_heads_in_stave(gray, binary, stave, staves,
                                              left_skip_px=clef_x_limit,
                                              top_margin_px=top_margin_px,
                                              bot_clip_y=int(stave['lines'][4] + LYRIC_SKIP_SP * stave['sp']),
                                              right_clip_px=right_clip_px)
            for (cx, cy, r) in heads:
                pos  = y_to_diatonic_pos(cy, stave)
                name = pitch_name(pos, stave['clef'], acc)

                # Convert image pixels → PDF points
                px   = cx / SCALE
                py   = cy / SCALE
                r_pt = r  / SCALE

                # Label: offset top-right of the note head
                lx = px + r_pt * 0.8
                ly = py - r_pt * 1.5

                page.insert_text(
                    fitz.Point(lx, ly),
                    name,
                    fontsize=FONT_SIZE,
                    color=BLUE,
                    fontname='helv',
                )
                total += 1

        pbar.set_postfix(staves=len(staves), notes=total)

    doc.save(output_path)
    doc.close()
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Annotate piano sheet music PDF with note-name labels.'
    )
    parser.add_argument('input', type=Path, help='Path to the input PDF file')
    parser.add_argument(
        '-o', '--output', type=Path, default=None,
        help='Path for the annotated output PDF (default: <input stem> - Annotated.pdf)'
    )
    parser.add_argument(
        '-k', '--key', type=int, default=KEY_SIG,
        metavar='N',
        help='Key signature: positive = sharps, negative = flats (default: %(default)s)'
    )
    args = parser.parse_args()

    pdf_path = args.input
    output_path = args.output or pdf_path.with_name(pdf_path.stem + ' - Annotated.pdf')

    annotate_pdf(pdf_path, output_path, args.key)
