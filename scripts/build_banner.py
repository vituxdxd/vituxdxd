#!/usr/bin/env python3
"""Build the animated GitHub profile banner (dark.svg / light.svg).

Pipeline: crop photo -> enhance -> segment bg (dark mode) -> 1-bit Floyd-Steinberg
serpentine dither -> SMIL layers (intro shimmer, drift bands, logo travellers).
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portrait import (load_crop, enhance_gray, subject_mask, floyd_steinberg,
                      GRID_W, GRID_H)

from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import kmeans2

# ----------------------------------------------------------------------------
# parameters
# ----------------------------------------------------------------------------
SRC = os.environ.get("PROFILE_PIC", "/home/vitor/Downloads/github_style/profile_pic.jpg")
OUT_DIR = os.environ.get("OUT_DIR", os.path.dirname(os.path.abspath(__file__)))
BOX = (0.244, 0.0151, 0.756, 0.595)          # head + shoulders crop
HI = (900, 1020)                              # segmentation/enhance resolution
LIGHT_GAMMA = 0.38                            # lift for the light theme

N_TRAVELLERS = 900
INTRO_GROUPS = 60
N_BANDS = 94
DRIFT_FRAC = 0.42
LOOP = 14.2
DELAY = 2.8                                   # loop starts after the intro

W, H = 1180, 610
FR = dict(x=36, y=78, w=436, h=494)
SX, SY = FR["w"] / GRID_W, FR["h"] / GRID_H
CELL = 1.47
PANEL_X, PANEL_R = 512, 1144
ROW_Y0, ROW_H = 190, 23

DARK = dict(bg="#0A101F", win="#0C1322", titlebar="#0E1626", border="#1E293B",
            frame="#0D1526", chrome="#22D3EE", portrait="#A78BFA",
            accent="#10B981", live="#FF4D5E", text="#94A3B8",
            strong="#F8FAFC", muted="#64748B", sep="#1E293B")
LIGHT = dict(bg="#F8FAFC", win="#FFFFFF", titlebar="#EEF3F9", border="#CBD5E1",
             frame="#F1F5FB", chrome="#0891B2", portrait="#7C3AED",
             accent="#10B981", live="#DC2626", text="#475569",
             strong="#0F172A", muted="#94A3B8", sep="#E2E8F0")

ROWS1 = [
    ("Subject", "Vitor Hugo F. Oliveira"),
    ("Role", "Estudante de Medicina & Dev"),
    ("Origin", "Brasília - DF, Brasil"),
    ("Education", "Medicina — UniCeplac (cursando)"),
    ("Status", "Building + Learning + Shipping"),
    ("ToolChain", "Android Studio · VS Code · Git"),
]
ROWS2 = [
    ("Core.Lang", "Kotlin · Python · C++"),
    ("Core.Frontend", "HTML/CSS · Android UI"),
    ("Core.Backend", "Python · Flask · REST"),
    ("Core.Database", "SQLite (WAL)"),
    ("Core.Infra", "Docker · Git · ESP32"),
]
ROWS3 = [
    ("Grid.Mail", "vitor.oliveira@medicina.uniceplac.edu.br"),
    ("Grid.Portfolio", "coming soon"),
    ("Grid.GitHub", "github.com/vituxdxd"),
]
HANDLE = "@vituxdxd"

# keyTimes for the 14.2 s loop
T = {k: round(v / LOOP, 4) for k, v in dict(
    hold=3.0, in1=4.3, hold1=6.3, hold2=7.6, hold3=9.6, hold4=10.9,
    hold5=12.9, fade_in=3.6, fade_out=13.49).items()}
KT_POS = f"0;{T['hold']};{T['hold1']};{T['hold2']};{T['hold3']};{T['hold4']};{T['hold5']};1"
KT_BAND = f"0;{T['hold']};{T['in1']};{T['hold5']};1"
KT_TOP = f"0;{T['hold']};{T['fade_in']};{T['hold5']};{T['fade_out']};1"


def fmt(v, nd=1):
    return f"{v:.{nd}f}".rstrip("0").rstrip(".")


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ----------------------------------------------------------------------------
# portrait dot grids
# ----------------------------------------------------------------------------
def build_portraits():
    crop_hi = load_crop(SRC, BOX, grid=HI)
    rgb = np.asarray(crop_hi).astype(np.float32)
    vals = np.asarray(enhance_gray(crop_hi).resize((GRID_W, GRID_H), Image.LANCZOS),
                      dtype=np.float64) / 255.0

    # dark: segment the background out
    m_hi = _lab_mask(crop_hi, rgb)
    m = np.asarray(Image.fromarray((m_hi * 255).astype(np.uint8))
                   .resize((GRID_W, GRID_H), Image.LANCZOS)) > 127
    dark = floyd_steinberg(vals, dark_mode=True, mask=m)
    # light: ink dots draw the lit subject on a white panel (the dark studio
    # background is removed); a faint speckle remains where the wall was.
    inv = 1.0 - np.clip(vals, 0, 1)
    light_vals = np.where(m, inv ** 0.92, np.full_like(vals, 0.965))
    light = floyd_steinberg(light_vals, dark_mode=False)
    return dark, light, m


def _srgb_to_lab(rgb):
    x = rgb / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def _lab_mask(crop_hi, rgb, thr=14.0):
    h, w, _ = rgb.shape
    ys, xs = np.mgrid[0:h, 0:w]
    xn, yn = xs / w, ys / h
    b = max(3, int(0.07 * min(h, w)))
    bor = np.zeros((h, w), bool)
    bor[:b, :] = bor[-b:, :] = bor[:, :b] = bor[:, -b:] = True
    A = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], -1)
    coef = np.linalg.lstsq(A[bor], rgb[bor], rcond=None)[0]
    bg = A @ coef
    dist = np.linalg.norm(_srgb_to_lab(rgb) - _srgb_to_lab(bg), axis=2)
    m = dist > thr
    m = ndimage.binary_closing(m, structure=np.ones((11, 11)))
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = 1 + np.where(sizes > 0.002 * m.size)[0]
        m = np.isin(lab, keep)
    m = ndimage.binary_closing(m, structure=np.ones((9, 9)))
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_erosion(m, iterations=2)
    return m


# ----------------------------------------------------------------------------
# runs / paths
# ----------------------------------------------------------------------------
def runs_for(assign, dots):
    """Merge horizontal dot runs that share the same group id."""
    runs = {}
    h, w = dots.shape
    for y in range(h):
        x = 0
        while x < w:
            if not dots[y, x]:
                x += 1
                continue
            g = assign[y, x]
            x2 = x + 1
            while x2 < w and dots[y, x2] and assign[y, x2] == g:
                x2 += 1
            runs.setdefault(int(g), []).append((y, x, x2))
            x = x2
    return runs


def runs_path(runs, fx, fy):
    parts = []
    ww = CELL
    for y, x0, x1 in runs:
        yy = fy + y * SY
        xx = fx + x0 * SX
        length = (x1 - x0) * SX + 0.06
        parts.append(f"M{fmt(xx)} {fmt(yy)}h{fmt(length)}v{fmt(ww)}H{fmt(xx)}Z")
    return "".join(parts)


# ----------------------------------------------------------------------------
# SVG pieces
# ----------------------------------------------------------------------------
def text_width(s, size):
    return len(s) * size * 0.6


def row(label, value, y, pal):
    lw = text_width(label, 14)
    vw = text_width(value, 14)
    x0 = PANEL_X + lw + 12
    x1 = PANEL_R - vw - 12
    leader = ("" if x1 <= x0 else
              f'<line x1="{x0:.1f}" y1="{y-4}" x2="{x1:.1f}" y2="{y-4}" '
              f'stroke="{pal["muted"]}" stroke-width="1" stroke-dasharray="0.5 5.5" '
              f'stroke-linecap="round" opacity="0.55"/>')
    return (
        f'<text x="{PANEL_X}" y="{y}" font-size="14" fill="{pal["text"]}" '
        f'textLength="{lw:.1f}" lengthAdjust="spacingAndGlyphs">{esc(label)}</text>'
        f'{leader}'
        f'<text x="{PANEL_R}" y="{y}" font-size="14" fill="{pal["strong"]}" '
        f'text-anchor="end" textLength="{vw:.1f}" lengthAdjust="spacingAndGlyphs">'
        f'{esc(value)}</text>'
    )


def info_panel(pal):
    out = []
    out.append(f'<text x="{PANEL_X}" y="104" font-size="13" letter-spacing="1.5" '
               f'fill="{pal["chrome"]}">SYSTEM.INFO</text>')
    out.append(f'<text x="{PANEL_R}" y="104" font-size="12" text-anchor="end" '
               f'fill="{pal["live"]}">LIVE</text>')
    out.append(f'<circle cx="{PANEL_R - 34}" cy="100" r="3.5" fill="{pal["live"]}">'
               f'<animate attributeName="opacity" values="1;0.15;1" dur="1.6s" '
               f'repeatCount="indefinite"/></circle>')
    out.append(f'<line x1="{PANEL_X}" y1="122" x2="{PANEL_R}" y2="122" '
               f'stroke="{pal["sep"]}" stroke-width="1"/>')
    # handle pill
    pw = text_width(HANDLE, 14) + 26
    out.append(f'<rect x="{PANEL_X}" y="136" width="{pw:.0f}" height="27" rx="13.5" '
               f'fill="{pal["accent"]}"/>')
    out.append(f'<text x="{PANEL_X + pw/2:.1f}" y="154.5" font-size="14" text-anchor="middle" '
               f'font-weight="600" fill="#04120D">{esc(HANDLE)}</text>')

    y = ROW_Y0
    for lab, val in ROWS1:
        out.append(row(lab, val, y, pal)); y += ROW_H
    y += 8
    out.append(f'<line x1="{PANEL_X}" y1="{y-20}" x2="{PANEL_R}" y2="{y-20}" '
               f'stroke="{pal["sep"]}" stroke-width="1"/>')
    y += 10
    for lab, val in ROWS2:
        out.append(row(lab, val, y, pal)); y += ROW_H
    y += 8
    out.append(f'<line x1="{PANEL_X}" y1="{y-20}" x2="{PANEL_R}" y2="{y-20}" '
               f'stroke="{pal["sep"]}" stroke-width="1"/>')
    y += 10
    for lab, val in ROWS3:
        out.append(row(lab, val, y, pal)); y += ROW_H
    return "".join(out)


def chrome(pal):
    out = []
    out.append(f'<rect x="1" y="1" width="{W-2}" height="{H-2}" rx="14" fill="{pal["win"]}" '
               f'stroke="{pal["border"]}" stroke-width="1"/>')
    out.append(f'<path d="M1 15a14 14 0 0 1 14-14h{W-30}a14 14 0 0 1 14 14v30H1Z" '
               f'fill="{pal["titlebar"]}"/>')
    out.append(f'<line x1="1" y1="45" x2="{W-1}" y2="45" stroke="{pal["border"]}" stroke-width="1"/>')
    for i, c in enumerate(("#FF5F56", "#FFBD2E", "#27C93F")):
        out.append(f'<circle cx="{24 + i*20}" cy="23" r="5.5" fill="{c}"/>')
    out.append(f'<text x="{W/2}" y="27" font-size="13" text-anchor="middle" '
               f'fill="{pal["muted"]}">profile.sh --live</text>')
    # portrait frame
    out.append(f'<text x="{FR["x"]}" y="{FR["y"]-10}" font-size="12" letter-spacing="1.5" '
               f'fill="{pal["chrome"]}">VISUAL.MAP</text>')
    out.append(f'<text x="{FR["x"]+FR["w"]}" y="{FR["y"]-10}" font-size="11" text-anchor="end" '
               f'fill="{pal["muted"]}">300×340 · 1bpp</text>')
    out.append(f'<rect x="{FR["x"]-1}" y="{FR["y"]-1}" width="{FR["w"]+2}" height="{FR["h"]+2}" '
               f'rx="6" fill="{pal["frame"]}" stroke="{pal["border"]}" stroke-width="1"/>')
    return "".join(out)


# ----------------------------------------------------------------------------
# main build
# ----------------------------------------------------------------------------
def build(theme, dark_dots, light_dots, logo_pts, stats, static=False,
          phase="portrait", frac=0.0):
    pal = DARK if theme == "dark" else LIGHT
    dots = dark_dots if theme == "dark" else light_dots
    fx, fy = FR["x"], FR["y"]

    rng = np.random.default_rng(11)
    ys, xs = np.where(dots)
    n_dots = len(xs)

    # intro groups
    gid_intro = rng.integers(0, INTRO_GROUPS, size=dots.shape)
    # drift bands: noise before clustering so boundaries stay organic
    pts = np.column_stack([xs, ys]).astype(np.float64)
    noisy = pts + rng.normal(0, 4.0, pts.shape)
    gid_band = np.full(dots.shape, -1, dtype=int)
    gid_band[ys, xs] = kmeans2(noisy, N_BANDS, minit="++", seed=3, iter=25)[1]

    # travellers: optimal-transport matching L1->L2->L3->L1
    p1, p2, p3 = (logo_pts[k].astype(np.float64) for k in ("kotlin", "python", "code"))
    C12 = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=2)
    s12 = linear_sum_assignment(C12)[1]
    s23 = linear_sum_assignment(np.linalg.norm(p2[:, None, :] - p3[None, :, :], axis=2))[1]
    t1 = p1
    t2 = p2[s12]
    t3 = p3[s23[s12]]  # dot i: L1 -> L2 -> L3 by matched chain

    def disp(p):
        return np.column_stack([fx + p[:, 0] * SX, fy + p[:, 1] * SY])

    d1, d2, d3 = disp(t1), disp(t2), disp(t3)

    # ----- portrait layers -----
    intro_runs = runs_for(gid_intro, dots)
    intro_g = []
    for g, runs in intro_runs.items():
        path = runs_path(runs, fx, fy)
        if not path:
            continue
        if static:
            intro_g.append(f'<g><path d="{path}" fill="{pal["portrait"]}"/></g>')
        else:
            begin = g * 2.0 / INTRO_GROUPS
            intro_g.append(
                f'<g opacity="0"><animate attributeName="opacity" from="0" to="1" '
                f'begin="{begin:.2f}s" dur="0.5s" fill="freeze"/>'
                f'<path d="{path}" fill="{pal["portrait"]}"/></g>')

    band_g = []
    l1x, l1y = d1[:, 0].mean(), d1[:, 1].mean()
    for g in range(N_BANDS):
        sel = gid_band == g
        if not sel.any():
            continue
        runs = runs_for(np.where(sel, g, -1), dots).get(g, [])
        path = runs_path(runs, fx, fy)
        if not path:
            continue
        bx = fx + pts[gid_band[ys, xs] == g, 0].mean() * SX
        by = fy + pts[gid_band[ys, xs] == g, 1].mean() * SY
        dx, dy = DRIFT_FRAC * (l1x - bx), DRIFT_FRAC * (l1y - by)
        if static:
            if phase == "dissolve":
                band_g.append(f'<g transform="translate({frac*dx:.1f} {frac*dy:.1f})" '
                              f'opacity="{1-frac:.2f}"><path d="{path}" fill="{pal["portrait"]}"/></g>')
            elif phase == "logo1":
                band_g.append("")
            else:
                band_g.append(f'<g><path d="{path}" fill="{pal["portrait"]}"/></g>')
        else:
            band_g.append(
                f'<g><animateTransform attributeName="transform" type="translate" '
                f'values="0 0;0 0;{dx:.1f} {dy:.1f};{dx:.1f} {dy:.1f};0 0" '
                f'keyTimes="{KT_BAND}" dur="{LOOP}s" begin="{DELAY}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="1;1;0;0;1" keyTimes="{KT_BAND}" '
                f'dur="{LOOP}s" begin="{DELAY}s" repeatCount="indefinite"/>'
                f'<path d="{path}" fill="{pal["portrait"]}"/></g>')

    # ----- traveller layer -----
    trav = []
    if static and phase == "logo1":
        for i in range(N_TRAVELLERS):
            trav.append(f'<circle r="1.15" cx="{d1[i,0]:.1f}" cy="{d1[i,1]:.1f}" '
                        f'fill="{pal["portrait"]}"/>')
    elif not static:
        for i in range(N_TRAVELLERS):
            x1, y1 = d1[i]; x2, y2 = d2[i]; x3, y3 = d3[i]
            cx = f"{x1:.1f};{x1:.1f};{x1:.1f};{x2:.1f};{x2:.1f};{x3:.1f};{x3:.1f};{x1:.1f}"
            cy = f"{y1:.1f};{y1:.1f};{y1:.1f};{y2:.1f};{y2:.1f};{y3:.1f};{y3:.1f};{y1:.1f}"
            trav.append(
                f'<circle r="1.15" cx="{x1:.1f}" cy="{y1:.1f}" fill="{pal["portrait"]}" opacity="0">'
                f'<animate attributeName="cx" values="{cx}" keyTimes="{KT_POS}" dur="{LOOP}s" '
                f'begin="{DELAY}s" repeatCount="indefinite"/>'
                f'<animate attributeName="cy" values="{cy}" keyTimes="{KT_POS}" dur="{LOOP}s" '
                f'begin="{DELAY}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values="0;0;1;1;0;0" keyTimes="{KT_TOP}" '
                f'dur="{LOOP}s" begin="{DELAY}s" repeatCount="indefinite"/></circle>')

    # ----- assemble -----
    if static:
        layers = f'<g>{"".join(band_g)}</g>' + (f'<g>{"".join(trav)}</g>' if trav else "")
    else:
        layers = (
            f'<g><animate attributeName="opacity" values="1;1;0" keyTimes="0;0.79;1" '
            f'dur="2.8s" fill="freeze"/>{"".join(intro_g)}</g>'
            f'<g><animate attributeName="opacity" values="0;0;1" keyTimes="0;0.8;1" '
            f'dur="3.0s" fill="freeze"/>{"".join(band_g)}</g>'
            f'<g>{"".join(trav)}</g>')

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,'
        f'\'DejaVu Sans Mono\',monospace">'
        f'<title>Vitor Hugo F. Oliveira — profile banner</title>'
        f'<rect width="{W}" height="{H}" fill="{pal["bg"]}"/>'
        f'{chrome(pal)}'
        f'{info_panel(pal)}'
        f'{layers}'
        f'</svg>'
    )
    return svg


def main():
    from PIL import Image
    dark_dots, light_dots, mask = build_portraits()
    np.save(os.path.join(OUT_DIR, "portrait_dark.npy"), dark_dots)
    np.save(os.path.join(OUT_DIR, "portrait_light.npy"), light_dots)

    z = np.load(os.path.join(OUT_DIR, "logo_points.npz"))
    logo_pts = {k: z[k] for k in z.files}

    for theme, dd in (("dark", dark_dots), ("light", light_dots)):
        svg = build(theme, dark_dots, light_dots, logo_pts, None, static=False)
        p = os.path.join(OUT_DIR, f"{theme}.svg")
        with open(p, "w") as f:
            f.write(svg)
        print(f"{theme}.svg  {os.path.getsize(p)/1024:.0f} KB")
        svg_s = build(theme, dark_dots, light_dots, logo_pts, None, static=True)
        ps = os.path.join(OUT_DIR, f"static_{theme}.svg")
        with open(ps, "w") as f:
            f.write(svg_s)
    print("dots dark", int(dark_dots.sum()), "light", int(light_dots.sum()))


if __name__ == "__main__":
    main()
