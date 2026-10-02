"""Portrait processing pipeline for the GitHub banner.
Crop -> enhance -> (segment) -> 1-bit Floyd-Steinberg serpentine dither.
"""
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy import ndimage

GRID_W, GRID_H = 300, 340


def load_crop(src, box_frac, grid=(GRID_W, GRID_H)):
    """box_frac = (x0, y0, x1, y1) as fractions of the source image."""
    img = Image.open(src).convert("RGB")
    W, H = img.size
    x0, y0, x1, y1 = (int(round(box_frac[0] * W)), int(round(box_frac[1] * H)),
                      int(round(box_frac[2] * W)), int(round(box_frac[3] * H)))
    return img.crop((x0, y0, x1, y1)).resize(grid, Image.LANCZOS)


def enhance_gray(crop):
    g = crop.convert("L")
    g = ImageOps.autocontrast(g, cutoff=1)
    g = ImageEnhance.Contrast(g).enhance(1.3)
    g = g.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
    return g


def subject_mask(crop, thresh=38.0, erode=1, local=False):
    """Segment the lit subject from the (dark) studio background."""
    rgb = np.asarray(crop).astype(np.float32)
    if local:
        # per-pixel background estimate via median filter
        bg = np.stack([ndimage.median_filter(rgb[:, :, c], size=25) for c in range(3)], axis=2)
    else:
        border = np.concatenate([
            rgb[:, :8].reshape(-1, 3), rgb[:, -8:].reshape(-1, 3),
            rgb[:8, :].reshape(-1, 3),
        ])
        bg = np.median(border, axis=0)[None, None, :]
    dist = np.linalg.norm(rgb - bg, axis=2)
    m = dist > thresh
    m = ndimage.binary_closing(m, structure=np.ones((5, 5)))
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (1 + int(np.argmax(sizes)))
    if erode:
        m = ndimage.binary_erosion(m, iterations=erode)
    return m


def floyd_steinberg(vals, dark_mode, mask=None):
    """1-bit FS dither, serpentine. Dark mode: dot on lit pixels (v>=.5).
    Light mode: dot on dark pixels. mask==False -> hard no-dot, no diffusion."""
    h, w = vals.shape
    img = vals.astype(np.float64).copy()
    out = np.zeros((h, w), np.uint8)
    for y in range(h):
        xs = range(w) if y % 2 == 0 else range(w - 1, -1, -1)
        d = 1 if y % 2 == 0 else -1
        for x in xs:
            if mask is not None and not mask[y, x]:
                continue  # hard clear: no dot, no error diffusion from bg
            v = img[y, x]
            b = int(v >= 0.5) if dark_mode else int(v < 0.5)
            q = b if dark_mode else (1 - b)
            out[y, x] = b
            e = v - q
            for dx, dy, wt in ((d, 0, 7 / 16), (-d, 1, 3 / 16), (0, 1, 5 / 16), (d, 1, 1 / 16)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    img[ny, nx] += e * wt
    return out.astype(bool)


def render_dots(dots, color, bg, size=(GRID_W, GRID_H), scale=3):
    """Render the boolean dot grid as a PNG for preview."""
    h, w = dots.shape
    img = Image.new("RGB", (w, h), bg)
    px = img.load()
    for y in range(h):
        row = dots[y]
        x = 0
        while x < w:
            if row[x]:
                x2 = x
                while x2 < w and row[x2]:
                    x2 += 1
                for xx in range(x, x2):
                    px[xx, y] = color
                x = x2
            else:
                x += 1
    return img.resize((w * scale, h * scale), Image.NEAREST)
