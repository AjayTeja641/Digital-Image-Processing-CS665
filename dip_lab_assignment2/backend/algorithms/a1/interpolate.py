"""
Assignment 1 — Interpolation & Geometric transforms
Shrink, Nearest-Neighbor, Bilinear, simple rotation.
Bicubic is available but intentionally limited for web performance.
"""
import numpy as np


def image_shrink(img, d):
    """Subsample by integer factor d (shows Moiré for large d)."""
    d = max(1, int(d))
    return img[::d, ::d].copy()


def nearest_neighbor_resize(img, scale):
    """
    Enlarge a 2-D (or multi-channel) image by integer scale using nearest neighbour.
    scale is the factor between original samples (as in the assignment).
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        m, n = img.shape
        new_m = scale * (m - 1) + 1
        new_n = scale * (n - 1) + 1
        out = np.empty((new_m, new_n), dtype=np.float64)
        # place original samples
        for i in range(m):
            for j in range(n):
                out[scale * i, scale * j] = img[i, j]
        # fill the rest
        for i in range(new_m):
            for j in range(new_n):
                if i % scale == 0 and j % scale == 0:
                    continue
                idx = i // scale
                idy = j // scale
                best = None
                best_dist = 1e18
                for di, dj in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    ni, nj = scale * (idx + di), scale * (idy + dj)
                    if ni >= new_m or nj >= new_n:
                        continue
                    dist = abs(ni - i) + abs(nj - j)
                    if dist < best_dist:
                        best_dist = dist
                        best = (ni, nj)
                out[i, j] = out[best]
        return out
    # multi-channel
    return np.stack([nearest_neighbor_resize(img[..., c], scale) for c in range(img.shape[-1])], -1)


def bilinear_resize(img, scale):
    """Enlarge by integer scale using bilinear interpolation (assignment style)."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        m, n = img.shape
        new_m = scale * (m - 1) + 1
        new_n = scale * (n - 1) + 1
        out = np.empty((new_m, new_n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                out[scale * i, scale * j] = img[i, j]
        for i in range(new_m):
            for j in range(new_n):
                if i % scale == 0 and j % scale == 0:
                    continue
                x1 = i // scale
                y1 = j // scale
                x2 = min(x1 + 1, m - 1)
                y2 = min(y1 + 1, n - 1)
                dx = (i / float(scale)) - x1
                dy = (j / float(scale)) - y1
                I1 = out[x1 * scale, y1 * scale] * (1 - dx) + out[x2 * scale, y1 * scale] * dx
                I2 = out[x1 * scale, y2 * scale] * (1 - dx) + out[x2 * scale, y2 * scale] * dx
                out[i, j] = I1 * (1 - dy) + I2 * dy
        return out
    return np.stack([bilinear_resize(img[..., c], scale) for c in range(img.shape[-1])], -1)


def _bilinear_sample(img, x, y):
    """Sample a multi-channel image at fractional (x, y) with bilinear."""
    h, w = img.shape[:2]
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = min(x1 + 1, w - 1)
    y2 = min(y1 + 1, h - 1)
    x1 = max(0, x1)
    y1 = max(0, y1)
    dx = x - x1
    dy = y - y1
    I1 = img[y1, x1] * (1 - dx) + img[y1, x2] * dx
    I2 = img[y2, x1] * (1 - dx) + img[y2, x2] * dx
    return I1 * (1 - dy) + I2 * dy


def _nearest_sample(img, x, y):
    h, w = img.shape[:2]
    xi = int(round(x))
    yi = int(round(y))
    xi = max(0, min(w - 1, xi))
    yi = max(0, min(h - 1, yi))
    return img[yi, xi]


def rotate_image(img, angle_deg, method="bilinear"):
    """
    Rotate image by angle_deg (degrees) around its centre.
    method: 'bilinear' or 'nearest'
    Output has the same shape as input.
    """
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    theta = np.radians(angle_deg)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    sample = _bilinear_sample if method == "bilinear" else _nearest_sample

    for y in range(h):
        for x in range(w):
            # inverse mapping
            xr = x - cx
            yr = y - cy
            xs = cos_t * xr + sin_t * yr + cx
            ys = -sin_t * xr + cos_t * yr + cy
            if 0 <= xs < w - 1 and 0 <= ys < h - 1:
                out[y, x] = sample(img, xs, ys)
    return out
