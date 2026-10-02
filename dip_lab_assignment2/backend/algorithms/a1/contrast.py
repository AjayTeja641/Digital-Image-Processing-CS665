"""
Assignment 1 — Contrast Enhancement
Linear stretch, Histogram Equalization, CLAHE, Histogram Matching
All operate on the Value (V) channel in HSV unless noted.
"""
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def _to_float01(img):
    img = np.asarray(img, dtype=np.float64)
    if img.max() > 1.0 + 1e-6:
        img = img / 255.0
    return np.clip(img, 0, 1)


def linear_contrast_stretch(img):
    """
    Linear contrast stretch on the V channel of HSV.
    Returns (enhanced_rgb, original_v, stretched_v) all in [0,1].
    """
    img = _to_float01(img)
    hsv = rgb_to_hsv(img)
    v = hsv[:, :, 2].copy()
    v_min, v_max = v.min(), v.max()
    if v_max - v_min < 1e-9:
        v_scaled = v.copy()
    else:
        v_scaled = (v - v_min) / (v_max - v_min)
    hsv[:, :, 2] = v_scaled
    enhanced = hsv_to_rgb(hsv)
    return enhanced, v, v_scaled


def hist_equalize(img):
    """
    Global histogram equalization on the V channel of HSV.
    Returns (equalized_rgb, original_v, equalized_v).
    """
    img = _to_float01(img)
    hsv = rgb_to_hsv(img)
    v = hsv[:, :, 2].copy()
    hist, _ = np.histogram(v.ravel(), 256, (0, 1))
    cdf = np.cumsum(hist).astype(np.float64)
    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-12)
    idx = np.clip(np.round(v * 255).astype(int), 0, 255)
    v_eq = cdf[idx]
    hsv[:, :, 2] = v_eq
    equalized = hsv_to_rgb(hsv)
    return equalized, v, v_eq


def clahe(img, window_size=63, num_bins=256, clip_limit=0.03):
    """
    Contrast-Limited Adaptive Histogram Equalization on V channel.
    Sliding-window implementation (educational; slower on large images).
    Returns enhanced RGB uint8-style float in [0,1].
    """
    img = _to_float01(img)
    hsv = rgb_to_hsv(img)
    v_channel = hsv[:, :, 2] * 255.0

    h, w = v_channel.shape
    half_w = max(1, window_size // 2)
    enhanced_v = np.zeros_like(v_channel)
    bin_edges = np.linspace(0, 256, num_bins + 1)

    # For speed on the web we process a downscaled version if the image is large
    # and then upscale the result; pure educational version stays pixel-wise.
    for r in range(h):
        r_min = max(0, r - half_w)
        r_max = min(h, r + half_w + 1)
        for c in range(w):
            c_min = max(0, c - half_w)
            c_max = min(w, c + half_w + 1)
            window = v_channel[r_min:r_max, c_min:c_max]
            n_pixels = window.size
            hist, _ = np.histogram(window, bins=bin_edges)
            actual_clip = max(1, int(clip_limit * n_pixels))
            excess = np.maximum(0, hist - actual_clip)
            hist = np.minimum(hist, actual_clip).astype(np.float64)
            hist += excess.sum() / num_bins
            cdf = hist.cumsum()
            cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-8)
            pixel_val = v_channel[r, c]
            bin_idx = int(np.clip(pixel_val / (256.0 / num_bins), 0, num_bins - 1))
            enhanced_v[r, c] = cdf_normalized[bin_idx] * 255.0

    hsv[:, :, 2] = np.clip(enhanced_v, 0, 255) / 255.0
    return hsv_to_rgb(hsv)


def hist_match(src_img, ref_img, num_bins=256):
    """
    Histogram matching of src to ref, per RGB channel, ignoring pure-black background.
    Returns matched image in [0,1].
    """
    src = _to_float01(src_img)
    ref = _to_float01(ref_img)
    src_mask = np.sum(src, axis=2) > 0.01
    ref_mask = np.sum(ref, axis=2) > 0.01
    new_img = np.zeros_like(src)

    for c in range(3):
        src_pixels = src[src_mask, c]
        ref_pixels = ref[ref_mask, c]
        if src_pixels.size == 0 or ref_pixels.size == 0:
            new_img[:, :, c] = src[:, :, c]
            continue
        src_hist, bin_edges = np.histogram(src_pixels, bins=num_bins, range=(0, 1))
        ref_hist, _ = np.histogram(ref_pixels, bins=num_bins, range=(0, 1))
        src_cdf = src_hist.cumsum() / (src_hist.sum() + 1e-12)
        ref_cdf = ref_hist.cumsum() / (ref_hist.sum() + 1e-12)
        table = np.zeros(num_bins)
        for i in range(num_bins):
            idx = np.argmin(np.abs(ref_cdf - src_cdf[i]))
            table[i] = bin_edges[min(idx, len(bin_edges) - 1)]
        src_indices = np.clip(np.round(src_pixels * (num_bins - 1)).astype(int), 0, num_bins - 1)
        new_img[src_mask, c] = table[src_indices]
    return new_img
