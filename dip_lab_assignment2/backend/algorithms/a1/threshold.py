"""
Assignment 1 — Thresholding
Manual, Otsu, Adaptive (local mean via integral image)
"""
import numpy as np


def rgb_to_gray(img):
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        return img
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def to_255(gray):
    gray = np.asarray(gray, dtype=np.float64)
    if gray.max() <= 1.0 + 1e-6:
        gray = gray * 255.0
    return np.clip(gray, 0, 255)


def manual_threshold(img, T, invert=False):
    """
    Global manual threshold.
    Returns (gray_0_255, binary_0_255, T)
    """
    gray = to_255(rgb_to_gray(img))
    binary = np.zeros_like(gray)
    binary[gray >= T] = 255
    if invert:
        binary = 255 - binary
    return gray, binary, float(T)


def otsu_threshold(img, invert=False):
    """
    Otsu's method (maximize between-class variance).
    Returns (gray_0_255, binary_0_255, best_T)
    """
    gray = to_255(rgb_to_gray(img))
    gray_u8 = gray.astype(np.uint8)
    hist = np.bincount(gray_u8.ravel(), minlength=256).astype(np.float64)
    total = gray_u8.size
    intensity = np.arange(256, dtype=np.float64)
    total_intensity = np.sum(intensity * hist)

    w_bg = 0.0
    sum_bg = 0.0
    best_var = -1.0
    best_T = 0

    for T in range(256):
        w_bg += hist[T]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += T * hist[T]
        mean_bg = sum_bg / w_bg
        mean_fg = (total_intensity - sum_bg) / w_fg
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var = var
            best_T = T

    binary = np.zeros_like(gray)
    binary[gray >= best_T] = 255
    if invert:
        binary = 255 - binary
    return gray, binary, float(best_T)


def adaptive_threshold(img, window_size=21, C=5, foreground="dark"):
    """
    Local adaptive threshold using integral image (mean - C).
    Returns (gray_0_255, binary_0_255, threshold_map)
    """
    gray = to_255(rgb_to_gray(img))
    if window_size % 2 == 0 or window_size < 3:
        window_size = max(3, window_size | 1)  # force odd

    radius = window_size // 2
    padded = np.pad(gray, radius, mode="reflect")
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(padded, axis=0), axis=1)

    local_sum = (
        integral[window_size:, window_size:]
        - integral[:-window_size, window_size:]
        - integral[window_size:, :-window_size]
        + integral[:-window_size, :-window_size]
    )
    local_mean = local_sum / (window_size * window_size)
    thresh_map = np.clip(local_mean - C, 0, 255)

    if foreground == "dark":
        binary = np.where(gray < thresh_map, 0, 255).astype(np.float64)
    else:
        binary = np.where(gray >= thresh_map, 0, 255).astype(np.float64)

    return gray, binary, thresh_map
