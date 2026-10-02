import numpy as np
import cv2


def disc_filter(diameter):
    """
    Create a circular disc kernel of odd size.

    diameter is treated as the approximate diameter; the actual
    kernel is always (2*r+1) x (2*r+1) with r = diameter // 2.
    """
    diameter = max(1, int(diameter))
    radius = diameter // 2
    size = 2 * radius + 1

    y, x = np.ogrid[:size, :size]
    center = radius
    mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2

    kernel = np.zeros((size, size), dtype=np.float64)
    kernel[mask] = 1.0

    total = kernel.sum()
    if total > 0:
        kernel /= total

    return kernel


def bokeh(image, fg, bg, diameter):
    """
    Apply circular-disc blur to the background while keeping the
    foreground completely sharp.

    Uses a normalized masked convolution so that foreground pixels
    never contribute to the blurred background (no colour leakage
    near the subject boundary).

    fg, bg : boolean masks of shape (H, W)
    """
    image = np.asarray(image, dtype=np.float64)
    fg = np.asarray(fg, dtype=bool)
    bg = np.asarray(bg, dtype=bool)

    kernel = disc_filter(diameter)

    # Normalised masked blur:
    #   blurred = filter(image * bg) / filter(bg)
    # This guarantees that only background pixels enter the sum.
    bg_f = bg.astype(np.float64)

    # Denominator is the same for every channel
    denom = cv2.filter2D(bg_f, -1, kernel, borderType=cv2.BORDER_REFLECT)

    # Avoid division by zero (pixels with no background support)
    denom = np.maximum(denom, 1e-12)

    blurred = np.empty_like(image)
    for c in range(image.shape[2]):
        num = cv2.filter2D(image[:, :, c] * bg_f, -1, kernel,
                           borderType=cv2.BORDER_REFLECT)
        blurred[:, :, c] = num / denom

    result = image.copy()
    result[bg] = blurred[bg]
    # Foreground stays exactly as the original
    result[fg] = image[fg]

    return result
