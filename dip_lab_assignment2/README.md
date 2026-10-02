# DIP Lab — CS663 Assignments 1 & 2

Interactive **Digital Image Processing** playground.

Upload your own images in the browser, tune parameters, and see the **actual Python algorithms** from the IIT Bombay CS663 coursework — not black-box OpenCV calls.

**Live Demo:** [https://digital-image-processing-cs665.onrender.com](https://digital-image-processing-cs665.onrender.com)

**Repo:** [AjayTeja641/Digital-Image-Processing-CS665](https://github.com/AjayTeja641/Digital-Image-Processing-CS665)

> Note: The free Render instance sleeps after inactivity. The first load after sleep may take 30–60 seconds.

---

## Features

### Assignment 1
| Algorithm | What it does |
|-----------|--------------|
| Linear Contrast Stretch | Stretch HSV Value channel to full range |
| Histogram Equalization | Global HE on luminance |
| CLAHE | Contrast-Limited Adaptive HE (sliding window) |
| Histogram Matching | Match colours of one image to another |
| Manual Threshold | Global T + optional invert |
| Otsu Threshold | Automatic optimal threshold |
| Adaptive Threshold | Local mean − C (handles lighting gradients) |
| Image Shrink | Subsampling (Moiré patterns) |
| Rotation | Bilinear / nearest-neighbour around centre |

### Assignment 2
| Algorithm | What it does |
|-----------|--------------|
| Canny Edge Detection | Full pipeline + intermediate stages |
| Harris / Shi-Tomasi | Structure tensor + visible corner markers |
| NCC Template Matching | Per-channel normalised cross-correlation |
| Unsharp Masking | Classical sharpening |
| Bokeh | Circular disc blur on background, sharp polygonal foreground |

---

## Project structure
