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

```
dip_lab_assignment2/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   └── algorithms/
│       ├── __init__.py
│       ├── canny.py
│       ├── corners.py
│       ├── ncc.py
│       ├── sharpen.py
│       ├── bokeh.py
│       └── a1/
│           ├── __init__.py
│           ├── contrast.py
│           ├── threshold.py
│           └── interpolate.py
├── frontend/
│   └── index.html
├── requirements.txt
├── Dockerfile
├── Procfile
├── LICENSE
├── .gitignore
└── README.md
```

There are **no API keys, secrets, or credentials** in this repository.

---

## Run locally

```bash
git clone https://github.com/AjayTeja641/Digital-Image-Processing-CS665.git
cd Digital-Image-Processing-CS665/dip_lab_assignment2
python -m venv .venv

# Windows:
#   .venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in your browser.

If port 8000 is busy, use port 8001:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

---

## Live deployment

Already deployed on Render:

**https://digital-image-processing-cs665.onrender.com**

To redeploy (Render / Railway / any host):

| Field | Value |
|-------|--------|
| **Build command** | `pip install -r requirements.txt` |
| **Start command** | `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |

Or use the included `Dockerfile`:

```bash
docker build -t dip-lab .
docker run -p 8000:8000 dip-lab
```

---

## Security notes

| Topic | Status |
|-------|--------|
| API keys / secrets in repo | **None** — safe to open-source |
| Upload type | Only `image/*` accepted |
| Upload size | Hard limit **8 MB** |
| Image dimensions | Auto-downscaled if longest side > 1200 px |
| Files written to disk | No — images stay in memory |
| Path traversal / RCE | Not possible with current code |
| CORS | Open (`*`) for public demo use |
| Rate limiting | Not included — add Cloudflare or `slowapi` if traffic grows |
| Authentication | None (open educational tool) |

---

## Design principle

```
Your Python algorithm  →  FastAPI  →  Browser
```

Intermediate results are shown on purpose so the site remains an **educational DIP laboratory**, not a generic photo-filter website.

---

## Authors

- **Kolanu Sarvajith** (24B0924)
- **Kondi Ajay Teja** (24B0617)

Course: **CS663 — Digital Image Processing**, IIT Bombay

---

## License

MIT — see [LICENSE](LICENSE). Free to use, modify, and redistribute with attribution.
