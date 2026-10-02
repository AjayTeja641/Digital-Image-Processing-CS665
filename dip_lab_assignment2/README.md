# DIP Lab — CS663 Assignments 1 & 2

Interactive **Digital Image Processing** playground.

Upload your own images in the browser, tune parameters, and see the **actual Python algorithms** from the IIT Bombay CS663 coursework — not black-box OpenCV calls.

**Live idea:** anyone can clone → run locally, or you can deploy once and share a public URL.

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
dip_lab/
├── backend/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app + all endpoints
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
│   └── index.html              # Single-page UI (no build step)
├── requirements.txt
├── Dockerfile                  # Optional one-command deploy
├── LICENSE                     # MIT
├── .gitignore
└── README.md
```

There are **no API keys, secrets, or credentials** in this repository.

---

## Run locally (anyone can do this)

```bash
git clone https://github.com/<your-username>/dip-lab.git
cd dip-lab
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in your browser.

Upload any PNG/JPEG, choose a tool, adjust parameters, click Process.

---

## Deploy so strangers can use it in their browser

You need **one always-on server**. Free options that work well:

### Option A — Render.com (easiest)
1. Push this repo to GitHub.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Connect the repo.
4. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python -m uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - Runtime: Python 3.11+
5. Deploy → you get a public URL like `https://dip-lab.onrender.com`

### Option B — Docker (Railway / Fly.io / any VPS)
```bash
docker build -t dip-lab .
docker run -p 8000:8000 dip-lab
```

### Option C — Hugging Face Spaces
Create a Space with Docker SDK, push the same Dockerfile.

After deployment, **anyone** opens the URL in their browser and uses the lab — no install required.

---

## Security notes (important for public deploy)

| Topic | Status |
|-------|--------|
| API keys / secrets in repo | **None.** Safe to open-source. |
| Environment files | `.env` is gitignored. Do not add secrets. |
| Upload type | Only `image/*` accepted. |
| Upload size | Hard limit **8 MB**. |
| Image dimensions | Auto-downscaled if longest side > 1200 px. |
| Path traversal / RCE | Images are decoded with Pillow in memory; never written to disk as user-controlled paths. |
| CORS | Open (`*`) so the static frontend can call the API. Fine for a public demo. |
| Rate limiting | **Not implemented.** For a busy public site you should add something like `slowapi` or put Cloudflare in front. |
| Authentication | None — this is an open educational tool. |
| Expensive algorithms | CLAHE / rotation / bokeh are limited by the size caps above to reduce DoS risk. |

**What a malicious user could still do**
- Spam the free tier with many large (but still ≤ 8 MB) images → higher CPU bills / rate limits on the host.
- Mitigate with host-level rate limits or Cloudflare.

**What they cannot do**
- Steal credentials (there are none).
- Write arbitrary files on the server.
- Execute shell commands via the image endpoints.

---

## Design principle

```
Your Python algorithm  →  FastAPI  →  Browser
```

Intermediate results are shown on purpose so the site remains an **educational DIP laboratory**, not a generic photo filter site.

---

## Authors

- Kolanu Sarvajith (24B0924)
- Kondi Ajay Teja (24B0617)

Course: **CS663 — Digital Image Processing**, IIT Bombay

---

## License

IIT Bombay — see [LICENSE](LICENSE). Free to use, modify, and redistribute with attribution.
