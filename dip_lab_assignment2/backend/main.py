
from fastapi import FastAPI,UploadFile,File,Form,HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pathlib import Path
import numpy as np,io,base64,json
from .algorithms.canny import canny
from .algorithms.corners import structure_tensor,detect
from .algorithms.ncc import resize_bilinear,myNCC,myNCC_masked
from .algorithms.sharpen import unsharp
from .algorithms.bokeh import bokeh
from .algorithms.a1.contrast import linear_contrast_stretch, hist_equalize, clahe, hist_match
from .algorithms.a1.threshold import manual_threshold, otsu_threshold, adaptive_threshold
from .algorithms.a1.interpolate import image_shrink, nearest_neighbor_resize, bilinear_resize, rotate_image

# ---- safety limits (public deployment) ----
MAX_UPLOAD_BYTES = 8 * 1024 * 1024   # 8 MB
MAX_SIDE = 1200                      # longest image side after load

app = FastAPI(
    title="DIP Lab — CS663 Assignments 1 & 2",
    description="Educational Digital Image Processing playground. Algorithms are the coursework implementations.",
    version="1.0.0",
)
# Allow browser access from any origin when self-hosted / demoed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

INDEX = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

async def _read_upload(upload: UploadFile) -> bytes:
    """Read upload with size + content-type guards."""
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(400, "Only image uploads are allowed.")
    data = await upload.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"Image too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB).")
    if len(data) == 0:
        raise HTTPException(400, "Empty file.")
    return data

def read_img(data: bytes):
    """Decode bytes → RGB float [0,1], optionally downscale large images."""
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image. Use PNG/JPEG/WebP.")
    w, h = im.size
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return np.asarray(im).astype(float) / 255.0

def encode(a):
    a=np.asarray(a)
    if a.ndim==2:
        lo,hi=np.percentile(a,[1,99]) if np.any(a) else (0,1)
        if hi<=lo:hi=lo+1e-9
        a=np.clip((a-lo)/(hi-lo),0,1)
        im=Image.fromarray((a*255).astype("uint8"))
    else: im=Image.fromarray((np.clip(a,0,1)*255).astype("uint8"))
    b=io.BytesIO(); im.save(b,"PNG")
    return "data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()

def result(items,meta={}):
    return {"images":[{"name":n,"data":encode(a)} for n,a in items],"meta":meta}

@app.get("/",response_class=HTMLResponse)
def home(): return INDEX.read_text(encoding="utf8")

@app.post("/api/canny")
async def api_canny(image:UploadFile=File(...),low:float=Form(50),high:float=Form(100),sigma:float=Form(1.4)):
    x=read_img(await _read_upload(image)); r=canny(x,5,sigma,low,high)
    return result([("Original",x),("Grayscale",r["gray"]),("Gaussian Blur",r["blurred"]),
                   ("Gradient Magnitude",r["magnitude"]),("NMS",r["nms"]),("Final Edges",r["edges"])],
                  {"low":low,"high":high,"sigma":sigma})

def mark_points(img, ys, xs, color, half=3):
    """Draw a small filled square marker around each (y, x) so corners are visible."""
    out = img.copy()
    H, W = out.shape[:2]
    for y0, x0 in zip(ys, xs):
        y1 = max(0, int(y0) - half)
        y2 = min(H, int(y0) + half + 1)
        x1 = max(0, int(x0) - half)
        x2 = min(W, int(x0) + half + 1)
        out[y1:y2, x1:x2] = color
    return out

@app.post("/api/corners")
async def api_corners(image:UploadFile=File(...),k:float=Form(.04),keep:float=Form(3)):
    x=read_img(await _read_upload(image)); e=structure_tensor(x,1,1.5); f=detect(e,k,7,keep)
    hy, hx = np.where(f["harris_bin"])
    sy, sx = np.where(f["shi_bin"])
    h = mark_points(x, hy, hx, [1, 0, 0], half=3)
    s = mark_points(x, sy, sx, [0, 1, 0], half=3)
    return result([("Original",x),("Largest Eigenvalue",e[...,1]),("Second Eigenvalue",e[...,0]),
                   ("Harris Cornerness",f["harris"]),("Harris Corners",h),("Shi-Tomasi Corners",s)],
                  {"harris_corners":int(f["harris_bin"].sum()),"shi_tomasi_corners":int(f["shi_bin"].sum())})

@app.post("/api/sharpen")
async def api_sharp(image:UploadFile=File(...),strength:float=Form(1.5)):
    x=read_img(await _read_upload(image)); return result([("Original",x),("Sharpened",unsharp(x,1,strength,5))],
                                                   {"strength":strength,"sigma":1,"kernel_size":5})

@app.post("/api/template")
async def api_template(scene:UploadFile=File(...),template:UploadFile=File(...),size:int=Form(51)):
    s=read_img(await _read_upload(scene)); t=read_img(await _read_upload(template))
    s=resize_bilinear(s,max(1,s.shape[0]//5),max(1,s.shape[1]//5)); t=resize_bilinear(t,size,size)
    names=["Red","Green","Blue"]; maps=[]; best=[]
    for c,nm in enumerate(names):
        n=myNCC(s[:,:,c],t[:,:,c]); maps.append(n)
        p=np.unravel_index(np.argmax(n),n.shape)
        best.append({"channel":nm,"ncc":float(n[p]),"row":int(p[0]),"col":int(p[1])})
    return result([(names[i]+" NCC",maps[i]) for i in range(3)],
                  {"scene_reduction":"5x","template_size":size,"best":best})

@app.post("/api/bokeh")
async def api_bokeh(image:UploadFile=File(...),points:str=Form(...),diameter:int=Form(50)):
    x=read_img(await _read_upload(image)); pts=np.asarray(json.loads(points),float)
    if len(pts)<3: raise HTTPException(400,"Select at least 3 foreground points.")
    from matplotlib.path import Path as MPath
    h,w=x.shape[:2]; yy,xx=np.mgrid[:h,:w]
    fg=MPath(pts).contains_points(np.c_[xx.ravel(),yy.ravel()]).reshape(h,w)
    return result([("Original",x),("Bokeh",bokeh(x,fg,~fg,diameter))],
                  {"diameter":diameter,"foreground_points":len(pts)})

@app.post("/api/masked-ncc")
async def api_masked(scene:UploadFile=File(...),template:UploadFile=File(...),
                     cx:float=Form(...),cy:float=Form(...),radius:float=Form(...),size:int=Form(201)):
    s=read_img(await _read_upload(scene)); t=read_img(await _read_upload(template))
    t=resize_bilinear(t,size,size); sx=size/template.shape[1]; sy=size/template.shape[0]
    y,x=np.ogrid[:size,:size]; mask=((x-cx*sx)**2+(y-cy*sy)**2 <= (radius*(sx+sy)/2)**2).astype(float)
    maps=[]; best=[]; names=["Red","Green","Blue"]
    for c,nm in enumerate(names):
        n=myNCC_masked(s[:,:,c],t[:,:,c],mask); maps.append(n); p=np.unravel_index(np.argmax(n),n.shape)
        best.append({"channel":nm,"ncc":float(n[p]),"row":int(p[0]),"col":int(p[1])})
    return result([(names[i]+" Masked NCC",maps[i]) for i in range(3)],
                  {"template_size":size,"best":best})

# ---------- Assignment 1 ----------

@app.post("/api/a1/stretch")
async def api_stretch(image: UploadFile = File(...)):
    x = read_img(await _read_upload(image))
    enh, v, vs = linear_contrast_stretch(x)
    return result([("Original", x), ("Contrast Stretched", enh),
                   ("Original V", v), ("Stretched V", vs)],
                  {"method": "linear_contrast_stretch"})

@app.post("/api/a1/histeq")
async def api_histeq(image: UploadFile = File(...)):
    x = read_img(await _read_upload(image))
    eq, v, ve = hist_equalize(x)
    return result([("Original", x), ("Histogram Equalized", eq),
                   ("Original V", v), ("Equalized V", ve)],
                  {"method": "histogram_equalization"})

@app.post("/api/a1/clahe")
async def api_clahe(image: UploadFile = File(...),
                    window: int = Form(63), clip: float = Form(0.03)):
    x = read_img(await _read_upload(image))
    # limit size for web responsiveness (CLAHE is O(H*W*win^2))
    h, w = x.shape[:2]
    if max(h, w) > 400:
        scale = 400 / max(h, w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        x = resize_bilinear(x, new_h, new_w)
    enh = clahe(x, window_size=window, clip_limit=clip)
    return result([("Original (possibly resized)", x), ("CLAHE", enh)],
                  {"window": window, "clip": clip, "note": "large images auto-resized for speed"})

@app.post("/api/a1/histmatch")
async def api_histmatch(source: UploadFile = File(...), reference: UploadFile = File(...),
                        bins: int = Form(128)):
    s = read_img(await _read_upload(source))
    r = read_img(await _read_upload(reference))
    matched = hist_match(s, r, num_bins=bins)
    return result([("Source", s), ("Reference", r), ("Matched", matched)],
                  {"bins": bins})

@app.post("/api/a1/manual-thresh")
async def api_manual_thresh(image: UploadFile = File(...), T: float = Form(128),
                            invert: bool = Form(False)):
    x = read_img(await _read_upload(image))
    gray, binary, t = manual_threshold(x, T, invert)
    return result([("Original", x), ("Grayscale", gray / 255.0),
                   ("Binary", binary / 255.0)],
                  {"T": t, "invert": invert})

@app.post("/api/a1/otsu")
async def api_otsu(image: UploadFile = File(...), invert: bool = Form(False)):
    x = read_img(await _read_upload(image))
    gray, binary, t = otsu_threshold(x, invert)
    return result([("Original", x), ("Grayscale", gray / 255.0),
                   ("Binary (Otsu)", binary / 255.0)],
                  {"T_otsu": t, "invert": invert})

@app.post("/api/a1/adaptive")
async def api_adaptive(image: UploadFile = File(...), window: int = Form(21),
                       C: float = Form(5), foreground: str = Form("dark")):
    x = read_img(await _read_upload(image))
    gray, binary, tmap = adaptive_threshold(x, window, C, foreground)
    return result([("Original", x), ("Grayscale", gray / 255.0),
                   ("Threshold Map", tmap / 255.0), ("Binary", binary / 255.0)],
                  {"window": window, "C": C, "foreground": foreground})

@app.post("/api/a1/shrink")
async def api_shrink(image: UploadFile = File(...), d: int = Form(2)):
    x = read_img(await _read_upload(image))
    out = image_shrink(x, d)
    return result([("Original", x), (f"Subsampled d={d}", out)],
                  {"d": d})

@app.post("/api/a1/rotate")
async def api_rotate(image: UploadFile = File(...), angle: float = Form(5.0),
                     method: str = Form("bilinear")):
    x = read_img(await _read_upload(image))
    # limit size
    h, w = x.shape[:2]
    if max(h, w) > 500:
        scale = 500 / max(h, w)
        x = resize_bilinear(x, max(1, int(h * scale)), max(1, int(w * scale)))
    out = rotate_image(x, angle, method=method)
    return result([("Original", x), (f"Rotated ({method})", out)],
                  {"angle": angle, "method": method})
