
import numpy as np

def resize_bilinear(image,new_H,new_W):
    H,W=image.shape[:2]
    if image.ndim==2:
        out=np.zeros((new_H,new_W),float); sy,sx=H/new_H,W/new_W
        for i in range(new_H):
            y=(i+.5)*sy-.5; y0=int(np.floor(y)); y1=y0+1
            y0=max(0,y0); y1=min(H-1,y1); wy=y-y0
            for j in range(new_W):
                x=(j+.5)*sx-.5; x0=int(np.floor(x)); x1=x0+1
                x0=max(0,x0); x1=min(W-1,x1); wx=x-x0
                top=(1-wx)*image[y0,x0]+wx*image[y0,x1]
                bot=(1-wx)*image[y1,x0]+wx*image[y1,x1]
                out[i,j]=(1-wy)*top+wy*bot
        return out
    return np.stack([resize_bilinear(image[:,:,c],new_H,new_W) for c in range(image.shape[2])],2)

def calculate_ncc(patch,template):
    p=patch.astype(float); t=template.astype(float)
    p-=p.mean(); t-=t.mean()
    d=np.sqrt(np.sum(p*p)*np.sum(t*t))
    return 0 if d<1e-12 else np.sum(p*t)/d

def myNCC(image,template):
    H,W=image.shape; h,w=template.shape
    out=np.zeros((H-h+1,W-w+1))
    for r in range(H-h+1):
        for c in range(W-w+1):
            out[r,c]=calculate_ncc(image[r:r+h,c:c+w],template)
    return out

def myNCC_masked(image,template,mask):
    image=image.astype(float); template=template.astype(float); mask=mask.astype(float)
    H,W=image.shape; h,w=template.shape; N=np.sum(mask)
    ts=np.sum(template*mask); ts2=np.sum(template*template*mask)
    shape=(H+h-1,W+w-1)
    imf=np.fft.fft2(image,s=shape); mf=np.fft.fft2(mask,s=shape)
    im2f=np.fft.fft2(image*image,s=shape)
    si=np.real(np.fft.ifft2(imf*np.conj(mf)))
    si2=np.real(np.fft.ifft2(im2f*np.conj(mf)))
    tf=np.fft.fft2(template*mask,s=shape)
    cross=np.real(np.fft.ifft2(imf*np.conj(tf)))
    si=si[h-1:H,w-1:W]; si2=si2[h-1:H,w-1:W]; cross=cross[h-1:H,w-1:W]
    num=cross-si*ts/N
    iv=si2-si*si/N; tv=ts2-ts*ts/N
    den=np.sqrt(np.maximum(iv*tv,0)); out=np.zeros_like(num)
    ok=den>1e-12; out[ok]=num[ok]/den[ok]
    return out
