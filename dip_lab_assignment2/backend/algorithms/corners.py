
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def gaussian_kernel1d(sigma,order=0,radius=None):
    if radius is None: radius=max(1,int(4*sigma+0.5))
    x=np.arange(-radius,radius+1,dtype=float); s2=sigma*sigma
    k=np.exp(-0.5*x*x/s2); k/=k.sum()
    if order==0:return k
    if order==1:
        k=-(x/s2)*k; k-=k.mean(); return k
    raise ValueError("invalid order")

def conv1d_reflect(arr,kernel,axis):
    r=len(kernel)//2; pad=[(0,0)]*arr.ndim; pad[axis]=(r,r)
    a=np.pad(arr,pad,mode="reflect")
    w=sliding_window_view(a,len(kernel),axis=axis)
    return np.tensordot(w,kernel[::-1],axes=([-1],[0]))

def gaussian_filter2d(arr,sigma,order=(0,0)):
    oy,ox=order
    arr=conv1d_reflect(arr,gaussian_kernel1d(sigma,ox),1)
    return conv1d_reflect(arr,gaussian_kernel1d(sigma,oy),0)

def to_gray(img):
    if img.ndim==2:return img.astype(float)
    return (0.299*img[...,0]+0.587*img[...,1]+0.114*img[...,2]).astype(float)

def structure_tensor(img,sigma_deriv=1.0,sigma_window=1.5):
    g=to_gray(img)
    ix=gaussian_filter2d(g,sigma_deriv,(0,1))
    iy=gaussian_filter2d(g,sigma_deriv,(1,0))
    ixx=gaussian_filter2d(ix*ix,sigma_window)
    iyy=gaussian_filter2d(iy*iy,sigma_window)
    ixy=gaussian_filter2d(ix*iy,sigma_window)
    tr=ixx+iyy; diff=ixx-iyy
    disc=np.sqrt((diff/2)**2+ixy**2)
    lmax=tr/2+disc; lmin=np.clip(tr/2-disc,0,None)
    return np.stack([lmin,lmax],axis=-1)

def maximum_filter2d(arr,size):
    r=size//2; a=np.pad(arr,r,mode="reflect")
    w=sliding_window_view(a,(size,size))
    return w.max(axis=(-1,-2))

def nms(response,size=7):
    m=maximum_filter2d(response,size)
    return np.where(response==m,response,0)

def harris(eig,k=.04):
    a,b=eig[...,0],eig[...,1]
    return a*b-k*(a+b)**2

def percentile_binarize(response,keep_pct):
    """Keep the strongest keep_pct percent of positive responses.
    Uses >= so that identical peak values (common after NMS) are retained.
    """
    v=response[response>0]
    if v.size==0:return np.zeros_like(response,bool)
    t=np.percentile(v,100-keep_pct)
    return response>=t

def detect(eig,k=.04,corner_window=7,keep_pct=3):
    h=harris(eig,k); s=eig[...,0]; edge=np.clip(-h,0,None)
    hn=nms(np.clip(h,0,None),corner_window)
    sn=nms(np.clip(s,0,None),corner_window)
    en=nms(edge,1)
    return {"harris":h,"shi":s,"harris_bin":percentile_binarize(hn,keep_pct),
            "shi_bin":percentile_binarize(sn,keep_pct),"harris_nms":hn,"shi_nms":sn,
            "edge_bin":percentile_binarize(en,18)}
