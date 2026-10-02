
import numpy as np
def gaussian_kernel(size,sigma):
    k=size//2; x,y=np.mgrid[-k:k+1,-k:k+1]
    g=np.exp(-(x*x+y*y)/(2*sigma*sigma)); return g/g.sum()
def convolve(image,kernel):
    kh,kw=kernel.shape; ph,pw=kh//2,kw//2
    p=np.pad(image,((ph,ph),(pw,pw),(0,0)),mode="reflect")
    k=kernel[:,:,None]; out=np.zeros_like(image,float)
    for i in range(kh):
        for j in range(kw): out+=p[i:i+image.shape[0],j:j+image.shape[1]]*k[i,j]
    return out
def unsharp(image,sigma=1,s=1.5,size=5):
    x=image.astype(float); lo,hi=x.min(),x.max()
    y=convolve(x,gaussian_kernel(size,sigma))
    return np.clip(x+s*(x-y),lo,hi)
