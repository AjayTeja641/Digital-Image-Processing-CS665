
import numpy as np

def rgb_to_grayscale(image):
    return 0.299*image[:,:,0] + 0.587*image[:,:,1] + 0.114*image[:,:,2]

def convolve2d(image, kernel):
    h,w=image.shape
    kh,kw=kernel.shape
    ph,pw=kh//2,kw//2
    padded=np.pad(image,((ph,ph),(pw,pw)),mode="constant")
    out=np.zeros_like(image,dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i,j]*padded[i:i+h,j:j+w]
    return out

def gaussian_blur(image,kernel_size=5,sigma=1.4):
    coords=np.arange(kernel_size)-kernel_size//2
    x,y=np.meshgrid(coords,coords)
    g=np.exp(-(x*x+y*y)/(2*sigma*sigma))
    g/=g.sum()
    return convolve2d(image,g)

def compute_gradients(image):
    sx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],float)
    sy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]],float)
    gx=convolve2d(image,sx); gy=convolve2d(image,sy)
    mag=np.sqrt(gx**2+gy**2)
    direction=np.arctan2(gy,gx)*180/np.pi
    direction[direction<0]+=180
    return gx,gy,mag,direction

def non_maximum_suppression(magnitude,direction):
    h,w=magnitude.shape
    out=np.zeros_like(magnitude)
    for i in range(1,h-1):
        for j in range(1,w-1):
            a=direction[i,j]
            if a<22.5 or a>=157.5: q,r=magnitude[i,j+1],magnitude[i,j-1]
            elif a<67.5: q,r=magnitude[i+1,j-1],magnitude[i-1,j+1]
            elif a<112.5: q,r=magnitude[i+1,j],magnitude[i-1,j]
            else: q,r=magnitude[i-1,j-1],magnitude[i+1,j+1]
            if magnitude[i,j]>=q and magnitude[i,j]>=r: out[i,j]=magnitude[i,j]
    return out

def double_threshold(image,low,high):
    e=np.zeros_like(image,dtype=np.uint8)
    e[image>=high]=255
    e[(image>=low)&(image<high)]=50
    return e

def hysteresis(edges):
    h,w=edges.shape
    stack=list(zip(*np.where(edges==255)))
    while stack:
        i,j=stack.pop()
        for di in (-1,0,1):
            for dj in (-1,0,1):
                ni,nj=i+di,j+dj
                if 0<=ni<h and 0<=nj<w and edges[ni,nj]==50:
                    edges[ni,nj]=255
                    stack.append((ni,nj))
    edges[edges!=255]=0
    return edges

def canny(image,kernel_size=5,sigma=1.4,low=50,high=100):
    gray=rgb_to_grayscale(image) if image.ndim==3 else image.astype(float)
    if gray.max()<=1: gray*=255
    blurred=gaussian_blur(gray,kernel_size,sigma)
    gx,gy,mag,direction=compute_gradients(blurred)
    nms=non_maximum_suppression(mag,direction)
    thresholded=double_threshold(nms,low,high)
    edges=hysteresis(thresholded)
    return {"gray":gray,"blurred":blurred,"gx":gx,"gy":gy,"magnitude":mag,
            "direction":direction,"nms":nms,"thresholded":thresholded,"edges":edges}
