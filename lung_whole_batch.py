#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
import os,time,subprocess,glob,re
import pandas
import argparse

# characteristic cycles
cond = [
#         {'dim': 2, 'b0':-220, 'b1':9999, 'l0': 10}, #fib1
#         {'dim': 0, 'b0':-1100, 'b1':-50, 'l0': 5}, # HC
#         {'dim': 1, 'b0':-1050, 'b1':-1030, 'l0': 15}, # HC
#         {'dim': 0, 'b0':-1700, 'b1':-610, 'l0': 370, 'l1': 5000}, # fib_long　　low = -1300 -- -1260, high = -800 -- -610
         {'name': 'fib25', 'dim': 0, 'b0':-1260, 'b1':-380, 'd0': -5000, 'd1': 5000, 'l0': 360, 'l1': 5000, 'th': 1}, # 
#         {'name': 'fib25h1', 'dim': 1, 'b0':-800, 'b1':-300,  'd0': -5000, 'd1': 5000, 'l0': 370, 'l1': 5000, 'th':0.1,}, # low: -700
#         {'dim': 0, 'b0':-1300, 'b1':-500, 'd0': -850, 'd1': 1000, 'l0': 360, 'l1': 5000, 'th': 0.4}, # fib_50
#         {'dim': 0, 'b0':-1100, 'b1':-1000, 'd0': -1020, 'd1': -970, 'l0': 30, 'l1': 5000, 'th': 40}, # emp_50
         {'name': 'emp', 'dim': 2, 'b0':-1020, 'b1':-900, 'd0': -5000, 'd1': 5000, 'l0': 20, 'l1': 90, 'th': 8.3}, # emp25_narrow   low = -1100 -- -980, 
       ]

# load dicom volume in a directory
def load_dicom(dirname, ftype="dcm"):
    num = lambda val : int(re.sub("\\D", "", val))
    fns = [os.path.join(dirname,f) for f in os.listdir(dirname) if (ftype in f) ]
    fns.sort(key=num)
    images = []
    if ftype == "dcm":
        try:
            import pydicom as dicom
        except:
            print("install pydicom by pip install pydicom")
            exit()
        for f in fns:
            ref_dicom_in = dicom.read_file(f, force=True)
            images.append(ref_dicom_in.pixel_array.astype(np.float64) +ref_dicom_in.RescaleIntercept)
    elif ftype == "npy":
        for f in fns:
            img = np.load(f)
            images.append(img.astype(np.float64))
    else:
        for f in fns:
            img = plt.imread(f)
            if len(img.shape)==3:
                img = img.transpose(2,0,1)
            images.append(img.astype(np.float64))
    return(np.squeeze(np.stack(images,axis=-1)))   ## [c,x,y,z]


def gaussian(h,sigma):
    x = np.arange(-h,h,1)
    y = np.arange(-h,h,1)
    z = np.arange(-h,h,1)
    xx, yy, zz = np.meshgrid(x,y,z)
    return(np.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2)).astype(np.float32))

def cycle_count(vol,pd,cond,h=11,sigma=1.0,gpu_id=0,conv=True, verbose=False):
    if verbose:
        print("couting relevant cycles...")
    mx,my,mz=vol.shape
    cycle = np.zeros((len(cond),mx,my,mz)).astype(np.float32)
    for c in pd:
        d = int(c[0]) # dim
        life = c[2]-c[1] # life
        x,y,z=int(c[3]),int(c[4]),int(c[5])
        for i,f in enumerate(cond):
            if(d==f['dim'] and f['b0']<c[1] and c[1]<f['b1'] and f['d0']<c[2] and c[2]<f['d1'] and f['l0'] < life and life < f['l1']):
                if(0<= z <mz):
                    cycle[i,x,y,z] += 1

    if conv==False:
        return(cycle * mx*my*mz / np.sum(vol>-2048))

    if verbose:
        print("computing cycle heatmap...")

    # convolute with gaussian kernel
    kernel = gaussian(h,sigma)
    return(conv_channel(cycle, vol, kernel, gpu_id, verbose=False))

def conv_channel(cycle, vol, kernel, gpu_id=0, verbose=False):
    if gpu_id >= 0:
#        from chainer.functions import convolution_nd
        import cupy as cp
        from cupyx.scipy.ndimage import convolve
        cp.cuda.Device(gpu_id).use()
        kernel = cp.asarray(kernel)
        cycle_conv = np.stack([ cp.asnumpy(convolve(cp.asarray(cycle[i]),kernel)) for i in range(len(cycle))])
#        kernel = cp.asarray(kernel[np.newaxis,np.newaxis,:])
#        cycle_conv = cp.asnumpy(convolution_nd(cp.asarray(cycle[:,np.newaxis,:]),kernel,pad=h))
        if verbose:
            print("normalising by local volume...")        
#        volume = cp.asnumpy(convolution_nd( cp.asarray((vol>-2048),dtype=np.float32)[np.newaxis,np.newaxis,:], cp.ones((1,1,h,h,h),dtype=np.float32) ))[0]
        vkernel = cp.ones(kernel.shape,dtype=np.float32)/np.prod(kernel.shape)
        volume = cp.asnumpy(convolve( cp.asarray((vol>-2048),dtype=np.float32), vkernel )[np.newaxis,:])
    else:
        from scipy.ndimage.filters import convolve
        cycle_conv = np.stack([convolve(cycle[i],kernel) for i in range(len(cycle))])
        if verbose:
            print("normalising by local volume...")        
        vkernel = np.ones(kernel.shape,dtype=np.float32)/np.prod(kernel.shape)
        volume = convolve( (vol>-2048).astype(np.float32), vkernel )[np.newaxis,:]
        
    # normalise by volume
    volume[:,vol<=-2048] = np.inf
    return(cycle_conv  / volume)  

def volume_stat(vol,cycle_norm, th):
    stats = np.zeros(len(cycle_norm)*4+3,dtype=np.float32)
    mask = (vol>-2048)
    stats[0]=np.sum(mask)
    c = cycle_norm[:,mask]
    for i in range(len(c)):
        stats[i+1]=np.sum(c[i])
        stats[i+1+len(c)]=stats[i+1]/stats[0]
        stats[i+1+2*len(c)]=np.percentile(c[i], 99)
        if i==1:
            stats[i+1+3*len(c)]=np.sum((c[i]>th[i])*(c[i-1]<=th[i-1]))/stats[0]
        else:
            stats[i+1+3*len(c)]=np.sum((c[i]>th[i]))/stats[0]

    stats[-2] = np.sum(vol>-200) / stats[0]
    stats[-1] = np.sum( (vol>-2048)*(vol<-950) ) / stats[0]
    return(stats)

def load_vol(fn, z_crop=None, verbose=False):
    start = time.time()
    bn = os.path.splitext(fn)[0]
    if os.path.isfile(fn):
        volz= np.load(fn)
        vol = volz[volz.files[0]]
        if verbose:
            print("volume loaded from numpy")
    elif os.path.isdir(bn):
        vol = load_dicom(bn)
        print("saving the volume to npz...")
        np.savez_compressed(fn,vol=vol)
        if verbose:
            print("volume loaded from dicom and saved")
    else:
        print("file not found.")
        return(None)
    if verbose:
        print ("elapsed_time:{} sec".format(time.time() - start))
    if z_crop:
        if len(vol.shape)==3:
            vol = vol[:,:,z_crop[0]:z_crop[1]]
        else:
            vol = vol[:,:,:,z_crop[0]:z_crop[1]]
    return(vol)

def load_pd(base_fn, vol=None, z_crop=None, verbose=False):
    pd_fn = base_fn+".npy_pd.npz"
    start = time.time()
    if os.path.isfile(pd_fn):
        pd = np.load(pd_fn)['pd']
        if z_crop:
            pd[:,5] -= z_crop[0]
        if verbose:
            print("pre-computed PD loaded")
    else:
        try:
            import cripser
        except:
            print("install cripser by pip install git+https://github.com/shizuo-kaji/CubicalRipser_3dim")
            exit()
        print("computing PH...")
        pd = cripser.computePH(vol)
        if z_crop is None:
            np.savez_compressed(pd_fn,pd=pd)
        if verbose:
            print("PD computed")
    if verbose:
        print ("elapsed_time:{} sec".format(time.time() - start))
    return(pd)

def load_cyc(cyc_fn,z_crop=None, verbose=False, recompute={'force': False}):
    start = time.time()
    if os.path.isfile(cyc_fn) and not recompute['force']:
        cycle_norm = np.load(cyc_fn)['cyc']
        if z_crop:
            cycle_norm = cycle_norm[:,:,:,z_crop[0]:z_crop[1]]
        if verbose:
            print("pre-computed cycle heatmap loaded")
    else:
        cycle_norm = cycle_count(recompute['vol'],recompute['pd'],recompute['cond'],h=recompute['h'],sigma=recompute['sigma'],gpu_id=recompute['gpu_id'],conv=True, verbose=verbose)
        if z_crop is None:
            np.savez_compressed(cyc_fn,cyc=cycle_norm)
        if verbose:
            print("cycle heatmap computed")
    if verbose:
        print("elapsed_time:{} sec".format(time.time() - start))
    return(cycle_norm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--datadir', type=str, default='/home/skaji/ipf/npy/', help='data directory')
    parser.add_argument('--index_file', type=str, default='idlist.csv', help='csv file containing the names of the volume files')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID (-1 means CPU)')
    parser.add_argument('--output', type=str, default='total.csv', help='name of the csv file to which summary will be saved')
    args = parser.parse_args()

    print(cond)
    np.set_printoptions(precision=5,suppress=True)
    
    # labelling
    dat = pandas.read_csv(args.index_file,header=0)
    names = dat['name'].tolist()


    ## characteristic cycle and gaussian kernel parameter
    #cycle_data_suffix, sigma, h = "_cyc200922.npz", 5.0, 25
    #cycle_data_suffix, sigma, h = "_cyc200925.npz", 4.0, 12
    cycle_data_suffix, sigma, h = "_cyc200930.npz", 10.0, 12

    ## Threshold
    th_def = [f['th'] for f in cond]
    total = np.zeros((len(names),len(cond)*4+3),dtype=np.float32)

    for k in range(len(names)):
        ## results will be cached under root_npy. remove files if you want to recompute.
        print("loading... {}/{}, {}, {}".format(k,len(names),names[k]))
        base_fn = os.path.join(args.datadir,names[k])

        vol = load_vol(base_fn+".npz")
        pd = load_pd(base_fn)
        recompute_data = {'vol': vol, 'pd': pd, 'cond': cond, 'h':h, 'sigma':sigma, 'gpu_id': args.gpu_id}
        print("Volume: ",vol.shape," PD: ",pd.shape)
        cycle_norm = load_cyc(base_fn+cycle_data_suffix,recompute=recompute_data,z_crop=None,verbose=True)

        total[k] = volume_stat(vol,cycle_norm, th_def)
        print("total: ",total[k])

    np.savetxt(args.output,total,fmt='%.5f',delimiter=",",header="vol,fib,emp,fib_r,emp_r,fib_99,emp_99,fib%def,emp%def,HAA%,LAA%")


