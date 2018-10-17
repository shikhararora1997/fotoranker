#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 00:30:58 2018

@author: shikhararora
"""

from scipy.stats import entropy
import cv2
import numpy as np
import pandas as pd
import os
import glob
import math
from PIL import Image
from PIL import ImageStat
from numpy import array


APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()
def brightness(image):
        im = Image.open(image)
        stat = ImageStat.Stat(im)
        r,g,b = stat.rms
        return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))
def contrast(image):
        img2gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        hist=cv2.calcHist([img2gray],[0], None, [256],[0,256])
        e=entropy(hist,qk=None, base=None)
        return e[0]
def vignette(rows,cols):
    kernel_x = cv2.getGaussianKernel(cols, 200)
    kernel_y = cv2.getGaussianKernel(rows, 200)
    kernel = kernel_y * kernel_x.T
    mask = 255 * kernel / np.linalg.norm(kernel)
    return mask
def sizeof(image):
    im=Image.open(image)
    return im.size
def normalizef(feature):
    maxf=max(feature)
    minf=min(feature)
    i=0
    for x in feature:
        feature[i]=(x-minf)/(maxf-minf)
        i=i+1
    return feature
            
# img_dir="/Users/shikhararora/Desktop/images"
# data_path=os.path.join(img_dir,'*g')
# files=sorted(glob.glob(data_path)
  
def perform(files):

    data=[]
    url=[]
    for f1 in files:
        img=cv2.imread(f1)
        data.append(img)
        url.append(f1[49:])
    rows=[]
    cols=[]
    channels=[]
    for x in data:
        rows.append(x.shape[0])
        cols.append(x.shape[1])
        channels.append(x.shape[2])
        
        
    blur=[]
    contrastar=[]
    vig=[]

    for x in data:
         k=0
         blur.append(variance_of_laplacian(x))
         contrastar.append(contrast(x))
         vig.append(vignette(rows[k],cols[k]))
         k=k+1
    blur=normalizef(blur)
    contrastar=normalizef(contrastar)

    size=[] 
    bright=[]
    i=0
    for f1 in files:
        bright.append(brightness(f1))
        size.append(sizeof(f1))
        i=i+1
    bright=normalizef(bright)
    pixels=[]
    for s in size:
        pixels.append(s[0]*s[1])
    pixels=normalizef(pixels)
    result=pd.DataFrame({'Brightness':bright,'Clarity':blur,'Contrast':contrastar,'Total Pixels':pixels, 'url':url})
    weight=array([1,-1,1,1])
    score=[]
    mat=result.values
    k=0
    while(k<mat.shape[0]):
        score.append(mat[k][0]*weight[0]+mat[k][1]*weight[1]+mat[k][2]*weight[2]+mat[k][3]*weight[3])
        k=k+1
    results=pd.DataFrame({'Brightness':bright,'Clarity':blur,'Contrast':contrastar,'Total Pixels':pixels,'url':url,'Total score':score})
    resultsF=results.sort_values(by=['Total score'], ascending=False)
    i=1
    rank=[]
    while(i<=resultsF.shape[0]):
        rank.append(i)
        i=i+1
    rank=np.asarray(rank)
    resultsF=resultsF.assign(Rank=rank)
    targetFilePath = os.path.join(APP_ROOT, 'result', 'results.csv')
    print(targetFilePath)
    #resultsF.to_csv('/Users/shikhararora/Desktop/featuresimage/templates/results.csv', index=False, encoding='utf8')
    resultsF.to_csv(targetFilePath, index=False, encoding='utf8')

