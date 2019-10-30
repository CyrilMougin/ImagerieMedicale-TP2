# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:42:10 2019

@author: Moi
"""

import numpy as np
import matplotlib.pyplot as plt
from HistoCo import JointHist 

# =============================================================================
# import images + normalisation intensité
# =============================================================================
I1 = plt.imread('DataTP2\Data\I1.png')[:,:,1]  
J1 = plt.imread('DataTP2\Data\J1.png')
I2 = np.float32(plt.imread('DataTP2\Data\I2.jpg'))/255
J2 = np.float32(plt.imread('DataTP2\Data\J2.jpg'))/255
I3 = np.float32(plt.imread('DataTP2\Data\I3.jpg'))/255
J3 = np.float32(plt.imread('DataTP2\Data\J3.jpg'))/255
I4 = np.float32(plt.imread('DataTP2\Data\I4.jpg') )/255 
J4 = np.float32(plt.imread('DataTP2\Data\J4.jpg'))/255
I5 = np.float32(plt.imread('DataTP2\Data\I5.jpg') ) /255
J5 = np.float32(plt.imread('DataTP2\Data\J5.jpg'))/255
I6 = np.float32(plt.imread('DataTP2\Data\I6.jpg') ) /255
J6 = np.float32(plt.imread('DataTP2\Data\J6.jpg'))/255

# =============================================================================
# Q5.a
# =============================================================================
def interpolationPlusProcheVoisin(x,y) :
    return [np.int32(np.round(x)), np.int32(np.round(y))]

def interpolationBiLineaire(x,y):
    xPrime=int(x)
    yPrime=int(y)
    a=np.abs(x-xPrime)
    b=np.abs(y-yPrime)
    
    return

def translation(I, p, q):
    Itranslat=np.zeros((np.int32(np.ceil(I.shape[0]+p)),np.int32(np.ceil(I.shape[1]+q))))
    for i in range (I.shape[0]) :
        for j in range (I.shape[1]):
            t=interpolationPlusProcheVoisin(i+p,j+q)
            Itranslat[t[0],t[1]]=I[i,j]
    return Itranslat

translation=,80.8)
plt.imshow(translation,cmap='gray')