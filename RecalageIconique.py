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
def interpolationPlusProcheVoisin(Itranslat,I,i,j,p,q) :
    t= [np.int32(np.round(i+p)), np.int32(np.round(j+q))]
    Itranslat[t[0],t[1]]=I[i,j]
    return Itranslat

def interpolationBiLineaire(Itranslat,I,i,j,p,q) :
    #Integer x and y coordinates of translated pixel to interpolate
    x=int(i+p)
    y=int(j+q)
    #Fractional x and y coordinates of source pixel
    src_x = i;
    src_y = j;
    #Integer coordinates of pixel to the left/right of the fractional source pixel:
    xPrime = int(np.floor(src_x));
    yPrime = int(np.floor(src_y));
        
    #Interpolate source pixel:
    a = src_x-xPrime;
    b = src_y-yPrime;
    
    Itranslat[x, y]=(1-a)*(1-b)*I[xPrime,yPrime]+(a)*(1-b)*I[xPrime+1,yPrime]+(1-a)*(b)*I[xPrime,yPrime+1]+(a)*(b)*I[xPrime+1,yPrime+1]
    return Itranslat

def translation(I, p, q):
    Itranslat=np.zeros((I.shape[0],I.shape[1]))
    for i in range (Itranslat.shape[0]) :
        for j in range (Itranslat.shape[1]):
            if (i+p<I.shape[0] and j+q<I.shape[1]):
                #Itranslat=interpolationPlusProcheVoisin(Itranslat,I,i,j,p,q)
                Itranslat=interpolationBiLineaire(Itranslat,I,i,j,p,q)
    return Itranslat

translation=translation(I2, 80,100)
plt.imshow(translation,cmap='gray')