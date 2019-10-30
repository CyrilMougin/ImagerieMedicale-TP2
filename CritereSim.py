# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:03:32 2019

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
# Q2.a SSD
# =============================================================================

def SSD(I,J):
    dif=I-J
    carre=np.power(dif, 2)
    somme=np.sum(carre)
    return np.round(somme)

Qa=SSD(I6, J6)

# =============================================================================
# Q2.b CR
# =============================================================================

def CR(I,J):
    Imoy=np.mean(I)
    Jmoy=np.mean(J)
    num=np.sum((I-Imoy)*(J-Jmoy))
    denum=(np.sqrt(np.sum(np.power(I-Imoy,2))))*(np.sqrt(np.sum(np.power(J-Jmoy,2))))
    p=num/denum
    return np.round(p,3)
    
Qb=CR(I6, J6)

# =============================================================================
# Q2.c IM
# =============================================================================
def IM(I,J):
    histoJoint=JointHist(I,J, bin)
    IM=0
    Pi=np.sum(histoJoint, axis=0)
    Pj=np.sum(histoJoint, axis=1)
    for i in range (I.shape[0]) :
        for j in range (I.shape[1]):
            IM+=histoJoint[i,j]*np.log(histoJoint[i,j]/(Pi[i]*Pj[j]))
    return Pi
            

#Qc=IM(I1, I1)