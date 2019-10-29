# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:03:32 2019

@author: Moi
"""

import numpy as np
import matplotlib.pyplot as plt
from HistoCo import JointHist 

# =============================================================================
# import images
# =============================================================================
I1 = plt.imread('DataTP2\Data\I1.png')[:,:,1]  
J1 = plt.imread('DataTP2\Data\J1.png')
I2 = np.int16(plt.imread('DataTP2\Data\I2.jpg'))
J2 = np.int16(plt.imread('DataTP2\Data\J2.jpg'))
I3 = np.int16(plt.imread('DataTP2\Data\I3.jpg'))  
J3 = np.int16(plt.imread('DataTP2\Data\J3.jpg'))
I4 = np.int16(plt.imread('DataTP2\Data\I4.jpg') ) 
J4 = np.int16(plt.imread('DataTP2\Data\J4.jpg'))
I5 = np.int16(plt.imread('DataTP2\Data\I5.jpg') ) 
J5 = np.int16(plt.imread('DataTP2\Data\J5.jpg'))
I6 = np.int16(plt.imread('DataTP2\Data\I6.jpg') ) 
J6 = np.int16(plt.imread('DataTP2\Data\J6.jpg'))

# =============================================================================
# Q2.a SSD
# =============================================================================

def SSD(I,J):
    dif=I-J
    carre=np.power(dif, 2)
    somme=np.sum(carre)
    return somme

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
    return p
    
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