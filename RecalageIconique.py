# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:42:10 2019

@author: Moi
"""

import numpy as np
import matplotlib.pyplot as plt
from CritereSim import SSD 

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

BrainMRI_1=np.float32(plt.imread('DataTP2\Data\BrainMRI_1.jpg'))/255
BrainMRI_2=np.float32(plt.imread('DataTP2\Data\BrainMRI_2.jpg'))/255
BrainMRI_3=np.float32(plt.imread('DataTP2\Data\BrainMRI_3.jpg'))/255
BrainMRI_4=np.float32(plt.imread('DataTP2\Data\BrainMRI_4.jpg'))/255

# =============================================================================
# Q5.a
# =============================================================================
def interpolationPlusProcheVoisin(Itranslat,I,i,j,p,q) :
    t= [np.int32(np.round(i+p)), np.int32(np.round(j+q))]
    if(t[0]<Itranslat.shape[0] and t[1]<Itranslat.shape[1]):
        Itranslat[t[0],t[1]]=I[i,j]
    return Itranslat

def interpolationBiLineaire(Itranslat,I,i,j,p,q) :
    x=int(i+p)
    y=int(j+q)
    xPrime = int(np.floor(i));
    yPrime = int(np.floor(j));
    a = i-xPrime;
    b = j-yPrime;

    if(xPrime+1<Itranslat.shape[0] and yPrime+1<Itranslat.shape[1]):
        Itranslat[x, y]=(1-a)*(1-b)*I[xPrime,yPrime]+(a)*(1-b)*I[xPrime+1,yPrime]+(1-a)*(b)*I[xPrime,yPrime+1]+(a)*(b)*I[xPrime+1,yPrime+1]
    return Itranslat

def translation(I, p, q):
    Itranslat=np.zeros((I.shape[0],I.shape[1]))
    for i in range (Itranslat.shape[0]) :
        for j in range (Itranslat.shape[1]):
            if (i+p<I.shape[0] and j+q<I.shape[1] and i+p>=0 and j+q>=0):
                Itranslat=interpolationPlusProcheVoisin(Itranslat,I,i,j,p,q)
                #Itranslat=interpolationBiLineaire(Itranslat,I,i,j,p,q)
    return Itranslat

#translation=translation(I2, -80.8,-100)
#plt.imshow(translation,cmap='gray')

# =============================================================================
# Q5.b
# =============================================================================
def recalage2DLucasKanade(I1,I2) : #p95
    Ix = np.gradient(I2)[0]
    Iy = np.gradient(I2)[1]
    It = I2-I1
    
    M=[[np.sum(Ix*Ix), np.sum(Iy*Ix)],[np.sum(Iy*Ix), np.sum(Iy*Iy)]]
    b=[[np.sum(Ix*It)], [np.sum(Iy*It)]]
    Minv=np.linalg.inv(M)  

    u=np.dot(-Minv,b)
   # print(u)
    translat=translation(I2,u[0], u[1])
   

    return translat


recalage=recalage2DLucasKanade(BrainMRI_1,BrainMRI_4)
plt.imshow(recalage,cmap='gray')


def recalage2DLucasKanadeIteratif(I,J) :
    energies=[SSD(I,J)]
    recalage=recalage2DLucasKanade(I,J)
    energies=np.append(energies,SSD(I,recalage))
    i=0;
    while (energies[i+1]<=energies[i]):
        J=recalage
        recalage=recalage2DLucasKanade(I,J)
        energies=np.append(energies,SSD(I,recalage))
        i+=1
   # print(energies)
    return J

#tmp=translation(I2, -80.8,-100)
#recalage=recalage2DLucasKanadeIteratif(I2,tmp)
#recalage=recalage2DLucasKanadeIteratif(BrainMRI_1,BrainMRI_4)
#plt.imshow(recalage,cmap='gray')
    