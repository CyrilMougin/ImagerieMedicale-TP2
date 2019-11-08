# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:42:10 2019

@author: Moi
"""

import numpy as np
import matplotlib.pyplot as plt
from  scipy import ndimage
from CritereSim import SSD
from PIL import Image

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
def recalage2DLucasKanade(I,J) : #p94
    gradient=np.gradient(J)
    Jx = gradient[0]
    Jy = gradient[1]
    Jt = J-I
    
    M=np.array(([np.sum(Jx*Jx), np.sum(Jy*Jx)],[np.sum(Jy*Jx), np.sum(Jy*Jy)]))
    b=np.array(([np.sum(Jx*Jt)], [np.sum(Jy*Jt)]))
    u=np.squeeze(np.linalg.solve(M,b))
    return u


def recalage2DLucasKanadeIteratif(I,J, i_max, afficherEnergie) :
    energies=[SSD(I,J)]
    recalage=J
    recalageFinal=J
    u=recalage2DLucasKanade(I,J)
    recalage=ndimage.interpolation.shift(recalage, u, mode='wrap')
    energies=np.append(energies,SSD(I,recalage))
    energieMin=energies[0]
    i=0;
    #while (energies[i+1]<=energies[i]):
    while (i<i_max):
        i+=1
        u+=recalage2DLucasKanade(I,recalage)
        recalage=ndimage.interpolation.shift(recalage, u, mode='wrap')
        energies=np.append(energies,SSD(I,recalage))
        if (energies[-1]<energieMin) :
            energieMin=energies[-1]
            recalageFinal=recalage
    if (afficherEnergie==True) :
        print("Energie minimale : " + str(energieMin))
        plt.plot(energies)
        plt.show()
    return recalageFinal

# =============================================================================
# #Débruitage
# BrainMRI_1_debruité=ndimage.gaussian_filter(BrainMRI_1, sigma=1)
# BrainMRI_2_debruité=ndimage.gaussian_filter(BrainMRI_2, sigma=1)
# BrainMRI_3_debruité=ndimage.gaussian_filter(BrainMRI_3, sigma=1)
# BrainMRI_4_debruité=ndimage.gaussian_filter(BrainMRI_4, sigma=1)
# =============================================================================


#decale=ndimage.interpolation.shift(BrainMRI_1_debruité, [10,10], mode='wrap')
#recalage=recalage2DLucasKanadeIteratif(BrainMRI_1_debruité,BrainMRI_2_debruité,500, True)
##plt.imshow(recalage,cmap='gray')
##
##plt.imshow(BrainMRI_2_debruité-recalage,cmap='gray')
###
#plt.imshow(recalage-BrainMRI_1_debruité)
#





# =============================================================================

def rotation(I,phi):
    width,length = np.shape(I)
    im=np.zeros((width*2,length*2))
    im[width:width*2,length:length*2]=I
    im = Image.fromarray(im)
    imRot = im.rotate(phi)
    w, h = imRot.size
    imRot=list(imRot.getdata())
    imRot=np.array(imRot)
    imRot=np.reshape(imRot,(height,width))
    return imRot
    #return ndimage.interpolation.rotate(I, phi, mode='nearest')

def recalageRotationSSD(I,J):
    phi=0
    epsilon = 0.5
    x = np.linspace(0,I.shape[0]-1, num=I.shape[0])
    y = np.linspace(0,I.shape[1]-1, num=I.shape[1])
    X,Y=np.meshgrid(x,y)
    gradient=np.gradient(I)
    for i in range(200):
        imRot = rotation(I,phi)
        gradX = rotation(gradient[0],phi)
        gradY = rotation(gradient[1],phi)
        dIdx=gradX*(-X*np.sin(phi)-Y*np.cos(phi))
        dIdy=gradY*(X*np.cos(phi)-Y*np.sin(phi))
        gradSSD=2*np.sum((imRot-J)*(dIdx+dIdy))
        phi=phi-epsilon*gradSSD
        print("phi = "+str(phi))
    return rotation(I,phi)

def afficherRecalageRotationSSD(I,phi):
    plt.figure(1)
    plt.imshow(I,cmap='gray')
    J = ndimage.rotate(I,phi)
    plt.figure(2)
    plt.imshow(J,cmap='gray')
    recalage = recalageRotationSSD(I,J)
    plt.figure(3)
    plt.imshow(recalage,cmap='gray')    
    return

def recalageTransformationRigideSSD(I,J):
    
    return


afficherRecalageRotationSSD(BrainMRI_1,20)
# =============================================================================
# plt.figure(1)
# plt.imshow(BrainMRI_1,cmap='gray')
# plt.figure(2)
# plt.imshow(rotation(BrainMRI_1,-10),cmap='gray')
# =============================================================================

#plt.imshow()