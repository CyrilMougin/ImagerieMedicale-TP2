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

#Débruitage
BrainMRI_1_debruité=ndimage.gaussian_filter(BrainMRI_1, sigma=1)
BrainMRI_2_debruité=ndimage.gaussian_filter(BrainMRI_2, sigma=1)
BrainMRI_3_debruité=ndimage.gaussian_filter(BrainMRI_3, sigma=1)
BrainMRI_4_debruité=ndimage.gaussian_filter(BrainMRI_4, sigma=1)

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
    energies=[]
    recalage=J
    recalageFinal=J
    u=[0,0]
    energies=np.append(energies,SSD(I,recalage))
    energieMin=[SSD(I,J)]
    i=0;
    while (i<i_max):
        u+=recalage2DLucasKanade(I,recalage)
        recalage=ndimage.interpolation.shift(recalage, u, mode='nearest')
        energies=np.append(energies,SSD(I,recalage))
        if (energies[-1]<energieMin) :
            energieMin=energies[-1]
            recalageFinal=recalage
        i+=1
    if (afficherEnergie==True) :
        print("Energie minimale : " + str(energieMin))
        plt.plot(energies)
        plt.show()
    return recalageFinal

def recalageTraslationDescente(I,J):
    u=[0,0]
    epsilon=0.01
    translation=J
    for i in range(500):
        print(u)
        translation=ndimage.interpolation.shift(J, u, mode='nearest')
        [dSSDdp,dSSDdq]=calculerGradSSDTranslation(u,translation,J,I)
        u[0]+=epsilon*dSSDdp
        u[1]+=epsilon*dSSDdq
        
    return translation

def calculerGradSSDTranslation(u,translation,J,I):
    gradient=np.gradient(translation)
    dJdx=gradient[0]
    dJdy=gradient[1]
    dSSDdp=2*np.sum((translation-I)*dJdx)
    dSSDdq=2*np.sum((translation-I)*dJdy)
    return [dSSDdp,dSSDdq]


# =============================================================================
# #Débruitage
# BrainMRI_1_debruité=ndimage.gaussian_filter(BrainMRI_1, sigma=1)
# BrainMRI_2_debruité=ndimage.gaussian_filter(BrainMRI_2, sigma=1)
# BrainMRI_3_debruité=ndimage.gaussian_filter(BrainMRI_3, sigma=1)
# BrainMRI_4_debruité=ndimage.gaussian_filter(BrainMRI_4, sigma=1)
# #
# #
# translationx=ndimage.interpolation.shift(BrainMRI_1_debruité, [50,0], mode='nearest')
# translationy=ndimage.interpolation.shift(BrainMRI_1_debruité, [0,20], mode='nearest')
# translationxy=ndimage.interpolation.shift(BrainMRI_1_debruité, [10,10], mode='nearest')
# 
# =============================================================================

def Q1(translation):
    recalage=recalageTraslationDescente(BrainMRI_1_debruité,translation)
    plt.figure(2)
    plt.imshow(recalage,cmap='gray')
    plt.figure(3)
    plt.imshow(translation-recalage,cmap='gray')
    plt.figure(4)
    plt.imshow(BrainMRI_1_debruité-recalage,cmap='gray')


# Q1(translationy)



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
    imRot=np.reshape(imRot,(h,w))
    return imRot[width:width*2,length:length*2]

def recalageRotationSSD(I,J):
    energies=[]
    phi=0
    epsilon = 0.000000005 
    x = np.linspace(0,I.shape[0]-1, num=I.shape[0])
    y = np.linspace(0,I.shape[1]-1, num=I.shape[1])
    X,Y=np.meshgrid(x,y)
    gradient=np.gradient(I)
    for i in range(300):
        phi=phi-epsilon*calculerGradSSDRotation(I,rotation(J,phi),0,gradient,X,Y)
        recalage= rotation(J,phi)
        energies=np.append(energies,SSD(recalage,I))
    plt.figure(1)    
    plt.plot(energies)
    plt.show()
    print(energies[-1])    
    return rotation(J,phi)

def calculerGradSSDRotation(I,J,phi,gradient,X,Y):
    imRot = rotation(I,phi)
    gradX = rotation(gradient[0],phi)
    gradY = rotation(gradient[1],phi)
    dIdx=gradX.T*(-X*np.sin(phi/180*np.pi)-Y*np.cos(phi/180*np.pi))
    dIdy=gradY.T*(X*np.cos(phi/180*np.pi)-Y*np.sin(phi/180*np.pi))
    gradSSD=2*np.sum((imRot-J).T*(dIdx+dIdy))
    return gradSSD

def afficherRecalageRotationSSD(I,phi):
    plt.figure(2)
    plt.imshow(I,cmap='gray')
    J = rotation(I,phi)
    plt.figure(3)
    plt.imshow(J,cmap='gray')
    recalage = recalageRotationSSD(I,J)
    plt.figure(4)
    plt.imshow(recalage,cmap='gray')  
    plt.figure(5)
    plt.imshow(recalage-I,cmap='gray')    
    return



#afficherRecalageRotationSSD(BrainMRI_1_debruité,20)



def recalageIconiqueRigide(I,J):
    p=1
    q=1
    phi=1
    epsilon = 0.000001
    i=0
    x = np.linspace(0,I.shape[0]-1, num=I.shape[0])
    y = np.linspace(0,I.shape[1]-1, num=I.shape[1])
    X,Y=np.meshgrid(x,y)
    gradient=np.gradient(I)
    energies=[]
    stop = False
    while i<500 and not stop:
        J2=rotation(ndimage.interpolation.shift(J, [p,q], mode='nearest'),phi)
        [gradP,gradQ]=calculerGradSSDTranslation([p,q],J2,I,J)
        gradPhi=calculerGradSSDRotation(I,J2,0,gradient,X,Y)
        p+=epsilon*gradP
        q+=epsilon*gradQ
        phi-=epsilon*gradPhi
        energies=np.append(energies,SSD(J2,I))
        print(SSD(J2,I))
        i+=1
        if (i>10):
            if np.power((energies[-1]-energies[-8]),2)<0.05 :
                stop = True
# =============================================================================
#         plt.subplots()
#         plt.imshow(J2,cmap='gray')
#         plt.show()
# =============================================================================
    
    plt.figure(1)    
    plt.plot(energies)
    plt.show()
    print(i)
    return J2
    
def afficherRecalageIconiqueRigide(I,J):
    plt.figure(2)
    plt.imshow(I,cmap='gray')
    plt.figure(3)
    plt.imshow(J,cmap='gray')
    recalage = recalageIconiqueRigide(I,J)
    plt.figure(4)
    plt.imshow(recalage-I,cmap='gray')    
    plt.figure(5)
    plt.imshow(recalage,cmap='gray')
    return

afficherRecalageIconiqueRigide(I2,J2)
    
# =============================================================================
# plt.figure(1)
# plt.imshow(BrainMRI_1,cmap='gray')
# plt.figure(2)
# plt.imshow(rotation(BrainMRI_1,-10),cmap='gray')
# =============================================================================

