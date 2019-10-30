# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:24:09 2019

@author: cmoug
"""


import numpy as np
import cv2 
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
  
# Save image in set directory 
# Read RGB image 
img1 = plt.imread('DataTP2\Data\I4.jpg')  
img2 = plt.imread('DataTP2\Data\J4.jpg')
  



#plt.imshow(img1,cmap=plt.get_cmap('gray'))

I = np.unique(img1).shape[0]
J = np.unique(img2).shape[0]






    
def JointHist(I,  J,  bin) :
    Conjoint =  np.zeros((I,J))
    for i in range (I.shape[0]) :
        for j in range (I.shape[1]):
            Conjoint[I[i,j],J[i,j]]+=1 
            if  (Conjoint[I[i,j],J[i,j]]>255):
                Conjoint[I[i,j],J[i,j]]-=1
        
    
    return Conjoint
    

#plt.imshow(JointHist(I,J,bin), vmin=0, vmax=255)
#
#cv2.imshow('image',JointHist(I,J,bin))
#cv2.waitKey(0)
#cv2.destroyAllWindows()


  
