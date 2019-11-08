# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:24:09 2019

@author: cmoug
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot



np.set_printoptions(suppress=True)




  
# Save image in set directory 
# Read RGB image 
img1 = plt.imread('DataTP2\Data\I2.jpg')  
img2 = plt.imread('DataTP2\Data\J2.jpg')
  

    
def JointHist(I,  J,  Bin) :
    Conjoint =  np.zeros((int(np.amax(I))+1,(int(np.amax(J))+1)))
    intervalle=255//Bin
    print (intervalle)
    for i in range (I.shape[0]) :
        for j in range (I.shape[1]):
            xMin=I[i,j]//intervalle
            yMin=J[i,j]//intervalle
            Conjoint[ xMin*intervalle:xMin*intervalle +intervalle,yMin*intervalle:yMin*intervalle+intervalle]+=1 
            
            
#            if  (Conjoint[I[i,j],J[i,j]]>255):
#                Conjoint[I[i,j],J[i,j]]-=1

    

    return  Conjoint
    
    
    


#histo=JointHist(img1,img2,2)
#plt.imshow(histo, origin='lower',cmap = 'jet')



  
