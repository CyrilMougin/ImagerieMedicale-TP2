# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:24:09 2019

@author: cmoug
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot








#I1 = np.int(plt.imread('DataTP2\Data\I1.png')[:,:,1]  ) * 255
#J1 = np.int(plt.imread('DataTP2\Data\J1.png')[:,:,1]  ) * 255
I2 = plt.imread('DataTP2\Data\I2.jpg')
J2 = plt.imread('DataTP2\Data\J2.jpg')
I3 = plt.imread('DataTP2\Data\I3.jpg')
J3 = plt.imread('DataTP2\Data\J3.jpg')
I4 = plt.imread('DataTP2\Data\I4.jpg')  
J4 = plt.imread('DataTP2\Data\J4.jpg')
I5 = plt.imread('DataTP2\Data\I5.jpg')  
J5 = plt.imread('DataTP2\Data\J5.jpg')
I6 = plt.imread('DataTP2\Data\I6.jpg') 
J6 = plt.imread('DataTP2\Data\J6.jpg')
  

    
    
def JointHist(I,  J,  Bin) :
    Conjoint =  np.zeros((int(np.amax(I))+1,(int(np.amax(J))+1)))
    intervalle=255//Bin
    print (intervalle)
    for i in range (I.shape[0]) :
        for j in range (I.shape[1]):
            xMin=I[i,j]//intervalle
            yMin=J[i,j]//intervalle
            Conjoint[ xMin*intervalle:xMin*intervalle +intervalle,yMin*intervalle:yMin*intervalle+intervalle]+=1 
            

    Conjoint = np.asarray(Conjoint)
    Conjoint = np.log(Conjoint)
    
   
    
    sum= 0
    
    for i in range (Conjoint.shape[0]) :
        for j in range (Conjoint.shape[1]):
            sum = sum + Conjoint[i,j]
    print(sum)

    return  Conjoint
    
    
    



# =============================================================================
# histo=JointHist(I6,J6,100)
# ig, ax = plt.subplots()
# ax.imshow(histo, origin='lower',cmap = 'jet')
# 
# plt.show()
#   
# =============================================================================
