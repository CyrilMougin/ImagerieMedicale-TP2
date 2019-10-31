# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:47:46 2019

@author: cmoug
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d 

fig = plt.figure()
ax = fig.gca(projection='3d')

N = 2
x,y,z = np.mgrid[0:20:2, 0:20:2, 0:20:1]
print(x )
print(y)
#T = np.exp(-x**2 - y**2 - z**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(x, y, z, alpha=0.5)

plt.tight_layout()

def trans_rigide(theta, omega, phi, p, q, r) :
    trans = np.array([[p,0,0],[0,q,0],[0,0,r]])
    rx = np.array([[1,0,0],[0,np.cos(theta),-np.sin(theta)],[0,np.sin(theta),np.cos(theta)]])
    ry = np.array([[np.cos(omega),0,np.sin(omega)],[0,1,0],[-np.sin(omega),0,np.cos(omega)]])
    rz = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
    return trans*rx*ry*rz
    
        
