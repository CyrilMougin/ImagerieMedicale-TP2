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
x,y,z = np.meshgrid(np.arange(0,20,1), np.arange(0,20,1),np.arange(0,20,1))
x2,y2,z2 = np.meshgrid(np.arange(0,20,1), np.arange(0,20,1),np.arange(0,20,1))

tot = np.array([x2,y2,z2,1])


#T = np.exp(-x**2 - y**2 - z**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(x, y, z, alpha=0.1)


plt.tight_layout()

def trans_rigide(theta, omega, phi, p, q, r) :
    trans = np.array([[1,0,0,p],[0,1,0,q],[0,0,1,r],[0,0,0,1]])
    rx = np.array([[1,0,0,0],[0,np.cos(theta),-np.sin(theta),0],[0,np.sin(theta),np.cos(theta),0],[0,0,0,1]])
    ry = np.array([[np.cos(omega),0,-np.sin(omega),0],[0,1,0,0],[np.sin(omega),0,np.cos(omega),0],[0,0,0,1]])
    rz = np.array([[np.cos(phi),-np.sin(phi),0,0],[np.sin(phi),np.cos(phi),0,0],[0,0,1,0],[0,0,0,1]])

    return trans.dot(rx).dot(ry).dot(rz)

def similitude(s,theta, omega, phi, p, q, r) :
    simi = np.array([[s,0,0,0],[0,s,0,0],[0,0,s,0],[0,0,0,1]])
    trans = np.array([[1,0,0,p],[0,1,0,q],[0,0,1,r],[0,0,0,1]])
    rx = np.array([[1,0,0,0],[0,np.cos(theta),-np.sin(theta),0],[0,np.sin(theta),np.cos(theta),0],[0,0,0,1]])
    ry = np.array([[np.cos(omega),0,-np.sin(omega),0],[0,1,0,0],[np.sin(omega),0,np.cos(omega),0],[0,0,0,1]])
    rz = np.array([[np.cos(phi),-np.sin(phi),0,0],[np.sin(phi),np.cos(phi),0,0],[0,0,1,0],[0,0,0,1]])

    return simi.dot(trans.dot(rx).dot(ry).dot(rz))



#tot = (similitude(1,0,0,0,0,0,40) ).dot(tot)
    
mat = np.array([[0.9045,-0.3847,-0.1840,10.0000],[0.2939,0.8750,-0.3847,10.0000],[0.3090,0.2939,0.9045,10.0000],[0,0,0,1.0000]])
tot = mat.dot(tot)
scat1 = ax.scatter(tot[0], tot[1], tot[2], alpha=0.5)


