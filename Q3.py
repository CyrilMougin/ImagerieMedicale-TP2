# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:47:46 2019

@author: cmoug
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d 
import math

fig = plt.figure()
ax = fig.gca(projection='3d')

N = 2
x,y,z = np.meshgrid(np.arange(0, 4, 1), np.arange(0, 4, 1),np.arange(0, 5, 1))
print(x)

#T = np.exp(-x**2 - y**2 - z**2)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
scat = ax.scatter(x, y, z, alpha=0.5)


plt.tight_layout()






def trans_rigide(theta, omega, phi, p, q, r) :
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(0.5), -math.sin(0.5) ],
                    [0,         math.sin(0.5), math.cos(0.5)  ]
                    ])
    
    
    return ax.scatter(R_x*x,y,z, alpha = 1)
        
trans_rigide(0,0,0,0,0,0)