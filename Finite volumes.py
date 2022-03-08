# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 16:22:17 2022

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import math as m

N = 100
x_0 = 0
x_N = m.pi/2

#equation: -y" + (cos(2x) * y)' = 4sin(2x) + 2 - 4sin^2(2x)
#solution: sin(2x)

y_solution = np.zeros(N)

# uneven grid formation

h_steps = np.zeros(N-1)

if (N % 2 == 1):
    
    for i in range (0,N-1):
        h_steps[i] = x_N / N + pow(-1,i % 2) * x_N / (2*N)
        
else:
    
    h_steps[0] = x_N / N
    for i in range (1,N-1):
        h_steps[i] = x_N / N + pow(-1,i % 2) * x_N / (2*N)
    
    
# vector of arguments formation

x = np.zeros(N)
x[0] = x_0

for i in range (1,N):
    x[i] = x[i-1] + h_steps[i-1]
    
x[N-1] = x_N

#print (np.sum(h_steps), x)

# vector of the y formation 

y = np.zeros(N)
y_solution = np.sin(2*x)

# vector of v(x) formation

v = np.cos(2*x)

# vector of f(x) formation

f = 4*np.sin(2*x) + 2 - 4*np.sin(2*x)*np.sin(2*x)

# quotients formation

A = np.zeros(N)
B = np.zeros(N)
C = np.zeros(N)

for i in range (1,N-1):
    C[i] = 1/(h_steps[i] + h_steps[i-1]) * (-2/h_steps[i] + (v[i+1]+v[i])/2)
    B[i] = 1/(h_steps[i] + h_steps[i-1]) * (2/h_steps[i] + 2/h_steps[i-1] + (v[i+1]-v[i-1])/2)
    A[i] = 1/(h_steps[i] + h_steps[i-1]) * (-2/h_steps[i-1] - (v[i]+v[i-1])/2)
    
C[0] = 2/y_solution[1]
B[0] = 1e7
A[N-1] = 2/y_solution[N-2]
B[N-1] = 1e7

F = np.zeros( (N,N) )

for i in range (1,N-1):
    F[i][i-1] = A[i]
    F[i][i] = B[i]
    F[i][i+1] = C[i]
    
F[0][0] = B[0]
F[0][1] = C[0]
F[N-1][N-2] = A[N-1]
F[N-1][N-1] = B[N-1]

l = np.zeros(N)
u = np.zeros(N)
l[0] = -F[0][1]/F[0][0]
u[0] = f[0]/F[0][0]
l[N-1] = 0
for i in range(1,N-1):
    l[i] = F[i][i+1] / (-F[i][i] - F[i][i-1]*l[i-1])
    u[i] = (F[i][i-1]*u[i-1] - f[i]) / (-F[i][i] - F[i][i-1]*l[i-1])
u[N-1] = (F[N-1][N-2]*u[N-2] - f[N-1]) / (-F[N-1][N-1] - F[N-1][N-2]*l[N-2])

#the reversive part
y[N-1] = u[N-1]
for i in range(N-2,-1,-1):
    y[i] = l[i]*y[i+1] + u[i]
    
plt.figure()
plt.plot(x,y)
plt.plot(x,y_solution)

print (np.max(np.abs(y - y_solution)))



