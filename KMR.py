from discrete_rv import discreteRV
from mc_tools import *
import numpy as np
import matplotlib.pyplot as plt

"""
�J�ڍs����쐬
N�̓v���C���[���Aips�̓v���C���[�������s�������m��
p�́u�헪1���Ƃ�v���C���[�����̊m�����傫����ΐ헪1���œK�����ƂȂ�v�m��
"""

def pmat(N, ips, p):
    P = np.zeros((N, N))
    P[0, 0] = 1 - (ips/2.0)
    P[0, 1] = ips/2.0
    P[N-1,N-2] = ips/2.0
    P[N-1,N-1] = 1 - (ips/2.0)
    
    for i in range(1,N-1):
        a, b = ips/2.0, ips/2.0
        
        if (i/ float(N)) < p:
            a = (a + (1-ips))
        else:
            b = (b + (1-ips))
        
        dec = (i/ float(N-1)) * a
        inc = ((N-i)/ float(N-1)) * b
                 
        P[i, i-1] = dec 
        P[i, i+1] = inc
        P[i, i]   = 1-(dec + inc)
        
    return P

A = pmat(4 , 0.1, 1.0/3.0)
y = mc_sample_path(A) 

def subplots(): #���A�ڐ����ݒ�
    fig, ax = plt.subplots()
    ax.set_title('KMRmodel')
    return (fig, ax)
fig, ax = subplots() 

output = 0 or 1 or 2 #0(X�̃v���b�g)�A1(X�̃q�X�g�O����)�A2(��핪�z�̕���)

if output == 0:
    ax.plot(y)
    plt.show()

if output == 1:
    ax.hist(y)
    plt.show()
    
if output == 2:
    sd = []
    for i in range(1, 101):
        ips = 1.0/i
        B = pmat(ips)
        x = mc_compute_stationary(B)
        sd.append(x)

    ax.hist(sd)
    plt.show()    