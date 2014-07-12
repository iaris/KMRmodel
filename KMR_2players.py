from discrete_rv import discreteRV
from mc_tools import *
import numpy as np
import matplotlib.pyplot as plt

def pmat(ips):
    P = np.array([[1-(ips*0.5),ips*0.5, 0], 
                  [(2-ips)*0.25, ips*0.5, (2-ips)*0.25], 
                  [0, ips*0.5, 1-(ips*0.5)]]) #�J�ڍs���ݒ�
    return P

A = pmat(0.1)
y = mc_sample_path(A) 

def subplots(): #���A�ڐ����ݒ�
    fig, ax = plt.subplots()
    ax.set_title('KMRmodel')
    return (fig, ax)
fig, ax = subplots() 

output = 0 or 1 or 2 #0(X�̃v���b�g) or 1(X�̃q�X�g�O����) or 2(��핪�z�̃q�X�g�O����)

if output == 0:
    ax.plot(y)
    plt.show()

if output == 1:
    ax.hist(y)
    plt.show()
    
if output == 2:
    sd = []
    for i in range(1, 1001):
        ips = 1/i
        B = pmat(ips)
        x = mc_compute_stationary(B)
        sd.append(x)

    ax.hist(sd)
    plt.show()
        