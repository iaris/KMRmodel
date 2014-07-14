# -*- coding: utf-8 -*-

from discrete_rv import discreteRV
from mc_tools import *
import numpy as np
import matplotlib.pyplot as plt

"""
�J�ڍs���������֐����쐬
�����ł͎������g�Ƒΐ킷�邱�Ƃ������Ă���
N�̓v���C���[���Aips�̓v���C���[�������s���������m��
p�́u�헪1���Ƃ��v���C���[�̊��������̊m�������傫�����ΐ헪1���œK�����ƂȂ��v�m��
�Ȃ��ǂ����̐헪���Ƃ��̂������ʂȎ��͐헪1�������Ƃ���
"""
def pmat(N, ips, p):
    P = np.zeros((N+1, N+1)) #���s�����쐬
    P[0, 0] = 1 - (ips/2.0) #��0�s�A��N�s�̐ݒ�
    P[0, 1] = ips/2.0
    P[N,N-1] = ips/2.0
    P[N,N] = 1 - (ips/2.0)
    
    #��1�s������N-1�s�܂ł̐ݒ�
    for i in range(1,N):
        a, b = ips/2.0, ips/2.0
        
        if (i/ float(N)) < p: #�헪0���œK�����ƂȂ�
            a = (a + (1-ips))
        else:                 #�헪1���œK�����ƂȂ�
            b = (b + (1-ips))
        
        #�헪1���Ƃ��v���C���[�������m���A�����m�����v�Z
        dec = (i/ float(N)) * a
        inc = ((N-i)/ float(N)) * b
        
        #�J�ڍs���ɂ��ꂼ���̊m��������
        P[i, i-1] = dec 
        P[i, i+1] = inc
        P[i, i]   = 1-(dec + inc)
        
    return P

#�v���C���[���A�Â̒l�Ap�̒l���ݒ�
players = 20
ipsilon = 0.25
plob = 1.0/3.0

#���ۂɑJ�ڍs�����쐬�A�V�~�����[�V����������
A = pmat(players , ipsilon , plob)
y = mc_sample_path(A, 0, 10000) 

def subplots(): #���A�ڐ������ݒ�
    fig, ax = plt.subplots()
    ax.set_title('KMRmodel')
    return (fig, ax)
fig, ax = subplots() 

output = 0 #0 or 1 or 2

if output == 0: #0(X�̃v���b�g)
    ax.plot(y)
    ax.plot(y, label= str(players) + 'players,ipsilon=' + str(ipsilon))
    ax.legend(loc='lower right')
    plt.show()

if output == 1: #1(X�̃q�X�g�O�������`��)
    ax.hist(y, label= str(players) + 'players,ipsilon=' + str(ipsilon))
    ax.legend()
    plt.show()
    
if output == 2: #2(���핪�z�̕���)
    x=mc_compute_stationary(A)
    ax.bar(range(players+1), x)
    plt.show()
