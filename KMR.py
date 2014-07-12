from discrete_rv import discreteRV
from mc_tools import *
import numpy as np
import matplotlib.pyplot as plt

"""
遷移行列を作る関数を作成
Nはプレイヤー数、ipsはプレイヤーが実験行動を取る確率
pは「戦略1をとるプレイヤーがこの確率より大きければ戦略1が最適反応となる」確率
なおどちらの戦略をとるのも無差別な時は戦略1を取るとした
"""
def pmat(N, ips, p):
    P = np.zeros((N+1, N+1)) #零行列を作成
    P[0, 0] = 1 - (ips/2.0) #第0行、第N行の設定
    P[0, 1] = ips/2.0
    P[N,N-1] = ips/2.0
    P[N,N] = 1 - (ips/2.0)
    
    #第1行から第N-1行までの設定
    for i in range(1,N):
        a, b = ips/2.0, ips/2.0
        
        if (i/ float(N)) < p:
            a = (a + (1-ips))
        else:
            b = (b + (1-ips))
        
        dec = (i/ float(N)) * a
        inc = ((N-i)/ float(N)) * b
                 
        P[i, i-1] = dec 
        P[i, i+1] = inc
        P[i, i]   = 1-(dec + inc)
        
    return P

#プレイヤー数、εの値、pの値を設定
players = 20
ipsilon = 0.25
plob = 1.0/3.0

#実際に遷移行列を作成、シミュレーションをする
A = pmat(players , ipsilon , plob)
y = mc_sample_path(A, 0, 10000) 

def subplots(): #軸、目盛りを設定
    fig, ax = plt.subplots()
    ax.set_title('KMRmodel')
    return (fig, ax)
fig, ax = subplots() 

output = 0 #0 or 1 or 2

if output == 0: #0(Xのプロット)
    ax.plot(y)
    ax.plot(y, label= str(players) + 'players,ipsilon=' + str(ipsilon))
    ax.legend(loc='lower right')
    plt.show()

if output == 1: #1(Xのヒストグラムを描く)
    ax.hist(y, label= str(players) + 'players,ipsilon=' + str(ipsilon))
    ax.legend()
    plt.show()
    
if output == 2: #2(定常分布の分析)
    sd = []
    for i in range(1, 101):
        ips = 1.0/i
        B = pmat(ips)
        x = mc_compute_stationary(B)
        sd.append(x)

    ax.hist(sd)
    plt.show()    