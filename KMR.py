from discrete_rv import discreteRV
from mc_tools import *
import numpy as np
import matplotlib.pyplot as plt

"""
遷移行列を作る関数を作成
ここでは自分自身と対戦することも許している
Nはプレイヤー数、ipsはプレイヤーが実験行動を取る確率
pは「戦略1をとるプレイヤーの割合がこの確率より大きければ戦略1が最適反応となる」確率
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
        
        if (i/ float(N)) < p: #戦略0が最適反応となる
            a = (a + (1-ips))
        else:                 #戦略1が最適反応となる
            b = (b + (1-ips))
        
        #戦略1をとるプレイヤーが減る確率、減る確率を計算
        dec = (i/ float(N)) * a
        inc = ((N-i)/ float(N)) * b
        
        #遷移行列にそれぞれの確率を代入
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
    x=mc_compute_stationary(A)
    ax.bar(range(players+1), x)
    plt.show()