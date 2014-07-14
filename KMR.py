# -*- coding: utf-8 -*-

from discrete_rv import discreteRV
from mc_tools import *
import numpy as np
import matplotlib.pyplot as plt

"""
遷移行列を作成
Nはプレイヤーの数、ipsは実験行動をとる確率、
pは「戦略1をとっているプレイヤーの割合がこれより大きければ戦略1が最適反応となる」確率
なお、どちらの戦略をとるのも無差別なときは戦略1をとるとし、自分自信と対戦することも許した
"""
def pmat(N, ips, p):
    P = np.zeros((N+1, N+1)) #零行列を作成
    P[0, 0] = 1 - (ips/2.0) #第0行、第N行を設定
    P[0, 1] = ips/2.0
    P[N,N-1] = ips/2.0
    P[N,N] = 1 - (ips/2.0)
    
    #第１行から第N-1行を設定
    for i in range(1,N):
        a, b = ips/2.0, ips/2.0 #aとbはそれぞれ戦略1、0をとるプレイヤーが選ばれたときに戦略を変更する確率
        
        if (i/ float(N)) < p: #戦略0をとるのが最適反応
            a = (a + (1-ips))
        else:                 #戦略1をとるのが最適反応
            b = (b + (1-ips))
        
        #戦略1をとるプレイヤーが減る確率、増える確率を計算
        dec = (i/ float(N)) * a     #(戦略1をとるプレイヤーが選ばれる)かつ(戦略を変更する)
        inc = ((N-i)/ float(N)) * b #(戦略0をとるプレイヤーが選ばれる)かつ(戦略を変更する)
        
        #遷移行列にそれぞれを入れる
        P[i, i-1] = dec 
        P[i, i+1] = inc
        P[i, i]   = 1-(dec + inc)
        
    return P

#プレイヤーの数、εの値、確率を設定
players = 20
ipsilon = 0.25
plob = 1.0/3.0

#遷移行列を作成してシミュレーションをする
A = pmat(players , ipsilon , plob)
y = mc_sample_path(A, 0, 10000) 

def subplots(): #軸の設定
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

if output == 1: #1(Xのヒストグラムを作成)
    ax.hist(y, label= str(players) + 'players,ipsilon=' + str(ipsilon))
    ax.legend()
    plt.show()
    
if output == 2: #2(定常状態を分析�)
    x=mc_compute_stationary(A)
    ax.bar(range(players+1), x)
    plt.show()
