# -*- coding: utf-8 -*-
#
# 最尤推定による回帰分析
#

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

N = 100
DIM = [0,1,3,9]

##################################################
# 正解データ
##################################################
def t(x):
        t = np.sin(2*np.pi*x)
        return(t)

##################################################
# データ作成
##################################################
def create_dataset(num):
        data_set = DataFrame(columns=['x','y'])
        for i in range(num):
                # sin(2*PI*x) + 正規分布(分散=0.3,平均=0)
                x = float(i)/float(num-1)
                y = t(x) + np.random.normal(scale=0.3)

                # Series:インデックス付けられる配列
                # ignore_index:連結する軸とは別に新たなインデックスを生成する
                data_set = data_set.append(Series([x,y], index=['x','y']),
                                           ignore_index=True)

        return(data_set)
	
##################################################
# 平方根平均二乗誤差(Root Mean Square Error)を計算
##################################################
def rms_error(data_set, f):
        err = 0.0
        for index, line in data_set.iterrows():
                x, y = line.x, line.y
                err += 0.5 * (y - f(x))**2
        rms = np.sqrt(2*err / len(data_set))
        return(rms)

##################################################
# 最大対数尤度(Maximum log likelihood)を計算
##################################################
def log_likelihood(data_set, f):
	err = 0.0
	for index, line in data_set.iterrows():
		x, y = line.x, line.y
		err += (y - f(x))**2
	
	n = float(len(data_set))
	ed = err * 0.5	# Ed
	dev = err / n	# σ^2
	beta = 1 / dev	# 1/σ^2
	lp = 0.5 * n * np.log(beta / (2*np.pi)) - (beta * ed)
	#lp = -(beta * err) + 0.5*n*np.log(0.5 * beta / np.pi)
	
	return(lp)
	
##################################################
# 最尤推定法で解を求める
##################################################
def resolve(data_set, dim):
	
        # Phi = { x0^0, x0^1, x0^2 }
        #       { x1^0, x1^1, x1^2 }
        #       { x2^0, x2^1, x2^2 }
        #       {       ...        }
        #       { xn^0, xn^1, xn^2 }
        t = data_set.y
        phi = DataFrame()
        for i in range(0,dim+1):
                p = data_set.x ** i                     # x^i : xのi乗
                p.name = "x**%d" % i                    # 名称
                phi = pd.concat([phi,p], axis=1)        # x^iの行列

        # dot:行列の積, inv:逆行列
        phi2 = np.dot(phi.T,phi)
        phi2inv = np.linalg.inv(phi2)

        # ws:求めたい係数
        ws = np.dot(np.dot(phi2inv, phi.T), t)
	
        # f(x) = w0 + w1*x + w2*x^2 + ...
        def f(x):
                y = 0
                for i, w in enumerate(ws):
                        y += w * (x ** i)
                return(y)

	# 分散σ^2
	sigma2 = 0.0
	for index, line in data_set.iterrows():
		sigma2 += (line.y - f(line.x))**2
	sigma2 /= len(data_set)
	
	return(f, ws, np.sqrt(sigma2))
	
##################################################
# Main
##################################################
if __name__ == '__main__':
	
	# データ作成
	train_set = create_dataset(N)
        test_set  = create_dataset(N)

	# 多項式近似の曲線を求めて表示
	fig = plt.figure()
	df_ws = DataFrame()
	for i, dim in enumerate(DIM):
		
		# 最尤推定法で係数を求める
		f, ws, sigma = resolve(train_set, dim)
		df_ws = df_ws.append(Series(ws, name='M=%d'%dim))
		
		# トレーニングセットを表示
		subplot = fig.add_subplot(2, 2, i+1)
		subplot.set_xlim(-0.05, 1.05)
		subplot.set_ylim(-1.5, 1.5)
		subplot.scatter(train_set.x, train_set.y,
				marker='o', color='blue', label=None)
		
		# 真の曲線を表示
		line_x = np.linspace(0, 1, 101)
		line_y = t(line_x)
		subplot.plot(line_x, line_y, color='green',
				linestyle='--', linewidth=3)
		
		
		# 学習した曲線を表示
		line_x = np.linspace(0, 1, 101)
		line_y = f(line_x)
		label = 'Sigma=%.2f' % sigma
		subplot.plot(line_x, line_y, color='red', label = label)
		subplot.plot(line_x, line_y+sigma, color='red', linestyle='--')
		subplot.plot(line_x, line_y-sigma, color='red', linestyle='--')
		subplot.legend(loc=1)
		
	# トレーニングセットとテストセットの最大対数尤度を計算
	train_mlh = []
	test_mlh = []
	for dim in range(0,9):
		f, ws, sigma = resolve(train_set, dim)
		train_mlh.append(log_likelihood(train_set, f))
		test_mlh.append(log_likelihood(test_set, f))
	df = DataFrame()
	df = pd.concat([df, 
			DataFrame(train_mlh, columns=['Training set']),
			DataFrame(test_mlh, columns=['Test set'])],
				# columns指定で縦方向の行列になる
			axis=1)	# axis=1で横方向の連結
	df.plot(title=('Log likelihood for N=%d' % N),
			grid=True, style=['-','--'])
	
	# グラフ表示
	plt.show()
	
