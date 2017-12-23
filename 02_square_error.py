# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

N = 100			# データ数
DIM = [0, 1, 3 ,9]	# 次元

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
# 最小二乗法で解を求める
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
		p = data_set.x ** i			# x^i : xのi乗
		p.name = "x**%d" % i			# 名称
		phi = pd.concat([phi,p], axis=1)	# x^iの行列
	
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
	
	return(f,ws)
	
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
		print 'i:%d,dim:%02d' % (i, dim)
		f, ws = resolve(train_set, dim)
		df_ws = df_ws.append(Series(ws, name='M=%d'%dim))
		
		# トレーニングセットを表示
		subplot = fig.add_subplot(2, 2, i+1)	# 2x2のうち(i+1)番目に表示
		subplot.set_xlim(-0.05, 1.05)		# xの表示範囲
		subplot.set_ylim(-1.5, 1.5)		# yの表示範囲
		subplot.set_title('M=%d' % dim)		# タイトル
		subplot.scatter(train_set.x, train_set.y,
		                marker='o', color='blue', label=None)	# oは丸
		
		# 真の曲線を表示
		line_x = np.linspace(0, 1, 101)
		line_y = t(line_x)
		subplot.plot(line_x, line_y, color='green', 
		             linestyle='--', linewidth=3)
		#subplot.plot(line_x, line_y, color='green', linestyle='--')
		
		# 学習した曲線を表示
		line_x = np.linspace(0, 1, 101)
		line_y = f(line_x)
		label='E(RMS)=%.2f' % rms_error(train_set, f)
		subplot.plot(line_x, line_y, color='red', label=label)
		subplot.legend(loc=1)
		
		
	# 係数の値を表示
	print 'Table of the coefficients'
	print df_ws.transpose()
	#fig.show()
	#plt.show()
	
	# トレーニングセットとテストセットの誤差表示
	df = DataFrame(columns=['Training set', 'Test set'])
	for dim in range(0,10):
		f, ws = resolve(train_set, dim)
		train_error = rms_error(train_set, f)
		test_error = rms_error(test_set, f)
		df = df.append(Series([train_error, test_error],
		                      index=['Training set', 'Test set']),
		               ignore_index=True)
				# columns名とindex名が対応する位置に追加
	df.plot(title='RMS Error', style=['-', '--'], grid=True, ylim=(0,0.9))
	plt.show()
