# -*- coding: utf-8 -*-
#
# パーセプトロンによる二項分類
#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

N1 = 20		# クラス t=+1のデータ数
M1 = [15,10]	# クラス t=+1の中心座標

N2 = 20		# クラス t=-1のデータ数
M2 = [0,0]	# クラス t=-1の中心座標

Variances = [15,50]	# 両クラス共通の分散(2種類の分散で計算を実施)

ITER=30		# 繰り返し計算の回数

##################################################
# データセット作成
##################################################
def prepare_dataset(var):
	
	# クラス t=+1のデータ作成
	cov1 = np.array( [ [var,0],[0,var] ] )	# 共分散行列
	df1 = DataFrame(np.random.multivariate_normal(M1, cov1, N1),
			columns=['x','y'])	# multivariate_normal:多次元正規分布
						# 平均:M1, 共分散行列:cov1, データ数:N1
	df1['type'] = 1.0			# 正解ラベル:1
	
	# クラス t=-1のデータ作成
	cov2 = np.array( [ [var,0],[0,var] ] )	# 共分散行列
	df2 = DataFrame(np.random.multivariate_normal(M2, cov2, N2),
			columns=['x','y'])	# multivariate_normal:多次元正規分布
						# 平均:M1, 共分散行列:cov1, データ数:N1
	df2['type'] = -1.0			# 正解ラベル:-1
	
	# クラス t=+1,t=-1のデータを結合	
	df = pd.concat([df1,df2], ignore_index=True)
	
	# データをランダムに並び替え
	shuffle_idx = np.random.permutation(df.index)	# permutation:コピーを生成
							# shuffle:本体を並び替え
	df = df.reindex(shuffle_idx).reset_index(drop=True)
						# reindex：引数のインデックスに振り直す
						# reset_index：先頭から0,1,...と振り直す
						# drop:旧インデックスを残さない
	return(df)
	
##################################################
# パーセプトロンを確率的勾配降下法で解く
##################################################
def simulation(train_set):
	
	# パラメータの初期化
	w = np.array([0.0 for i in range(3)])
	#w = [0 for i in range(3)]
	bias = 0.5 * (train_set.x.mean() + train_set.y.mean())
	
	# 繰り返し学習
	paramhist = DataFrame( [w], columns=['w0','w1','w2'])
	for i in range(ITER):
		for index, point in train_set.iterrows():
			x, y, t = point.x, point.y, point.type
			z = w[0]*bias + w[1]*x + w[2]*y
			if (t*z) <= 0:
				w += t*np.array([bias, x, y])
				#w[0] += t*bias
				#w[1] += t*x
				#w[2] += t*y
		paramhist = paramhist.append(
				Series(w, index=['w0','w1','w2']),
				ignore_index=True)
	
	# 判定誤差の計算
	err = 0
	for index, point in train_set.iterrows():
		x, y, t = point.x, point.y, point.type
		z = w[0]*bias + w[1]*x + w[2]*y
		if (t*z) < 0:
			err += 1
	err_rate = err * 100 / len(train_set)
	
	return(w, bias, err_rate, paramhist)
		
	
##################################################
# Main
##################################################
if __name__ == '__main__':
	
	# 2種類の分散で実行
	fig = plt.figure()
	for i, var in enumerate(Variances):
		
		# 2x2のうち、どこに表示するか
		data_subplot = fig.add_subplot(2, 2, i*2+1)
		param_subplot = fig.add_subplot(2, 2, i*2+2)
		
		# データ群作成
		train_set = prepare_dataset(var)
		
		# パーセプトロンを解く
		w, bias, err_rate, paramhist = simulation(train_set)
		
		# データ群表示
		xmin, xmax = train_set.x.min()-5, train_set.x.max()+5
		ymin, ymax = train_set.y.min()-5, train_set.y.max()+5
		data_subplot.set_xlim([xmin, xmax])
		data_subplot.set_ylim([ymin, ymax])
		train_set1 = train_set[train_set['type']==1]
		train_set2 = train_set[train_set['type']==-1]
		data_subplot.scatter(train_set1.x, train_set1.y, marker='o', label=None)
		data_subplot.scatter(train_set2.x, train_set2.y, marker='x', label=None)
		
		# 分割する直線表示
		line_x = np.arange(xmin-5, xmax+5)	# 始点:xmin-5, 終点:xmax+5
		line_y = -(w[0]*bias + w[1]*line_x) / w[2]
		print 'w:[%f,%f,%f]' % (w[0],w[1],w[2])
		label = 'ERR %0.2f%%' % err_rate
		data_subplot.plot(line_x, line_y, label=label, color='red')
		data_subplot.legend(loc=1)
		
		# パラメータ(重み,バイアス)の推移表示
		paramhist.plot(ax=param_subplot)	# ax:座標軸
		param_subplot.legend(loc=1)
		
	# 表示
	#fig.show()
	plt.show()
