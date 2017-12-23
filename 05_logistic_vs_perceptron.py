# -*- coding: utf-8 -*-
#
# ロジスティック回帰とパーセプトロンによる二項分類
#
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

N1 = 20		# クラス t=+1のデータ数
M1 = [15,0]	# クラス t=+1の中心座標

N2 = 20		# クラス t=-1のデータ数
M2 = [-3,-3]	# クラス t=-1の中心座標

Variances = [5,10,30,50]	# 両クラス共通の分散(2種類の分散で計算を実施)

ITER=30		# 繰り返し計算の回数
EPS=0.001	# 繰り返し計算の閾値

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
	df2['type'] = 0				# 正解ラベル:0
	
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
# パーセプトロン＋確率的勾配降下法でパラメータ推定
##################################################
def run_perceptron(train_set):
	
	# パラメータの初期化
	w = np.array([0.1 for i in range(3)])
	bias = 0.5 * (train_set.x.mean() + train_set.y.mean())
	
	# 繰り返し学習
	for i in range(ITER):
		# 確率的勾配降下法でパラメータ修正
		for index, point in train_set.iterrows():
			x, y, t = point.x, point.y, point.type
			z = w[0]*bias + w[1]*x + w[2]*y
			t = t*2 - 1	# t=1,0 -> t=1,-1に変換する
			if (t*z) <= 0:
				w += t*np.array([bias, x, y])
	
	# 判定誤差の計算
	err = 0
	for index, point in train_set.iterrows():
		x, y, t = point.x, point.y, point.type
		z = w[0]*bias + w[1]*x + w[2]*y
		if (t*z) < 0:
			err += 1
	err_rate = err * 100 / len(train_set)
	
	return(w, bias, err_rate)
		
##################################################
# ロジスティック回帰でパラメータ推定
##################################################
def run_logistic(train_set):
	
	# パラメータの初期化
        #   Phi = { 1, x0, y0 }
        #         { 1, x1, y1 }
        #         { 1, x2, y2 }
        #         { 1, ...    }
        #         { 1, xn, yn }

	w = np.array([[0],[0.1],[0.1]])
	phi = train_set[['x','y']]
	phi['bias'] = 1
	phi = phi.as_matrix(columns=['bias','x','y'])
			# biasが先頭に来るように並び替える
	t = train_set[['type']]
	t = t.as_matrix()	# DataFrameから行列(ndarray)に変換
	
	# 繰り返し学習
	for i in range(ITER):
		# IRIS法でパラメータ修正
		z = np.array([])
		for line in phi:
			a = np.dot(line, w)	# a=w*Φ
			z = np.append(z, [ 1.0/(1.0+np.exp(-a)) ] )
						# ロジスティック関数z=σ(a)
		r = np.diag(z*(1-z))		# R=z*(1-z)の対角行列
		z = z[np.newaxis,:].T		# [z0,z1 ,... ,zn]から
		#z = z[np.newaxis].T		# [z0,z1 ,... ,zn]から
						# [ [z0],
						#   [z1],
						#   ... ,
						#   [zn] ]に変換
		#z = z.as_matrix()
		phit_r_phi = np.dot(np.dot(phi.T,r),phi)	# Φt * R * Φ
		inv = np.linalg.inv(phit_r_phi)
		phit_z_t = np.dot(phi.T, (z-t))			# Φt * (z-t)
		w_new = w - np.dot(inv, phit_z_t)
		#w_new = w - np.dot(np.linalg.inv(phit_r_phi), phit_z_t)
		
		# パラメータの変化が閾値未満になったら終了
		e = np.dot( (w_new-w).T, (w_new-w) )	# (w_new - w_old)^2
		w_old = np.dot( w.T, w )		# (w_old)^2
		if (e/w_old) < EPS:
			w = w_new
			break
		w = w_new
		
	
	# 判定誤差の計算
	err = 0
	for index, point in train_set.iterrows():
		x, y, t = point.x, point.y, point.type
		z = w[0] + w[1]*x + w[2]*y
		if (t*z) < 0:
			err += 1
	err_rate = err * 100 / len(train_set)
	
	return(w, err_rate)
		
	
##################################################
# Main
##################################################
if __name__ == '__main__':
	
	# 2種類の分散で実行
	fig = plt.figure()
	for i, var in enumerate(Variances):
		
		# 2x2のうち、どこに表示するか
		data_subplot = fig.add_subplot(2, 2, i+1)
		
		# データ群作成
		train_set = prepare_dataset(var)
		
		# データ群表示
		xmin, xmax = train_set.x.min()-5, train_set.x.max()+5
		ymin, ymax = train_set.y.min()-5, train_set.y.max()+5
		data_subplot.set_xlim([xmin, xmax])
		data_subplot.set_ylim([ymin, ymax])
		train_set1 = train_set[train_set['type']==1]
		train_set2 = train_set[train_set['type']==0]
		data_subplot.scatter(train_set1.x, train_set1.y, marker='o', label=None)
		data_subplot.scatter(train_set2.x, train_set2.y, marker='x', label=None)
		
		# パーセプトロンの結果表示
		w, bias, err_rate = run_perceptron(train_set)
		line_x = np.arange(xmin-5, xmax+5)	# 始点:xmin-5, 終点:xmax+5
		line_y = -(w[0]*bias + w[1]*line_x) / w[2]
		label = 'ERR %0.2f%%' % err_rate
		data_subplot.plot(line_x, line_y, label=label, color='red', linestyle='--')
		data_subplot.legend(loc=1)
		
		# ロジスティック回帰の結果表示
		w, err_rate = run_logistic(train_set)
		line_y = -(w[0] + w[1]*line_x) / w[2]
		label = 'ERR %0.2f%%' % err_rate
		data_subplot.plot(line_x, line_y, label=label, color='blue')
		data_subplot.legend(loc=1)
		
	# 表示
	#fig.show()
	plt.show()
