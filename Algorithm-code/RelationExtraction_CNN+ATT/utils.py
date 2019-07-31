# -*- coding: utf-8 -*-

import numpy as np 
import time

def now():
	return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def save_pr(out_dir, name, epoch, pre, rec, fp_res=None, opt=None):
	if opt is None:
		out = open('{}/{}_{}_PR.txt'.format(out_dir, name, epoch + 1), 'w')
	else:
		out = out = open('{}/{}_{}_{}_PR.txt'.format(out_dir, name, opt, epoch + 1), 'w')
	if fp_res is not None:
		fp_out = open('{}/{}_{}_FP.txt'.format(out_dir, name, epoch + 1), 'w')
		for idx, r, p in fp_res:
			fp_out.write('{}{}{}\n'.format(idx, r, p))
	fp_out.close()

	for p, r in zip(pre, rec):
		out.write('{} {}\n'.format(p, r))
	out.close()


def eval_metric(true_y, pred_y, pred_p):
	'''
	calculate the precision and recall for p-r curve
	reglect the NA relation
	'''
	assert len(true_y) == len(pred_y)
	positive_num = len([i for i in true_y if i[0] > 0])
	index = np.argsort(pred_p)[::-1]

	tp = 0
	fp = 0
	fn = 0
	all_pre = [0]
	all_rec = [0]
	fp_res = []

	for idx in range(len(true_y)):
		i = true_y[idx]
		j = pred_y[idx]

		if i[0] == 0:
			if j > 0:
				fp_res.append(index[idx], j, pred_p[index[idx]])
				fp += 1
		else:
			if j == 0:
				fn += 1
			else:
				for k in i:
					if k == -1:
						break
					if k == j:
						tp += 1
						break
		if fp + tp == 0:
			precision = 1.0
		else:
			precision = tp * 1.0 / (fp + tp)
        recall = tp * 1.0 / positive_num
        if precision != all_pre[-1] or recall != all_rec[-1]:
        	all_pre.append(precision)
        	all_rec.append(recall)


	print("tp={}; fp={}; fn={}; positive_num={}".format(tp, fp, fn, positive_num))
	return all_pre[1:], all_rec[1:], fp_res
