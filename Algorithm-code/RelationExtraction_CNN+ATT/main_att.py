# -*- coding: utf-8 -*-

from config import opt
import models
import datasets
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import save_pr, now, eval_metric


def collate_fn(batch):

	data, label = zip(*batch)
	return data, label

def test(**kwargs):
	pass

def train(**kwargs):
    kwargs.update({'model': 'PCNN_ATT'})
    opt.parse(kwargs)
    if opt.use_gpu:
		torch.cuda.set_device(opt.gpu_id)

    model = getattr(models, 'PCNN_ATT')(opt)
    if opt.use_gpu:
		model.cuda()

	# loading data
    DataModel = getattr(datasets, opt.data + 'Data')
    train_data = DataModel(opt.data_root, train=True)
    train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn = collate_fn)
    
    test_data = DataModel(opt.data_root, train=False)
    test_data_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn = collate_fn)
    print('{} train data: {}; test data: {}'.format(now(), len(train_data), len(test_data)))

    optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)


    # train
    for epoch in range(opt.num_epochs):
		total_loss = 0
		for idx, (data, label_set) in enumerate(train_data_loader):

		    label = [l[0] for l in label_set]

		    optimizer.zero_grad()
		    model.batch_size = opt.batch_size
		    loss = model(data, label)
		    if opt.use_gpu:
		        label = torch.LongTensor(label).cuda()
		    else:
		        label = torch.LongTensor(label)
		    loss.backward()
		    optimizer.step()
		    total_loss += loss.item()

		if epoch > 2:
			pred_res, p_num = predict_var(model, test_data_loader)
			all_pre, all_rec = eval_metric_var(pred_res, p_num)
			last_pre, last_rec = all_pre[-1], all_rec[-1]
			if last_pre > 0.24 and last_rec > 0.24:
				save_pr(opt.result_dir, model.model_name, epoch, all_pre, all_rec, opt=opt.print_opt)
				print('{} Epoch {} save pr'.format(now, epoch + 1))
			print('{} Epoch {}/{}: train loss: {}; test precision: {}, test recall {}'.format(now(), epoch + 1, opt.num_epochs, total_loss, last_pre, last_rec))
		else:
			print('{} Epoch {}/{}: train loss: {};'.format(now(), epoch + 1, opt.num_epochs, total_loss))


def predict_var(model, test_data_loader):
	'''
	Apply the prediction method in Lin2016
	'''
	model.eval()

	res = []
	true_y = []
	for idx, (data, labels) in enumerate(test_data_loader):
		out = model(data)
		true_y.extend(labels)
		if opt.use_gpu:
			out = out.data.cpu().numpy().tolist()
		else:
			out = out.data.numpy().tolist()
		for r in range(1, opt.rel_num):
			for j in range(len(out[0])):
				res.append([labels[j], r, out[r][j]])
	model.train()
	positive_num = len([i for i in true_y if i[0] > 0])
	return res, positive_num


def eval_metric_var(pred_res, p_num):
	'''
	Apply the evalation method in lin 2016
	'''
	pred_res_sort = sorted(pred_res, key=lambda x: -x[2])
	correct = 0.0
	all_pre = []
	all_rec = []

	for i in range(2000):
		true_y = pred_res_sort[i][0]
		pred_y = pred_res_sort[i][1]
		for j in true_y:
			if pred_y == j:
				correct += 1
				break
		precision = correct / (i + 1)
		recall = correct / p_num
		all_pre.append(precision)
		all_rec.append(recall)

	print("positive_num: {}; correct: {}".format(p_num, correct))
	return all_pre, all_rec

def predict(model, test_data_loader):
	'''
	Apply the prediction method in zeng 2015
	'''
	model.eval()

	pred_y = []
	true_y = []
	pred_p = []

	for idx, (data, labels) in enumerate(test_data_loader):
		true_y.extend(labels)
		out = model(data)
		res = torch.max(out, 1)
		if model.opt.use_gpu:
			pred_y.extend(res[1].data.cpu().numpy().tolist())
			pred_p.extend(res[0].data.cpu().numpy().tolist())
		else:
			pred_y.extend(res[1].data.numpy().tolist())
			pred_p.extend(res[0].data.numpy().tolist())
	size = len(test_data_loader.dataset)
	assert len(pred_y) == size and len(true_y) == size
	assert len(pred_y) == len(pred_p)
	model.train()
	return true_y, pred_y, pred_p

if __name__ == '__main__':
	import fire
	fire.Fire()
		


