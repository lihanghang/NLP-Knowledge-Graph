# -*- coding: utf-8 -*-

from config import opt
import models
import dataset
import torch
from utils.data import DataLoader
import torch.optim as optim
from utils import save_pr, now, eval_metric


def collate_fn(batch):

	data, label = zip(*batch)
	return data, label

def test(**kwargs):
	pass

def train(**kwargs):
	kwargs.update({'model': 'pcnnAtt'})
	opt.parse(kwargs)
	if opt.use_gpu:
		torch.cuda.set_device(opt.gpu_id)
	model = getattr(models, "pcnnAtt")(opt)
	if opt.use_gpu:
		model.cuda()

	# loading data
	DataModel = getattr(dataset, opt.data + "Data")
	train_data = DataModel(opt.data_root, train=True)
	train_data_loader = DataLoader(train_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn = collate_fn)

	test_data = DataModel(opt.data_root, train=False)
	test_data_loader = DataLoader(test_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn = collate_fn)
	print('{} train data: {}; test data: {}'.format(now(), len(train_data), len(test_data)))

	optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=1e-6)


