# -*- coding:utf-8 -*-

import torch
import time

class BasicModule(torch.nn.Module):


	def __init__(self):
		super(BasicModule, self).__init__()
		self.model_name = str(type(self))

	def load(self, path):
		'''
        loading model path
		'''
		self.load_state_dict(torch.load(path))

	def save(self, name=None):
		'''
		save model, name+date
		'''
		prefix = "checkpoints/"
		if name is None:
			name = prefix + self.model_name + '_'
			name = time.strftime(name + '%m%d_%M:%S.pth')
		else:
			name = prefix + self.model_name + '_' + str(name) + '.pth'
		torch.save(self.load_state_dict(), name)
		return name