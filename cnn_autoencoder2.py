import numpy as np
import chainer
from chainer import cuda, Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, reporter
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import Link, Chain, ChainList
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import reporter as reporter_module
from chainer import function
import matplotlib.pyplot as plt
import os.path
import copy

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)
#  matplotlib (1.5.3)
#  chainer (1.20.0.1)

#  ****
#  This is a fault one.
#  eval loss of middle layer in/out, not real in/out


class CNN_Autoencoder2(Chain):
	def __init__(self,num_filter, size_filter, stride0=1, net=None):
		super(CNN_Autoencoder2, self).__init__(
			conv1 = L.Convolution2D(1, num_filter, size_filter, stride=stride0,
			                        initialW=net.conv1_W if net else None ,initial_bias=net.conv1_b if net else None),
			conv2 = L.Convolution2D(num_filter, num_filter * 2, size_filter, stride=stride0),
			dcnv2 = L.Deconvolution2D(num_filter * 2, num_filter, size_filter, stride=stride0),
			dcnv1 = L.Deconvolution2D(num_filter, 1, size_filter, stride=stride0,
			                        initialW=net.dconv1_W if net else None ,initial_bias=net.dconv1_b if net else None),
		)
		self.train = True

	def __call__(self, x):
		h3, h1 = self.sub(x)
		h4 = F.relu(self.dcnv1(h3))
		return h4

	def sub(self, x):
		h1 = F.relu(self.conv1(x))
		h2 = F.relu(self.conv2(h1))
		h3 = F.relu(self.dcnv2(h2))
		return h3,h1

class DelGradient(object):
	name = 'DelGradient'
	def __init__(self, delTgt):
		self.delTgt = delTgt

	def __call__(self, opt):
		for name,param in opt.target.namedparams():
			for d in self.delTgt:
				if d in name:
					#print ('avoid ', d)
					grad = param.grad
					with cuda.get_device(grad):
						grad*=0


class Custom_Updater(training.StandardUpdater):
	def __init__(self, iterator, generator, optimizers, converter=convert.concat_examples, device=None,):
		if isinstance(iterator, iterator_module.Iterator):
			iterator = {'main': iterator}
		self._iterators = iterator
		self.gen = generator
		self._optimizers = optimizers
		self.converter = converter
		self.device = device
		self.iteration = 0

	def update_core(self):
		batch = self._iterators['main'].next()
		in_arrays = self.converter(batch, self.device)
		x_data = in_arrays
		
		batchsize = x_data.shape[0]
		z= x_data

		y_gen,x_gen = self.gen.sub(z)

		loss_gen =  F.mean_squared_error(x_gen, y_gen)
		loss = loss_gen / batchsize

		for optimizer in self._optimizers.values():
			optimizer.target.cleargrads()

		# compute gradients all at once
		loss.backward()

		for optimizer in self._optimizers.values():
			optimizer.update()

		# loss will be summaried and compute_mean() per epoch
		reporter.report(
			{'loss': loss})


class Custom_Evaluator(extensions.Evaluator):
	def evaluate(self):
		iterator = self._iterators['main']
		target = self._targets['main']
		eval_func = target
		it = copy.copy(iterator)

		summary = reporter_module.DictSummary()

		for batch in it:
			observation = {}
			with reporter_module.report_scope(observation):
				in_arrays = self.converter(batch, self.device)
				with function.no_backprop_mode():
						y_z,x_z=eval_func.sub(in_arrays)
						loss_validation =  F.mean_squared_error(y_z, x_z) / in_arrays.shape[0]

			summary.add({'validation/loss':loss_validation })
			summary.add(observation)


		return summary.compute_mean()


def plot_figure(test, OUT_DIR='autoencoder2'):
	@training.make_extension(trigger=(1, 'epoch'))
	def _plot_figure(trainer):
		number=5
		fig, axs = plt.subplots(2,number)
		
		with function.no_backprop_mode():
			for i in range(number):
				x = Variable(test[i].reshape(1, 1, 64, 64) ) 
				y = updater.gen(x)
				x_data = test[i]
				y_data = y.data[0]
				axs[0,i].imshow(x_data.reshape(64,64), cmap='gray')
				axs[1,i].imshow(y_data.reshape(64,64), cmap='gray')

		plt.savefig( os.path.join(OUT_DIR,( 'epoch' + str(updater.epoch) +  '.png')))
		
	return _plot_figure


def get_dataset(IN_DIR='DataSet', train_ratio=9):
	# load data set and convert to tuple in this.py
	train_data = np.load(os.path.join(IN_DIR,'train_data.npy'))
	#train_label = np.load(os.path.join(IN_DIR,'train_label.npy'))
	#print ( train_data.shape[0])
	# dvide train and test per the ratio
	threshold = np.int32(train_data.shape[0]/10*train_ratio)
	train = train_data[0:threshold]
	test  = train_data[threshold:]
	return train, test


def save_auto_wb(model, OUT_DIR='autoencoder2'):
	if not os.path.exists(OUT_DIR):
		os.mkdir(OUT_DIR)
	conv2_W = model.conv2.W.data
	conv2_b = model.conv2.b.data
	dconv2_W = model.dcnv2.W.data
	dconv2_b = model.dcnv2.b.data
	np.save(os.path.join(OUT_DIR,'conv2_W.npy'), conv2_W)
	np.save(os.path.join(OUT_DIR,'conv2_b.npy'), conv2_b)
	np.save(os.path.join(OUT_DIR,'dconv2_W.npy'), dconv2_W)
	np.save(os.path.join(OUT_DIR,'dconv2_b.npy'), dconv2_b)
	print ('saved conv W and b')


class Class_net(object):
	def __init__(self, default_Wb):
		self.Wb = default_Wb
	@property
	def conv1_W(self):
		return self.Wb[0]
	@property
	def conv1_b(self):
		return self.Wb[1]
	@property
	def dconv1_W(self):
		return self.Wb[2]
	@property
	def dconv1_b(self):
		return self.Wb[3]


def load_init_Wb(IN_DIR='log_Autoencoder'):
	list0=['conv1_W','conv1_b','dconv1_W','dconv1_b']
	for i, listx in enumerate(list0):
		f=os.path.join(IN_DIR+ str(int(i/4)+1) ,(listx+'.npy'))
		if os.path.exists(f):
			list0[i] = np.load(f)
			print (' load of ', f)
		else:
			list0[i] = None
			print (' no file of ', f)
	return Class_net(list0)


if __name__=='__main__':
	
	# load upper layer W and b
	net0=load_init_Wb()

	model = CNN_Autoencoder2(num_filter=16, size_filter=6,stride0=2, net=net0)

	train, test = get_dataset()

	optimizer = {'gen': chainer.optimizers.Adam()}
	optimizer['gen'].setup(model)
	
	# To avoid weight update of specified layer
	optimizer['gen'].add_hook(DelGradient(["conv1","dcnv1"]))
	
	
	train_iter = chainer.iterators.SerialIterator(train, batch_size=30)
	test_iter = chainer.iterators.SerialIterator(test, batch_size=30, repeat=False, shuffle=False)

	updater = Custom_Updater(train_iter,  model, optimizer, device=-1)
	
	### output dir "logs"
	OUT_DIR='log_Autoencoder2'
	if not os.path.exists(OUT_DIR):
		os.mkdir(OUT_DIR)
	
	trainer = training.Trainer(updater, (30, 'epoch'), out=OUT_DIR)
	trainer.extend(Custom_Evaluator(test_iter, model, device=-1))
	
	# If plot in/out comparison figure 
	trainer.extend(plot_figure(test,OUT_DIR))
	
	trainer.extend(extensions.LogReport())
	
	trainer.extend(extensions.PrintReport( ['epoch', 'loss', 'validation/loss',  'elapsed_time']))
	trainer.extend(extensions.ProgressBar())
	
	trainer.extend(extensions.PlotReport(['loss','validation/loss'], x_key='epoch', file_name='loss.png'))
	
	
	
	trainer.run()

	save_auto_wb(model, OUT_DIR)



# This file uses TAB




