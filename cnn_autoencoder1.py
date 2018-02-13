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


class CNN_Autoencoder1(Chain):
	def __init__(self,num_filter, size_filter, stride0=1):
		super(CNN_Autoencoder1, self).__init__(
			conv1 = L.Convolution2D(1, num_filter, size_filter, stride=stride0),
			dcnv1 = L.Deconvolution2D(num_filter, 1, size_filter, stride=stride0)
		)
		self.train = True

	def __call__(self, x):
		f1 = F.relu(self.conv1(x))
		f2 = F.relu(self.dcnv1(f1))
		return f2

		
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

		#global x_gen
		x_gen = self.gen(z)

		loss_gen =  F.mean_squared_error(x_gen, z)
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
						z=eval_func(in_arrays)
						loss_validation =  F.mean_squared_error(in_arrays, z) / in_arrays.shape[0]

			summary.add({'validation/loss':loss_validation })
			summary.add(observation)
		
		#
		#for batch in it:

		return summary.compute_mean()


def plot_figure(test, OUT_DIR='autoencoder'):
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


def save_auto_wb(model, OUT_DIR='autoencoder'):
	if not os.path.exists(OUT_DIR):
		os.mkdir(OUT_DIR)
	conv1_W = model.conv1.W.data
	conv1_b = model.conv1.b.data
	np.save(os.path.join(OUT_DIR,'conv1_w.npy'), conv1_W)
	np.save(os.path.join(OUT_DIR,'conv1_b.npy'), conv1_b)
	print ('saved conv W and B')




if __name__=='__main__':

	model = CNN_Autoencoder1(num_filter=16, size_filter=6,stride0=2)

	train, test = get_dataset()

	optimizer = {'gen': chainer.optimizers.Adam()}
	optimizer['gen'].setup(model)
	train_iter = chainer.iterators.SerialIterator(train, batch_size=30)
	test_iter = chainer.iterators.SerialIterator(test, batch_size=30, repeat=False, shuffle=False)

	updater = Custom_Updater(train_iter,  model, optimizer, device=-1)
	
	### output dir "logs"
	OUT_DIR='log_Autoencoder1'
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




