import numpy as np
import chainer
from chainer import cuda, Function, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, reporter
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer.datasets import tuple_dataset
from chainer import Link, Chain, ChainList
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import os.path

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy (1.14.0)
#  chainer (1.20.0.1)

class SimpleCNN3(Chain):
	def __init__(self,num_filter, size_filter, stride0=1):
		super(SimpleCNN3, self).__init__(
			conv1 = L.Convolution2D(1, num_filter, size_filter, stride=stride0),
			conv2 = L.Convolution2D(num_filter, num_filter * 2, size_filter, stride=stride0),
			conv3 = L.Convolution2D(num_filter * 2, num_filter * 2, size_filter, stride=stride0),
			l1 = L.Linear(512, 10),
		)
		self.train = True
		#print ('num_filter', num_filter)
		#print ('size_filter', size_filter)


	def __call__(self, x):
		h1 = F.relu(self.conv1(x))
		#print ('h1.shape ', h1.data.shape)
		h2 = F.relu(self.conv2(h1))
		#print ('h2.shape ', h2.data.shape)
		h3 = F.relu(self.conv3(h2))
		#print ('h3.shape ', h3.data.shape)
		h4 = self.l1(h3)
		return h4


def get_dataset(IN_DIR='DataSet', train_ratio=9):
	# load data set and convert to tuple in this.py
	train_data = np.load(os.path.join(IN_DIR,'train_data.npy'))
	train_label = np.load(os.path.join(IN_DIR,'train_label.npy'))
	#print ( train_data.shape[0])
	# dvide train and test per the ratio
	threshold = np.int32(train_data.shape[0]/10*train_ratio)
	train = tuple_dataset.TupleDataset(train_data[0:threshold], train_label[0:threshold])
	test  = tuple_dataset.TupleDataset(train_data[threshold:],  train_label[threshold:])
	return train, test



if __name__=='__main__':

	model = L.Classifier(SimpleCNN3(num_filter=16, size_filter=6,stride0=2))


	train,test=get_dataset()

	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)
	train_iter = chainer.iterators.SerialIterator(train, batch_size=30)
	test_iter = chainer.iterators.SerialIterator(test, batch_size=30, repeat=False, shuffle=False)

	updater = training.StandardUpdater(train_iter, optimizer, device=-1)
	
	### output dir "logs"
	OUT_DIR='result'
	trainer = training.Trainer(updater, (10, 'epoch'), out=OUT_DIR)

	trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
	trainer.extend(extensions.LogReport())
	trainer.extend(extensions.PrintReport( ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
	trainer.extend(extensions.ProgressBar())
	
	trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
	trainer.extend(extensions.PlotReport(['main/accuracy','validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))

	# output result as save_npz
	trainer.extend(extensions.snapshot())
	
	trainer.run()



# This file uses TAB




