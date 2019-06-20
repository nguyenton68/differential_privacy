'''
preprocess data
https://www.simonwenkel.com/2018/09/26/revisiting-ml-scania-aps-failure.html
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from distutils.version import LooseVersion
from privacy.analysis import privacy_ledger
from privacy.analysis.rdp_accountant import compute_rdp_from_ledger
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers import dp_optimizer

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.preprocessing import MaxAbsScaler


if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
  GradientDescentOptimizer = tf.train.GradientDescentOptimizer
else:
  GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 128, 'Batch size')
flags.DEFINE_integer('epochs', 1, 'Number of epochs')
# flags.DEFINE_integer('num_steps', 1000, 'Number of steps')
flags.DEFINE_integer('num_classes', 2, 'Number of classes')
flags.DEFINE_integer('microbatches', 128, 'Number of microbatches ''(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')


class EpsilonPrintingTrainingHook(tf.estimator.SessionRunHook):
	"""Training hook to print current value of epsilon after an epoch."""

	def __init__(self, ledger):
		"""Initalizes the EpsilonPrintingTrainingHook.
		Args:
		ledger: The privacy ledger.
		"""
		self._samples, self._queries = ledger.get_unformatted_ledger()

	def end(self, session):
		orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
		samples = session.run(self._samples)
		queries = session.run(self._queries)
		formatted_ledger = privacy_ledger.format_ledger(samples, queries)
		rdp = compute_rdp_from_ledger(formatted_ledger, orders)
		eps = get_privacy_spent(orders, rdp, target_delta=1e-5)[0]
		print('For delta=1e-5, the current epsilon is: %.2f' % eps)



def get_data():

	df_train = pd.read_csv('data_original/aps_failure_training_set.csv')
	df_test = pd.read_csv('data_original/aps_failure_test_set.csv')
	# 
	df_train.replace('na','-1', inplace=True)
	df_test.replace('na','-1', inplace=True)
	# categorical for label: 0: neg, 1: pos
	df_train['class'] = pd.Categorical(df_train['class']).codes
	df_test['class']  = pd.Categorical(df_test['class']).codes

	# split data into x and y
	Y_train = df_train['class'].copy(deep=True)
	X_train = df_train.copy(deep=True)
	X_train.drop(['class'], inplace=True, axis=1)

	Y_test = df_test['class'].copy(deep=True)
	X_test = df_test.copy(deep=True)
	X_test.drop(['class'], inplace=True, axis=1)

	# strings to float
	X_train = X_train.astype('float64')
	X_test  = X_test.astype('float64')

	# scale the dataset
	scaler = MaxAbsScaler()
	scaler.fit(X_train)
	X_train = scaler.transform(X_train)
	X_test  = scaler.transform(X_test)

	return X_train, Y_train, X_test, Y_test


def linear_layer(x_dict):
	x = x_dict['images']
	out_layer = tf.keras.layers.Dense(FLAGS.num_classes).apply(x)
	return out_layer


def model_fn(features, labels, mode):
	logits = linear_layer(features)

	# vector loss: each component of the vector correspond to an individual training point and label.
	# Use for per example gradient later.
	vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
		labels=tf.cast(labels, dtype=tf.int64))#=labels) # change compare w mnist


	scalar_loss = tf.reduce_mean(vector_loss)
	print('*******************')
	print(vector_loss.dtype)
	print(scalar_loss.dtype)
	if mode == tf.estimator.ModeKeys.TRAIN:

		if FLAGS.dpsgd:
			ledger = privacy_ledger.PrivacyLedger(
				population_size=60000,
				selection_probability=(FLAGS.batch_size / 60000))

			optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
				l2_norm_clip=FLAGS.l2_norm_clip,
				noise_multiplier=FLAGS.noise_multiplier,
				num_microbatches=FLAGS.microbatches,
				ledger=ledger,
				learning_rate=FLAGS.learning_rate)
			training_hooks = [
				EpsilonPrintingTrainingHook(ledger)
			]
			opt_loss = vector_loss
		else:
			optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
			#train_op  = optimizer.minimize(scalar_loss,
			#	global_step=tf.train.get_global_step())
			opt_loss = scalar_loss
			training_hooks = []
		global_step = tf.train.get_global_step()		
		train_op = optimizer.minimize(loss=opt_loss,
			global_step=global_step)
		return tf.estimator.EstimatorSpec(mode=mode,
			loss=scalar_loss,
			train_op=train_op,
			training_hooks=training_hooks)
	elif mode == tf.estimator.ModeKeys.EVAL:
		# pred_probas  = tf.nn.softmax(logits) # should I remove this ?
		pred_classes = tf.argmax(logits, axis=1)
		acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
		return tf.estimator.EstimatorSpec(mode=mode,
			loss=scalar_loss,
			eval_metric_ops={'accuracy':acc_op})
	#if mode == tf.estimator.ModeKeys.PREDICT:
#		return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)



def main(unused_argv):

	tf.logging.set_verbosity(tf.logging.INFO)
	if FLAGS.dpsgd and FLAGS.batch_size % FLAGS.microbatches != 0:
		raise ValueError('Number of microbatches should divide evenly batch_size')

	# get data: train_data, train_label, test_data, test_label
	x_train, y_train, x_test, y_test = get_data()
	# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

	# Init estimator
	# model_fn, model_dir
	model = tf.estimator.Estimator(model_fn)

	# define train input
	input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'images':x_train},
		y=y_train,
		batch_size=FLAGS.batch_size,
		num_epochs=None,
		shuffle=True)

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={'images': x_test},
		y=y_test,
		batch_size=FLAGS.batch_size,
		shuffle=False)

	# for plotting
	number_epoch = []
	test_accuracy= []


	step_per_epoch = 60000 // FLAGS.batch_size
	# train model on train input
	for epoch in range(FLAGS.epochs):
		model.train(input_fn, steps=step_per_epoch)
		# e = model.evaluate(eval_input_fn)
		# print("Epoch %d, Testing accuracy: %.3f" % (epoch, e['accuracy']))
		# # save to txt
		# # output.write("%d \t %.3f \n" % (epoch, e['accuracy']))
		number_epoch.append(epoch)
		# test_accuracy.append(e['accuracy'])
	plt.plot(number_epoch, test_accuracy)
	plt.show()
	# output.close()

if __name__ == "__main__":
	app.run(main)

	# labels = X_train.iloc[0:1, :]
	# bins = X_train['ab_000'].values
	# plt.hist(X_train['ad_000'].values, bins=30)
	# plt.show()



	# plot histograms
	# fig, axs = plt.subplots(4, 5)
	# axs = axs.ravel()

	# offset = 0
	# for idx, ax in enumerate(axs):
	# 	ax.hist(X_train[:,offset+idx], bins=30)
	# 	index = offset + idx
	# 	ax.set_title('h%d' % index)
	# plt.tight_layout()
	# plt.show()