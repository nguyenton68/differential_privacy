# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DP Logistic Regression on MNIST.

DP Logistic Regression on MNIST with support for privacy-by-iteration analysis.
Vitaly Feldman, Ilya Mironov, Kunal Talwar, and Abhradeep Thakurta.
"Privacy amplification by iteration."
In 2018 IEEE 59th Annual Symposium on Foundations of Computer Science (FOCS),
pp. 521-532. IEEE, 2018.
https://arxiv.org/abs/1808.06651.
"""
# Rachel 07/2019
# Training DP with APS dataset

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl import app
from absl import flags

from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf
import pandas as pd
# import utils
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.impute import SimpleImputer
from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers import dp_optimizer
import time



if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
    GradientDescentOptimizer = tf.train.GradientDescentOptimizer
else:
    GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 0.6,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_integer('batch_size', 125, 'Batch size')
flags.DEFINE_integer('epochs', 20, 'Number of epochs')
flags.DEFINE_float('regularizer', 0.0 , 'L2 regularizer coefficient')
flags.DEFINE_string('model_dir', None, 'Model directory')
flags.DEFINE_float('data_l2_norm', 5.0, 'Bound on the L2 norm of normalized data')


def lr_model_fn(features, labels, mode, nclasses, dim):
    """Model function for logistic regression."""
    input_layer = tf.reshape(features['x'], tuple([-1]) + dim)
    logits = tf.layers.dense(inputs=input_layer, units=nclasses,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.regularizer),
                             bias_regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.regularizer))

    # Calculate loss as a vector (to support microbatches in DP-SGD).
    vector_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) + tf.losses.get_regularization_loss()
    # Define mean of loss across minibatch (for reporting through tf.Estimator).
    scalar_loss = tf.reduce_mean(vector_loss)

    # Configure the training op (for TRAIN mode).
    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.dpsgd:
        # The loss function is L-Lipschitz with L = sqrt(2*(||x||^2 + 1)) where
        # ||x|| is the norm of the data.
        # We don't use microbatches (thus speeding up computation), since no
        # clipping is necessary due to data normalization.
            optimizer = dp_optimizer.DPGradientDescentGaussianOptimizer(
                l2_norm_clip=math.sqrt(2 * (FLAGS.data_l2_norm**2 + 1)),
                noise_multiplier=FLAGS.noise_multiplier,
                num_microbatches=1,
                learning_rate=FLAGS.learning_rate)
            opt_loss = vector_loss
        else:
            optimizer = GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
            opt_loss = scalar_loss
        global_step = tf.train.get_global_step()
        train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
        # In the following, we pass the mean of the loss (scalar_loss) rather than
        # the vector_loss because tf.estimator requires a scalar loss. This is only
        # used for evaluation and debugging by tf.estimator. The actual loss being
        # minimized is opt_loss defined above and passed to optimizer.minimize().
        return tf.estimator.EstimatorSpec(mode=mode, loss=scalar_loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode).
    elif mode == tf.estimator.ModeKeys.EVAL:
        pred_classes = tf.argmax(logits, axis=1)
        acc_op    = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
        recall_op = tf.metrics.recall(labels=labels, predictions=pred_classes)
        precision_op = tf.metrics.precision(labels=labels, predictions=pred_classes)
        auc_op = tf.metrics.auc(labels=labels, predictions=pred_classes)
        return tf.estimator.EstimatorSpec(mode=mode,loss=scalar_loss,
                        eval_metric_ops={'recall':recall_op, 'accuracy':acc_op,
                                         'precision':precision_op, 'auc':auc_op})


def normalize_data(data, data_l2_norm):
    """Normalizes data such that each samples has bounded L2 norm.
    Args:
    data: the dataset. Each row represents one samples.
    data_l2_norm: the target upper bound on the L2 norm.
    """

    for i in range(data.shape[0]):
        norm = np.linalg.norm(data[i])
        # print(norm, data[i])
        if norm > data_l2_norm:
            data[i] = data[i] / norm * data_l2_norm


def feature_idxes_with_fair_nan_ratio(df, max_nan_ratio=0.8):
    nan_ratio = df.isnull().sum() / df.shape[0]
    leq_max_nan_ratio = nan_ratio <= max_nan_ratio

    print("High NaN ratio features:", (~leq_max_nan_ratio).sum())
    return leq_max_nan_ratio.index[leq_max_nan_ratio]


def feature_idxes_with_fair_corr(df, max_corr_coef=0.99):
    corr = df.corr()
    corr_upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    high_corr = (corr_upper > max_corr_coef).any()

    print("Highly correlated features:", high_corr.sum())
    return high_corr.index[~high_corr]


def feature_idxes_with_fair_var(df, min_var=1e-3):
    gt_min_var = df.var() > min_var

    print("Low variance features:", (~gt_min_var).sum())
    return gt_min_var.index[gt_min_var]

def standard_scaling(df_train, df_tests):
    scaler = StandardScaler()
    df_train_std = scaler.fit_transform(df_train)
    df_tests_std = [scaler.transform(df) for df in df_tests]

    return df_train_std, df_tests_std


def fill_missing_values(X_train, X_tests,
                        missing_values=np.nan,
                        strategy='most_frequent'):
    nan_imputer = SimpleImputer(missing_values=missing_values,
                                strategy=strategy)
    X_train_imputed = nan_imputer.fit_transform(X_train)
    X_tests_imputed = [nan_imputer.transform(X) for X in X_tests]

    print("Fill missing values with '%s'" % strategy)

    return X_train_imputed, X_tests_imputed

def sample_preprocessing(df_train, df_tests):
    # Feature standardization (remove the mean and scale to unit variance)
    X_train_std, X_tests_std = standard_scaling(df_train, df_tests)

    # Missing value imputation
    X_train_imputed, X_tests_imputed = fill_missing_values(X_train_std, X_tests_std)

    return X_train_imputed, X_tests_imputed

def preprocessing(df_X_train, df_X_tests):
    keep_cols = set(df_X_train.columns)

    # Only keep features that have reasonable amount of missing values
    fair_nan_ratio_idxes = feature_idxes_with_fair_nan_ratio(df_X_train[keep_cols], 0.8)
    keep_cols = keep_cols.intersection(fair_nan_ratio_idxes)

    # Only keep features that are not too highly correlated
    fair_corr_idxes = feature_idxes_with_fair_corr(df_X_train[keep_cols], 0.99)
    keep_cols = keep_cols.intersection(fair_corr_idxes)

    # Do not keep features with low variance
    fair_var_idxes = feature_idxes_with_fair_var(df_X_train[keep_cols], min_var=1e-3)
    keep_cols = keep_cols.intersection(fair_var_idxes)

    df_X_train_col_filtered = df_X_train[keep_cols]
    df_X_tests_col_filtered = [df[keep_cols] for df in df_X_tests]

    # Modify sample values without modifying the shape of the dataset
    X_train_processed, X_tests_processed = sample_preprocessing(df_X_train_col_filtered,
                                                                df_X_tests_col_filtered)


    return X_train_processed, X_tests_processed, list(keep_cols)


def load_aps(data_l2_norm=float('inf')):
    """Loads APS and preprocesses to combine training and validation data."""
    # train, test = tf.keras.datasets.mnist.load_data()
    df_training = pd.read_csv('data/aps_failure_training_set.csv', na_values='na')
    df_test = pd.read_csv('data/aps_failure_test_set.csv', na_values='na')
    df_X_eval = pd.read_csv('data/aps_failure_evaluation_set.csv', na_values='na')
    y_eval = pd.read_csv('data/aps_failure_evaluation_set_target.csv')

    # Convert string labels to numerical labels
    label_mapping = {'neg': 0, 'pos': 1}
    df_training['class'] = df_training['class'].map(label_mapping)
    df_test['class'] = df_test['class'].map(label_mapping)
    y_eval = y_eval['class'].map(label_mapping)

    # Extract feature matrix X and target label vector Y
    df_X_train = df_training.drop('class', axis=1)
    y_train = df_training['class']
    df_X_test = df_test.drop('class', axis=1)
    y_test = df_test['class']

    X_train, [X_test, X_eval], keep_cols = preprocessing(df_X_train, [df_X_test, df_X_eval])

    # train_data = df_X_train.to_numpy(dtype='float32')
    # train_labels= y_train.to_numpy()
    # test_data  = df_X_test.to_numpy(dtype='float32')
    # test_labels = y_test.to_numpy()
    # print('Training dataset shape:',  X_train.shape, X_train.dtype, type(X_train))
    # print('Test dataset shape:', X_test.shape, type(y_train))
    # print('Eval dataset shape:', df_X_eval.shape)


    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    idx = np.random.permutation(len(X_train))  # shuffle data once
    train_data = X_train[idx]
    train_labels = y_train[idx]


    #
    normalize_data(train_data, data_l2_norm)
    normalize_data(X_test, data_l2_norm)
    #
    # train_data = np.array(train_data, dtype=np.float32)
    # test_data  = np.array(test_data,  dtype=np.float32)
    # train_labels = np.array(train_labels, dtype=np.int32)
    # test_labels = np.array(test_labels, dtype=np.int32)
    # print(train_data.dtype, train_labels.dtype)
    # return train_data, train_labels, test_data, test_labels
    train_data = train_data.astype(np.float32)
    X_test = X_test.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return train_data, train_labels, X_test, y_test
    # return train_data, train_labels, test_data, test_labels


def get_data(data_l2_norm=float('inf')):

    df_train = pd.read_csv('data_original/aps_failure_training_set.csv')
    df_test = pd.read_csv('data_original/aps_failure_test_set.csv')

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
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    Y_train = Y_train.astype(np.int32)
    Y_test  = Y_test.astype(np.int32)

    # convert to numpy array ?
    X_train = X_train.to_numpy(dtype='float32')
    X_test  = X_test.to_numpy(dtype='float32')

    # scale the dataset
    scaler = MaxAbsScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    normalize_data(X_train, data_l2_norm)
    normalize_data(X_test, data_l2_norm)
    return X_train, Y_train, X_test, Y_test


def print_privacy_guarantees(epochs, batch_size, samples, noise_multiplier):
    """Tabulating position-dependent privacy guarantees."""
    if noise_multiplier == 0:
        print('No differential privacy (additive noise is 0).')
        return
    print('In the conditions of Theorem 34 (https://arxiv.org/abs/1808.06651) '
        'the training procedure results in the following privacy guarantees.')

    print('Out of the total of {} samples:'.format(samples))

    steps_per_epoch = samples // batch_size
    orders = np.concatenate([np.linspace(2, 20, num=181),
                             np.linspace(20, 100, num=81)])
    delta = 1e-5
    for p in (.5, .9, .99):
        steps = math.ceil(steps_per_epoch * p)  # Steps in the last epoch.
        coef = 2 * (noise_multiplier * batch_size)**-2 * (
        # Accounting for privacy loss
        (epochs - 1) / steps_per_epoch +  # ... from all-but-last epochs
        1 / (steps_per_epoch - steps + 1))  # ... due to the last epoch
        # Using RDP accountant to compute eps. Doing computation analytically is
        # an option.
        rdp = [order * coef for order in orders]
        eps, _, _ = get_privacy_spent(orders, rdp, target_delta=delta)
        print('\t{:g}% enjoy at least ({:.2f}, {})-DP'.format(
        p * 100, eps, delta))

    # Compute privacy guarantees for the Sampled Gaussian Mechanism.
    rdp_sgm = compute_rdp(batch_size / samples, noise_multiplier,
                        epochs * steps_per_epoch, orders)
    eps_sgm, _, _ = get_privacy_spent(orders, rdp_sgm, target_delta=delta)
    print('By comparison, DP-SGD analysis for training done with the same '
          'parameters and random shuffling in each epoch guarantees '
          '({:.2f}, {})-DP for all samples.'.format(eps_sgm, delta))


def main(unused_argv):
    start_time = time.perf_counter()
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.data_l2_norm <= 0:
        raise ValueError('data_l2_norm must be positive.')
    if FLAGS.dpsgd and FLAGS.learning_rate > 8 / FLAGS.data_l2_norm**2:
        raise ValueError('The amplification-by-iteration analysis requires'
                     'learning_rate <= 2 / beta, where beta is the smoothness'
                     'of the loss function and is upper bounded by ||x||^2 / 4'
                     'with ||x|| being the largest L2 norm of the samples.')

    # Load training and test data.
    # Smoothness = ||x||^2 / 4 where ||x|| is the largest L2 norm of the samples.
    # To get bounded smoothness, we normalize the data such that each sample has a
    # bounded L2 norm.
    train_data, train_labels, test_data, test_labels = get_data(data_l2_norm=FLAGS.data_l2_norm)
    print('Train/test size = ', train_data.shape, test_data.shape)
    # Instantiate tf.Estimator.
    # pylint: disable=g-long-lambda
    nclasses = len(np.unique(train_labels)) # make sure it right
    # print('number of class ', nclasses)
    model_fn = lambda features, labels, mode: lr_model_fn(
      features, labels, mode, nclasses=nclasses, dim=train_data.shape[1:])
    aps_classifier = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir)

    # Create tf.Estimator input functions for the training and test data.
    # To analyze the per-user privacy loss, we keep the same orders of samples in
    # each epoch by setting shuffle=False.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': train_data},
      y=train_labels,
      batch_size=FLAGS.batch_size,
      num_epochs=FLAGS.epochs,
      shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': test_data}, y=test_labels, num_epochs=1, shuffle=False)

    # Train the model.
    num_samples = train_data.shape[0]
    steps_per_epoch = num_samples // FLAGS.batch_size
    #
    # output = open("dp_lr_0p1_noise_0p6_l2_5p0_batch_125_epoch_20.txt", "w+")

    for i in range(100):
        aps_classifier.train(input_fn=train_input_fn,
                         steps=steps_per_epoch* FLAGS.epochs)

        # Evaluate the model and print results.
        eval_results = aps_classifier.evaluate(input_fn=eval_input_fn)
        accuracy = eval_results['accuracy']
        recall = eval_results['recall']
        precision = eval_results['precision']
        auc = eval_results['auc']
        print('After {} epochs : accuracy {:.2f} recall {:.2f} precision {:.2f} auc {:.2f}'.format(FLAGS.epochs,
                accuracy, recall, precision, auc))
        # output.write("%d \t %.3f \t %.3f \t %.3f \t %.3f \n" % (i, accuracy, recall, precision, auc))
    if FLAGS.dpsgd:
        print_privacy_guarantees(
        epochs=FLAGS.epochs,
        batch_size=FLAGS.batch_size,
        samples=num_samples,
        noise_multiplier=FLAGS.noise_multiplier)

    print('Total time = ', time.perf_counter() - start_time, ' seconds')
    # output.close()
if __name__ == '__main__':
    app.run(main)
