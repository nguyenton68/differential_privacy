'''
Use DP on MNIST (balanced dataset)
- Compare between non-private and private model

'''
# Rachel 07/2019
# Training IBM privlib for MNIST dataset

from sklearn.datasets import fetch_openml
# from sklearn import datasets
from sklearn.model_selection import train_test_split
import diffprivlib.models as dp 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from diffprivlib.models.logistic_regression import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from sklearn import linear_model
from sklearn.utils import check_random_state
import time


start_time = time.perf_counter()

train_size = 60000
test_size  = 10000
# pre-process image
# small dataset
# dataset = datasets.load_digits() 
# dataset = datasets.load_iris()
# X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)
# X_train = [1437, 64]
# X_test  = [360, 64]

# Full dataset
#
# X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# random_state = check_random_state(0)
# permutation = random_state.permutation(X.shape[0])
# X = X[permutation]
# y = y[permutation]
# X = X.reshape((X.shape[0], -1))
# X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_size,test_size=test_size)
# scale the dataset
# scaler = MaxAbsScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test  = scaler.transform(X_test)

train, test = tf.keras.datasets.mnist.load_data()
train_data, train_labels = train
test_data, test_labels = test

train_data = np.array(train_data, dtype=np.float32) / 255
test_data = np.array(test_data, dtype=np.float32) / 255

train_data = train_data.reshape(train_data.shape[0], -1)
X_test = test_data.reshape(test_data.shape[0], -1)

idx = np.random.permutation(len(train_data))  # shuffle data once
X_train = train_data[idx]
y_train = train_labels[idx]

y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(test_labels, dtype=np.int32)




# # define list of epsilon
epsilons = np.logspace(-2, 2, 50)

acc_w_dp = list()
# acc_wo_dp = list()
output = open("ibm_mnist_dp_data_norm_18_70k_images.txt", "w+")
for epsilon in epsilons:
# for i in range(1):
# 	# logreg wo DP
# 	logreg_wo_dp = LogisticRegression(epsilon=float('inf'), data_norm=28, max_iter=1000)

# normal linear model wo dp
	#linear_model.LogisticRegression(solver="lbfgs", multi_class="ovr", max_iter=1000)
# 	# logreg from DP
	logreg_w_dp  = LogisticRegression(epsilon=epsilon, data_norm=18, max_iter=20)
# 	# l2 norm = sqrt(all_columns), there are 64 columns -> data_norm = 8

	# logreg_wo_dp.fit(X_train, y_train)
	logreg_w_dp.fit(X_train, y_train)
	score_dp = logreg_w_dp.score(X_test, y_test)
	# score    = logreg_wo_dp.score(X_test, y_test)
	# print('Accuracy = ', score_dp)
# 	acc_wo_dp.append(score)
	acc_w_dp.append(score_dp)
	output.write("%.3f \t %.3f\n" % (epsilon, score_dp))

print('Total time = ', time.perf_counter() - start_time, ' seconds')
plt.plot(epsilons, acc_w_dp)
# plt.plot(epsilons, acc_wo_dp)

# fig = plt.figure()
# ax  = plt.subplot(111)
# ax.plot(epsilons, acc_wo_dp, label="No Privacy")
# ax.plot(epsilons, acc_w_dp,  label="With Privacy")

# plt.title("IBM differential privacy")
# # plt.semilogx(epsilons, acc)
# plt.xlabel('epsilon')
# plt.ylabel('accuracy')
# ax.legend()

output.close()
plt.show()