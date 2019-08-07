'''
Rachel 07/2019
Use IBM privlib on Credit Fraud detection (balanced dataset)
- Compare between non-private and private model

'''
import diffprivlib.models as dp
from diffprivlib.models.logistic_regression import LogisticRegression
from sklearn import linear_model
import time
import utils
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix


start_time = time.perf_counter()

# read input
df = pd.read_csv("./input/creditcard.csv")

# drop feature that has same distribution between fraud and normal
# 31cols - 11 cols = 20 cols
df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)

# condition to separate fraud from normal
# 20cols + 17 cols = 37 cols
df['V1_'] = df.V1.map(lambda x: 1 if x < -3 else 0)
df['V2_'] = df.V2.map(lambda x: 1 if x > 2.5 else 0)
df['V3_'] = df.V3.map(lambda x: 1 if x < -4 else 0)
df['V4_'] = df.V4.map(lambda x: 1 if x > 2.5 else 0)
df['V5_'] = df.V5.map(lambda x: 1 if x < -4.5 else 0)
df['V6_'] = df.V6.map(lambda x: 1 if x < -2.5 else 0)
df['V7_'] = df.V7.map(lambda x: 1 if x < -3 else 0)
df['V9_'] = df.V9.map(lambda x: 1 if x < -2 else 0)
df['V10_'] = df.V10.map(lambda x: 1 if x < -2.5 else 0)
df['V11_'] = df.V11.map(lambda x: 1 if x > 2 else 0)
df['V12_'] = df.V12.map(lambda x: 1 if x < -2 else 0)
df['V14_'] = df.V14.map(lambda x: 1 if x < -2.5 else 0)
df['V16_'] = df.V16.map(lambda x: 1 if x < -2 else 0)
df['V17_'] = df.V17.map(lambda x: 1 if x < -2 else 0)
df['V18_'] = df.V18.map(lambda x: 1 if x < -2 else 0)
df['V19_'] = df.V19.map(lambda x: 1 if x > 1.5 else 0)
df['V21_'] = df.V21.map(lambda x: 1 if x > 0.6 else 0)

#Create a new feature for normal (non-fraudulent) transactions
# Add 1 more column to dataframe "normal"
# 37cols + 1col -> 38cols
df.loc[df.Class == 0, 'Normal'] = 1
df.loc[df.Class == 1, 'Normal'] = 0

#Rename 'Class' to 'Fraud'.
df = df.rename(columns={'Class': 'Fraud'})

#Create dataframes of only Fraud and Normal transactions
# Fraud: 492x38
Fraud = df[df.Fraud == 1]
# Normal: 284807x38
Normal = df[df.Normal == 1]

# Set X_train equal to 80% of the fraudulent transactions.
# X_train = 394x38
X_train = Fraud.sample(frac=0.8)
count_Frauds = len(X_train)

# Add 80% of the normal transactions to X_train.
# X_train = 227846x38
X_train = pd.concat([X_train, Normal.sample(frac = 0.8)], axis = 0)

# X_test contains all the transaction not in X_train.
# X_test = 56961x38: here Fraud + normal
# ===> Modify here: I want to apply Log Reg with only 1 col in prediction: only the Fraud
# X_train = X_train.drop(['Normal'], axis = 1) # <======

X_test = df.loc[~df.index.isin(X_train.index)]

#Shuffle the dataframes so that the training is done in a random order.
X_train = shuffle(X_train)
X_test = shuffle(X_test)

#Add our target features to y_train and y_test.
# Add [Fraud] + [Normal] = [Fraud, Normal]
# 227846x1 + 227486x1 = 227486x2
y_train = X_train.Fraud
# ===> Modify here: no need to add Normal to label
y_train = pd.concat([y_train, X_train.Normal], axis=1)

y_test = X_test.Fraud
# ===> Modify here: No need to add Normal to label
y_test = pd.concat([y_test, X_test.Normal], axis=1)

#Drop target features from X_train and X_test.
# ===> Modify here: Only drop Fraud
X_train = X_train.drop(['Fraud','Normal'], axis = 1)
X_test = X_test.drop(['Fraud','Normal'], axis = 1)
# X_train = X_train.drop(['Fraud'], axis = 1)
# X_test = X_test.drop(['Fraud'], axis = 1)

'''
Due to the imbalance in the data, ratio will act as an equal weighting system for our model. 
By dividing the number of transactions by those that are fraudulent, ratio will equal the value that when multiplied
by the number of fraudulent transactions will equal the number of normal transaction. 
Simply put: # of fraud * ratio = # of normal
'''
# ratio = len(X_train)/count_Frauds
# Make 1 -> 1*ratio
# y_train.Fraud *= ratio
# y_test.Fraud *= ratio


y_train = y_train.drop(['Normal'], axis=1)
y_test  = y_test.drop(['Normal'], axis=1)

#Names of all of the features in X_train.
features = X_train.columns.values

#Transform each feature in features so that it has a mean of 0 and standard deviation of 1;
#this helps with training the neural network.
for feature in features:
    mean, std = df[feature].mean(), df[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std

# Convert dataframe to numpy array
X_train = np.array(X_train, dtype=np.float32)
X_test  = np.array(X_test, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
y_test = np.array(y_test, dtype=np.int32)

# # define list of epsilon
epsilons = [1]# np.logspace(-2, 2, 50)

acc_w_dp = list()

# output = open("ibm_mnist_dp_data_norm_18_70k_images.txt", "w+")
for epsilon in epsilons:
	logreg_w_dp  = LogisticRegression(epsilon=epsilon, data_norm=500, max_iter=1000)
# 	# l2 norm = sqrt(all_columns), there are 64 columns -> data_norm = 8
	logreg_w_dp.fit(X_train, y_train.ravel())
	# score_dp = logreg_w_dp.score(X_test, y_test)
	accuracy, recall, precision, auc = utils.predict_score(logreg_w_dp, X_test, y_test)
	# output.write("%.3f \t %.3f \t %.3f \n" % (epsilon, recall, precision))
	print(accuracy, recall, precision, auc)
	# acc_w_dp.append(score_dp)
	# output.write("%.3f \t %.3f\n" % (epsilon, score_dp))

print('Total time = ', time.perf_counter() - start_time, ' seconds')
# plt.plot(epsilons, acc_w_dp)

# output.close()
# plt.show()