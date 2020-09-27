import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
#load csv file
data = pd.read_csv("GOOG.csv")
test = pd.read_csv("data_stocks.csv")
test = test.drop(['DATE'], 1)
data = data[['Close']]

print(data.count)
plt.figure(figsize=(16, 8))
plt.title('Stock Prediction')
plt.xlabel('Days', fontsize=18)
plt.ylabel('Close Price ($)', fontsize=18)
plt.plot(data)
#plt.show()

n = data.shape[0]
p = data.shape[1]

# Make data a np.array
data = data.values

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
print(data_train.shape)
#data_train = np.reshape(data_train, (1, np.product(data_train.shape)))
#data_train = data_train.flatten()
data_test = data[np.arange(test_start, test_end), :]
#data_test = np.reshape(data_test, (1, np.product(data_test.shape)))
#data_test = data_test.flatten()

# Scale data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

data_train = np.reshape(data_train, (1, np.product(data_train.shape)))
data_test = np.reshape(data_test, (1, np.product(data_test.shape)))
print(data_test.shape)
# data_train = data_train.flatten()
# data_test = data_test.flatten()

# Build X and y
# X_train = data_train[:, :data_train.size - 10]
# y_train = data_train[:, data_train.size - 10:]
# y_train = y_train.flatten()
# print(y_train.shape)
X_test = data_test[:, :data_test.size - 10]
y_test = data_test[:, data_test.size - 10:]
y_test = y_test.flatten()
# print(y_test.shape)

# Number of stocks in training data
n_stocks = 242
# print(X_train.shape[0])

# Neurons
n_neurons_1 = 512
n_neurons_2 = 256
n_neurons_3 = 128
n_neurons_4 = 64
n_neurons_5 = 32
n_target = 10

# Session
net = tf.compat.v1.InteractiveSession()

# Placeholder
X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, None])
Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.compat.v1.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.compat.v1.zeros_initializer()

# Hidden weights
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
W_hidden_5 = tf.Variable(weight_initializer([n_neurons_4, n_neurons_5]))
bias_hidden_5 = tf.Variable(bias_initializer([n_neurons_5]))

# Output weights
W_output = tf.Variable(weight_initializer([n_neurons_5, n_target]))
bias_output = tf.Variable(bias_initializer([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
hidden_5 = tf.nn.relu(tf.add(tf.matmul(hidden_4, W_hidden_5), bias_hidden_5))

# Output layer (transpose!)
output = tf.transpose(tf.add(tf.matmul(hidden_5, W_output), bias_output))

# Cost function
mse = tf.reduce_mean(tf.compat.v1.squared_difference(output, Y))

# Optimizer
opt = tf.compat.v1.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.compat.v1.global_variables_initializer())

# Setup plot
# plt.ion()
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# line1, = ax1.plot(y_test)
# line2, = ax1.plot(y_test * 0.5)
# plt.show()

# Fit neural net
batch_size = 252
mse_train = []
mse_test = []
pred = []

# Run
epochs = 1000
for e in range(epochs):
    start_batchx = 0
    end_batchx = start_batchx+242
    for i in range(data_train.shape[1]//batch_size):
        batch_x = data_train[:, start_batchx:end_batchx]
        batch_y = data_train[:, end_batchx:end_batchx+10]
        batch_y = batch_y.flatten()
        start_batchx = end_batchx+10
        end_batchx=start_batchx+242
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

print(X_test.shape)
# mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
pred.append(net.run(output, feed_dict={X: X_test}))


#print(mse_train)
print(mse_test)
print(pred)



