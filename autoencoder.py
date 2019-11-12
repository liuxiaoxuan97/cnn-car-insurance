import pandas as pd
df=pd.read_csv('C:/Users/lenovo/Desktop/trip.csv')


import numpy as np
from scipy.stats import norm
from keras.layers import Input,Dense,Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics

import tensorflow as tf
import matplotlib.pyplot as plt
#导入数据
#from tensorflow.examples.tutorials.mnist import input_data


learning_rate = 0.01
training_epochs = 10
batch_size = 100
original_dim=320
display_step = 1
n_input = 320
X = tf.placeholder("float", [None, n_input])
n_hidden_1 = 7
n_hidden_2 = 2

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], )),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], )),
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], )),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], )),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    # 为了便于编码层的输出，编码层随后一层不使用激活函数
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    sess.run(tf.global_variables_initializer())
    total_batch = int(df.train.num_examples / batch_size)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = df.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

print(layer_2)
