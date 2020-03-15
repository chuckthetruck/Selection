# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:28:22 2019

@author: User
"""

import tensorflow as tf
import pandas as pd


def neural_net(x):
    #hidden layer 1
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)#activation
    #hideen layer 2
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_1 = tf.nn.relu(layer_2)#activation
    # output layer
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out_1']
    return (out_layer)

x = pd.read_csv('C:/Users/User/Documents/basketball/trainseeds.csv')
x_hat = pd.read_csv('C:/Users/User/Documents/basketball/testseeds.csv')


ytrain = x[['Seed']]
ytest = x_hat[['Seed']]

xtrain = x[['Win-Loss%','SOS','Conf Champ']].astype(float)   
xtrain.fillna(0, inplace=True)
traindesc = xtrain.describe().transpose()
xtrain = (xtrain-traindesc['mean'])/traindesc['std']


xtest = x_hat[['Win-Loss%','SOS','Conf Champ']].astype(float)   
xtest.fillna(0, inplace=True)
testdesc = xtest.describe().transpose()
xtest = (xtest-testdesc['mean'])/testdesc['std']

features_shape = xtrain.shape[1]
labels_shape = ytrain.shape[1]

X = tf.placeholder(tf.float32,[None,features_shape],name='X')
Y = tf.placeholder(tf.float32,[None,labels_shape],name='Y')

is_training=tf.Variable(True,dtype=tf.bool)

training_epochs = 10000
learning_rate = 0.001


weights = {
    'h1': tf.Variable(tf.random_normal([3, 10]),name='h1'),#4 inputs 10  nodes in h1 layer
    'h2': tf.Variable(tf.random_normal([10, 10]),name='h2'),# 10 nodes in h2 layer
    'out': tf.Variable(tf.random_normal([10, 1]),name='out')# 1 ouput label
}
biases = {
    'b1': tf.Variable(tf.random_normal([10]),name='b1'),
    'b2': tf.Variable(tf.random_normal([10]),name='b2'),
    'out_1': tf.Variable(tf.random_normal([1]),name='out_1')
}


Y_hat=neural_net(X)
Y_hat = tf.identity(Y_hat,name='Y_hat')
loss_op=tf.losses.mean_squared_error(Y,Y_hat)#loss function
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)#define optimizer # play around with learning rate
train_op = optimizer.minimize(loss_op)#minimize losss
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for i in range(0,training_epochs):
        sess.run(train_op,feed_dict={X:xtrain,Y:ytrain})
        loss=sess.run(loss_op,feed_dict={X:xtrain,Y:ytrain})
        if(i%100==0):
            print("epoch no"+str(i),(loss))
            
    saver.save(sess,"C:/Users/User/Documents/basketball/Seeder/seeder.ckpt") 
    
    print(sess.run(weights))
        
    out1 = sess.run(Y_hat,feed_dict={X:xtest})
    
x_hat['Seed'] = out1