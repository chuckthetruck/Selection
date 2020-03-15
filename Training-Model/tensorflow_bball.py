# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 13:35:19 2019

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def fullyearscraper(year):
    firstread = pd.read_html('https://www.sports-reference.com/cbb/seasons/'+ str(year) + '-advanced-school-stats.html')
   
    data = firstread[0]
    
    data.columns = ["Rank","School","Games","Wins", "Losses", "Win-Loss%", "SRS", "SOS", 
                    "Conf Wins", "Conf Losses", "Home Wins", "Home Losses", "Away Wins", "Away Losses",
                    "Points For", "Points Against","Blank", "Pace", "ORtg","FTr","3PAr","TS%","TRB%","AST%","STL%","BLK%",
                    "eFG%","TOV%","ORB%","FT/FGA"]
    data = data.dropna(subset=["Rank"])
    
    data = data[data["Rank"] != "Rk"]
    
    data["NCAA"] = data["School"].str.contains("NCAA")

    data["School"] = data.School.str.replace('NCAA' , '').str.strip()
    
    return data

def confchampscraper(year,data):
    firstread = pd.read_html('https://www.sports-reference.com/cbb/seasons/'+ str(year) +'.html')
   
    confchamp = firstread[0]
    
    confchamp = confchamp[['Tournament Champ']]
    
    s = data['School'].isin(confchamp['Tournament Champ'])

    return s

i=2000
train_dict = {}
while i < 2019:
    df = fullyearscraper(i)
    
    df['Conf Champ'] = confchampscraper(i,df)
    
    train_dict.update({str(i):df})
    
    i+=1

i=2019
test_dict = {}
while i < 2020:
    df = fullyearscraper(i)
    
    df['Conf Champ'] = confchampscraper(i,df)
    
    test_dict.update({str(i):df})
    
    i+=1
    

x = pd.concat(train_dict).drop(['Blank','Pace'], axis=1)
x_hat = pd.concat(test_dict).drop(['Blank','Pace'], axis=1)


xtrain = x[['Win-Loss%','SOS','Points Against','ORtg','FTr','3PAr','TRB%','AST%','BLK%',
            'eFG%','TOV%','FT/FGA','Conf Champ']].astype(float)   

xtrain.fillna(0, inplace=True)

traindesc = xtrain.describe().transpose()

xtrain = (xtrain-traindesc['mean'])/traindesc['std']

# xtrain['Win-Loss%'] = 10*xtrain['Win-Loss%']
# xtrain['eFG%'] = 10*xtrain['eFG%']
# xtrain[['FTr','3PAr','FT/FGA']] = 10*xtrain[['FTr','3PAr','FT/FGA']]
# xtrain[['ORtg','TRB%']] = xtrain[['ORtg','TRB%']]/10
# xtrain['Points Against'] = xtrain['Points Against']/100

ytrain = x[['NCAA']].astype(int)

ytest = x_hat[['NCAA']].astype(int)    

xtest = x_hat[['Win-Loss%','SOS','Points Against','ORtg','FTr','3PAr','TRB%','AST%','BLK%',
                  'eFG%','TOV%','FT/FGA','Conf Champ']].astype(float)   

xtest.fillna(0, inplace=True)

testdesc = xtest.describe().transpose()

xtest = (xtest-testdesc['mean'])/testdesc['std']

# xtest['Win-Loss%'] = 10*xtest['Win-Loss%']
# xtest['eFG%'] = 10*xtest['eFG%']
# xtest[['FTr','3PAr','FT/FGA']] = 10*xtest[['FTr','3PAr','FT/FGA']]
# xtest[['ORtg','TRB%']] = xtest[['ORtg','TRB%']]/10
# xtest['Points Against'] = xtest['Points Against']/100

features_shape = xtrain.shape[1]
labels_shape = ytrain.shape[1]

X = tf.placeholder(tf.float32,[None,features_shape],name='X')
Y = tf.placeholder(tf.float32,[None,labels_shape],name='Y')
is_training=tf.Variable(True,dtype=tf.bool)

hidden_layers = int(features_shape/2)
training_epochs = 10000
learning_rate = 0.001
cost_history = np.empty(shape=[1],dtype=float)

initializer = tf.contrib.layers.xavier_initializer()
h0 = tf.layers.dense(X, hidden_layers, activation=tf.nn.relu, kernel_initializer=initializer)
h0=tf.identity(h0, name='h0')
# h0 = tf.nn.dropout(h0, 0.95)
h1 = tf.layers.dense(h0, labels_shape, activation=None)
h1=tf.identity(h1, name='h1')


cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1)
cost = tf.reduce_mean(tf.square(cross_entropy))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# prediction = tf.argmax(h0, 1)
# correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predicted = tf.nn.sigmoid(h1,name='predicted')
# predicted=tf.identity(predicted, name='predicted')
correct_pred = tf.equal(tf.round(predicted), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# accuracy=tf.identity(accuracy, name='accuracy')

# session
Losses = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        sess.run(optimizer, feed_dict={X: xtrain, Y: ytrain})
        loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={
                                 X: xtrain, Y: ytrain})
        cost_history = np.append(cost_history, acc)
        if step % 500 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))
        Losses.append(loss)    
    # Test model and check accuracy
    # output =  sess.run(predicted, feed_dict={X: xtest})
        
    tensor_info_input = tf.saved_model.utils.build_tensor_info(X)
    tensor_info_output = tf.saved_model.utils.build_tensor_info(predicted)
    
    prediction_signature = (
    tf.saved_model.signature_def_utils.build_signature_def(inputs={'X':tensor_info_input},outputs={'predicted':tensor_info_output},
                                                        ))

    
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder('C:/Users/User/Documents/basketball/InOut2')
    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.SERVING],
      signature_def_map={'seed_prediction':prediction_signature})
    
    builder.save()
    
    # tf.saved_model.simple_save(sess,'C:/Users/User/Documents/basketball/InOut',inputs={'X':X},outputs={'predicted':predicted})
    
    # tf.compat.v1.train.Saver().save(sess=sess,save_path='C:/Users/User/Documents/basketball/InOut.ckpt')
    
# x_hat['Pred'] = output