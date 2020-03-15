# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:22:27 2020

@author: User
"""

import pandas as pd
import tensorflow as tf
import numpy as np

pred_in = pd.read_csv('C:/Users/User/Documents/basketball/pred_in.csv')

xtest = pred_in[['Win-Loss%','SOS','Conf Champ']].astype(float)   
xtest.fillna(0, inplace=True)
testdesc = xtest.describe().transpose()
xtest = (xtest-testdesc['mean'])/testdesc['std']

with tf.Session() as sess:
    
    graph = tf.get_default_graph()
    saver = tf.train.import_meta_graph('C:/Users/User/Documents/basketball/Seeder/seeder.ckpt.meta')
    saver.restore(sess, 'C:/Users/User/Documents/basketball/Seeder/seeder.ckpt')
    
    graph = tf.get_default_graph()
    
    # print(sess.run(['h1:0','h2:0','out']))
    
    # testw = {
    # 'h1': graph.get_tensor_by_name('h1:0'),
    # 'h2': graph.get_tensor_by_name('h2:0'),
    # 'out': graph.get_tensor_by_name('out:0')
    # }
    #  = {
    # 'b1': graph.get_tensor_by_name('b1:0'),
    # 'b2': graph.get_tensor_by_name('b2:0'),
    # 'out_1': graph.get_tensor_by_name('out_1:0')
    # }
    
    output = sess.run('Y_hat:0',feed_dict={'X:0':xtest})

    
pred_in['Seed'] = output

pred_in['Pred_Seed'] = 0

pred_in.sort_values('Seed',inplace=True)

counter = 16
s = 0
for i in range(len(pred_in) -1,-1,-1):
    if counter == 16:
        if s < 6:
            pred_in['Pred_Seed'].iloc[i] = counter
            s+=1
        
        else:
            counter -=1
            pred_in['Pred_Seed'].iloc[i] = counter
            s = 1
    
    elif counter == 12 :
        if s < 6:
            pred_in['Pred_Seed'].iloc[i] = counter
            s+=1
        
        else:
            counter -=1
            pred_in['Pred_Seed'].iloc[i] = counter
            s = 1
        
        
    else:
        if s < 4:
            pred_in['Pred_Seed'].iloc[i] = counter
            s+=1
        
        else:
            counter -= 1
            pred_in['Pred_Seed'].iloc[i] = counter
            s = 1
            
check = pred_in[['School','Pred_Seed','Conf Champ']]