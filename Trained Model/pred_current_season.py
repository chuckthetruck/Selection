# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:20:10 2020

@author: User
"""

import pandas as pd
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


conferences = ['aac','acc','america-east','atlantic-10','big-12','big-east',
               'big-sky','big-ten','big-west','colonial','cusa','horizon','maac','mac',
               'meac','northeast','pac-12','patriot','sec','southland','swac',
               'summit','sun-belt','wcc','wac']

standings = ['Liberty','Winthrop','Yale','Bradley','Utah State','Belmont','East Tennessee State']
for i in conferences:

    df = pd.read_html('https://www.sports-reference.com/cbb/conferences/'+ i +'/2020.html')[0]
    df.columns = df.columns.droplevel()
    if i == 'mac':
        df.columns = ['Rk','School','W','L','W-L%','W','L','All W-L%','Own','Opp','SRS','SOS','Notes']
        df.sort_values('W-L%',ascending=False,inplace=True)
        df.reset_index(drop=True,inplace=True)
    standings.append(df['School'].iloc[0])
    
x_hat = fullyearscraper(2020) 

x_hat['Conf Champ'] = x_hat['School'].isin(standings)


tf.reset_default_graph()
    
x_hat.drop(['Blank','Pace'], axis=1,inplace=True)

xtest = x_hat[['Win-Loss%','SOS','Points Against','ORtg','FTr','3PAr','TRB%','AST%','BLK%',
                  'eFG%','TOV%','FT/FGA','Conf Champ']].astype(float)   

xtest.fillna(0, inplace=True)

testdesc = xtest.describe().transpose()

xtest = (xtest-testdesc['mean'])/testdesc['std']    

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], 'C:/Users/User/Documents/basketball/InOut')
    graph = tf.get_default_graph()
    output1 = sess.run('predicted:0',feed_dict={'X:0':xtest})
    
x_hat['PredTourn'] = output1

x_hat.sort_values('PredTourn',ascending=False,inplace=True)

pred_in = x_hat.iloc[0:68]

pred_in.to_csv('C:/Users/User/Documents/basketball/pred_in.csv',index=False)