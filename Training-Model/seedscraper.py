# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 10:34:50 2019

@author: User
"""

import pandas as pd

def seedreader(year):
    check_dl_dict = {}
    
    for i in range(2000,year+1):
        
        check_dl_dict[i] = []
        
        if i < 2007 or i == 2017:
            test = pd.read_html('https://en.wikipedia.org/wiki/'+str(i)+'_NCAA_Division_I_Men%27s_Basketball_Tournament')
            
            for df in test:
                
                while (df.columns.nlevels > 1):
                    df.columns = df.columns.droplevel(0)
                
                if 'Seed' in df.columns and 'Flagship station' not in df.columns and df.shape[0] >= 16 and df.shape[0] <=19  :
                    check_dl_dict[i].append(df.astype(str))
                    
        else:
            test = pd.read_html('https://en.wikipedia.org/wiki/'+str(i)+'_NCAA_Division_I_Men%27s_Basketball_Tournament:_qualifying_teams')
            
            if i == 2019:
                test = pd.read_html('https://en.wikipedia.org/wiki/'+str(i)+'_NCAA_Division_I_Men%27s_Basketball_Tournament:_qualifying_teams',encoding = "ISO-8859-1")
            
            for df in test:
                
                while (df.columns.nlevels > 1):
                    df.columns = df.columns.droplevel(0)
                
                if 'Seed' in df.columns and df.shape[0] >= 16 and df.shape[0] <= 19:
                    check_dl_dict[i].append(df.astype(str))
                    
    for key in check_dl_dict.keys():
        for df in check_dl_dict[key]:
            df['Seed'] = df['Seed'] = df['Seed'].str.strip('No. ')
            df['Seed'] = df['Seed'] = df['Seed'].str.strip('#')
            df['Seed'] = df['Seed'] = df['Seed'].str.strip('*')
            df['Seed'] = df['Seed'] = df['Seed'].str.strip('B')
            
            df['Seed'] = df['Seed'].astype(int)
            
        check_dl_dict[key] = pd.concat(check_dl_dict[key])
    
    return check_dl_dict


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
    
    df = df[df['NCAA'] == True]
    
    train_dict.update({i:df})
    
    i+=1

i=2019
test_dict = {}
while i < 2020:
    df = fullyearscraper(i)
    
    df['Conf Champ'] = confchampscraper(i,df)
    
    df = df[df['NCAA'] == True]
    
    test_dict.update({i:df})
    
    i+=1
    

teamstoteams = pd.read_csv('C:/Users/User/Documents/basketball/teamstoteams.csv',encoding = "ISO-8859-1")

tournyteams = seedreader(2019)

for i in range(2000,2020):
    if i < 2019:
        tttmask = (teamstoteams['Input'].isin(tournyteams[i]['School']))
        temp = teamstoteams[tttmask].copy()
        temp.sort_values('Input',inplace=True)           
        temp.reset_index(drop=True,inplace=True)         
        
        train_dict[i].sort_values('School',inplace=True)
        train_dict[i].reset_index(drop=True,inplace=True)
        
        tournyteams[i].sort_values('School',inplace=True)
        tournyteams[i].reset_index(drop=True,inplace=True)
        
        tournyteams[i]['School'] = temp['Output']
        tournyteams[i].sort_values('School',inplace=True)
        tournyteams[i].reset_index(drop=True,inplace=True)
        
        train_dict[i]['Seed'] = tournyteams[i]['Seed']
                  
        
        
        
    
    else:   
        tttmask = (teamstoteams['Input'].isin(tournyteams[i]['School']))
        temp = teamstoteams[tttmask].copy()
        temp.sort_values('Input',inplace=True)           
        temp.reset_index(drop=True,inplace=True)
        
        test_dict[i].sort_values('School',inplace=True)
        test_dict[i].reset_index(drop=True,inplace=True)
        
        tournyteams[i].sort_values('School',inplace=True)
        tournyteams[i].reset_index(drop=True,inplace=True)
        
        tournyteams[i]['School'] = temp['Output']
        tournyteams[i].sort_values('School',inplace=True)
        tournyteams[i].reset_index(drop=True,inplace=True)
        
        test_dict[i]['Seed'] = tournyteams[i]['Seed']


x = pd.concat(train_dict).drop(['Blank','Pace'], axis=1)
x_hat = pd.concat(test_dict).drop(['Blank','Pace'], axis=1)

x.to_csv('C:/Users/User/Documents/basketball/trainseeds.csv')
x_hat.to_csv('C:/Users/User/Documents/basketball/testseeds.csv')




