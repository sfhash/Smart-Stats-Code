#importing libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import os
import glob

from sklearn.ensemble import RandomForestRegressor

data0=pd.read_csv('input2.csv')

#adding pressure column
def pressure_generator(data0):
    data0['Pr1']=[(1 if x>=0.1 else 10*x) if x<=0.5 else 2*(1-x) for x in data0['p1']]
    data0['Pr2']=[(1 if x>=0.1 else 10*x) if x<=0.5 else 2*(1-x) for x in data0['p2']]
    return data0


def function1_RR_BR(data1):
    n0,m0=data1.shape
    no_balls=[0]*n0  
    balls=[0]*n0
    total_balls=[0]*n0  #total balls including no_balls
    BR=[0]*n0  # balls remaining
    total_runs=[0]*n0
    run_rate=[0]*n0

    no_balls[0]= 1 if data1.iloc[0]['runs_noballs']>0 else 0
    balls[0]=1
    total_balls[0]=balls[0]+no_balls[0]
    BR[0]=119
    total_runs[0]=data1.iloc[0]['runs_batsman']
    run_rate[0]=total_runs[0]

    first_innings_balls=0

    for i in range(1,n0):
        if data1.iloc[i]['match_pkey']==data1.iloc[i-1]['match_pkey']:
            no_balls[i]=no_balls[i-1]+1 if data1.iloc[i-1]['runs_noballs']>0 else no_balls[i-1]
            total_runs[i]=total_runs[i-1]+data1.iloc[i]['runs_batsman']

            if data1.iloc[i]['innings']==2 and data1.iloc[i-1]['innings']==1:
                first_innings_balls=(6*(data1.iloc[i-1]['over_num']-1)+data1.iloc[i-1]['ball_num'])

        else:
            no_balls[i]= 1 if data1.iloc[i]['runs_noballs']>0 else 0
            total_runs[i]=data1.iloc[i]['runs_batsman']
            first_innings_balls=0
        balls[i]=(6*(data1.iloc[i]['over_num']-1)+data1.iloc[i]['ball_num'])+first_innings_balls
        BR[i]=120-(6*(data1.iloc[i]['over_num']-1)+data1.iloc[i]['ball_num'])
        total_balls[i]=balls[i]+no_balls[i]
        run_rate[i]=total_runs[i]/total_balls[i]


    total_runs=pd.DataFrame(total_runs)
    run_rate=pd.DataFrame(run_rate)
    balls=pd.DataFrame(balls)
    no_balls=pd.DataFrame(no_balls)
    total_balls=pd.DataFrame(total_balls)
    BR=pd.DataFrame(BR)

    data1.insert(data1.shape[1],"no_balls",no_balls)#number of no balls in the innings
    data1.insert(data1.shape[1],"BB",balls) #Bowls bowled without counting no balls
    data1.insert(data1.shape[1],"Total_BB",total_balls)# balls+no_balls
    data1.insert(data1.shape[1],"BR",BR)# balls remaining in the innings
    data1.insert(data1.shape[1],"Runs",total_runs)#Total runs scored by all batsman (Byes not included) in the match till now
    data1.insert(data1.shape[1],"RR",run_rate)# run rate till that instant in tha match

    return data1


def function2_Pressure(data2):
    n2,m2=data2.shape

    final_pressure1=[0]*n2
    final_pressure2=[0]*n2
    final_pressure1[n2-1]=data2.iloc[n2-1]['Pr1']
    final_pressure2[n2-1]=data2.iloc[n2-1]['Pr2']


    for j in range(n2-2,-1,-1):
        if data2.iloc[j]['match_pkey'] != data2.iloc[j+1]['match_pkey']:
            final_pressure1[j]=data2.iloc[j]['Pr1']
            final_pressure2[j]=data2.iloc[j]['Pr2']
        elif data2.iloc[j]['innings']==1 and data2.iloc[j+1]['innings']==2:
            final_pressure1[j]=data2.iloc[j]['Pr1']
            final_pressure2[j]=data2.iloc[j]['Pr2']
        else:
            final_pressure1[j]=max(final_pressure1[j+1],data2.iloc[j]['Pr1'])
            final_pressure2[j]=max(final_pressure2[j+1],data2.iloc[j]['Pr2'])

    final_pressure1=pd.DataFrame(final_pressure1)
    final_pressure2=pd.DataFrame(final_pressure2)
    data2.insert(data2.shape[1],'Pr_bat',final_pressure1)
    data2.insert(data2.shape[1],'Pr_ball',final_pressure2)
    return data2


def function3_making_balls_unique(data3):
    n3,m3=data3.shape

    BR_unique=[-1]*n3
    BB_unique=[-1]*n3

    for i in range(0,n3-1):
        if data3.iloc[i]['BR']!=data3.iloc[i+1]['BR']:
            BR_unique[i]=data3.iloc[i]['BR']
        if data3.iloc[i]['Total_BB']!=data3.iloc[i+1]['Total_BB']:
            BB_unique[i]=data3.iloc[i]['Total_BB']        
    BR_unique[n3-1]=data3.iloc[n3-1]['BR']
    BB_unique[n3-1]=data3.iloc[n3-1]['Total_BB']

    BR_unique=pd.DataFrame(BR_unique)
    BB_unique=pd.DataFrame(BB_unique)
    data3.insert(data3.shape[1],'BR_unique',BR_unique)
    data3.insert(data3.shape[1],'BB_unique',BB_unique)
    return data3


def data_rf_runs_per_ball_generator(data4):
    # dropping row with wides, after doing this data4 contains only unique values of BB_unique for a match_pkey
    data4=data4[data4['BB_unique']>=0]
    # reseting the index after dropping some rows
    data4.reset_index(inplace = True, drop = True)

    # generating the training data for our random forest model as mentioned in paper
    data_rf_runs_per_ball=[]
    for i in data4['BR'].unique():
            data_rf_runs_per_ball.append((i,data4[data4['BR']==i]['runs_batsman'].mean()))
    data_rf_runs_per_ball=pd.DataFrame(data_rf_runs_per_ball)
    data_rf_runs_per_ball.rename(columns = {0:'BR', 1:'runs_per_balls'}, inplace = True)
    data_rf_runs_per_ball.to_csv('data_rf_runs_per_ball.csv',index=False)
    return data_rf_runs_per_ball


# updating the data4 table with random forest prediction

def Add_random_forest_model(data4,data_rf_runs_per_ball):
    # dropping row with wides, after doing this data4 contains only unique values of BB_unique for a match_pkey
    data4=data4[data4['BB_unique']>=0]
    # reseting the index after dropping some rows
    data4.reset_index(inplace = True, drop = True)
    
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    X=data_rf_runs_per_ball[['BR']]
    y=data_rf_runs_per_ball[['runs_per_balls']]
    regressor.fit(X, y)
    Random_forest_output=pd.DataFrame(regressor.predict(data4[['BR']]))
    data4.insert(data4.shape[1],'RF_output',Random_forest_output)
    return data4


# calculating smart_runs_batsman and smart_runs_bowler and adding them to data4

def smart_runs(data4):
    data4['smart_runs_batsman']=data4['runs_batsman']+data4['Pr_bat']*(data4['runs_batsman']-data4['RR']*data4['RF_output']/1.30)
    data4['smart_runs_bowler']=data4['runs_batsman']+data4['Pr_ball']*(data4['runs_batsman']-data4['RR']*data4['RF_output']/1.30)
    return data4


##Smart Wickets starting from here
# first we calculate all the default value which will be useful if a batsman played less than 15 innings or 
# less than 200 balls
# for that we separate the data into top_order, middle_order and lower_order

def batsman_list_generator(data4):
    top_order=data4[data4['position']<=4]
    middle_order=data4[data4['position']<=7][data4['position']>=5]
    lower_order=data4[data4['position']>7]

    # Default Smart Average for top_order, middle_order and lower_order batsmen
    SA_T_default=top_order['smart_runs_batsman'].sum()/top_order['is_dismissal'].sum()
    SA_M_default=middle_order['smart_runs_batsman'].sum()/middle_order['is_dismissal'].sum()
    SA_L_default=lower_order['smart_runs_batsman'].sum()/lower_order['is_dismissal'].sum()

    # Default Dismissal Rate for top_order, middle_order and lower_order batsmen
    top_order1=[]
    for i in range(len(top_order)):
        top_order1.append((top_order.iloc[i]['match_pkey'],top_order.iloc[i]['batsman_pkey']))
    top_order1=list(set(top_order1))
    DI_T_default=top_order['is_dismissal'].sum()/len(top_order1)

    # Default Dismissal Rate for top_order, middle_order and lower_order batsmen
    middle_order1=[]
    for i in range(len(middle_order)):
        middle_order1.append((middle_order.iloc[i]['match_pkey'],middle_order.iloc[i]['batsman_pkey']))
    middle_order1=list(set(middle_order1))
    DI_M_default=middle_order['is_dismissal'].sum()/len(middle_order1)

    # Default Dismissal Rate for top_order, middle_order and lower_order batsmen
    lower_order1=[]
    for i in range(len(lower_order)):
        lower_order1.append((lower_order.iloc[i]['match_pkey'],lower_order.iloc[i]['batsman_pkey']))
    lower_order1=list(set(lower_order1))
    DI_L_default=lower_order['is_dismissal'].sum()/len(lower_order1)

    # Default Smart Strike Rate for top_order, middle_order and lower_order batsmen
    SSR_T_default=top_order['smart_runs_batsman'].sum()/len(top_order['is_dismissal'])
    SSR_M_default=middle_order['smart_runs_batsman'].sum()/len(middle_order['is_dismissal'])
    SSR_L_default=lower_order['smart_runs_batsman'].sum()/len(lower_order['is_dismissal'])
    
    
    
    # manipulating data4 to get runs, dismissals, balls and no_innings for each player
    # Calculating Smart Average (SA), Smart strike rate (SSR), R_0 and R for each batsmen
    # R_0=SSR+SA
    # Batsmen Rating  R=(R_0-R_0_min)/(R_0_max-R_0_min)  
    # adding it to the batsman_list1 dataframe
    batsman_list=[]
    for i in data4['batsman_pkey'].unique():
        batsman_list.append((i,data4[data4['batsman_pkey']==i]['smart_runs_batsman'].sum(),
                             data4[data4['batsman_pkey']==i]['runs_batsman'].sum(),
                  len(data4[data4['batsman_pkey']==i][data4['is_dismissal']!=0]),len(data4[data4['batsman_pkey']==i]),
                 len(data4[data4['batsman_pkey']==i]['match_pkey'].unique())))
    batsman_list=pd.DataFrame(batsman_list)
    batsman_list.rename(columns = {0:'batsman_pkey', 1:'smart_runs',2:'runs',3:'dismissals', 4: 'balls', 5: 'no_innings'}, inplace = True)
    #here matches != innings as a batsman may not bat in a match which will not count towards the no of innings
    batsman_list1=batsman_list.copy()

    SA=[]
    SSR=[]
    average_nominal=[]
    strike_rate_nominal=[]
    for i in batsman_list['batsman_pkey']:
        Smart_Runs=batsman_list[batsman_list['batsman_pkey']==i].iloc[0]['smart_runs']
        Runs=batsman_list[batsman_list['batsman_pkey']==i].iloc[0]['runs']
        No_Dismissals=batsman_list[batsman_list['batsman_pkey']==i].iloc[0]['dismissals']
        balls=batsman_list[batsman_list['batsman_pkey']==i].iloc[0]['balls']

        if batsman_list[batsman_list['batsman_pkey']==i].iloc[0]['no_innings']>=15:
            SA.append(Smart_Runs/No_Dismissals)
        else:
            NI_T=len(top_order[top_order['batsman_pkey']==i]['match_pkey'].unique())
            NI_M=len(middle_order[middle_order['batsman_pkey']==i]['match_pkey'].unique())
            NI_L=len(lower_order[lower_order['batsman_pkey']==i]['match_pkey'].unique())
            NI=NI_T+NI_M+NI_L
            SA_0=(NI_T*SA_T_default+NI_M*SA_M_default+NI_L*SA_L_default)/NI
            DI_0=(NI_T*DI_T_default+NI_M*DI_M_default+NI_L*DI_L_default)/NI
            SA.append((Smart_Runs+(15-NI)*DI_0*SA_0)/(No_Dismissals+(15-NI)*DI_0))


        if batsman_list[batsman_list['batsman_pkey']==i].iloc[0]['balls']>=200:
            SSR.append(100*Smart_Runs/balls)
        else:
            BF_T=len(top_order[top_order['batsman_pkey']==i]['match_pkey'].unique())
            BF_M=len(middle_order[middle_order['batsman_pkey']==i]['match_pkey'].unique())
            BF_L=len(lower_order[lower_order['batsman_pkey']==i]['match_pkey'].unique())
            BF=BF_T+BF_M+BF_L
            SSR_0=(BF_T*SSR_T_default+BF_M*SSR_M_default+BF_L*SSR_L_default)/BF
            DI_0=(NI_T*DI_T_default+NI_M*DI_M_default+NI_L*DI_L_default)/NI
            SSR.append(100*(Smart_Runs+(200-BF)*SSR_0)/200)
        
        average_nominal.append(Runs/No_Dismissals) 
        strike_rate_nominal.append(100*Runs/balls)
        


    SA=pd.DataFrame(SA)
    SSR=pd.DataFrame(SSR)
    average_nominal=pd.DataFrame(average_nominal)
    strike_rate_nominal=pd.DataFrame(strike_rate_nominal)

    batsman_list1.insert(batsman_list1.shape[1],'average_nominal',average_nominal)#Smart Average
    batsman_list1.insert(batsman_list1.shape[1],'strike_rate_nominal',strike_rate_nominal)#Smart Strike Rate    
    batsman_list1['R_0_nominal']=(batsman_list1['average_nominal']-batsman_list1['average_nominal'].min())/(batsman_list1['average_nominal'].max()-batsman_list1['average_nominal'].min())+(batsman_list1['strike_rate_nominal']-batsman_list1['strike_rate_nominal'].min())/(batsman_list1['strike_rate_nominal'].max()-batsman_list1['strike_rate_nominal'].min())
    batsman_list1['R_nominal']=(batsman_list1['R_0_nominal']-batsman_list1['R_0_nominal'].min())/(batsman_list1['R_0_nominal'].max()-batsman_list1['R_0_nominal'].min())
     
    batsman_list1.insert(batsman_list1.shape[1],'SA',SA)#Smart Average
    batsman_list1.insert(batsman_list1.shape[1],'SSR',SSR)#Smart Strike Rate
    batsman_list1['R_0']=(batsman_list1['SA']-batsman_list1['SA'].min())/(batsman_list1['SA'].max()-batsman_list1['SA'].min())+(batsman_list1['SSR']-batsman_list1['SSR'].min())/(batsman_list1['SSR'].max()-batsman_list1['SSR'].min())
    batsman_list1['R']=(batsman_list1['R_0']-batsman_list1['R_0'].min())/(batsman_list1['R_0'].max()-batsman_list1['R_0'].min())
    return batsman_list1


def combine_batsmanR_data(data5):
    temp2=[]
    # Adding R for each batsmen in data5
    for i in data5['batsman_pkey']:
        temp2.append(batsman_list1[batsman_list1['batsman_pkey']==i].iloc[0]['R'])
    temp2=pd.DataFrame(temp2)
    data5.insert(data5.shape[1],'R',temp2)

    # Finally calculating smart wickets based on Pr_ball and R value whenever there is dismissal
    def func(df):
        if df['is_dismissal']==0:
            return 0
        elif df['BR']>=18:
            return df['Pr_ball']+df['R']
        else:
            return df['Pr_ball']+df['R']*df['BR']/18

    data5['Smart_wicket']=data5.apply(func,axis=1)
    return data5


#bowler wise
def bowler_list_generator(data5):
    bowler_list=[]
    for i in data5['bowler_pkey'].unique():
        bowler_list.append((i,data5[data5['bowler_pkey']==i]['Smart_wicket'].sum(),
                            len(data5[data5['bowler_pkey']==i][data5['is_dismissal']!=0]),
                            len(data5[data5['bowler_pkey']==i]['match_pkey'].unique()),
                            data5[data5['bowler_pkey']==i]['smart_runs_bowler'].sum(),
                            data5[data5['bowler_pkey']==i]['runs_batsman'].sum(), len(data5[data5['bowler_pkey']==i]),
                len(data5[data5['bowler_pkey']==i][data5['runs_noballs']>0])))
    bowler_list=pd.DataFrame(bowler_list)
    bowler_list.rename(columns = {0:'bowler_pkey', 1:'Smart_wicket',2:'Actual_wicket',3: 'no_innings_bowled', 
                                  4:'smart_runs_conceded', 5:'Actual_runs_conceded',
                                   6: 'balls', 7:'no_balls'}, inplace = True)
    
    smart_bowl_average=[]
    economy_rate=[]
    bowl_average_nominal=[]
    economy_rate_nominal=[]
    for i in bowler_list['bowler_pkey']:
        Smart_wicket=bowler_list[bowler_list['bowler_pkey']==i].iloc[0]['Smart_wicket']
        Actual_wicket=bowler_list[bowler_list['bowler_pkey']==i].iloc[0]['Actual_wicket']
        smart_runs_conceded=bowler_list[bowler_list['bowler_pkey']==i].iloc[0]['smart_runs_conceded']
        Actual_runs_conceded=bowler_list[bowler_list['bowler_pkey']==i].iloc[0]['Actual_runs_conceded']
        balls=bowler_list[bowler_list['bowler_pkey']==i].iloc[0]['balls']
        no_balls=bowler_list[bowler_list['bowler_pkey']==i].iloc[0]['no_balls']
        
        if Actual_wicket==0:
            Actual_wicket=-1*Actual_runs_conceded
            Smart_wicket=-1*smart_runs_conceded
        
        smart_bowl_average.append(smart_runs_conceded/Smart_wicket)
        economy_rate.append(smart_runs_conceded/(balls-no_balls))
        bowl_average_nominal.append(Actual_runs_conceded/Actual_wicket)
        economy_rate_nominal.append(Actual_runs_conceded/(balls-no_balls))
    
    smart_bowl_average=pd.DataFrame(smart_bowl_average)
    economy_rate=pd.DataFrame(economy_rate)
    bowl_average_nominal=pd.DataFrame(bowl_average_nominal)
    economy_rate_nominal=pd.DataFrame(economy_rate_nominal)
    
    bowler_list.insert(bowler_list.shape[1],'smart_bowl_average',smart_bowl_average)
    bowler_list.insert(bowler_list.shape[1],'economy_rate',economy_rate)    
    bowler_list.insert(bowler_list.shape[1],'bowl_average_nominal',bowl_average_nominal)
    bowler_list.insert(bowler_list.shape[1],'economy_rate_nominal',economy_rate_nominal) 
    return bowler_list


def impact_score(data6):
    data6['r0_par_runs']=data6['RF_output']*data6['RR']/1.3
    IS_bat=[]
    IS_bowl=[]
    for i in data6['match_pkey'].unique():
        for j in data6[data6['match_pkey']==i]['batsman_pkey'].unique():
            IS_bat.append((i,j,data6[data6['match_pkey']==i][data6['batsman_pkey']==j]['smart_runs_batsman'].sum()))
        for j in data6[data6['match_pkey']==i]['bowler_pkey'].unique():
            IS_bowl.append((i,j,(data6[data6['match_pkey']==i][data6['bowler_pkey']==j]['r0_par_runs'].sum()-data6[data6['match_pkey']==i][data6['bowler_pkey']==j]['smart_runs_bowler'].sum())+25*data6[data6['match_pkey']==i][data6['bowler_pkey']==j]['Smart_wicket'].sum()))
    IS_bat=pd.DataFrame(IS_bat)
    IS_bowl=pd.DataFrame(IS_bowl)
    IS_bat.rename(columns = {0:'match_pkey', 1:'batsman_pkey',2:'IS_bat'}, inplace = True)
    IS_bowl.rename(columns = {0:'match_pkey', 1:'bowler_pkey',2:'IS_bowl'}, inplace = True)
    return IS_bat,IS_bowl


data=data0.copy()
data=pressure_generator(data)
data=function1_RR_BR(data)
data=function2_Pressure(data)
data=function3_making_balls_unique(data)
data_rf_runs_per_ball=data_rf_runs_per_ball_generator(data)
data=Add_random_forest_model(data,data_rf_runs_per_ball)
data=smart_runs(data)
batsman_list1=batsman_list_generator(data) 

data=combine_batsmanR_data(data)
bowler_list=bowler_list_generator(data)
IS_bat,IS_bowl=impact_score(data)

data.to_csv('results_smart_stats.csv', index=False)
batsman_list1.to_csv('smart_stats_by_batsman.csv',index=False)
bowler_list.to_csv('smart_stats_by_bowler.csv',index=False)
IS_bat.to_csv('impact_score_batsman.csv',index=False)
IS_bowl.to_csv('impact_score_bowler.csv',index=False)