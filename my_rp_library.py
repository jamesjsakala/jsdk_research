#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 13:51:32 2021

@author: james
"""
###Libraries###
#numpy
import numpy as np
#pandas
import pandas as pd
#matplotlib
import matplotlib.pyplot as plt
#matplotlib
import matplotlib
#groupby from itertools
import more_itertools as groupby
#datetime
import datetime
#odeint
from scipy.integrate import odeint
#optimize
from scipy import optimize
#time and date functions
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
#itertools
import itertools
#stats
from scipy import stats
#from scipy.integrate import solve_ivp
#########Data while testing#######
#%%
####################################################
#           ELT GLOBAL DATASET FUNCTIONS START
#
####################################################
def my_covid_data_set(array_index):
    #my_covid_data_set('infected')
    #my_covid_data_set('recorvered')
    #my_covid_data_set('deaths')
    #my_covid_data_set('global')    
    ##"Segmented COVID-19 Global Dataset" Files##
    infected_csv_file = "/home/james/Desktop/Skulu/Semister III/Dissertation/Test1/FINAL/CODE/DATA/ELT_DATA/global1/time_series_covid19_confirmed_global_narrow.csv"
    recorvered_csv_file = "/home/james/Desktop/Skulu/Semister III/Dissertation/Test1/FINAL/CODE/DATA/ELT_DATA/global1/time_series_covid19_recovered_global_narrow.csv"
    deaths_csv_file = "/home/james/Desktop/Skulu/Semister III/Dissertation/Test1/FINAL/CODE/DATA/ELT_DATA/global1/time_series_covid19_deaths_global_narrow.csv"
    ## COVID-19 Global Dataset##
    global_covid_stats_xslx_file = "/home/james/Desktop/Skulu/Semister III/Dissertation/Test1/FINAL/CODE/DATA/ELT_DATA/global1/COVID-19-geographic-disbtribution-worldwide.xlsx"
    ##Dictionary the paths for easy access#
    path_arrays = {'infected':infected_csv_file,'recorvered':recorvered_csv_file,'deaths':deaths_csv_file,'global':global_covid_stats_xslx_file}
    if array_index in path_arrays.keys():
        return path_arrays[array_index]
    else:
        return {}
 
##Extract Population Data###
def elt_population_from_xslx_file(file_name):
    data = pd.read_excel(file_name)
    return list(data.loc[data['countriesAndTerritories'] == 'Zambia']['popData2019'])[0]

###Function To Extract Target Rows and Columns From CSV##
def do_elt_on_csv(csv_file_path,dict_target_rowname_value,lst_fields_to_get):
    return_list = []
    df = pd.read_csv(csv_file_path)
    for key, val in dict_target_rowname_value.items():
        for rname in lst_fields_to_get:
            return_list.append(list(df.loc[df[key] == val][rname]))
    return return_list

##Function to Transform Communulative List To Daily Wise##
def cummum_to_day_wise(lst_list_of_commulative):
    lst_list_of_commulative.append(0)
    return_list = []
    i = 0
    while i < (len(lst_list_of_commulative) -1 ):
        return_list.append(lst_list_of_commulative[i] - lst_list_of_commulative[i + 1])
        i += 1
    return return_list

##Tested and OK####
def breakup_dates_to_months_array_index(given_list,infected,deaths,recorvered,susceptible):
    monthDict={'01':'January', '02':'February', '03':'March', '04':'April', '05':'May', '06':'June', '07':'July', '08':'August', '09':'September', '10':'October', '11':'November', '12':'December'}
    s = {}
    for i in given_list:
        g = i[:7]
        h = monthDict[g[5:]]+' '+str(g[:4])
        if g not in s:
            s[g] = {'Label':h,'infected':[[],[]],'deaths':[[],[]],'recorvered':[[],[]],'susceptible':[[],[]]}
        j = given_list.index(i)
        s[g]['infected'][0].append(i)
        s[g]['infected'][1].append(infected[j])
        s[g]['deaths'][0].append(i)
        s[g]['deaths'][1].append(deaths[j])
        s[g]['recorvered'][0].append(i)
        s[g]['recorvered'][1].append(recorvered[j])
        s[g]['susceptible'][0].append(i)
        s[g]['susceptible'][1].append(susceptible[j])        
    return s

####Tested and OK. Depends on all function above it###########
def go_get_me_my_eltd_dataset():
    xdate = do_elt_on_csv(my_covid_data_set('infected'),{'Country/Region':'Zambia'},['Date'])[0]
    xdate.reverse()
    infected_cummulative = do_elt_on_csv(my_covid_data_set('infected'),{'Country/Region':'Zambia'},['Value'])[0]
    infected = cummum_to_day_wise(infected_cummulative)
    infected.reverse()
    infected_cummulative.reverse()
    infected_cummulative.remove(0)
    deaths_cummulative = do_elt_on_csv(my_covid_data_set('deaths'),{'Country/Region':'Zambia'},['Value'])[0]
    deaths = cummum_to_day_wise(deaths_cummulative)
    deaths.reverse()
    deaths_cummulative.reverse()
    deaths_cummulative.remove(0)
    recorvered_cummulative = do_elt_on_csv(my_covid_data_set('recorvered'),{'Country/Region':'Zambia'},['Value'])[0]
    recorvered = cummum_to_day_wise(recorvered_cummulative)
    recorvered.reverse()
    recorvered_cummulative.reverse()
    recorvered_cummulative.remove(0)
    #zambia_population = int(round(elt_population_from_xslx_file(my_covid_data_set('global'))))
    susceptible = []
    zambia_population = 17861034
    for i in range(0,len(deaths_cummulative)-1):
        if(len(susceptible) == 0):
            susceptible.append(zambia_population - (deaths_cummulative[i] + infected_cummulative[i] + recorvered_cummulative[i]))
        else:
            susceptible.append(susceptible[i-1] - (deaths_cummulative[i] + infected_cummulative[i] + recorvered_cummulative[i]))
            #susceptible.append(zambia_population - (deaths[i+1] + infected[i+1] + recorvered[i+1]))
    susceptible.append(susceptible[-1])
    
    y = breakup_dates_to_months_array_index(xdate,infected,deaths,recorvered,susceptible)
    z = breakup_dates_to_months_array_index(xdate,infected_cummulative,deaths_cummulative,recorvered_cummulative,susceptible)
    
    return {'Contenious': {'date':xdate,'infected_cummulative':infected_cummulative,'infected':infected,'recorvered_cummulative':recorvered_cummulative,'recorvered':recorvered,'deaths_cummulative':deaths_cummulative,'deaths':deaths,'recorvered':recorvered,'susceptible':susceptible} , 'Monthly':{'Daywise':y,'Commulative':z}}

#%%
####################################################
#           ELT GLOBAL DATASET FUNCTIONS END
#
####################################################
#--------------------------------------------------------------------
####################################################
#           ELT ZNPHI DATASET FUNCTIONS START
#                   START
####################################################
def my_znphi_data_set():
    input_xslx_file_path ="/home/james/Desktop/Skulu/Semister III/Dissertation/Test1/FINAL/CODE/DATA/ELT_DATA/znphi/Combined_Harvested_Dataset.xlsx"
    interesting_fields = {'New_Cases':'New_Cases','Recovered Cases':'Recovered Cases','Fatality Cases':'Fatality Cases'}
    #my_ripped_data = {}
    dfj = pd.read_excel(input_xslx_file_path, sheet_name = list(interesting_fields.keys()))
    return dfj

def make_my_provintial_totals(znphi_data_frame):
    ytowns = {}
    filter_list = ['TOTAL','Date']
    xcases_names = list(znphi_data_frame.keys())
    for x in list(xcases_names):
        xtowns = list(znphi_data_frame[x].keys())
    for x in xcases_names:
        for y in xtowns:
            if y not in filter_list and y not in ytowns: 
               ytowns[y] = {}
               ytowns[y]['Label'] = y
               ytowns[y]['deaths'] = []
               ytowns[y]['infected'] = []
               ytowns[y]['recorvered'] = []
            if x == 'New_Cases' and y not in filter_list:    
                ytowns[y]['infected'].append(list(znphi_data_frame[x]['Date']))
                ytowns[y]['infected'].append(list(znphi_data_frame[x][y])) 
            if x == 'Fatality Cases' and y not in filter_list:   
                ytowns[y]['deaths'].append(list(znphi_data_frame[x]['Date']))
                ytowns[y]['deaths'].append(list(znphi_data_frame[x][y]))
            if x == 'Recovered Cases' and y not in filter_list:    
                ytowns[y]['recorvered'].append(list(znphi_data_frame[x]['Date']))
                ytowns[y]['recorvered'].append(list(znphi_data_frame[x][y]))
    return ytowns

def make_my_cases_totals_for_znphi_dataset(znphi_data_frame):
    infected = 0
    recorvered = 0
    deaths = 0
    filter_list = ['TOTAL','Date']
    xcases_names = list(znphi_data_frame.keys())
    for x in list(xcases_names):
        xtowns = list(znphi_data_frame[x].keys())
    for x in xcases_names:
        for y in xtowns:
            if x == 'New_Cases' and y not in filter_list: 
                infected = infected + sum(list(znphi_data_frame[x][y]))
            if x == 'Fatality Cases' and y not in filter_list:   
                deaths = deaths + sum(list(znphi_data_frame[x][y]))
            if x == 'Recovered Cases' and y not in filter_list:    
                recorvered = recorvered + sum(list(znphi_data_frame[x][y]))
    return {'infected':infected,'recorvered':recorvered,'deaths':deaths}                  

def summarize_znphi_data(my_znphi_data): 
    summed_days = {'date':[]}
    #my_znphi_data = my_znphi_data_set()
    province_names = list(my_znphi_data['New_Cases'].keys())[1:12]
    xkeys = list(my_znphi_data.keys())
    list(my_znphi_data[xkeys[0]]['Date'])
    for i in xkeys:
        if i not in summed_days:
            summed_days[i] = []
        for j in range(0,len(my_znphi_data[i])):
            summed_days[i].append(sum(list(my_znphi_data[i].iloc[j])[1:12]))
            if str(list(my_znphi_data[i].iloc[j])[0]) not in summed_days['date']:
                summed_days['date'].append(str(list(my_znphi_data[i].iloc[j])[0]))
      
    month_sums = {}
    week_sums = {}
    for i in range(0,len(summed_days['date'])):    
        xd = datetime.strptime(str(list(summed_days['date'])[i]), '%d/%m/%y')
        xk = str(str(xd.year) + '_' + str(xd.strftime("%B")))
        xw = str(xd.year) + '_Week_' + str(xd.isocalendar()[1])  + '_(' + xd.strftime("%B") + ')'
        if xw not in week_sums:
            week_sums[xw] = {}
            for l in xkeys:
                week_sums[xw][l] = []
        for m in xkeys:
            week_sums[xw][m].append(summed_days[m][i])
        
        if xk not in month_sums:
            month_sums[xk] = {}
            for h in xkeys:
                month_sums[xk][h] = []
        for j in xkeys:
            month_sums[xk][j].append(summed_days[j][i])
    return {'DaySums':summed_days,'Monthly_Segments':month_sums,'Weekly_Segments':week_sums}


def make_line_graph_totals(X):
    line_graph_data = {'infected':[],'deaths':[],'recorvered':[],'susceptible':[],'labels':[],'dates':[]}
    for i in X['Monthly']['Daywise'].keys():
        line_graph_data['infected'].append(sum(X['Monthly']['Daywise'][i]['infected'][1]))
        line_graph_data['deaths'].append(sum(X['Monthly']['Daywise'][i]['deaths'][1]))
        line_graph_data['recorvered'].append(sum(X['Monthly']['Daywise'][i]['recorvered'][1]))
        line_graph_data['susceptible'].append(sum(X['Monthly']['Daywise'][i]['susceptible'][1]))
        line_graph_data['dates'].append(i)
        line_graph_data['labels'].append(X['Monthly']['Daywise'][i]['Label'])
    return line_graph_data
####################################################
#           ELT ZNPHI DATASET FUNCTIONS END
#                   END
####################################################

#%%
####################################################
#       EXPONENTIAL FIT METHOD FUNCTIONS TEST START
#               EXPERIMENT CODE
####################################################
def exponential_fit(cases,start,length,dd):
    def resid(beta):
        prediction = cases[start]*np.exp(beta*(dd-start))
        return prediction[start:start+length]-cases[start:start+length]

    soln = optimize.least_squares(resid,0.2)
    beta = soln.x[0]
    #print('Estimated value of beta: {:.3f}'.format(beta))
    return beta
def plot_zambia_case_march_18_t0_dec_31(infected_cummulativex):    
    fig, ax= plt.subplots(figsize= (12,8))
    ax.plot(infected_cummulativex, label= 'Infected',marker= 'o', markersize= 4, alpha=0.7, color= 'orange')
    plt.xticks(rotation= 90)
    plt.legend(loc= 0)
    plt.grid(1)
    plt.show()

def plot_fit(cases,start,end,dd):
    length=end-start
    plt.plot(cases)
    beta = exponential_fit(cases,start,length,dd)
    prediction = cases[start]*np.exp(beta*(dd-start))
    plt.plot(dd[start:start+length],prediction[start:start+length],'--k');
    plt.legend(['Cases','Best Fit']);
    plt.xlabel('Days'); plt.ylabel('Total cases');
    return beta

####################################################
#          EXPONENTIAL FIT METHOD FUNCTIONS ENDS
#
####################################################

#%%
####################################################
#           LEAST SQUARE METHOD FUNCTIONS START
#
####################################################

#############New 2021 April#########    
def do_calculate_beta_gamma_r0_m_with_least_square_method(total,recovered,death,st,en):
    ###################################
    #total = cummulative_infected
    #death = day_wise_deaths
    #recovered = day_wise_recovered
    ###################################
    R=[x + y for x,y in zip(death, recovered)]
    I=[x + y for x,y in zip(total, R)]
   
    y=np.log(I[st:en])
    t=np.array(range(st,en))
    
    m , b = np.polyfit(t,y,1)
    plt.plot(t,(y),'red')
    plt.plot(t, (m*t + b),color='black',ls=('dashed'))
    #plt.title('m estimate: %s' %m)
    plt.title('Testing  :: Point/Day A = : ' + str(st) + ' , Point/Day B : ' + str(en))
    plt.grid()
    plt.figure()

    g=[]
    for i in range(st,en-1):
        oo=(R[i+1]-R[i])/I[i]
        g.append(oo)
    
    #plt.plot(g)
    #plt.grid(I)
    gamma = np.mean(g)
    #plt.title('gamma estimate: %s' %gamma)
    beta = m + gamma
    print('gamma=%g beta=%g R0=%g' %(gamma , beta , beta/gamma))
    
    return {'beta':beta,'gamma':gamma,'R0':beta/gamma,'m':m}


def index_to_array(xarray,xindex):
    oo = []
    for i in xindex:
        oox = []
        for j in i:
            oox.append(xarray[j])
        oo.append(oox)
    return oo

######Remove Negative Values######
def kill_negatives_in_list(given_list):
    ret_val = []
    for i in given_list:
        if 0 > int(i):
            ret_val.append(0)
        else:
            ret_val.append(i)
    return ret_val

###########New Segmenter#################
def get_consecutive_numbers_in_list_x(given_list):
    X = list(np.nonzero(given_list)[0])
    K = [list(group) for group in groupby.consecutive_groups(X)]
    ox = []   
    for y in K:
        ox.append(len(y))
    sd = []
    td = index_to_array(given_list,K)
    for i in td:
        sd.append(int(round(np.std(i))))
    return [K,K[ox.index((max(ox)))]]

def calculate_my_gamma_beta(K,I,R):
    en = K[-1]
    st= K[0]
    y=np.log(I[st:en])
    t=np.array(range(st,en))
    m, b = np.polyfit(t, y, 1)
    g = []
    for i in range(st, en):
        oo = (R[i+1] - R[i])/I[i]
        g.append(oo)
    gamma = np.mean(g)
    beta = m + gamma
    return {'gamma':gamma,'beta':beta,'R0':beta/gamma,'m':m}

##Do Final Calculations#######
def calculate_gamma_beta_nppoly_method(total,recovered,death): 
    ####Grab Data From DataFrame, Testing######
    #total = list(my_data['Total'])
    #death = list(my_data['Deaths'])
    #recovered = list(my_data['Recorvered'])
    ######Calculate R and I ########
    R = [x + y for x, y in zip(death, recovered)]
    #I = [x - y for x, y in zip(total, R)]
    I =  kill_negatives_in_list([x - y for x, y in zip(total, R)])
    ###Get Consecutive Non Zero Cases split up into Arrays#######
    J = get_consecutive_numbers_in_list_x(I)
    #####Caalculate_beta,gamma and m#########
    X = []
    for i in J[0]:
        X.append(calculate_my_gamma_beta(i,I,R))
    return X

####################################################
#          LEAST SQUARE METHOD FUNCTIONS START END
#
####################################################

#%%
####################################################
#           REVERSE FUNCTIONS START
#               EXPERIMENT
####################################################

def calculate_gamma_beta_equation_reverse_method(infected,recorvered,deaths,susceptible,dt):
    beta = [0]
    gamma = [0]
    for i in range(1,(len(infected) - 1)):
        try:
            x = (susceptible[i] - susceptible[i+1]) / (susceptible[i] * infected[i] * dt)
        except ZeroDivisionError as e:
            x = 0
        beta.append(x)
        #beta.append((susceptible[i] - susceptible[i+1]) / (susceptible[i] * infected[i] * dt))
        #gamma.append((recorvered[i+1] - recorvered[i]) / (infected[i] * dt))
        try:
            y = (recorvered[i+1] - recorvered[i]) / (infected[i] * dt)
        except ZeroDivisionError as e:
            y = 0
        gamma.append(y)
    beta.insert(0,0)
    gamma.insert(0,0)
    dr = np.sum(deaths)/np.sum(infected)
    rr = np.sum(recorvered)/np.sum(infected)
    try:
        z = (beta[(len(beta)-1)])/(gamma[(len(gamma)-1)])
    except ZeroDivisionError as e:
        z = 0
    return {'gamma':gamma[(len(gamma)-1)],'beta':beta[(len(beta)-1)],'rr':rr,'dr':dr,'R0':z}
    #return {'gamma':gamma,'beta':beta,'R0':(np.mean(beta))/(np.mean(gamma))}

####################################################
#           REVERSE FUNCTIONS END
#
#################################################### 
#%%
####################################################
#           SIR FUNCTIONS START
#
####################################################
    
def do_sir_model(I0,R0,beta,gamma,N,x,y):
    #Parameters required 
    #do_sir_model(InitiallyInfected,InitiallyRecovered,beta,gamma,Population,StartAt,DayPeriod)
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    
    # A grid of time points (in days)
    t = np.linspace(x, y, y)
    
    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt
    
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    #S, I, R = ret.T
    return [ret,t]

def plot_sir_model(S,I,R,t):
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/1000, 'o', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()

####################################################
#           SIR FUNCTIONS END
#
####################################################
#%%
####################################################
#           SEIRD FUNCTIONS START
#
####################################################    
def do_calculate_seird(N,D,IP,R_0,DR,IUD,DaysForLine):
    # N => Total population (Raw/Unfactored)
    # D => infection->recovered lasts six days
    # IP => Incubation Period In days
    # R_0 => Value of R/R0
    # DaysForLine => Days for line eg 365
    # DR => Death-Rate ( Fraction)
    def deriv(y, t, N, beta, gamma, delta, alpha, rho):
        #############################
        #alpha => Death Rate as a fraction
        #rho => 1/X , X = [days it takes from infection until death]
        #N => Total population (Raw/Unfactored)
        #gamma => 1/X , X = [Number of Infections in the last Y days]
        #delta => 1/X , X = [Incubation period in days]
        #beta => Transmission Rate [ Reproduction Number * gamma ]
        #t = Grid of time points (in days)
        #y = S,E,I,R,D
        ##############################
        S, E, I, R, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - delta * E
        dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
        dRdt = (1 - alpha) * gamma * I
        dDdt = alpha * rho * I
        return dSdt, dEdt, dIdt, dRdt, dDdt

    ##N = 17_915_567
    ##D = 6.0 # infection-recovered lasts six days
    gamma = 1.0 / D
    ##delta = 1.0 / 6.0  # incubation period of six days
    delta = 1.0 / IP
    ##R_0 = 2.65
    beta = R_0 * gamma  # R_0 = beta / gamma, so beta = R_0 * gamma
    ##alpha = 0.01119  # 1% death rate
    alpha = DR
    ##rho = 1/7.7  # 7.7 days from infection until death
    rho = 1/IUD
    S0, E0, I0, R0, D0 = N-1, 1, 0, 0, 0  # initial conditions: one exposed
    
    #t = np.linspace(0, 365, 365) # Grid of time points (in days)
    t = np.linspace(0, DaysForLine, DaysForLine) # Grid of time points (in days)

    y0 = S0, E0, I0, R0, D0 # Initial conditions vector
    
    #  Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta, alpha, rho))
    #S, E, I, R, D = ret.T
    return [ret,t]

def plotseird(t, S, E, I, R, D, title, end):
    #title = > tititle of graph
    #end => X , X = end day count to end from start. egg 407 
    f, ax = plt.subplots(1,1,figsize=(12,8))
    plt.title(title, size= 14)
   
    ax.plot(t, S, 'b', alpha=0.7, linewidth=1.5, label='Susceptible')
    ax.plot(t, E, 'gold', alpha=0.7, linewidth=1.5, label='Exposed')
    ax.plot(t, I, 'r', alpha=0.7, linewidth=1.5, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=1.5, label='Recovered')
    ax.plot(t, D, 'k', alpha=0.7, linewidth=1.5, label='Fatalities')
    #ax.plot(t, S+E+I+R+D, 'c--', alpha=0.5, linewidth=2, label='Total')
      
  
    ax.set_ylabel('Cases')
    ax.set_xlabel('Days')
    
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
      
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
      
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
        plt.grid(color= 'lightgrey', )
      
    #plt.savefig(path_output+'SEIRD_completo.jpg', dpi= 500, bbox_inches='tight')
    plt.show()
    
def plotseird_t(t, E, I, R, D, title, start, end, advance):
    # start => Days from start. Eg = 50 
    # end   => End day count from start . EG = 407
    # advance  => Jump by count. EG = 5
    f, ax = plt.subplots(1,1,figsize=(12,8))
      
    plt.title(title, size= 14)
      
    ax.plot(t, E, 'gold', alpha=0.5, linewidth=1.5, label='Exposed')
    ax.plot(t, I, 'r', alpha=0.7, linewidth=1.5, label='Infected')
    ax.plot(t, D, 'k', alpha=0.7, linewidth=1.5, label='Fatalities')
      
    plt.axvline(x= end, label='Pandemic day', color= 'deeppink', linestyle= '--')
    plt.arrow(end, 23000, 2, 0, head_width = 5000, head_length= 2, ec= 'deeppink')
    plt.annotate('Today ', (end+2, 17500), size= 10, color= 'deeppink', weight='bold')
      
    ax.set_ylabel('Variables (Time)')
    ax.set_xlabel('Time (pandemic days)')
    
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=1, ls='-')
      
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
      
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.grid(color= 'lightgrey', )
    plt.yticks(np.arange(start, max(E), 10000))
    plt.xticks(np.arange(start, end+advance, 5))
      
    plt.show()

def plot_predicted_cases(given_df):
    fig, ax= plt.subplots(figsize= (12,8))

    ax.plot(given_df['infectados_activos'], label= 'Global Dataset Reported - Currently Infected',marker= 'x', markersize= 4, alpha=0.7, color= 'r')
    ax.plot(given_df['new_cases_SEIRD'], label= 'SEIRD Model Prediction - Currently Infected', alpha=0.7, color= 'r')

    # Set title and labels for axes
    ax.set(xlabel='Time (Date)',ylabel='Currently Infected',title='New Cases Reported Global Dataset and SEIRD model result')

    # Define the date format
    date_form = DateFormatter("%d-%m-%Y")
    ax.xaxis.set_major_formatter(date_form)

    # Ensure a major tick for each week using (interval=1) 
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    plt.xticks(rotation= 45)
    plt.legend(loc= 0)
    plt.grid()
    #plt.savefig(path_output+'infected_SEIRD.jpg', dpi= 500, bbox_inches='tight')
    plt.show()

    #Plotting new deaths

    fig, ax= plt.subplots(figsize= (12,8))
    ax.plot(given_df['new_deaths'].cumsum(), label= 'Global Dataset Reported Data, Sum: '+str(given_df['new_deaths'].sum()), marker= 'x', markersize= 4, alpha=0.7, color= 'k')
    ax.plot(given_df['new_deaths_SEIRD'], label= 'SEIRD Model Prediction, Sum: '+str(round(given_df['new_deaths_SEIRD'].diff().sum(),0)), alpha=0.7, color= 'k')

    # Set title and labels for axes
    ax.set(xlabel='Time (Date)',
        ylabel='Accumulated deaths',
        title='Deaths / Day Reported Global Dataset and SEIRD model Result')

    # Define the date format
    date_form = DateFormatter("%d-%m-%Y")
    ax.xaxis.set_major_formatter(date_form)

    # Ensure a major tick for each week using (interval=1) 
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    plt.xticks(rotation= 45)
    plt.legend(loc= 0)
    plt.grid()
    #plt.savefig(path_output+'new_death_reported_SEIRD.jpg', dpi= 500, bbox_inches='tight')
    plt.show()
    
def append_column_to_df(original_df,list_to_add):
    xheader = list_to_add[0]
    xlist = list_to_add[1]
    
    df_size = len(original_df)
    lst_size = len(list_to_add[1])
    if df_size > lst_size:
        x = df_size - lst_size
        z = guatemala.drop(guatemala.tail(5).index)
        z[xheader] = xlist
    if lst_size > df_size:
        z = original_df
        x = lst_size - df_size - 1
        z[xheader] = xlist[0:x]
    else:
        z = original_df
        z[xheader] = xlist 
    return z

def segment_seird(given_list,ret):
    S, E, I, R, D = ret.T
    monthDict={'01':'January', '02':'February', '03':'March', '04':'April', '05':'May', '06':'June', '07':'July', '08':'August', '09':'September', '10':'October', '11':'November', '12':'December'}
    s = {}
    w = {}
    for i in list(range(0,len(given_list) - 1)):
        g = given_list[i][:7]
        h = monthDict[str(g[:7])[5:7]] + ' ' + str(g[:7])[:4]
        wn = str(datetime.strptime(str(given_list[i]), '%Y-%m-%d').isocalendar()[1])
        wns = str('Week '+ wn + ' (' + h + ')')
        wymn = str(str(g[:7])[:4] + '_' + monthDict[str(g[:7])[5:7]] + '_Week_' + str(wn))
        if g not in s:
            s[g] = {'Label':h,'S':[[],[]],'E':[[],[]],'I':[[],[]],'R':[[],[]],'D':[[],[]]}
        if wymn not in w:
            w[wymn] = {'Label':wns,'S':[[],[]],'E':[[],[]],'I':[[],[]],'R':[[],[]],'D':[[],[]]}
        s[g]['S'][0].append(given_list[i])
        s[g]['S'][1].append(S[i])
        s[g]['E'][0].append(given_list[i])
        s[g]['E'][1].append(E[i])
        s[g]['I'][0].append(given_list[i])
        s[g]['I'][1].append(I[i])
        s[g]['R'][0].append(given_list[i])
        s[g]['R'][1].append(R[i])
        s[g]['D'][0].append(given_list[i])
        s[g]['D'][1].append(D[i])        
        w[wymn]['S'][0].append(given_list[i])
        w[wymn]['S'][1].append(S[i])
        w[wymn]['E'][0].append(given_list[i])
        w[wymn]['E'][1].append(E[i])
        w[wymn]['I'][0].append(given_list[i])
        w[wymn]['I'][1].append(I[i])
        w[wymn]['R'][0].append(given_list[i])
        w[wymn]['R'][1].append(R[i])
        w[wymn]['D'][0].append(given_list[i])
        w[wymn]['D'][1].append(D[i])
    return [s,w]            


def sum_up_segmented_seird(js):
    s = {}
    for i in js[1].keys():
        if i not in s:
            s[i] = {'Label':'','S':0,'E':0,'I':0,'R':0,'D':0}
        s[i]['S'] = int(sum(js[1][i]['S'][1]))
        s[i]['E'] = int(sum(js[1][i]['E'][1]))
        s[i]['I'] = int(sum(js[1][i]['I'][1]))
        s[i]['R'] = int(sum(js[1][i]['R'][1]))
        s[i]['D'] = int(sum(js[1][i]['D'][1]))
        s[i]['Label'] = i
        #s[i]['Label'] = js[1][i]['Label']
    
    x = {}
    for i in js[0].keys():
        if i not in x:
            x[i] = {'Label':'','S':0,'E':0,'I':0,'R':0,'D':0}
        x[i]['S'] = int(sum(js[0][i]['S'][1]))
        x[i]['E'] = int(sum(js[0][i]['E'][1]))
        x[i]['I'] = int(sum(js[0][i]['I'][1]))
        x[i]['R'] = int(sum(js[0][i]['R'][1]))
        x[i]['D'] = int(sum(js[0][i]['D'][1]))
        x[i]['Label'] = js[0][i]['Label']
    return [x,s]
            


############Testing SEIRD##########
    
def the_end():
    print("Dumny Function")

#########Testing SEIRD#############
####################################################
#           SEIRD FUNCTIONS END
#
####################################################     
#%%
####################################################
#           VISUALIZATION FUNCTIONS START
#
####################################################
def make_my_provintial_totals(znphi_data_frame):
    ytowns = {}
    filter_list = ['TOTAL','Date']
    xcases_names = list(znphi_data_frame.keys())
    for x in list(xcases_names):
        xtowns = list(znphi_data_frame[x].keys())
    for x in xcases_names:
        for y in xtowns:
            if y not in filter_list and y not in ytowns: 
               ytowns[y] = {}
               ytowns[y]['Label'] = y
               ytowns[y]['deaths'] = []
               ytowns[y]['infected'] = []
               ytowns[y]['recorvered'] = []
            if x == 'New_Cases' and y not in filter_list:    
                ytowns[y]['infected'].append(list(znphi_data_frame[x]['Date']))
                ytowns[y]['infected'].append(list(znphi_data_frame[x][y])) 
            if x == 'Fatality Cases' and y not in filter_list:   
                ytowns[y]['deaths'].append(list(znphi_data_frame[x]['Date']))
                ytowns[y]['deaths'].append(list(znphi_data_frame[x][y]))
            if x == 'Recovered Cases' and y not in filter_list:    
                ytowns[y]['recorvered'].append(list(znphi_data_frame[x]['Date']))
                ytowns[y]['recorvered'].append(list(znphi_data_frame[x][y]))
    return ytowns

def make_monthly_bar_graph_x(brokenup_data_set,lst_titiles):
    barWidth = 0.25
    infected_sums = []
    recovered_sums = []
    deaths_sums = []
    date_labels = []
    for i in brokenup_data_set.keys():
        infected_sums.append(sum(brokenup_data_set[i]['infected'][1]))
        recovered_sums.append(sum(brokenup_data_set[i]['recorvered'][1]))
        deaths_sums.append(sum(brokenup_data_set[i]['deaths'][1]))
        date_labels.append(brokenup_data_set[i]['Label'])
    # Set position of bar on X axis
    r1 = np.arange(len(infected_sums))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    # Make the plot
    plt.bar(r1, infected_sums, color='orange', width=barWidth, edgecolor='white', label='Infected')
    plt.bar(r2, recovered_sums, color='green', width=barWidth, edgecolor='white', label='Recorvered')
    plt.bar(r3, deaths_sums, color='red', width=barWidth, edgecolor='white', label='Deaths')
    # Add xticks on the middle of the group bars
    #plt.xlabel('group', fontweight='bold')
    plt.xlabel(lst_titiles[0],fontweight='bold')
    plt.ylabel(lst_titiles[1],fontweight='bold')
    plt.title(lst_titiles[2],fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(infected_sums))], date_labels , rotation = 85)
    # Create legend & Show graphic
    plt.legend()
    plt.show()   

def make_me_a_pie_chart(labels,titile,sizes,colors,explodelist,pietype):
    i1 = len(labels)
    i2 = 0
    new_labels = []
    for i in labels:
        txt = str(labels[i2]) + '(' + str(sizes[i2]) + ')'
        new_labels.append(txt)
        i2 += 1
        
    if pietype == 0:
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
        #centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        #fig.gca().add_artist(centre_circle)
        plt.title(titile)
        ax1.axis('equal')  
        plt.tight_layout()
        plt.show()
    elif pietype == 1:
        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(aspect="equal"))
        wedges, texts = ax.pie(sizes, colors = colors , wedgeprops = {'width':0.5,'linewidth' : 1, 'edgecolor' : "white" }, startangle=-40)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        kw = dict(arrowprops=dict(arrowstyle="-"),bbox=bbox_props, zorder=0, va="center")
        for i, p in enumerate(wedges):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),horizontalalignment=horizontalalignment, **kw)
        ax.set_title(title)
        plt.show()
    elif pietype == 3:
        x = np.char.array(labels)
        y = np.array(sizes)
        porcent = 100.*y/y.sum()
        patches, texts = plt.pie(y, colors=colors,wedgeprops = {'width':0.5,'linewidth' : 1, 'edgecolor' : "white" }, startangle=90, radius=1.2)
        labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]
        
        sort_legend = True
        if sort_legend:
            patches, labels, dummy =  zip(*sorted(zip(patches, labels, y), key=lambda x: x[2], reverse=True))
        
        plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),fontsize=12)
        plt.title(titile)
        plt.show()
    else:
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.title(titile)
        ax1.axis('equal')  
        plt.tight_layout()
        #plt.legend(labels,loc=3)
        plt.show()
######tested= OK###########
###Depends On Above Function##
def make_monthly_pie_charts(brokenup_data_set):
    colors = ['blue','green','red']
    explodelist = [0.3,0.3,0.3]
    for i in brokenup_data_set.keys():
        x = sum(brokenup_data_set[i]['infected'][1])
        y = sum(brokenup_data_set[i]['recorvered'][1])
        z = sum(brokenup_data_set[i]['deaths'][1])
        labels = ['Infected ('+str(x)+')','Recovered ('+str(y)+')','Deaths ('+str(z)+')']
        titile = brokenup_data_set[i]['Label']
        sizes = [x,y,z]
        make_me_a_pie_chart(labels,titile,sizes,colors,explodelist,0)
        #print(labels)

def make_my_cases_nested_pie_chart(cases,xtext,colors,title):
    infected1 = [xtext[0]+'\n ('+str(cases[0]) + ')']
    data1 = cases[:1]
    colors_gender1 = colors[:1]
    
    recdeath = [xtext[1]+' ('+str(cases[1]) + ')\n['+str(round(cases[1]/cases[0]*100,2))+'%]', xtext[0]+' ('+str(cases[2]) + ')\n['+str(round((cases[2]/cases[0]*100),2))+'%]']
    data = cases[1:]
    colors_gender = colors[1:]
    
    fig, ax = plt.subplots()
    
    wedges, texts = ax.pie(data, radius=1.2, colors=colors_gender, wedgeprops = {'width':0.5,'linewidth' : 1, 'edgecolor' : "white" }, startangle=-40)
    wedges2, texts2 = ax.pie(data1, radius=1.3-0.3,colors=colors_gender1 , wedgeprops = {'width':0.5,'linewidth' : 1, 'edgecolor' : "white" })
    
    bbox_props = dict(boxstyle="round,pad=0.5", fc="w", ec="k", lw=2)
    kw = dict(arrowprops=dict(arrowstyle="->"),bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(recdeath[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y), verticalalignment='top', horizontalalignment=horizontalalignment, **kw)
    
    arrowprops = dict(arrowstyle="->")
    ax.annotate(infected1[0], xy=(0.6, 0.55), xytext=(-0.25, 0),bbox=bbox_props, arrowprops=arrowprops)
    ax.set_title(title)
    ###Draw Ka White Center Circle#######
    centre_circle = plt.Circle((0,0),0.8,color='black', fc='white',linewidth=0)
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.show()


###Plot A Line Graph (OK)###
def plot_infected_deaths_recovered_line_graph(lst_date,lst_infected,lst_deaths,lst_recovered,txt_title,txt_xlabel,txt_ylabel):
    plt.figure()
    #fig, ax= plt.subplots(figsize= (12,8))
    plt.plot(lst_date,lst_infected,marker = 'o',color='orange')
    plt.plot(lst_recovered,marker = 'o',color='green')
    plt.plot(lst_deaths,marker = 'o',color='red')
    plt.legend(['Infected','Recovered','Deaths'])
    plt.grid(1)
    plt.xticks(lst_date, lst_date, rotation = 85)
    plt.title(txt_title)
    plt.xlabel(txt_xlabel)
    plt.ylabel(txt_ylabel)
    plt.figure()    

def plot_infected_deaths_recovered_line_graphx(lst_date,lst_infected,lst_deaths,lst_recovered,txt_title,txt_xlabel,txt_ylabel):
    fig, ax= plt.subplots(figsize= (12,8))

    ax.plot(lst_infected, label= 'Infected',marker= 'o', markersize= 4, alpha=0.7, color= 'orange')
    ax.plot(lst_deaths, label= 'Deaths', alpha=0.7, color= 'red')
    ax.plot(lst_recovered, label= 'Recovered', alpha=0.7, color= 'green')

    plt.xticks(rotation= 90)
    plt.legend(loc= 0)
    plt.grid(1)
    plt.show()
    
    
def get_color_group(color):
    if color   == 1: 
        color_sq = ['#1abc9c','#e8f8f5','#d1f2eb','#a3e4d7','#76d7c4','#48c9b0','#1abc9c','#17a589','#148f77','#117864','#0e6251']; 
        colors = 'Turquoise';
    elif color == 2:
        color_sq = ['#2ecc71','#eafaf1','#d5f5e3','#abebc6','#82e0aa','#58d68d','#2ecc71','#28b463','#239b56','#1d8348','#186a3b']; 
        colors = 'Emerald';
    elif color == 3: 
        color_sq = ['#f39c12','#fef5e7','#fdebd0','#fad7a0','#f8c471','#f5b041','#f39c12','#d68910','#b9770e','#9c640c','#7e5109']; 
        colors = 'Orange';
    elif color == 4:                        
        color_sq = ['#7f8c8d','#f2f4f4','#e5e8e8','#ccd1d1','#b2babb','#99a3a4','#7f8c8d','#707b7c','#616a6b','#515a5a','#424949'];
        colors = 'Asbestos';
    elif color == 5:                        
        color_sq = ['#95a5a6','#f4f6f6','#eaeded','#d5dbdb','#bfc9ca','#aab7b8','#95a5a6','#839192','#717d7e','#5f6a6a','#4d5656'];
        colors = 'Concrete';
    elif color == 6:                        
        color_sq = ['#bdc3c7','#f8f9f9','#f2f3f4','#e5e7e9','#d7dbdd','#cacfd2','#bdc3c7','#a6acaf','#909497','#797d7f','#626567'];
        colors = 'Silver';
    elif color == 7:                        
        color_sq = ['#3498db','#ebf5fb','#d6eaf8','#aed6f1','#85c1e9','#5dade2','#3498db','#2e86c1','#2874a6','#21618c','#1b4f72'];
        colors = 'PeterRiver';
    elif color == 9: 
        color_sq = ['#f1c40f','#fef9e7','#fcf3cf','#f9e79f','#f7dc6f','#f4d03f','#f1c40f','#d4ac0d','#b7950b','#9a7d0a','#7d6608']
        colors = 'IDK';
    elif color == 10: 
        color_sq = ['#FF8600','#FFFF00','#FF2400','#E066FF','#CD00CD','#A020F0','#00008B','#000000','#5DFC0A','#660000','#808000']
        colors = 'Bright1';
    elif color == 11: 
        color_sq = ['#FF0000','#00FFFF','#0000FF','#0000A0','#ADD8E6','#800080','#00FF00','#FF00FF','#C0C0C0','#808080','#FFA500','#A52A2A','#800000','#008000','#808000']
        colors = 'Bright2';                   
    else: 
        color_sq = ['#ecf0f1','#fdfefe','#fbfcfc','#f7f9f9','#f4f6f7','#f0f3f4','#ecf0f1','#d0d3d4','#b3b6b7','#979a9a','#7b7d7d']; 
        colors = 'Clouds';
    return color_sq, colors;

def draw_line_graph(lst_xlist,lst_lst_values,lst_labels_legend,lst_colors,txt_title,txt_xlabel,txt_ylabel):
    plt.figure()
    for i in range(0,len(lst_lst_values)):
        plt.plot(lst_lst_values[i],marker = 'o',color=lst_colors[i])
    plt.legend(lst_labels_legend)
    plt.grid(1)
    plt.xticks(lst_xlist, lst_xlist, rotation = 85)
    plt.title(txt_title)
    plt.xlabel(txt_xlabel)
    plt.ylabel(txt_ylabel)
    plt.figure()


def plot_simple_line_graph(lst_dates,lst_size,lst_lst_points,lst_lst_labels,lst_lst_colors,lst_lst_marker,lst_txt_title_label):
    fig, ax= plt.subplots(figsize= (lst_size[0],lst_size[1]))
    for i in range(0,len(lst_lst_points)):
            if i == 0 and len(lst_dates) > 0:
                ax.plot(lst_dates,lst_lst_points[i], label= lst_lst_labels[i],marker=lst_lst_marker[i], markersize= 4, alpha=0.7, color= lst_lst_colors[i])
            else:
                ax.plot(lst_lst_points[i], label= lst_lst_labels[i],marker=lst_lst_marker[i], markersize= 4, alpha=0.7, color= lst_lst_colors[i])

    plt.xticks(rotation= 80)
    plt.legend(loc= 0)
    plt.grid(1)
    plt.title(lst_txt_title_label[0])
    plt.xlabel(lst_txt_title_label[1])
    plt.ylabel(lst_txt_title_label[2])  
    plt.show() 

def plot_simple_line_graph_dashed(lst_dates,lst_size,lst_lst_points,lst_lst_labels,lst_lst_colors,lst_lst_marker,lst_txt_title_label):
    fig, ax= plt.subplots(figsize= (lst_size[0],lst_size[1]))
    for i in range(0,len(lst_lst_points)):
            if i == 0 and len(lst_dates) > 0:
                ax.plot(lst_dates,lst_lst_points[i], label= lst_lst_labels[i],marker=lst_lst_marker[i], markersize= 4, linewidth=1.5, ls=('dashed'), alpha=0.7, color= lst_lst_colors[i])
            else:
                ax.plot(lst_lst_points[i], label= lst_lst_labels[i],marker=lst_lst_marker[i], markersize= 4, linewidth=1.5, ls=('dashed'), alpha=0.7, color= lst_lst_colors[i])

    plt.xticks(rotation= 80)
    plt.legend(loc= 0)
    plt.grid(1)
    plt.title(lst_txt_title_label[0])
    plt.xlabel(lst_txt_title_label[1])
    plt.ylabel(lst_txt_title_label[2])  
    plt.show()


def comparison_line_graph(lst_dates,lst_size,lst_lst_points,lst_lst_labels,lst_lst_colors,lst_lst_marker,lst_vert,lst_xshade,lst_label):
    fig, ax= plt.subplots(figsize= (lst_size[0],lst_size[1]))
    for i in range(0,len(lst_lst_points)):
            if i == 0 and len(lst_dates) > 0:
                ax.plot(lst_dates,lst_lst_points[i][0], label= lst_lst_labels[i][0],marker=lst_lst_marker[i][0], markersize=lst_lst_marker[i][2], alpha=0.7, color= lst_lst_colors[i][0])
                ax.plot(lst_lst_points[i][1], label= lst_lst_labels[i][1],marker=lst_lst_marker[i][1], markersize=lst_lst_marker[i][2], alpha=0.7, color= lst_lst_colors[i][1], ls=('dashed'))
            else:
                ax.plot(lst_lst_points[i][0], label= lst_lst_labels[i][0],marker=lst_lst_marker[i][0], markersize=lst_lst_marker[i][2], alpha=0.7, color= lst_lst_colors[i][0])
                ax.plot(lst_lst_points[i][1], label= lst_lst_labels[i][1],marker=lst_lst_marker[i][1], markersize=lst_lst_marker[i][2], alpha=0.7, color= lst_lst_colors[i][1], ls=('dashed')) 
               
    #ax.plot(lst_deaths, label= 'Deaths', alpha=0.7, color= 'red')
    #ax.plot(lst_recovered, label= 'Recovered', alpha=0.7, color= 'green')
    for i in lst_vert:
        ax.axvline(x = i[0], color = i[1], label = i[2])
    for i in lst_xshade:
        ax.axvspan(i[0],i[1],color =i[2])
       
    plt.xticks(rotation= 90)
    plt.legend(loc= 0)
    plt.grid(1)
    plt.title(lst_label[0])
    plt.xlabel(lst_label[1])
    plt.ylabel(lst_label[2])     
    plt.show() 
    

def seird_multiplier(ret,x):
    return list(ret.T[0]), list(np.multiply(ret.T[1],x)), list(np.multiply(ret.T[2],x)), list(np.multiply(ret.T[3],x)), list(np.multiply(ret.T[4],x))
  
#######################################    
###########Testing Comparison##########

def plotseirdx(t, S, E, I, R, D, N, A, title, end):
    #title = > tititle of graph
    #end => X , X = end day count to end from start. eg 407 
    # A = [Infected,Recovered,Deaths,Susceptible]

    f, ax = plt.subplots(1,1,figsize=(12,8))
    plt.title(title, size= 14)
   
    #ax.plot(t, S, 'blue', alpha=0.7, linewidth=1.5, label='Susceptible')
    #ax.plot(t, E, 'gold', alpha=0.7, linewidth=1.5, label='Exposed')
    ax.plot(t, I, 'red', alpha=0.7, linewidth=1.5, label='Infected')
    ax.plot(t, R, 'green', alpha=0.7, linewidth=1.5, label='Recovered')
    ax.plot(t, D, 'black', alpha=0.7, linewidth=1.5, label='Fatalities')
    #ax.plot(t, S+E+I+R+D, 'c--', alpha=0.5, linewidth=2, label='Total')

    if len(A) > 0:
        ax.plot(t, A[0], 'red', alpha=0.7, linewidth=1.5, label='Infected (Actual)',ls=('dashed'))
        ax.plot(t, A[1], 'green', alpha=0.7, linewidth=1.5, label='Recovered (Actual)',ls=('dashed'))
        ax.plot(t, A[2], 'black', alpha=0.7, linewidth=1.5, label='Fatalities (Actual)',ls=('dashed'))
        ax.plot(t, A[3], 'blue', alpha=0.7, linewidth=1.5, label='Susceptible (Actual)',ls=('dashed'))

    ax.set_ylabel('Case')
    ax.set_xlabel('Day #')
    
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
      
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
      
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
        plt.grid(color= 'lightgrey', )
      
    #plt.savefig(path_output+'SEIRD_completo.jpg', dpi= 500, bbox_inches='tight')
    plt.show()



def ode_model(z, t, beta, sigma, gamma, mu):
    S, E, I, R, D = z
    N = S + E + I + R + D
    dSdt = -beta*S*I/N
    dEdt = beta*S*I/N - sigma*E
    dIdt = sigma*E - gamma*I - mu*I
    dRdt = gamma*I
    dDdt = mu*I
    return [dSdt, dEdt, dIdt, dRdt, dDdt]

def ode_solver(t, initial_conditions, params):
    initE, initI, initR, initN, initD = initial_conditions
    beta, sigma, gamma, mu = params
    initS = initN - (initE + initI + initR + initD)
    res = odeint(ode_model, [initS, initE, initI, initR, initD], t, args=(beta, sigma, gamma, mu))
    return res


#%%
def ecdc_polynomial_diff_model(dDx,days_to_forecast_fwd):
    y0 = dDx
    xlen = len(y0)    
    x0 = []
    for i in range(0,xlen): x0.append(i)
    model = np.poly1d(np.polyfit(x0, y0, 6))
    polyline = np.linspace(1, xlen+days_to_forecast_fwd, xlen+days_to_forecast_fwd)
    modD = model(polyline)
    return modD    

#%%
#Regression Functions
def create_linear_model_array(y):
    xlen = len(y)
    x = []
    for i in range(0,xlen): x.append(i)
    slope, intercept, r, p, std_err = stats.linregress(x, y)
    def myfunc(x):
        return slope * x + intercept
    model = list(map(myfunc, x))
    return {'fit':model,'x':x,'y':y,'slope':slope,'intercept':intercept,'std_err':std_err,'r':r,'p':p}


def linear_monthly_predict_ecdc_hdx(X):
    def create_linear_model_array(y):
        xlen = len(y)
        x = []
        for i in range(0,xlen): x.append(i)
        slope, intercept, r, p, std_err = stats.linregress(x, y)
        def myfunc(x):
            return slope * x + intercept
        model = list(map(myfunc, x))
        return {'fit':model,'x':x,'y':y,'slope':slope,'intercept':intercept,'std_err':std_err,'r':r,'p':p}

    #xlimx = [[56,70],[70,100],[100,131],[131,161],[161,192],[192,223],[223,253],[253,284],[284,314],[314,344]]
    xlimx = [[56,344]]
    predikt = {'infected_cummulative':[],'recorvered_cummulative':[],'deaths_cummulative':[]}
    case_names = ['infected_cummulative','recorvered_cummulative','deaths_cummulative']
    for i in case_names:
        for g in xlimx:
            yy0 = X['Contenious'][i][g[0]:g[1]]
            j = create_linear_model_array(yy0,3)
            predikt[i].append(list(j['fit'])) 
        
    predikt['infected'] = list(itertools.chain.from_iterable(predikt['infected_cummulative']))
    predikt['deaths'] = list(itertools.chain.from_iterable(predikt['deaths_cummulative']))
    predikt['recorvered'] = list(itertools.chain.from_iterable(predikt['recorvered_cummulative']))            

    xlen = len(predikt['infected'])
    x0 = []
    for i in range(0,xlen): x0.append(i)
    
    predikt['x0'] = x0
    
    return predikt
    
def create_poly_model_array(y0,deg):
    xlen = len(y0)    
    x0 = []
    for i in range(0,xlen): x0.append(i)
    model = np.poly1d(np.polyfit(x0, y0, deg))
    polyline = np.linspace(1, xlen, xlen)
    return {'fit':model(polyline),'x':x0,'y':y0,'polyline':polyline}    
    
    
def polynomial_predict_ecdc_hdx(X):
    def create_poly_model_array(y0,deg):
        xlen = len(y0)    
        x0 = []
        for i in range(0,xlen): x0.append(i)
        model = np.poly1d(np.polyfit(x0, y0, deg))
        polyline = np.linspace(1, xlen, xlen)
        return {'fit':model(polyline),'x':x0,'y':y0,'polyline':polyline}
    
    xlimx = [[56,70],[70,100],[100,131],[131,161],[161,192],[192,223],[223,253],[253,284],[284,314],[314,345],[345,354]]
    predikt = {'infected_cummulative':[],'recorvered_cummulative':[],'deaths_cummulative':[]}
    case_names = ['infected_cummulative','recorvered_cummulative','deaths_cummulative']
    for i in case_names:
        for g in xlimx:
            yy0 = X['Contenious'][i][g[0]:g[1]]
            j = create_poly_model_array(yy0,3)
            predikt[i].append(list(j['fit'])) 
        
    predikt['infected'] = list(itertools.chain.from_iterable(predikt['infected_cummulative']))
    predikt['deaths'] = list(itertools.chain.from_iterable(predikt['deaths_cummulative']))
    predikt['recorvered'] = list(itertools.chain.from_iterable(predikt['recorvered_cummulative']))            

    xlen = len(predikt['infected'])
    x0 = []
    for i in range(0,xlen): x0.append(i)
    
    predikt['x0'] = x0
    
    return predikt    

#%%
def scale_ird(SI,SR,SD,AI,AR,AD):
    IX = list(np.multiply(SI,np.max(AI)/np.max(list(SI)[-1])))
    RX = list(np.multiply(SR,np.max(AR)/np.max(list(SR)[-1])))
    DX = list(np.multiply(SD,np.max(AD)/np.max(list(SD)[-1])))
    #dI = np.subtract(AI,IX)
    #dR = np.subtract(AR,RX)
    #dD = np.subtract(AD,DX)
    #xm = np.max(AI)/np.max(list(SI)[-1]
    #xr = np.max(AR)/np.max(list(SR)[-1]
    #xd = np.max(AD)/np.max(list(SD)[-1]
    return IX,RX,DX

def polynomial_diff_model(dDx,days_to_forecast_fwd,ylen):
    y0 = dDx
    xlen = len(y0)    
    x0 = []
    for i in range(0,xlen): x0.append(i)
    model = np.poly1d(np.polyfit(x0, y0, 3))
    polyline = np.linspace(1, xlen+days_to_forecast_fwd, xlen+days_to_forecast_fwd)
    modD = model(polyline)
   
    return [modD]

def get_jhu_csse_dataset():
    #Because covid_seird doesnt get Recovered. Get The Yourself
    jhu_csse_recovered_csv = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
    jhu_deaths_csv = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
    jhu_confirmed_csv = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"

    jhu_csse_recovered_pd = pd.read_csv(jhu_csse_recovered_csv)
    jhu_deaths_pd = pd.read_csv(jhu_deaths_csv)
    jhu_confirmed_pd = pd.read_csv(jhu_confirmed_csv)

    xlst = ['Province/State', 'Country/Region', 'Lat', 'Long']
    recovered = jhu_csse_recovered_pd.loc[jhu_csse_recovered_pd.loc[jhu_csse_recovered_pd['Country/Region'] == 'Zambia'].index]
    deaths = jhu_deaths_pd.loc[jhu_deaths_pd.loc[jhu_deaths_pd['Country/Region'] == 'Zambia'].index]
    confirmed = jhu_confirmed_pd.loc[jhu_confirmed_pd.loc[jhu_confirmed_pd['Country/Region'] == 'Zambia'].index]

    for i in xlst:
        recovered.pop(i)
        deaths.pop(i)
        confirmed.pop(i)
        
    #The 56 is there because for Zambia. Covid cases start at index 56 (18 March 2020)
    jhu_csse_zambia_dataset = {'date':list(recovered.keys())[56:],'recovered':list(recovered.values[0])[56:],'deaths':list(deaths.values[0])[56:],'infected':list(confirmed.values[0])[56:]}
    return jhu_csse_zambia_dataset


def my_cum_to_stepwise(lst_list_of_commulative):
    i = len(lst_list_of_commulative) - 1
    return_list = []
    while i > 0:
        return_list.append(lst_list_of_commulative[i] - lst_list_of_commulative[i - 1])
        i -= 1
    return_list.append(lst_list_of_commulative[0] - 0)
    return_list.reverse()
    return return_list 
    
def make_conf_matrix(pd_frame,actual,forecast,actual_lbl,forecast_lbl,title):
    pick_len = [451,432,402,372,332,302,272,232,202,172,132,102,172,132,102,72,42,0]
    
    y0 = []
    y1 = []
    for i in  pick_len:
        y0.append(list(pd_frame[actual])[i])
        y1.append(round(list(pd_frame[forecast])[i]))
    y_actu = pd.Series(y0, name=actual_lbl)
    y_pred = pd.Series(y1, name=forecast_lbl)
    
    df_confusion = pd.crosstab(y_actu, y_pred)
    
    
    def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
        plt.matshow(df_confusion, cmap=cmap) # imshow
        #plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(df_confusion.columns))
        plt.xticks(tick_marks, df_confusion.columns, rotation=90)
        plt.yticks(tick_marks, df_confusion.index)
        #plt.tight_layout()
        plt.ylabel(df_confusion.index.name)
        plt.xlabel(df_confusion.columns.name)
        plt.title(title)

    plot_confusion_matrix(df_confusion)