#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 19:32:41 2021

@author: james
"""
#%% Import Libraries
#Import function from experiment lib file
import my_rp_library as mrpc
#covid_seird
from  covid_seird import country_covid_seird
#matplotlib
import matplotlib.pyplot as plt 
#numpy
import numpy as np
#pandas
import pandas as pd
#datetime
import datetime
#scipy_lerans ks_2samp for correllation 
from scipy.stats import ks_2samp

#%% Actual Mining Starts
#Set Days To forecast 32 days = (almost) 1 months #
days_to_seird_model = 32

#Select Zambia
zambia = country_covid_seird.CountryCovidSeird("ZM")

#Fit Zambia Curve
zambia.fit()

#Best Fit Zambia
best_fit = zambia.best_fit

#Plot best fit for Zambia
zambia.plot_fit("Zambia Plot Fit")

#Set SEIRD to look forward upto x days
zambia.simulation(days_ahead=days_to_seird_model)

#Get JHU Dataset For Zambia Directly
jhu_zambia_dataset_direct_dict = mrpc.get_jhu_csse_dataset()
combined_dictionary = {}
combined_dictionary = jhu_zambia_dataset_direct_dict

## SEIRD simulation divides population by 1000000#
xmultiplier = 1000000

#Set the number of days to evaluate model from#
days_to_evaluate = len(jhu_zambia_dataset_direct_dict['infected'])

#Build combined dict to make pd#
combined_dictionary['infected (Forecast)'] = list(np.multiply(xmultiplier,list(zambia.curves['infected'])))[:days_to_evaluate]
combined_dictionary['deaths (Forecast)'] = list(np.multiply(xmultiplier,list(zambia.curves['dead'])))[:days_to_evaluate]
combined_dictionary['recovered (Forecast)'] = list(np.multiply(xmultiplier,list(zambia.curves['recovered'])))[:days_to_evaluate]
pandas_frame_for_evaluation = pd.DataFrame.from_dict(combined_dictionary)

#%%
##Make COnfusion Matices
mrpc.make_conf_matrix(pandas_frame_for_evaluation,'infected','infected (Forecast)',' Infected (Actual)',' Infected (Predicted)','Infections')
mrpc.make_conf_matrix(pandas_frame_for_evaluation,'deaths','deaths (Forecast)',' Fatal (Actual)',' Fatal (Predicted)','Fatalities')
mrpc.make_conf_matrix(pandas_frame_for_evaluation,'recovered','recovered (Forecast)',' Recovered (Actual)',' Recovered (Predicted)','Recovered')

#%%
#Evaluate forecast numerically#
correlation_infected_num = round(pandas_frame_for_evaluation['infected'].corr(pandas_frame_for_evaluation['infected (Forecast)']),3)
correlation_infected_pec = round((pandas_frame_for_evaluation['infected'].corr(pandas_frame_for_evaluation['infected (Forecast)'])* 100),3)
infected_forecast_max = max(round(pandas_frame_for_evaluation['infected (Forecast)'],2))
infected_actual_max = max(round(pandas_frame_for_evaluation['infected'],2))
#Calculate Correlation for recovered#
correlation_recovered_num = round(pandas_frame_for_evaluation['recovered'].corr(pandas_frame_for_evaluation['recovered (Forecast)']),3)
correlation_recovered_pec = round((pandas_frame_for_evaluation['recovered'].corr(pandas_frame_for_evaluation['recovered (Forecast)'])* 100),3)
recovered_forecast_max = max(round(pandas_frame_for_evaluation['recovered (Forecast)'],2))
recovered_actual_max = max(round(pandas_frame_for_evaluation['recovered'],2))
#Calculate Correlation for deaths#
correlation_fatal_num = round(pandas_frame_for_evaluation['deaths'].corr(pandas_frame_for_evaluation['deaths (Forecast)']),3)
correlation_fatal_pec = round((pandas_frame_for_evaluation['deaths'].corr(pandas_frame_for_evaluation['deaths (Forecast)'])* 100),3)
fatal_forecast_max = max(round(pandas_frame_for_evaluation['deaths (Forecast)'],2))
fatal_actual_max = max(round(pandas_frame_for_evaluation['deaths'],2))

#%%
#Plot JHU SEIRD (IRD) #
#print(ks_2samp(pandas_frame_for_evaluation['infected'], pandas_frame_for_evaluation['infected (Forecast)']))
plt.subplots(figsize= (8,6))
plt.plot(pandas_frame_for_evaluation['infected'],c='orange')
plt.plot(pandas_frame_for_evaluation['infected (Forecast)'],ls=('dashed'),c='orange')
plt.legend(['Infected (Actual)','Infected (Forecast)'])
plt.title('Actual Vs Forecast Cases (Cumulative)'+' \n Correlation = ' + str(correlation_infected_num) + '(' + str(correlation_infected_pec) + '%)\n Max(Actaul/Forecast)' + str(infected_actual_max) + '/' + str(infected_forecast_max) )
plt.xlabel('Day # (Since 18 Mar 2020)')
plt.grid(1)

#%%
#Plot JHU SEIRD (IRD) #
#print(ks_2samp(pandas_frame_for_evaluation['deaths'], pandas_frame_for_evaluation['deaths (Forecast)']))
plt.subplots(figsize= (8,6))
plt.plot(pandas_frame_for_evaluation['deaths'],c='red')
plt.plot(pandas_frame_for_evaluation['deaths (Forecast)'],ls=('dashed'),c='red')
plt.legend(['Fatal (Actual)','Fatal (Forecast)'])
plt.title('Actual Vs Forecast Cases (Cumulative)'+' \n Correlation = ' + str(correlation_fatal_num) + '(' + str(correlation_fatal_pec) + '%)\n Max(Actaul/Forecast)' + str(fatal_actual_max) + '/' + str(fatal_forecast_max)  )
plt.xlabel('Day # (Since 18 Mar 2020)')
plt.grid(1)

#%%
#print(ks_2samp(pandas_frame_for_evaluation['recovered'], pandas_frame_for_evaluation['recovered (Forecast)']))
plt.subplots(figsize= (8,6))
plt.plot(pandas_frame_for_evaluation['recovered'],c='green')
plt.plot(pandas_frame_for_evaluation['recovered (Forecast)'],ls=('dashed'),c='green')
plt.legend(['Recovered (Actual)','Recovered (Forecast)'])
plt.title('Actual Vs Forecast Cases (Cumulative)'+' \n Correlation = ' + str(correlation_recovered_num) + '(' + str(correlation_recovered_pec) + '%)\n Max(Actaul/Forecast)' + str(recovered_actual_max) + '/' + str(recovered_forecast_max) )
plt.xlabel('Day # (Since 18 Mar 2020)')
plt.grid(1)

#%%
#Export Simulated Values to Excel Sheet Thru PD and Dict and add Date column#
# SEIRD simulation divides population by 1000000#
xmultiplier = 1000000
zambia_simulation_dict = {}
#COVID cases start 18 March 2020#
StartDate = "03/18/20"
xlen = len(zambia.curves['infected']) - days_to_seird_model
start = datetime.datetime.strptime(StartDate, "%m/%d/%y")
end = start + datetime.timedelta(days=len(zambia.curves['dead']))
zambia_simulation_dict['Date'] = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)][xlen:]
zambia_simulation_dict['Infected'] = np.multiply(xmultiplier,list(zambia.curves['infected'][xlen:]))
zambia_simulation_dict['Recovered'] = np.multiply(xmultiplier,list(zambia.curves['recovered'][xlen:]))
zambia_simulation_dict['Fatal'] = np.multiply(xmultiplier,list(zambia.curves['dead'][xlen:]))
zambia_simulation_dict['Susceptible'] = np.multiply(xmultiplier,list(zambia.curves['susceptible'][xlen:]))
zambia_simulation_dict['Exposed'] = np.multiply(xmultiplier,list(zambia.curves['exposed'][xlen:]))
zambia_simulation_pd = pd.DataFrame.from_dict(zambia_simulation_dict)
file_name = "/home/james/Desktop/Skulu/Semister III/Dissertation/Test1/FINAL/CODE/DATA/new_30day_prediction_data.csv"
#zambia_simulation_pd.to_csv(file_name)
forecast_actual_export_file = "/home/james/Desktop/Skulu/Semister III/Dissertation/Test1/FINAL/CODE/DATA/new_forecast_actual_export_data.csv"
pandas_frame_for_evaluation.to_csv(forecast_actual_export_file)