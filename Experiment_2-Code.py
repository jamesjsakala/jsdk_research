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

#%% Actual mining starts
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

population = zambia.population
r0 = zambia.r0
r2 = zambia.r2

#Set SEIRD to look forward upto x days
zambia.simulation(days_ahead=days_to_seird_model)

# Plot SEIRD Model
zambia.plot_simulation("Zambia SEIRD")

#%%
#Plot JHU SEIRD (IRD) (Multiplier is used because simulation scales down graph)#
xmultiplier = 1
#xmultiplier = 1000000
plt.subplots(figsize= (8,6))
plt.plot(np.multiply(xmultiplier,list(zambia.curves['infected'])),c='orange')
plt.plot(np.multiply(xmultiplier,list(zambia.curves['recovered'])),c='green')
plt.plot(np.multiply(xmultiplier,list(zambia.curves['dead'])),c='red')
plt.legend(['Infected','Recovered','Fatal'])
plt.title('SEIRD Cases')
#plt.xlabel('Day # (Since '+datetime.datetime.now().strftime("%B %Y")+')')
plt.xlabel('Day # (Since 18 Mar 2020)')
plt.ylabel('Cumulative Cases (x1000000)')
plt.grid(1)

#Get JHU Dataset For Zambia Directly
jhu_zambia_dataset_direct_dict = mrpc.get_jhu_csse_dataset()

#%%
#Plot Actual (IRD)
plt.subplots(figsize= (8,6))
plt.plot(jhu_zambia_dataset_direct_dict['infected'],c='orange')
plt.plot(jhu_zambia_dataset_direct_dict['recovered'],c='green')
plt.plot(jhu_zambia_dataset_direct_dict['deaths'],c='red')
plt.legend(['Infected','Recovered','Fatal'])
plt.title('JHU CSSE Cases')
plt.xlabel('Day # (Since 18 Mar 2020)')
plt.ylabel('Cumulative Cases')
plt.grid(1)

#%%
#Plot JHU CSSE (IRD)
plt.subplots(figsize= (8,6))
plt.plot(mrpc.my_cum_to_stepwise(jhu_zambia_dataset_direct_dict['infected']),c='orange')
plt.plot(mrpc.my_cum_to_stepwise(jhu_zambia_dataset_direct_dict['recovered']),c='green')
plt.plot(mrpc.my_cum_to_stepwise(jhu_zambia_dataset_direct_dict['deaths']),c='red')
plt.legend(['Infected','Recovered','Fatal'])
plt.title('JHU CSSE Cases')
plt.xlabel('Day # (Since March 2020)')
plt.ylabel('Daily (Daywise Cases)')
plt.grid(1)

#%%
#Export Simulated Values to Excel Sheet Thru PD and Dict and add Date column#
# SEIRD simulation divides population by 1000000#
xmultiplier = 1000000
zambia_simulation_dict = {}
#COVID cases start 18 March 2020#
StartDate = "03/18/20"
start = datetime.datetime.strptime(StartDate, "%m/%d/%y")
end = start + datetime.timedelta(days=len(zambia.curves['dead']))
zambia_simulation_dict['Date'] = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
zambia_simulation_dict['Infected'] = np.multiply(xmultiplier,list(zambia.curves['infected']))
zambia_simulation_dict['Recovered'] = np.multiply(xmultiplier,list(zambia.curves['recovered']))
zambia_simulation_dict['Fatal'] = np.multiply(xmultiplier,list(zambia.curves['dead']))
zambia_simulation_dict['Susceptible'] = np.multiply(xmultiplier,list(zambia.curves['susceptible']))
zambia_simulation_dict['Exposed'] = np.multiply(xmultiplier,list(zambia.curves['exposed']))
zambia_simulation_pd = pd.DataFrame.from_dict(zambia_simulation_dict)
file_name = "/home/james/Desktop/Skulu/Semister III/Dissertation/Test1/FINAL/CODE/DATA/simulation_data.csv"
#zambia_simulation_pd.to_csv(file_name)
