#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 21:37:57 2021

@author: james
"""
#%% Import Libraries
#Import function from experiment lib file
import my_rp_library as mrpc
#numpy
import numpy as np
#pandas
import pandas as pd
#shapes
import shapefile as shp
#matplot lib
import matplotlib.pyplot as plt
#geopandas
import geopandas as gpd
#point
from shapely.geometry import Point
#functions to make heatmap range
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib as mpl

#Get ZNPHI Dataset#
X = mrpc.go_get_me_my_eltd_dataset()
######################Exponential Fit Experiment###########
# To Find R_0 ,  Beta and gamma using Exponential Fit#####
infected = X['Contenious']['infected_cummulative']
dd = np.arange(len(infected))
AB_RANGES = [[56,300],[56,185],[120,355],[56,200],[56,360],[56,310]]
#AB_RANGES = [[60,360]]
for i in AB_RANGES:
    A = i[0]
    B = i[1]
    R0 = mrpc.exponential_fit(infected,A,B,dd)
    beta = mrpc.plot_fit(infected,A,B,dd)
    gamma = beta/R0
    print("beta = "+str(beta) + " , gamma = "+ str(gamma) + ", R0 = "+ str(R0))
 
 
##############Variable Transposition Method Experiment###########
# To Find R_0 ,  Beta and gamma using variable reversal Fit#
infected_cummulative = X['Contenious']['infected_cummulative']  
deaths_cummulative = X['Contenious']['deaths_cummulative']
recorvered_cummulative  = X['Contenious']['recorvered_cummulative']
susceptible = []
populationlist = []
population = 17861034
for i in range(0,len(deaths_cummulative)):
    populationlist.append(17861034)
for i in range(0,len(deaths_cummulative)):
    susceptible.append(populationlist[i] - (infected_cummulative[i] + recorvered_cummulative[i] + deaths_cummulative[i]))
xlens = [[56,69],[70,99],[100,130],[131,160],[161,191],[192,222],[223,252],[253,283],[284,313],[314,244],[56,244]]
Z = {'beta':[],'gamma':[],'R0':[],'RR':[],'DR':[]}
for i in xlens:
    A = i[0]
    B = i[1]
    V = mrpc.calculate_gamma_beta_equation_reverse_method(infected_cummulative[A:B],recorvered_cummulative[A:B],deaths_cummulative[A:B],susceptible[A:B],1)
    Z['R0'].append(V['R0'])
    Z['beta'].append(V['beta'])
    Z['gamma'].append(V['gamma'])
    Z['RR'].append(V['rr'])
    Z['DR'].append(V['dr'])
Z1 = pd.DataFrame.from_dict(Z)
print(Z)   
#########Variable Reversal Method Experiment END ###########

###############Least Square Method Testing#################
#####Least Square Test#########
AB_RANGE = [[56,110],[120,150],[170,230],[50,130],[160,180]]
total = X['Contenious']['infected_cummulative'][56:300]
death = X['Contenious']['deaths'][56:300]
recovered = X['Contenious']['recorvered'][56:300]

datesx = []
xdata = [total]
sizes = [8,6]
labels = ['Infected']
colors = ['red']
markers = ['x']
lst_lab = ['Cases Overall','Day','Cases']
mrpc.plot_simple_line_graph(datesx,sizes,xdata,labels,colors,markers,lst_lab)

Z1 = []
for i in AB_RANGE:
    st = i[0]
    en = i[1]
    Z1.append(mrpc.do_calculate_beta_gamma_r0_m_with_least_square_method(total,recovered,death,st,en))
###############BUGGY OK TEST, LSM METHOD START##############