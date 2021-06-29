#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 21:02:57 2021

@author: james
"""

#%% Import Libraries
#Import function from experiment lib file
import my_rp_library as mrpc


#%%
# Plot ZNPHI Dataset as Bargraph
znphi_data_frame = mrpc.make_my_provintial_totals(mrpc.my_znphi_data_set())
mrpc.make_monthly_bar_graph_x(znphi_data_frame,['Province','Count','Cases By Province'])

#%%
## Plot Pie charts from ZNPHI Dataset
#znphi_data_frame = mrpc.make_my_provintial_totals(mrpc.my_znphi_data_set())
colors = mrpc.get_color_group(11)
postfixtitles = ' at Provicial Level'
piechart_data = {'infected':[],'deaths':[],'recovered':[],'labels':[],'explodelist':[]}
for i in znphi_data_frame.keys():
    piechart_data['labels'].append(i)
    piechart_data['infected'].append(sum(znphi_data_frame[i]['infected'][1]))
    piechart_data['deaths'].append(sum(znphi_data_frame[i]['deaths'][1]))
    piechart_data['recovered'].append(sum(znphi_data_frame[i]['recorvered'][1]))
    piechart_data['explodelist'].append(0.90)

for j in list(piechart_data.keys())[:3]:
    title = str(j).capitalize() + postfixtitles
    labels = piechart_data['labels']
    sizes = piechart_data[j]    
    explodelist = piechart_data['explodelist']
    print(title)
    mrpc.make_me_a_pie_chart(labels,title,sizes,colors[0],explodelist,3)
    
    
#%%
## Plot Line Graph of I,R,D from ECDC and HDX Datasets    
X = mrpc.go_get_me_my_eltd_dataset()
datesx = X['Contenious']['date']
ydata = {'L':[],'I':[],'R':[],'D':[]}
zxc = mrpc.breakup_dates_to_months_array_index(datesx,X['Contenious']['infected'],X['Contenious']['deaths'],X['Contenious']['recorvered'],X['Contenious']['recorvered'])
for i in zxc:
    ydata['L'].append(zxc[i]['Label'])
    ydata['I'].append(sum(zxc[i]['infected'][1]))
    ydata['R'].append(sum(zxc[i]['recorvered'][1]))
    ydata['D'].append(sum(zxc[i]['deaths'][1]))

datesx = ydata['L']
xdata = [ydata['I'],ydata['R'],ydata['D']]
sizes = [12,8]
labels = ['Infected','Recovered','Deaths']
colors = ['orange','green','red']
markers = ['x','x','x']
lst_lab = ['Cases Per Month','Month','Cases']
mrpc.plot_simple_line_graph(datesx,sizes,xdata,labels,colors,markers,lst_lab)


#%%
## Plot Line Graph of I,R,D from ECDC and HDX Datasets with bounds
lx1 = [0, 0 ,0, 120, 240, 360, 480, 600, 720, 840, 960, 1080 ]
lx2 = [0, 0, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]
X = mrpc.go_get_me_my_eltd_dataset()
datesx = X['Contenious']['date']
ydata = {'L':[],'I':[],'R':[],'D':[]}
zxc = mrpc.breakup_dates_to_months_array_index(datesx,X['Contenious']['infected'],X['Contenious']['deaths'],X['Contenious']['recorvered'],X['Contenious']['recorvered'])
for i in zxc:
    ydata['L'].append(zxc[i]['Label'])
    ydata['I'].append(sum(zxc[i]['infected'][1]))
    ydata['R'].append(sum(zxc[i]['recorvered'][1]))
    ydata['D'].append(sum(zxc[i]['deaths'][1]))

datesx = ydata['L']
xdata = [ydata['I'],ydata['R'],ydata['D'],lx1 , lx2]
sizes = [12,8]
labels = ['Infected','Recovered','Deaths','Lower Bound','Upper Bound']
colors = ['orange','green','red','grey','cyan']
markers = ['x','x','x','x','x']
lst_lab = ['Cases Per Month','Month','Cases']
mrpc.plot_simple_line_graph(datesx,sizes,xdata,labels,colors,markers,lst_lab)

#%%
### Plot ECDC and HDX Datasets. Daywise and Cumulative
import numpy as np
X = mrpc.go_get_me_my_eltd_dataset()
datesx = []
sizes = [12,8]
labels = ['Infected','Recovered','Fatalities']
colors = ['orange','green','red']
markers = ['x','x','x']
A = 70
B = 300

lst_lab = ['Daywise Cases','Day','Count']
xdata = [X['Contenious']['infected'][A:B],X['Contenious']['recorvered'][A:B],X['Contenious']['deaths'][A:B]]
mrpc.plot_simple_line_graph(datesx,sizes,xdata,labels,colors,markers,lst_lab)

lst_lab = ['Cumulative Cases','Day','Count']
xdata = [X['Contenious']['infected_cummulative'][A:B],X['Contenious']['recorvered_cummulative'][A:B],X['Contenious']['deaths_cummulative'][A:B]]
mrpc.plot_simple_line_graph(datesx,sizes,xdata,labels,colors,markers,lst_lab)

#%%
import numpy as np
X = mrpc.go_get_me_my_eltd_dataset()
datesx = []
sizes = [8,6]
labels = ['Actual Infected','Projected Infected','Actual Recovered','Projected Recovered','Actual Deaths','Projected Deaths']
colors = ['orange','green','black','gray','blue','pink']
markers = ['x','o','x','o','x','o']

A = list(X['Contenious']['infected_cummulative'][56:])[0]
B = list(X['Contenious']['infected_cummulative'][56:])[-1]
C = len(X['Contenious']['infected_cummulative'][56:])
D = np.linspace(A, B, C)

E = np.min(np.nonzero(X['Contenious']['recorvered_cummulative'][56:]))
F = list(X['Contenious']['recorvered_cummulative'][56:])[-1]
G = np.linspace(E, F, C)

H = np.min(np.nonzero(X['Contenious']['deaths_cummulative'][56:]))
I = list(np.nonzero(X['Contenious']['deaths_cummulative'][56:]))[-1]
J = np.linspace(H, I, C)

lst_lab = ['Cumulative Cases','Day','Count']
xdata = [X['Contenious']['infected_cummulative'][56:],D,X['Contenious']['recorvered_cummulative'][56:],G]
mrpc.plot_simple_line_graph_dashed(datesx,sizes,xdata,labels,colors,markers,lst_lab)
