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
#shape so we can read map shape files
import shapefile as shp
#matplotlib
import matplotlib.pyplot as plt
#geopandas
import geopandas as gpd
#geometry
from shapely.geometry import Point
#mcolors for range key of heatmap
import matplotlib.colors as mcolors
#SCalarMappable
from matplotlib.cm import ScalarMappable
#Normalize
from matplotlib.colors import Normalize
#mpl
import matplotlib as mpl

#%% csv and shape files to use
health_facilities_facilities_shp_file = "/home/james/Desktop/Skulu/Semester II/Data Mining/MINING_PAPER/DATA/UNZIPPED/hotosm_zmb_health_facilities_points_shp/hotosm_zmb_health_facilities_points.shp"
provinces_shp_file_path = "/home/james/Desktop/Skulu/Semester II/Data Mining/MINING_PAPER/DATA/UNZIPPED/data-zambia-shapefiles-master/data-zambia-shapefiles-master/cso-shapefiles/provices/zambia_new_province_2012.shp"
provinces_translation_dict = {'Central':'CEN', 'Copperbelt':'CB', 'Western':'WST', 'Southern':'STN', 'North Western':'NW', 'North-Western':'NW', 'Lusaka':'LSK', 'Eastern':'EST', 'Luapula':'LUAP', 'Muchinga':'MUC', 'Northern':'NORT','Unspecified':'LSK'}
education_facilities_shp_file ="/home/james/Desktop/Skulu/Semester II/Data Mining/MINING_PAPER/DATA/UNZIPPED/hotosm_zmb_education_facilities_points_shp/hotosm_zmb_education_facilities_points.shp"

#%%Quick Functions
def  get_totals_from_data_frame(brokenup_data_set):
    infected_sums = []
    recovered_sums = []
    deaths_sums = []
    date_labels = []
    for i in brokenup_data_set.keys():
        infected_sums.append(sum(brokenup_data_set[i]['infected'][1]))
        recovered_sums.append(sum(brokenup_data_set[i]['recorvered'][1]))
        deaths_sums.append(sum(brokenup_data_set[i]['deaths'][1]))
        date_labels.append(brokenup_data_set[i]['Label'])
    return {'I':infected_sums,'R':recovered_sums,'D':deaths_sums,'L':date_labels}
    


def calculate_area_size_kmsq(shp_pathx):
    gpd_xz = gpd.read_file(shp_pathx)
    tost = gpd_xz.copy()
    tost = tost.to_crs({'init': 'epsg:32633'})
    tost["area"] = tost['geometry'].area/ 10**6
    return tost["area"]

def get_heat_map_color(figurex,minx,maxx,start='lightblue',emd='red',the_type = 1):
    clist = [(0, start), (1, emd)]
    color_map = mcolors.LinearSegmentedColormap.from_list("", clist)
    if(the_type == 1):
        return color_map(round((figurex/maxx),1))
    elif(the_type == 2):
        figure, axes = plt.subplots(figsize =(11, 1))
        figure.subplots_adjust(bottom = 0.5)
        normlizer = mpl.colors.Normalize(vmin = 0, vmax = maxx)
        figure.colorbar(mpl.cm.ScalarMappable(norm = normlizer,cmap = color_map),cax = axes, orientation ='horizontal',label ='')
        return None


def plot_map_with_markers(sf, points_to_plot_arrays, points_labels_array, name_to_val_dict ,the_title,start_color,emd_color , x_lim = None, y_lim = None, figsize = (11,9)):
    ###set the maximum value###
    the_max = max(points_to_plot_arrays) + 200


    the_legend = []
    plt.figure(figsize = figsize)
    id=0
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        #######Plot Map Outline#####
        plt.plot(x, y, 'k')
        #fill the shape with color
        plt.fill(x, y, color = get_heat_map_color(points_to_plot_arrays[name_to_val_dict[points_labels_array[id]]],0,the_max,start_color,emd_color,1))
        the_legend.append(points_labels_array[id]+'('+str(id)+')')
        # Add A  Label on province
        if (x_lim == None) & (y_lim == None):
            x0 = np.mean(x)
            y0 = np.mean(y)
            plt.text(x0, y0, id, fontsize=13)
        id = id+1
    #legend   
    plt.legend(the_legend)
    #title
    plt.title(the_title)
    #make range
    get_heat_map_color(None,0,the_max,start_color,emd_color,2)


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
        color_sq = ['#FFC300','#FFFF00','#F5B7B1','#D7CCC8','#DAF7A6','#A020F0','#A9DFBF','#D7BDE2','#5DFC0A','#D7DBDD','#808000']
        colors = 'Bright1';                    
    elif color == 11: 
        color_sq = ['#EAECEE','#FBEEE6','#D0ECE7','#F4ECF7','#E8F8F5','#FEF9E7','#F8F9F9','#EAFAF1','#C0C0C0','#DCDCDC']
        colors = 'LightColors1';
    else: 
        color_sq = ['#ecf0f1','#fdfefe','#fbfcfc','#f7f9f9','#f4f6f7','#f0f3f4','#ecf0f1','#d0d3d4','#b3b6b7','#979a9a','#7b7d7d']; 
        colors = 'Clouds';
                    
    return color_sq, colors;


def read_lats_lon_from_sites_shape(filepath,geo_col_name):
    latsx = []
    lonsx = []    
    gpd_data = gpd.read_file(filepath)
    for index,row in gpd_data.iterrows():
      latsx.append(Point(row[geo_col_name]).centroid.y)
      lonsx.append(Point(row[geo_col_name]).centroid.x)
    return latsx, lonsx

#%% Actual Mining Starts
########read Provinces Shape File#########
provinces_shp_reader_data = shp.Reader(provinces_shp_file_path)

########Define Rectangle Frame To Use##########
y_lim = (-18.94795,21.0482838) # latitude 
x_lim = (-7.239207, 34.3864652) # longitude

name_to_val_dict = {'Central':1, 'Copperbelt':7, 'Western':8, 'Southern':3, 'North Western':6, 'Lusaka':0, 'Eastern':2, 'Luapula':9, 'Muchinga':4, 'Northern':5, 'North Western':6}

original_province_names = gpd.read_file(provinces_shp_file_path)['NAME1_'].tolist()

io = 0
short_province_names = []
for i in original_province_names:
    short_province_names.append(str(i) + '(' + str(io) + ')')
    io += 1

znphi_data_frame = mrpc.make_my_provintial_totals(mrpc.my_znphi_data_set())
province_data_val_list = get_totals_from_data_frame(znphi_data_frame)

my_map_title = 'Infected Cases Total Heat-Map'
plot_map_with_markers(provinces_shp_reader_data,province_data_val_list['I'], original_province_names, name_to_val_dict , my_map_title,'lightblue','orange')

my_map_title = 'Recovered Cases Total Heat-Map'
plot_map_with_markers(provinces_shp_reader_data,province_data_val_list['R'], original_province_names, name_to_val_dict , my_map_title,'lightblue','green')

my_map_title = 'Fatal Cases Total Heat-Map'
plot_map_with_markers(provinces_shp_reader_data,province_data_val_list['D'], original_province_names, name_to_val_dict , my_map_title,'lightblue','red')