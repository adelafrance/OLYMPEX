#Andrew DeLaFrance
#18 July 2019
#Compare found Secondary, Bright Bands and NARR melting levels for the east
#builds on comparebb2narr_east/west.py

import numpy as np
from numpy import genfromtxt
import matplotlib
#matplotlib.use('Agg')
import pandas as pd
import datetime
from datetime import timedelta
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib import dates
from scipy.stats import skewnorm
from scipy.stats import kurtosis, skew
import sys

#########

"""
THRESHOLDS AND VARIABLES
"""

plot_NARR = True
plot_NPOL_0C = True
plot_NPOL_15C = False
plot_BB = True
plot_Secondary = True
plot_Citation = True

#dates to start and end plot
mon_start = 11
day_start = 12
mon_end = 11
day_end = 13

layer_within = 1.0 # +/- (Z) kilometers of other neighboring layers found.

num_bins = 30 #histogram bins

dir = sys.argv[1] #input from script call, east or west after script name

plot_both_dirs = True if dir == 'both' else False

"""
SETUP - DATA INPUT/OUTPUT
"""
output_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/BrightBands/' #location of previous output
secondary_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/Secondary/'
data_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Data/' #directory for local data
bb_data_east = ''.join(['brightbandsfound_v6_r_6_time0x35.0pcntx25.0_withrhohv_0.910.97_','east','.npy'])
bb_data_west = ''.join(['brightbandsfound_v6_r_6_time0x15.0pcntx25.0_withrhohv_0.910.97_','west','.npy'])

#secondary_data = ''.join(['secondary_C_15X4excd_',dir,'.npy'])
secondary_data_east = ''.join(['secondary_E_15X4excd_','east','.npy'])
secondary_data_west = ''.join(['secondary_E_15X4excd_','west','.npy'])

secondary_vals_data_east = ''.join(['secondary_E_15X4excd_','east','_vals.npy'])
secondary_vals_data_west = ''.join(['secondary_E_15X4excd_','west','_vals.npy'])

sounding_data = 'NPOL_sounding_0_levs.npy'
NARR_data = 'NARR_at_NPOL.csv'

if plot_both_dirs:
    save_name_fig = ''.join(['BB_w_Secondary_','both','_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.png'])
    save_name_dist_fig = ''.join(['Secondary_dist_','both','_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.png'])
else:
    save_name_fig = ''.join(['BB_w_Secondary_',dir,'_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.png'])
    save_name_dist_fig = ''.join(['Secondary_dist_',dir,'_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.png'])

save_name_data_csv_east = ''.join(['BB_w_Secondary_','east','_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.csv'])
save_name_data_csv_west = ''.join(['BB_w_Secondary_','west','_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.csv'])

bb_fn_east = ''.join([output_dir,bb_data_east])
bb_fn_west = ''.join([output_dir,bb_data_west])
secondary_fn_east = ''.join([secondary_dir,secondary_data_east])
secondary_fn_west = ''.join([secondary_dir,secondary_data_west])
secondary_vals_fn_east = ''.join([secondary_dir,secondary_vals_data_east])
secondary_vals_fn_west = ''.join([secondary_dir,secondary_vals_data_west])
sounding_fn = ''.join([output_dir,sounding_data])
NARR_fn = ''.join([data_dir,NARR_data])
save_fn_data_csv_east = ''.join([secondary_dir,save_name_data_csv_east])
save_fn_data_csv_west = ''.join([secondary_dir,save_name_data_csv_west])
save_fn_fig = ''.join([secondary_dir,save_name_fig])
save_fn_dist_fig = ''.join([secondary_dir,save_name_dist_fig])

bright_bands_east = np.load(bb_fn_east)#time,bbfound, top, btm, bbcrit1, bbcrit2
bright_bands_west = np.load(bb_fn_west)
secondary_east = np.load(secondary_fn_east)
secondary_west = np.load(secondary_fn_west)
secondary_vals_east = np.load(secondary_vals_fn_east)
secondary_vals_west = np.load(secondary_vals_fn_west)
NPOL_data = np.load(sounding_fn)
df=pd.read_csv(NARR_fn, sep=',',header=None)
NARR_data = np.array(df) #NARR Time,IVT,Melting Level (m),925speed (kt),925dir,Nd,Nm
n_bbs_east = bright_bands_east.shape[0]
n_bbs_west = bright_bands_west.shape[0]
n_NARRs = NARR_data.shape[0]
n_NPOLs = NPOL_data.shape[0]
n_secondary_east = secondary_east.shape[0]
n_secondary_west = secondary_west.shape[0]

dates_2_flip_east = []
dates_2_flip_west = []

"""
CITATIAON FLIGHT TIMES - FROM OLYMPEX WEBSITE
"""
flight1_st = '12/11/2015 19:17:00'
flight1_end = '12/11/2015 22:34:00'
flight2_st = '13/11/2015 15:04:00'
flight2_end = '13/11/2015 18:26:00'
flight3_st = '14/11/2015 19:45:00'
flight3_end = '14/11/2015 22:29:00'
flight4_st = '18/11/2015 21:30:00'
flight4_end = '19/11/2015 00:21:00'
flight5_st = '23/11/2015 15:08:00'
flight5_end = '23/11/2015 18:16:00'
flight6_st = '23/11/2015 20:42:00'
flight6_end = '23/11/2015 23:31:00'
flight7_st = '24/11/2015 16:12:00'
flight7_end = '24/11/2015 17:40:00'
flight8_st = '24/11/2015 18:53:00'
flight8_end = '24/11/2015 21:42:00'
flight9_st = '02/12/2015 23:44:00'
flight9_end = '03/12/2015 02:48:00'
flight10_st = '03/12/2015 14:02:00'
flight10_end = '03/12/2015 17:04:00'
flight11_st = '04/12/2015 13:06:00'
flight11_end = '04/12/2015 16:00:00'
flight12_st = '05/12/2015 14:35:00'
flight12_end = '05/12/2015 18:00:00'
flight13_st = '10/12/2015 14:32:00'
flight13_end = '10/12/2015 17:02:00'
flight14_st = '12/12/2015 16:57:00'
flight14_end = '12/12/2015 20:13:00'
flight15_st = '13/12/2015 15:53:00'
flight15_end = '13/12/2015 19:11:00'
flight16_st = '13/12/2015 20:05:00'
flight16_end = '13/12/2015 23:19:00'
flight17_st = '18/12/2015 01:23:00'
flight17_end = '18/12/2015 04:30:00'
flight18_st = '18/12/2015 05:45:00'
flight18_end = '18/12/2015 08:39:00'
flight19_st = '19/12/2015 00:56:00'
flight19_end = '19/12/2015 03:59:00'

flight_starts = [flight1_st,flight2_st,flight3_st,flight4_st,flight5_st,flight6_st,flight7_st,flight8_st,flight9_st,flight10_st,\
                flight11_st,flight12_st,flight13_st,flight14_st,flight15_st,flight16_st,flight17_st,flight18_st,flight19_st]
flight_ends = [flight1_end,flight2_end,flight3_end,flight4_end,flight5_end,flight6_end,flight7_end,flight8_end,flight9_end,flight10_end,\
                flight11_end,flight12_end,flight13_end,flight14_end,flight15_end,flight16_end,flight17_end,flight18_end,flight19_end]

for i in range(0,len(flight_starts)):
    flight_starts[i] = datetime.datetime.strptime(flight_starts[i], "%d/%m/%Y %H:%M:%S")
    flight_ends[i] = datetime.datetime.strptime(flight_ends[i], "%d/%m/%Y %H:%M:%S")

#restrict plotting to only use times that a bright band occurred
for i in range(0,n_bbs_east):
    if bright_bands_east[i,1] != '1':
        bright_bands_east[i,2] = float('NaN')
for i in range(0,n_bbs_west):
    if bright_bands_west[i,1] != '1':
        bright_bands_west[i,2] = float('NaN')

"""
ATTEMPT LAYER SORTING IN THE CASE OF MULTIPLE ENHANCED LAYERS OR UPPER LAYERS ONLY - RECORD DATES TO FLIP VALUES AROUND IN VALUES ARRAY
"""
for i in range(0,n_secondary_east):
    if secondary_east[i,1] != '1': #only use times that a secondary enhancement occurs
        secondary_east[i,2] = float('NaN')
        secondary_east[i,3] = float('NaN')
    else:
        if np.isnan(np.float64(secondary_east[i,2])) and ~np.isnan(np.float64(secondary_east[i,4])): #enhancement found but in the second layer and nothing in lower- see if it pairs up with the other first layers
            #does it look like the scan earlier?
            if float(secondary_east[i-1,2]) - layer_within <= float(secondary_east[i,4]) <= float(secondary_east[i-1,2]) + layer_within:
                secondary_east[i,2], secondary_east[i,3] = secondary_east[i,4], secondary_east[i,5]
                dates_2_flip_east.append(secondary_east[i,0])
            #does it look like the scan after?
            elif float(secondary_east[i+1,2]) - layer_within <= float(secondary_east[i,4]) <= float(secondary_east[i+1,2]) + layer_within:
                secondary_east[i,2], secondary_east[i,3] = secondary_east[i,4], secondary_east[i,5]
                dates_2_flip_east.append(secondary_east[i,0])
        elif ~np.isnan(np.float64(secondary_east[i,2])) and ~np.isnan(np.float64(secondary_east[i,4])): #two modes, try to determine which to use
            #does either the previous or later scan have only one layer?
            if ~np.isnan(np.float64(secondary_east[i-1,2])) and np.isnan(np.float64(secondary_east[i-1,4])):
                #is the lower or upper closer to the period before
                if abs(np.float64(secondary_east[i,4]) - np.float64(secondary_east[i-1,2])) < abs(np.float64(secondary_east[i,2]) - np.float64(secondary_east[i-1,2])): #upper is closer to earlier period
                    temp_a, temp_b = secondary_east[i,2], secondary_east[i,3] #swap them around to plot the upper with the other lowers
                    secondary_east[i,2], secondary_east[i,3] = secondary_east[i,4], secondary_east[i,5]
                    secondary_east[i,4], secondary_east[i,5] = temp_a, temp_b
                    dates_2_flip_east.append(secondary_east[i,0])
            elif ~np.isnan(np.float64(secondary_east[i+1,2])) and np.isnan(np.float64(secondary_east[i+1,4])):
                if abs(np.float64(secondary_east[i,4]) - np.float64(secondary_east[i+1,2])) < abs(np.float64(secondary_east[i,2]) - np.float64(secondary_east[i+1,2])): #upper is closer to earlier period
                    temp_a, temp_b = secondary_east[i,2], secondary_east[i,3] #swap them around to plot the upper with the other lowers
                    secondary_east[i,2], secondary_east[i,3] = secondary_east[i,4], secondary_east[i,5]
                    secondary_east[i,4], secondary_east[i,5] = temp_a, temp_b
                    dates_2_flip_east.append(secondary_east[i,0])

for i in range(0,n_secondary_west):
    if secondary_west[i,1] != '1': #only use times that a secondary enhancement occurs
        secondary_west[i,2] = float('NaN')
        secondary_west[i,3] = float('NaN')
    else:
        if np.isnan(np.float64(secondary_west[i,2])) and ~np.isnan(np.float64(secondary_west[i,4])): #enhancement found but in the second layer and nothing in lower- see if it pairs up with the other first layers
            #does it look like the scan earlier?
            if float(secondary_west[i-1,2]) - layer_within <= float(secondary_west[i,4]) <= float(secondary_west[i-1,2]) + layer_within:
                secondary_west[i,2], secondary_west[i,3] = secondary_west[i,4], secondary_west[i,5]
                dates_2_flip_west.append(secondary_west[i,0])
            #does it look like the scan after?
            elif float(secondary_west[i+1,2]) - layer_within <= float(secondary_west[i,4]) <= float(secondary_west[i+1,2]) + layer_within:
                secondary_west[i,2], secondary_west[i,3] = secondary_west[i,4], secondary_west[i,5]
                dates_2_flip_west.append(secondary_west[i,0])
        elif ~np.isnan(np.float64(secondary_west[i,2])) and ~np.isnan(np.float64(secondary_west[i,4])): #two modes, try to determine which to use
            #does either the previous or later scan have only one layer?
            if ~np.isnan(np.float64(secondary_west[i-1,2])) and np.isnan(np.float64(secondary_west[i-1,4])):
                #is the lower or upper closer to the period before
                if abs(np.float64(secondary_west[i,4]) - np.float64(secondary_west[i-1,2])) < abs(np.float64(secondary_west[i,2]) - np.float64(secondary_west[i-1,2])): #upper is closer to earlier period
                    temp_a, temp_b = secondary_west[i,2], secondary_west[i,3] #swap them around to plot the upper with the other lowers
                    secondary_west[i,2], secondary_west[i,3] = secondary_west[i,4], secondary_west[i,5]
                    secondary_west[i,4], secondary_west[i,5] = temp_a, temp_b
                    dates_2_flip_west.append(secondary_west[i,0])
            elif ~np.isnan(np.float64(secondary_west[i+1,2])) and np.isnan(np.float64(secondary_west[i+1,4])):
                if abs(np.float64(secondary_west[i,4]) - np.float64(secondary_west[i+1,2])) < abs(np.float64(secondary_west[i,2]) - np.float64(secondary_west[i+1,2])): #upper is closer to earlier period
                    temp_a, temp_b = secondary_west[i,2], secondary_west[i,3] #swap them around to plot the upper with the other lowers
                    secondary_west[i,2], secondary_west[i,3] = secondary_west[i,4], secondary_west[i,5]
                    secondary_west[i,4], secondary_west[i,5] = temp_a, temp_b
                    dates_2_flip_west.append(secondary_west[i,0])

"""
BUILD SINGLE ARRAYS (EAST AND WEST) FOR ALL DATA TOGETHER
"""
BrightBands_w_NARR_east = bright_bands_east[:,[0,2]] #assign bright bands found in bbidv6 to the array
BrightBands_w_NARR_west = bright_bands_west[:,[0,2]] #assign bright bands found in bbidv6 to the array
NARR_melt_levs_east = bright_bands_east[:,[0,2]] #place holder for data just to build array, will be replaced later
NARR_melt_levs_west = bright_bands_west[:,[0,2]] #place holder for data just to build array, will be replaced later
NPOL_melt_levs_east = bright_bands_east[:,[0,2]] #place holder for data just to build array, will be replaced later
NPOL_melt_levs_west = bright_bands_west[:,[0,2]] #place holder for data just to build array, will be replaced later
secondary_levs_east = secondary_east[:,[0,2,3]] #grab dates, low enhance lev, high enhnace lev
secondary_levs_west = secondary_west[:,[0,2,3]] #grab dates, low enhance lev, high enhnace lev

#array structure = NPOL date, BB top found in algorithm, NARR date, melting layer
BrightBands_w_NARR_east = np.hstack((BrightBands_w_NARR_east,NARR_melt_levs_east,NPOL_melt_levs_east,secondary_levs_east))
BrightBands_w_NARR_west = np.hstack((BrightBands_w_NARR_west,NARR_melt_levs_west,NPOL_melt_levs_west,secondary_levs_west))

BrightBands_w_NARR_east[:,[3,5]] = float('NaN') #empty slot for NARR and NPOL values
BrightBands_w_NARR_west[:,[3,5]] = float('NaN') #empty slot for NARR and NPOL values

items = []
for h in range(0,n_NARRs-1):
    items.append(datetime.datetime.strptime(NARR_data[h+1,0], "%Y-%m-%d %H:%M:%S"))

items_NPOL = []
for h in range(0,n_NPOLs-1):
    items_NPOL.append(datetime.datetime.strptime(NPOL_data[h+1,0], "%m/%d/%y %H:%M:%S:"))

start_index_found_east = False
start_index_found_west = False
start_index_east = 0
start_index_west = 0
end_index_east = n_bbs_east-1
end_index_west = n_bbs_west-1

BrightBands_w_NARR_east = BrightBands_w_NARR_east[1:BrightBands_w_NARR_east.shape[0],:] #remove first row of non-data, index values
BrightBands_w_NARR_east = BrightBands_w_NARR_east[BrightBands_w_NARR_east[:,0].argsort()]
BrightBands_w_NARR_west = BrightBands_w_NARR_west[1:BrightBands_w_NARR_west.shape[0],:] #remove first row of non-data, index values
BrightBands_w_NARR_west = BrightBands_w_NARR_west[BrightBands_w_NARR_west[:,0].argsort()]

secondary_vals_east = secondary_vals_east[1:secondary_vals_east.shape[0],:] #remove first row of non-data, index values
secondary_vals_east = secondary_vals_east[secondary_vals_east[:,0].argsort()]
secondary_vals_west = secondary_vals_west[1:secondary_vals_west.shape[0],:] #remove first row of non-data, index values
secondary_vals_west = secondary_vals_west[secondary_vals_west[:,0].argsort()]

"""
FIND NEAREST NARR AND NPOL SOUNDING TIMES TO ASSIGN MELT LEVELS AND SOUDING HEIGHTS TO ARRAYS IN THE PROPER TIMES
"""
###Build the east set
end_list = []
#loop through all bb times to find nearest NARR melting level
for i in range(0,n_bbs_east-1):
    datetime_object = datetime.datetime.strptime(BrightBands_w_NARR_east[i,0], "%m/%d/%y %H:%M:%S")
    pivot = datetime_object

    timedeltas = []
    for j in range(0,len(items)):
        timedeltas.append(np.abs(pivot-items[j]))
    min_index = timedeltas.index(np.min(timedeltas)) +1
    d = datetime.datetime.strptime(NARR_data[min_index,0], "%Y-%m-%d %H:%M:%S").strftime('%m/%d/%y %H:%M:%S')
    melt_layer = NARR_data[min_index,2]
    BrightBands_w_NARR_east[i,2] = d
    BrightBands_w_NARR_east[i,3] = float(NARR_data[min_index,2].replace(',',''))/1000 #assign the NARR melting layer to my array

    timedeltas = []
    for j in range(0,len(items_NPOL)):
        timedeltas.append(np.abs(pivot-items_NPOL[j]))
    min_index = timedeltas.index(np.min(timedeltas)) +1
    d2 = datetime.datetime.strptime(NPOL_data[min_index,0], '%m/%d/%y %H:%M:%S:').strftime('%m/%d/%y %H:%M:%S')
    melt_layer2 = NPOL_data[min_index,1]
    BrightBands_w_NARR_east[i,4] = d2
    BrightBands_w_NARR_east[i,5] = float(NPOL_data[min_index,1])/1000 #assign the NPOL melting layer to my array

    secondary_datetime_object = datetime.datetime.strptime(BrightBands_w_NARR_east[i,6], "%m/%d/%y %H:%M:%S")
    if secondary_datetime_object != datetime_object:
        print('index mismatch')
    BrightBands_w_NARR_east[i,7] = float(BrightBands_w_NARR_east[i,7])
    BrightBands_w_NARR_east[i,8] = float(BrightBands_w_NARR_east[i,8])

    month = datetime_object.strftime("%m")
    day = datetime_object.strftime("%d")
    if not start_index_found_east:
        if int(month) == mon_start and int(day) == day_start: #grabs the first one it comes across
            start_index_found_east = True
            start_index_east = i
    if int(month) == mon_end and int(day) == day_end: #grabs the first one it comes across
        end_list.append(i)

end_index_east = max(end_list)
end_date_value_east = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR_east[end_index_east,0], "%m/%d/%y %H:%M:%S")])

start_date_east = datetime.datetime.strptime(BrightBands_w_NARR_east[start_index_east,0], "%m/%d/%y %H:%M:%S")
end_date_east = datetime.datetime.strptime(BrightBands_w_NARR_east[end_index_east,0], "%m/%d/%y %H:%M:%S")

###Build the west set
end_list = []
#loop through all bb times to find nearest NARR melting level
for i in range(0,n_bbs_west-1):
    datetime_object = datetime.datetime.strptime(BrightBands_w_NARR_west[i,0], "%m/%d/%y %H:%M:%S")
    pivot = datetime_object

    timedeltas = []
    for j in range(0,len(items)):
        timedeltas.append(np.abs(pivot-items[j]))
    min_index = timedeltas.index(np.min(timedeltas)) +1
    d = datetime.datetime.strptime(NARR_data[min_index,0], "%Y-%m-%d %H:%M:%S").strftime('%m/%d/%y %H:%M:%S')
    melt_layer = NARR_data[min_index,2]
    BrightBands_w_NARR_west[i,2] = d
    BrightBands_w_NARR_west[i,3] = float(NARR_data[min_index,2].replace(',',''))/1000 #assign the NARR melting layer to my array

    timedeltas = []
    for j in range(0,len(items_NPOL)):
        timedeltas.append(np.abs(pivot-items_NPOL[j]))
    min_index = timedeltas.index(np.min(timedeltas)) +1
    d2 = datetime.datetime.strptime(NPOL_data[min_index,0], '%m/%d/%y %H:%M:%S:').strftime('%m/%d/%y %H:%M:%S')
    melt_layer2 = NPOL_data[min_index,1]
    BrightBands_w_NARR_west[i,4] = d2
    BrightBands_w_NARR_west[i,5] = float(NPOL_data[min_index,1])/1000 #assign the NPOL melting layer to my array

    secondary_datetime_object = datetime.datetime.strptime(BrightBands_w_NARR_west[i,6], "%m/%d/%y %H:%M:%S")
    if secondary_datetime_object != datetime_object:
        print('index mismatch')
    BrightBands_w_NARR_west[i,7] = float(BrightBands_w_NARR_west[i,7])
    BrightBands_w_NARR_west[i,8] = float(BrightBands_w_NARR_west[i,8])

    month = datetime_object.strftime("%m")
    day = datetime_object.strftime("%d")
    if not start_index_found_west:
        if int(month) == mon_start and int(day) == day_start: #grabs the first one it comes across
            start_index_found_west = True
            start_index_west = i
    if int(month) == mon_end and int(day) == day_end: #grabs the first one it comes across
        end_list.append(i)

end_index_west = max(end_list)
end_date_value_west = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR_west[end_index_west,0], "%m/%d/%y %H:%M:%S")])

start_date_west = datetime.datetime.strptime(BrightBands_w_NARR_west[start_index_west,0], "%m/%d/%y %H:%M:%S")
end_date_west = datetime.datetime.strptime(BrightBands_w_NARR_west[end_index_west,0], "%m/%d/%y %H:%M:%S")

#how big is the plotting window-used for adjusting labeling
delta_east = end_date_east - start_date_east
delta_west = end_date_west - start_date_west

"""
FIND BEGINNING AND ENDING TIMES FOR REFLECTIVITY VALUES ARRAYS TO GRAB CORRECT VALUES LATER IN PLOTTING HISTOGRAM
"""
start_index_found_east = False
end_list = []
for i in range(len(secondary_vals_east)):
    datetime_object = datetime.datetime.strptime(secondary_vals_east[i,0], "%m/%d/%y %H:%M:%S")
    month = datetime_object.strftime("%m")
    day = datetime_object.strftime("%d")
    if secondary_vals_east[i,0] in dates_2_flip_east:
        temp = secondary_vals_east[i,1]
        secondary_vals_east[i,1] = secondary_vals_east[i,2]
        secondary_vals_east[i,2] = temp
    if not start_index_found_east:
        if int(month) == mon_start and int(day) == day_start: #grabs the first one it comes across
            start_index_found_east = True
            start_index_vals_east = i
    if int(month) == mon_end and int(day) == day_end: #grabs the first one it comes across
        end_list.append(i)
end_index_vals_east = max(end_list)


start_index_found_west = False
end_list = []
for i in range(len(secondary_vals_west)):
    datetime_object = datetime.datetime.strptime(secondary_vals_west[i,0], "%m/%d/%y %H:%M:%S")
    month = datetime_object.strftime("%m")
    day = datetime_object.strftime("%d")
    if secondary_vals_west[i,0] in dates_2_flip_west:
        temp = secondary_vals_west[i,1]
        secondary_vals_west[i,1] = secondary_vals_west[i,2]
        secondary_vals_west[i,2] = temp
    if not start_index_found_west:
        if int(month) == mon_start and int(day) == day_start: #grabs the first one it comes across
            start_index_found_west = True
            start_index_vals_west = i
    if int(month) == mon_end and int(day) == day_end: #grabs the first one it comes across
        end_list.append(i)
end_index_vals_west = max(end_list)

"""
RESTRICT DATA TO WITHIN PLOTTING WINDOW
"""
#limit the values arrays to within start/end dates for histogram
secondary_vals_east = secondary_vals_east[start_index_vals_east:end_index_vals_east+1,1]
secondary_vals_west = secondary_vals_west[start_index_vals_west:end_index_vals_west+1,1]

#limit the plotting to days within defined start end periods
if plot_both_dirs:
    if start_date_east < start_date_west:
        start_date = start_date_east
        start_index = start_index_east
    else:
        start_date = start_date_west
        start_index = start_index_west
    if end_date_east > end_date_west:
        end_date = end_date_east
        end_index = end_index_east
    else:
        end_date = end_date_west
        end_index = end_index_west
    BrightBands_w_NARR_east = BrightBands_w_NARR_east[start_index:end_index+1]
    pd.DataFrame(BrightBands_w_NARR_east).to_csv(save_fn_data_csv_east) #save the data
    BrightBands_w_NARR_west = BrightBands_w_NARR_west[start_index:end_index+1]
    pd.DataFrame(BrightBands_w_NARR_west).to_csv(save_fn_data_csv_west) #save the data
elif dir == 'east':
    start_date = start_date_east
    start_index = start_index_east
    end_date = end_date_east
    end_index = end_index_east
    BrightBands_w_NARR_east = BrightBands_w_NARR_east[start_index:end_index+1]
    pd.DataFrame(BrightBands_w_NARR_east).to_csv(save_fn_data_csv_east) #save the data
elif dir == 'west':
    start_date = start_date_west
    start_index = start_index_west
    end_date = end_date_west
    end_index = end_index_west
    BrightBands_w_NARR_west = BrightBands_w_NARR_west[start_index:end_index+1]
    pd.DataFrame(BrightBands_w_NARR_west).to_csv(save_fn_data_csv_west) #save the data


"""
BUILD X DATA FOR PLOTTING OF EACH VALUE - DATES TO NUMBERS
"""
xdatesBB_east = np.empty(BrightBands_w_NARR_east.shape[0])
for xi in range(0,BrightBands_w_NARR_east.shape[0]):
    xdatesBB_east[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR_east[xi,0], '%m/%d/%y %H:%M:%S')])

xdatesBB_west = np.empty(BrightBands_w_NARR_west.shape[0])
for xi in range(0,BrightBands_w_NARR_west.shape[0]):
    xdatesBB_west[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR_west[xi,0], '%m/%d/%y %H:%M:%S')])

xdatesNARR_east = np.empty(BrightBands_w_NARR_east.shape[0])
for xi in range(0,BrightBands_w_NARR_east.shape[0]):
    xdatesNARR_east[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR_east[xi,2], '%m/%d/%y %H:%M:%S')])
    if xdatesNARR_east[xi]>end_date_value_east:
        xdatesNARR_east[xi] = end_date_value_east
        BrightBands_w_NARR_east[xi,3] = float('NaN')

xdatesNARR_west = np.empty(BrightBands_w_NARR_west.shape[0])
for xi in range(0,BrightBands_w_NARR_west.shape[0]):
    xdatesNARR_west[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR_west[xi,2], '%m/%d/%y %H:%M:%S')])
    if xdatesNARR_west[xi]>end_date_value_west:
        xdatesNARR_west[xi] = end_date_value_west
        BrightBands_w_NARR_west[xi,3] = float('NaN')

xdatesNPOL_east = np.empty(BrightBands_w_NARR_east.shape[0])
for xi in range(0,BrightBands_w_NARR_east.shape[0]):
    xdatesNPOL_east[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR_east[xi,4], '%m/%d/%y %H:%M:%S')])
    if xdatesNPOL_east[xi]>end_date_value_east:
        xdatesNPOL_east[xi] = end_date_value_east
        BrightBands_w_NARR_east[xi,5] = float('NaN')

xdatesNPOL_west = np.empty(BrightBands_w_NARR_west.shape[0])
for xi in range(0,BrightBands_w_NARR_west.shape[0]):
    xdatesNPOL_west[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR_west[xi,4], '%m/%d/%y %H:%M:%S')])
    if xdatesNPOL_west[xi]>end_date_value_west:
        xdatesNPOL_west[xi] = end_date_value_west
        BrightBands_w_NARR_west[xi,5] = float('NaN')

xdatesSecondary_east = np.empty(BrightBands_w_NARR_east.shape[0])
for xi in range(0,BrightBands_w_NARR_east.shape[0]):
    xdatesSecondary_east[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR_east[xi,6], '%m/%d/%y %H:%M:%S')])

xdatesSecondary_west = np.empty(BrightBands_w_NARR_west.shape[0])
for xi in range(0,BrightBands_w_NARR_west.shape[0]):
    xdatesSecondary_west[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR_west[xi,6], '%m/%d/%y %H:%M:%S')])

"""
PLOT TIME SERIES
"""
#plt.rcParams["figure.figsize"] = (20,5)
fig, ax = plt.subplots()
#fig=plt.figure()
#gs=matplotlib.gridspec.GridSpec(2,2) # 2 rows, 2 columns
#ax1=fig.add_subplot(gs[0,:]) # First row, span both columns
#ax2=fig.add_subplot(gs[1,0]) # Second row, first column
#ax3=fig.add_subplot(gs[1,1]) # Second row, second column


days = dates.DayLocator(interval = 2)
hours = dates.HourLocator(interval = 2)
d_fmt = dates.DateFormatter('%m/%d/%y')
h_fmt = dates.DateFormatter('%m/%d/%y %H:%M')

if plot_both_dirs:
    if delta_east >= delta_west:
        delta = delta_east
    else:
        delta = delta_west
    if plot_NARR:
        ax.scatter(xdatesNARR_east,BrightBands_w_NARR_east[:,3], label = 'NARR Melt Level', color = '#e66101',marker = 'o', s = 12)
    if plot_NPOL_0C:
        ax.scatter(xdatesNPOL_east,BrightBands_w_NARR_east[:,5], label = 'NPOL Sounding 0'+ '\u00b0'+ 'C',color = "mediumblue",marker = '^', s = 12)
    if plot_BB:
        ax.plot(xdatesBB_east,BrightBands_w_NARR_east[:,1], label = 'NPOL Radar Bright Band - EAST', color = 'darkgreen',linestyle = '-.', linewidth = 2.0)
        ax.plot(xdatesBB_west,BrightBands_w_NARR_west[:,1], label = 'NPOL Radar Bright Band - WEST', color = 'dimgray',linestyle = '-.', linewidth = 2.0)
    if plot_Secondary:
        upper_limit_east = np.array(BrightBands_w_NARR_east[:,8], dtype = float)
        lower_limit_east = np.array(BrightBands_w_NARR_east[:,7], dtype = float)
        upper_limit_west = np.array(BrightBands_w_NARR_west[:,8], dtype = float)
        lower_limit_west = np.array(BrightBands_w_NARR_west[:,7], dtype = float)
        ax.fill_between(xdatesSecondary_east, upper_limit_east,lower_limit_east, color="#1b9e77", alpha=0.6)
        ax.plot(xdatesSecondary_east,lower_limit_east, label = 'Secondary Enhancement - EAST', color = 'darkgreen', linestyle = '-', linewidth = 1.0)
        ax.plot(xdatesSecondary_east,upper_limit_east, color = 'darkgreen',linestyle = '-', linewidth = 1.0)
        ax.fill_between(xdatesSecondary_west, upper_limit_west,lower_limit_west, color='gray', alpha=0.6) #"#9a0bad" <- purple
        ax.plot(xdatesSecondary_west,lower_limit_west, label = 'Secondary Enhancement - WEST', color = 'dimgray', linestyle = '-', linewidth = 1.0)
        ax.plot(xdatesSecondary_west,upper_limit_west, color = 'dimgray',linestyle = '-', linewidth = 1.0)

else:
    if dir == 'east':
        BrightBands_w_NARR = BrightBands_w_NARR_east[:,:]
        xdatesBB = xdatesBB_east[:]
        xdatesNPOL = xdatesNPOL_east[:]
        xdatesNARR = xdatesNARR_east[:]
        xdatesSecondary = xdatesSecondary_east[:]
        delta = delta_east
    elif dir == 'west':
        BrightBands_w_NARR = BrightBands_w_NARR_west[:,:]
        xdatesBB = xdatesBB_west[:]
        xdatesNPOL = xdatesNPOL_west[:]
        xdatesNARR = xdatesNARR_west[:]
        xdatesSecondary = xdatesSecondary_west[:]
        delta = delta_west

    if plot_NARR:
        ax.scatter(xdatesNARR,BrightBands_w_NARR[:,3], label = 'NARR Melt Level', color = '#e66101',marker = 'o', s = 12)
    if plot_NPOL_0C:
        ax.scatter(xdatesNPOL,BrightBands_w_NARR[:,5], label = 'NPOL Sounding 0'+ '\u00b0'+ 'C',color = "mediumblue",marker = '^', s = 12)
    if plot_BB:
        ax.plot(xdatesBB,BrightBands_w_NARR[:,1], label = 'NPOL Radar Bright Band', color = 'dimgray',linestyle = '-.', linewidth = 2.0)
    if plot_Secondary:
        upper_limit = np.array(BrightBands_w_NARR[:,8], dtype = float)
        lower_limit = np.array(BrightBands_w_NARR[:,7], dtype = float)
        ax.fill_between(xdatesSecondary, upper_limit,lower_limit, color="#1b9e77", alpha=0.6)
        ax.plot(xdatesSecondary,lower_limit, label = 'Secondary Enhancement', color = 'darkgreen', linestyle = '-', linewidth = 1.0)
        ax.plot(xdatesSecondary,upper_limit, color = 'darkgreen',linestyle = '-', linewidth = 1.0)


if plot_Citation:
    for i in range(0,len(flight_starts)):
        if flight_starts[i] >= start_date and flight_ends[i] <= end_date:
            x1 = dates.date2num(flight_starts[i])
            x2 = dates.date2num(flight_ends[i])
            ax.axvspan(x1, x2, alpha=0.4, color='lightgrey')
        elif flight_starts[i] < start_date and flight_ends[i] >= start_date and flight_ends[i] <= end_date:
            x1 = dates.date2num(start_date)
            x2 = dates.date2num(flight_ends[i])
            ax.axvspan(x1, x2, alpha=0.4, color='lightgrey')
        elif flight_starts[i] >= start_date and flight_starts[i] <= end_date and flight_ends[i] > end_date:
            x1 = dates.date2num(flight_starts[i])
            x2 = dates.date2num(end_date)
            ax.axvspan(x1, x2, alpha=0.4, color='lightgrey')

#ax.xticks(xdatesNPOL,BrightBands_w_NARR[:,0])
if delta.days < 2:
    hours = dates.HourLocator(interval = 2)
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(h_fmt)
elif 2 <= delta.days <= 10:
    days = dates.DayLocator(interval = 1)
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(d_fmt)
else:
    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(d_fmt)
ax.grid(True, linestyle = '--', linewidth = 0.5)
if plot_Secondary:
    ax.set_ylim([0.25,6.5])
else:
    ax.set_ylim([0.25,3.5])
#ax.set_title(''.join(['OLYMPEX Bright Band Identification - ',dir]))
ax.set_ylabel('Height (km)')
plt.setp(ax.get_xticklabels(), rotation=90)
#plt.legend(loc = 'upper right', ncol = 3, fontsize = 10)
lgd = plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", frameon = False)
fig.tight_layout()
plt.show()

#plt.savefig(save_fn_fig,bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 300)
plt.close()


"""
PLOT HISTOGRAMS
"""

secondary_vals_east = [float(i) for i in secondary_vals_east if i != 'nan']
secondary_vals_west = [float(i) for i in secondary_vals_west if i != 'nan']
secondary_vals_east = [float(i) for i in secondary_vals_east if ~np.isnan(i)]
secondary_vals_west = [float(i) for i in secondary_vals_west if ~np.isnan(i)]

skew_east = pd.Series(secondary_vals_east).skew()
kurt_east = pd.Series(secondary_vals_east).kurtosis()
skew_west = pd.Series(secondary_vals_west).skew()
kurt_west = pd.Series(secondary_vals_west).kurtosis()

mu_east = np.nanmean(secondary_vals_east)
std_east = np.nanstd(secondary_vals_east)
n_east = len(secondary_vals_east)
mu_west = np.nanmean(secondary_vals_west)
std_west = np.nanstd(secondary_vals_west)
n_west = len(secondary_vals_west)

if plot_both_dirs:
    fig, ax = plt.subplots()
    ax.hist(secondary_vals_east, num_bins, normed=True, facecolor="#1b9e77", alpha=0.6)
    ax.hist(secondary_vals_west, num_bins, normed=True, facecolor='gray', alpha=0.6)
    fit_shape_east, fit_loc_east, fit_scale_east = skewnorm.fit(secondary_vals_east)
    fit_shape_west, fit_loc_west, fit_scale_west = skewnorm.fit(secondary_vals_west)

    #mu_east_f, std_east_f = skewnorm.fit(secondary_vals_east)
    #mu_west_f, std_west_f = skewnorm.fit(secondary_vals_west)
    xmin_east, xmax_east = np.nanmin(secondary_vals_east), np.nanmax(secondary_vals_east)
    xmin_west, xmax_west = np.nanmin(secondary_vals_west), np.nanmax(secondary_vals_west)
    x_east = np.linspace(xmin_east, xmax_east, 100)
    x_west = np.linspace(xmin_west, xmax_west, 100)
    p_east = skewnorm.pdf(x_east, fit_shape_east, fit_loc_east, fit_scale_east)
    p_west = skewnorm.pdf(x_west, fit_shape_west, fit_loc_west, fit_scale_west)
    ax.plot(x_east, p_east, 'darkgreen', linewidth=2, label = 'East')
    ax.plot(x_west, p_west, 'dimgray', linewidth=2, label = 'West')
    #ax.vlines(mu_east, ymin=0, ymax=np.nanmax(p_east), color='darkgreen', linestyle='--',)
    #ax.vlines(mu_west, ymin=0, ymax=np.nanmax(p_west), color='dimgray', linestyle='--')
    ax.set_xlim((0,45))
    ax.set_xlabel('Reflectivity (dBZ)')
    ax.set_ylabel('Probability')
    title = "East: mean = %.2f,  skew = %.2f, kurtosis = %.2f\nWest: mean = %.2f,  skew = %.2f, kurtosis = %.2f" % (mu_east, skew_east, kurt_east, mu_west, skew_west, kurt_west)
    plt.title(title)
    plt.legend()
    plt.show()
    #plt.savefig(save_fn_dist_fig, dpi = 300)
    plt.close()
