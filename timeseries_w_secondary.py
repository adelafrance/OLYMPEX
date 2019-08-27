#Andrew DeLaFrance
#18 July 2019
#Compare found Secondary, Bright Bands and NARR melting levels for the east
#builds on comparebb2narr_east/west.py


## requires an input for direction with program call (i.e. python3 timeseries_w_secondary.py SW WSW)

import numpy as np
from numpy import genfromtxt
import matplotlib
#matplotlib.use('Agg')
import pandas as pd
import datetime
from datetime import timedelta
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib import dates
from scipy.stats import skewnorm
from scipy.stats import kurtosis, skew
import sys

#########

"""
THRESHOLDS AND VARIABLES
"""

plot_NARR = False
plot_NPOL_0C = True
plot_NPOL_15C = False
plot_BB = True
plot_Secondary = True
plot_Citation = True
save_data = False

#dates to start and end plot
mon_start = 12
day_start = 8
mon_end = 12
day_end = 9

layer_within = 1.0 # +/- (Z) kilometers of other neighboring layers found.

num_bins = 30 #histogram bins

dir = []
#dir = sys.argv[1] #input from script call, east or west after script name
for i in range(1,len(sys.argv)):
    dir.append(sys.argv[i].upper())

dir_str = dir[0].upper()
if len(dir) > 1:
    for i in range(1,len(dir)):
        dir_str = dir_str+'_'+dir[i]

plot_all_sectors = True if 'ALL' in dir else False


"""
SETUP - DATA INPUT/OUTPUT
"""
output_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/BrightBands/' #location of previous output
bb_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/BrightBands/' #output directory for saved images
secondary_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/Secondary/'
data_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Data/' #directory for local data
bb_data_east = ''.join(['brightbandsfound_v6_r_6_time0x35.0pcntx25.0_withrhohv_0.910.97_','east','.npy'])
bb_data_west = ''.join(['brightbandsfound_v6_r_6_time0x15.0pcntx25.0_withrhohv_0.910.97_','west','.npy'])

#bb_fn_east = ''.join([output_dir,bb_data_east])
#bb_fn_west = ''.join([output_dir,bb_data_west])

bb_data_NE = 'brightbandsfound_V7_x35.0pcntx25.0_withrhohv_0.910.97_NE.npy'
bb_data_SW = 'brightbandsfound_V7_x35.0pcntx25.0_withrhohv_0.910.97_SW.npy'
bb_data_WSW = 'brightbandsfound_V7_x35.0pcntx25.0_withrhohv_0.910.97_WSW.npy'
bb_data_WNW = 'brightbandsfound_V7_x35.0pcntx25.0_withrhohv_0.910.97_WNW.npy'
bb_data_NW = 'brightbandsfound_V7_x35.0pcntx25.0_withrhohv_0.910.97_NW.npy'

bb_fn_NE = ''.join([bb_dir, bb_data_NE])
bb_fn_SW = ''.join([bb_dir, bb_data_SW])
bb_fn_WSW = ''.join([bb_dir, bb_data_WSW])
bb_fn_WNW = ''.join([bb_dir, bb_data_WNW])
bb_fn_NW = ''.join([bb_dir, bb_data_NW])

sounding_data = 'NPOL_sounding_0_levs.npy'
NARR_data = 'NARR_at_NPOL.csv'

sounding_fn = ''.join([output_dir,sounding_data])
NARR_fn = ''.join([data_dir,NARR_data])

if plot_all_sectors:
    save_name_fig = ''.join(['BB_w_Secondary_Sectors_','ALL','_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.png'])
    save_name_dist_fig = ''.join(['Secondary_Sectors_dist_','ALL','_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.png'])
else:
    save_name_fig = ''.join(['BB_w_Secondary_Sectors_',dir_str,'_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.png'])
    save_name_dist_fig = ''.join(['Secondary_Sectors_dist_',dir_str,'_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.png'])

#grab the secondary data
secondary_data_NE = ''.join(['secondary_E_8X4excd_','NE','.npy'])
secondary_data_SW = ''.join(['secondary_E_8X4excd_','SW','.npy'])
secondary_data_WSW = ''.join(['secondary_E_8X4excd_','WSW','.npy'])
secondary_data_WNW = ''.join(['secondary_E_8X4excd_','WNW','.npy'])
secondary_data_NW = ''.join(['secondary_E_8X4excd_','NW','.npy'])

secondary_vals_data_NE = ''.join(['secondary_E_8X4excd_vals_NE.npy'])
secondary_vals_data_SW = ''.join(['secondary_E_8X4excd_vals_SW.npy'])
secondary_vals_data_WSW = ''.join(['secondary_E_8X4excd_vals_WSW.npy'])
secondary_vals_data_WNW = ''.join(['secondary_E_8X4excd_vals_WNW.npy'])
secondary_vals_data_NW = ''.join(['secondary_E_8X4excd_vals_NW.npy'])

save_name_data_csv_NE = ''.join(['BB_w_Secondary_','NE','_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.csv'])
save_name_data_csv_SW = ''.join(['BB_w_Secondary_','SW','_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.csv'])
save_name_data_csv_WSW = ''.join(['BB_w_Secondary_','WSW','_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.csv'])
save_name_data_csv_WNW = ''.join(['BB_w_Secondary_','WNW','_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.csv'])
save_name_data_csv_NW = ''.join(['BB_w_Secondary_','NW','_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.csv'])

secondary_fn_NE = ''.join([secondary_dir,secondary_data_NE])
secondary_fn_SW = ''.join([secondary_dir,secondary_data_SW])
secondary_fn_WSW = ''.join([secondary_dir,secondary_data_WSW])
secondary_fn_WNW = ''.join([secondary_dir,secondary_data_WNW])
secondary_fn_NW = ''.join([secondary_dir,secondary_data_NW])

secondary_vals_fn_NE = ''.join([secondary_dir,secondary_vals_data_NE])
secondary_vals_fn_SW = ''.join([secondary_dir,secondary_vals_data_SW])
secondary_vals_fn_WSW = ''.join([secondary_dir,secondary_vals_data_WSW])
secondary_vals_fn_WNW = ''.join([secondary_dir,secondary_vals_data_WNW])
secondary_vals_fn_NW = ''.join([secondary_dir,secondary_vals_data_NW])

save_fn_data_csv_NE = ''.join([secondary_dir,save_name_data_csv_NE])
save_fn_data_csv_SW = ''.join([secondary_dir,save_name_data_csv_SW])
save_fn_data_csv_WSW = ''.join([secondary_dir,save_name_data_csv_WSW])
save_fn_data_csv_WNW = ''.join([secondary_dir,save_name_data_csv_WNW])
save_fn_data_csv_NW = ''.join([secondary_dir,save_name_data_csv_NW])

save_fn_fig = ''.join([secondary_dir,save_name_fig])
save_fn_dist_fig = ''.join([secondary_dir,save_name_dist_fig])

#bright_bands_east = np.load(bb_fn_east)#time,bbfound, top, btm, bbcrit1, bbcrit2
#bright_bands_west = np.load(bb_fn_west)
NPOL_data = np.load(sounding_fn)
df=pd.read_csv(NARR_fn, sep=',',header=None)
NARR_data = np.array(df) #NARR Time,IVT,Melting Level (m),925speed (kt),925dir,Nd,Nm

bright_bands_NE = np.load(bb_fn_NE)
bright_bands_SW = np.load(bb_fn_SW)
bright_bands_WSW = np.load(bb_fn_WSW)
bright_bands_WNW = np.load(bb_fn_WNW)
bright_bands_NW = np.load(bb_fn_NW)

secondary_NE = np.load(secondary_fn_NE)
secondary_SW = np.load(secondary_fn_SW)
secondary_WSW = np.load(secondary_fn_WSW)
secondary_WNW = np.load(secondary_fn_WNW)
secondary_NW = np.load(secondary_fn_NW)

secondary_vals_NE = np.load(secondary_vals_fn_NE)
secondary_vals_SW = np.load(secondary_vals_fn_SW)
secondary_vals_WSW = np.load(secondary_vals_fn_WSW)
secondary_vals_WNW = np.load(secondary_vals_fn_WNW)
secondary_vals_NW = np.load(secondary_vals_fn_NW)

#n_bbs_east = bright_bands_east.shape[0]
#n_bbs_west = bright_bands_west.shape[0]
n_bbs_NE = bright_bands_NE.shape[0]
n_bbs_SW = bright_bands_SW.shape[0]
n_bbs_WSW = bright_bands_WSW.shape[0]
n_bbs_WNW = bright_bands_WNW.shape[0]
n_bbs_NW = bright_bands_NW.shape[0]

n_NARRs = NARR_data.shape[0]
n_NPOLs = NPOL_data.shape[0]

n_secondary_NE = secondary_NE.shape[0]
n_secondary_SW = secondary_SW.shape[0]
n_secondary_WSW = secondary_WSW.shape[0]
n_secondary_WNW = secondary_WNW.shape[0]
n_secondary_NW = secondary_NW.shape[0]

dates_2_flip_NE = []
dates_2_flip_SW = []
dates_2_flip_WSW = []
dates_2_flip_WNW = []
dates_2_flip_NW = []

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

#nov13_zero = '13/11/2015 00:00:00'

if day_start == 12 and mon_start == 11:
    date_start = '12/11/2015 14:00:00'
else:
    date_start = ''.join([str(day_start),'/',str(mon_start),'/2015 00:00:00'])

date_end = ''.join([str(day_end+1),'/',str(mon_end),'/2015 00:00:00'])

#nov13_zero = datetime.datetime.strptime(nov13_zero, "%d/%m/%Y %H:%M:%S")

date_start = datetime.datetime.strptime(date_start, "%d/%m/%Y %H:%M:%S")
date_end = datetime.datetime.strptime(date_end, "%d/%m/%Y %H:%M:%S")

#restrict plotting to only use times that a bright band occurred
for i in range(0,n_bbs_NE):
    if bright_bands_NE[i,1] != '1':
        bright_bands_NE[i,2] = float('NaN')
for i in range(0,n_bbs_SW):
    if bright_bands_SW[i,1] != '1':
        bright_bands_SW[i,2] = float('NaN')
for i in range(0,n_bbs_WSW):
    if bright_bands_WSW[i,1] != '1':
        bright_bands_WSW[i,2] = float('NaN')
for i in range(0,n_bbs_WNW):
    if bright_bands_WNW[i,1] != '1':
        bright_bands_WNW[i,2] = float('NaN')
for i in range(0,n_bbs_NW):
    if bright_bands_NW[i,1] != '1':
        bright_bands_NW[i,2] = float('NaN')



"""
ATTEMPT LAYER SORTING IN THE CASE OF MULTIPLE ENHANCED LAYERS OR UPPER LAYERS ONLY - RECORD DATES TO FLIP VALUES AROUND IN VALUES ARRAY
"""
#sectors = ['NE', 'SW', 'WSW', 'WNW', 'NW']
#sector_names = ['NORTHEAST', 'SOUTHWEST', 'WEST-SOUTHWEST', 'WEST-NORTHWEST', 'NORTHWEST']

sectors = ['NE', 'NW', 'WNW', 'WSW', 'SW']
sector_names = ['NORTHEAST', 'NORTHWEST', 'WEST-NORTHWEST', 'WEST-SOUTHWEST', 'SOUTHWEST']

for s in range(0,len(sectors)):
    #reference the corresponding data
    if sectors[s] == 'NE':
        secondary = secondary_NE
        dates_2_flip = dates_2_flip_NE
        n_secondary = n_secondary_NE
    elif sectors[s] == 'SW':
        secondary = secondary_SW
        dates_2_flip = dates_2_flip_SW
        n_secondary = n_secondary_SW
    elif sectors[s] == 'WSW':
        secondary = secondary_WSW
        dates_2_flip = dates_2_flip_WSW
        n_secondary = n_secondary_WSW
    elif sectors[s] == 'WNW':
        secondary = secondary_WNW
        dates_2_flip = dates_2_flip_WNW
        n_secondary = n_secondary_WNW
    elif sectors[s] == 'NW':
        secondary = secondary_NW
        dates_2_flip = dates_2_flip_NW
        n_secondary = n_secondary_NW

    for i in range(0,n_secondary):
        if secondary[i,1] != '1': #only use times that a secondary enhancement occurs
            secondary[i,2] = float('NaN')
            secondary[i,3] = float('NaN')
        else:
            if np.isnan(np.float64(secondary[i,2])) and ~np.isnan(np.float64(secondary[i,4])): #enhancement found but in the second layer and nothing in lower- see if it pairs up with the other first layers
                #does it look like the scan earlier?
                if float(secondary[i-1,2]) - layer_within <= float(secondary[i,4]) <= float(secondary[i-1,2]) + layer_within:
                    secondary[i,2], secondary[i,3] = secondary[i,4], secondary[i,5]
                    dates_2_flip.append(secondary[i,0])
                #does it look like the scan after?
                elif float(secondary[i+1,2]) - layer_within <= float(secondary[i,4]) <= float(secondary[i+1,2]) + layer_within:
                    secondary[i,2], secondary[i,3] = secondary[i,4], secondary[i,5]
                    dates_2_flip.append(secondary[i,0])
            elif ~np.isnan(np.float64(secondary[i,2])) and ~np.isnan(np.float64(secondary[i,4])): #two modes, try to determine which to use
                #does either the previous or later scan have only one layer?
                if ~np.isnan(np.float64(secondary[i-1,2])) and np.isnan(np.float64(secondary[i-1,4])):
                    #is the lower or upper closer to the period before
                    if abs(np.float64(secondary[i,4]) - np.float64(secondary[i-1,2])) < abs(np.float64(secondary[i,2]) - np.float64(secondary[i-1,2])): #upper is closer to earlier period
                        temp_a, temp_b = secondary[i,2], secondary[i,3] #swap them around to plot the upper with the other lowers
                        secondary[i,2], secondary[i,3] = secondary[i,4], secondary[i,5]
                        secondary[i,4], secondary[i,5] = temp_a, temp_b
                        dates_2_flip.append(secondary[i,0])
                elif ~np.isnan(np.float64(secondary[i+1,2])) and np.isnan(np.float64(secondary[i+1,4])):
                    if abs(np.float64(secondary[i,4]) - np.float64(secondary[i+1,2])) < abs(np.float64(secondary[i,2]) - np.float64(secondary[i+1,2])): #upper is closer to earlier period
                        temp_a, temp_b = secondary[i,2], secondary[i,3] #swap them around to plot the upper with the other lowers
                        secondary[i,2], secondary[i,3] = secondary[i,4], secondary[i,5]
                        secondary[i,4], secondary[i,5] = temp_a, temp_b
                        dates_2_flip.append(secondary[i,0])

"""
BUILD ARRAYS FOR EACH SECTOR
"""

for s in range(0,len(sectors)):
    #reference the corresponding data
    if sectors[s] == 'NE':
        BrightBands_w_NARR = bright_bands_NE[:,[0,2]] #assign bright bands found in bbidv6 to the array
        secondary_levs, secondary_vals = secondary_NE[:,[0,2,3]], secondary_vals_NE  #grab dates, low enhance lev, high enhnace lev / values of enhancement
        n_bbs = n_bbs_NE
        dates_2_flip = dates_2_flip_NE
        save_fn_data_csv = save_fn_data_csv_NE
    elif sectors[s] == 'SW':
        BrightBands_w_NARR = bright_bands_SW[:,[0,2]] #assign bright bands found in bbidv6 to the array
        secondary_levs, secondary_vals = secondary_SW[:,[0,2,3]], secondary_vals_SW
        n_bbs = n_bbs_SW
        dates_2_flip = dates_2_flip_SW
        save_fn_data_csv = save_fn_data_csv_SW
    elif sectors[s] == 'WSW':
        BrightBands_w_NARR = bright_bands_WSW[:,[0,2]] #assign bright bands found in bbidv6 to the array
        secondary_levs, secondary_vals = secondary_WSW[:,[0,2,3]], secondary_vals_WSW
        n_bbs = n_bbs_WSW
        dates_2_flip = dates_2_flip_WSW
        save_fn_data_csv = save_fn_data_csv_WSW
    elif sectors[s] == 'WNW':
        BrightBands_w_NARR = bright_bands_WNW[:,[0,2]] #assign bright bands found in bbidv6 to the array
        secondary_levs, secondary_vals = secondary_WNW[:,[0,2,3]], secondary_vals_WNW
        n_bbs = n_bbs_WNW
        dates_2_flip = dates_2_flip_WNW
        save_fn_data_csv = save_fn_data_csv_WNW
    elif sectors[s] == 'NW':
        BrightBands_w_NARR = bright_bands_NW[:,[0,2]] #assign bright bands found in bbidv6 to the array
        secondary_levs, secondary_vals = secondary_NW[:,[0,2,3]], secondary_vals_NW
        n_bbs = n_bbs_NW
        dates_2_flip = dates_2_flip_NW
        save_fn_data_csv = save_fn_data_csv_NW

    NARR_melt_levs = np.copy(BrightBands_w_NARR) #place holder for data just to build array, will be replaced later
    NPOL_melt_levs = np.copy(BrightBands_w_NARR) #place holder for data just to build array, will be replaced later

    #array structure = NPOL date, BB top found in algorithm, NARR date, melting layer
    BrightBands_w_NARR = np.hstack((BrightBands_w_NARR,NARR_melt_levs,NPOL_melt_levs,secondary_levs))

    BrightBands_w_NARR[:,[3,5]] = float('NaN') #empty slot for NARR and NPOL values

    items = []
    for h in range(0,n_NARRs-1):
        items.append(datetime.datetime.strptime(NARR_data[h+1,0], "%Y-%m-%d %H:%M:%S"))

    items_NPOL = []
    for h in range(0,n_NPOLs-1):
        items_NPOL.append(datetime.datetime.strptime(NPOL_data[h+1,0], "%m/%d/%y %H:%M:%S:"))


    BrightBands_w_NARR = BrightBands_w_NARR[1:BrightBands_w_NARR.shape[0],:] #remove first row of non-data, index values
    BrightBands_w_NARR = BrightBands_w_NARR[BrightBands_w_NARR[:,0].argsort()]

    secondary_vals = secondary_vals[1:secondary_vals.shape[0],:] #remove first row of non-data, index values
    secondary_vals = secondary_vals[secondary_vals[:,0].argsort()]

    start_index_found = False
    end_index = n_bbs-1


    """
    FIND NEAREST NARR AND NPOL SOUNDING TIMES TO ASSIGN MELT LEVELS AND SOUDING HEIGHTS TO ARRAYS IN THE PROPER TIMES
    """

    end_list = []
    #loop through all bb times to find nearest NARR melting level
    for i in range(0,n_bbs-1):
        datetime_object = datetime.datetime.strptime(BrightBands_w_NARR[i,0], "%m/%d/%y %H:%M:%S")
        pivot = datetime_object

        timedeltas = []
        for j in range(0,len(items)):
            timedeltas.append(np.abs(pivot-items[j]))
        min_index = timedeltas.index(np.min(timedeltas)) + 1
        d = datetime.datetime.strptime(NARR_data[min_index,0], "%Y-%m-%d %H:%M:%S").strftime('%m/%d/%y %H:%M:%S')
        melt_layer = NARR_data[min_index,2]
        BrightBands_w_NARR[i,2] = d
        BrightBands_w_NARR[i,3] = float(NARR_data[min_index,2].replace(',',''))/1000 #assign the NARR melting layer to my array

        timedeltas = []
        for j in range(0,len(items_NPOL)):
            timedeltas.append(np.abs(pivot-items_NPOL[j]))
        min_index = timedeltas.index(np.min(timedeltas)) + 1
        d2 = datetime.datetime.strptime(NPOL_data[min_index,0], '%m/%d/%y %H:%M:%S:').strftime('%m/%d/%y %H:%M:%S')
        melt_layer2 = NPOL_data[min_index,1]
        BrightBands_w_NARR[i,4] = d2
        BrightBands_w_NARR[i,5] = float(NPOL_data[min_index,1])/1000 #assign the NPOL melting layer to my array

        secondary_datetime_object = datetime.datetime.strptime(BrightBands_w_NARR[i,6], "%m/%d/%y %H:%M:%S")
        if secondary_datetime_object != datetime_object:
            print('index mismatch')
            print(datetime_object, secondary_datetime_object)
        BrightBands_w_NARR[i,7] = float(BrightBands_w_NARR[i,7])
        BrightBands_w_NARR[i,8] = float(BrightBands_w_NARR[i,8])

        month = datetime_object.strftime("%m")
        day = datetime_object.strftime("%d")
        if not start_index_found:
            if int(month) == mon_start and int(day) == day_start: #grabs the first one it comes across
                start_index_found = True
                start_index = i
        if int(month) == mon_end and int(day) == day_end: #grabs the first one it comes across
            end_list.append(i)

    end_index = max(end_list)
    end_date_value = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR[end_index,0], "%m/%d/%y %H:%M:%S")])
    start_date = datetime.datetime.strptime(BrightBands_w_NARR[start_index,0], "%m/%d/%y %H:%M:%S")
    end_date = datetime.datetime.strptime(BrightBands_w_NARR[end_index,0], "%m/%d/%y %H:%M:%S")

    #update time window and save the array
    BrightBands_w_NARR = BrightBands_w_NARR[start_index:end_index+1]
    if save_data:
        pd.DataFrame(BrightBands_w_NARR).to_csv(save_fn_data_csv) #save the data


    #how big is the plotting window-used for adjusting labeling
    delta = end_date - start_date

    """
    FIND BEGINNING AND ENDING TIMES FOR REFLECTIVITY VALUES ARRAYS TO GRAB CORRECT VALUES LATER IN PLOTTING HISTOGRAM
    """
    start_index_found = False
    end_list = []
    for i in range(len(secondary_vals)):
        datetime_object = datetime.datetime.strptime(secondary_vals[i,0], "%m/%d/%y %H:%M:%S")
        month = datetime_object.strftime("%m")
        day = datetime_object.strftime("%d")
        if secondary_vals[i,0] in dates_2_flip:
            temp = secondary_vals[i,1]
            secondary_vals[i,1] = secondary_vals[i,2]
            secondary_vals[i,2] = temp
        if not start_index_found:
            if int(month) == mon_start and int(day) == day_start: #grabs the first one it comes across
                start_index_found = True
                start_index_vals = i
        if int(month) == mon_end and int(day) == day_end: #grabs the first one it comes across
            end_list.append(i)
    end_index_vals = max(end_list)

    """
    RESTRICT DATA TO WITHIN PLOTTING WINDOW AND COUNT NUMBER OF SCANS WITH BRIGHT BAND AND WITH SECONDARY ENHANCEMENT
    """
    #limit the values arrays to within start/end dates for histogram
    secondary_vals = secondary_vals[start_index_vals:end_index_vals+1,:]

    time_list_sec = []
    time_list_bb = []

    count_unique_enh_times_sec = 0
    count_unique_enh_times_bb = 0

    for d in range(end_index_vals-start_index_vals+1):
        if secondary_vals[d,0] not in time_list_sec and ~np.isnan(np.float64(secondary_vals[d,1])):
            time_list_sec.append(secondary_vals[d,0])
            count_unique_enh_times_sec += 1

    for d in range(end_index-start_index+1):
        if BrightBands_w_NARR[d,0] not in time_list_bb and ~np.isnan(np.float64(BrightBands_w_NARR[d,1])):
            time_list_sec.append(BrightBands_w_NARR[d,0])
            count_unique_enh_times_bb += 1

    secondary_vals = secondary_vals[:,1]



    """
    UPDATE GLOBAL ARRAYS
    """
    if sectors[s] == 'NE':
        BrightBands_w_NARR_NE = BrightBands_w_NARR
        start_index_NE, end_index_NE, start_date_NE, end_date_NE, end_date_value_NE = start_index, end_index, start_date, end_date, end_date_value
        secondary_vals_NE = secondary_vals
        delta_NE, times_NE_sec, times_NE_bb = delta, count_unique_enh_times_sec, count_unique_enh_times_bb
    elif sectors[s] == 'SW':
        BrightBands_w_NARR_SW = BrightBands_w_NARR
        start_index_SW, end_index_SW, start_date_SW, end_date_SW, end_date_value_SW = start_index, end_index, start_date, end_date, end_date_value
        secondary_vals_SW = secondary_vals
        delta_SW, times_SW_sec, times_SW_bb = delta, count_unique_enh_times_sec, count_unique_enh_times_bb
    elif sectors[s] == 'WSW':
        BrightBands_w_NARR_WSW = BrightBands_w_NARR
        start_index_WSW, end_index_WSW, start_date_WSW, end_date_WSW, end_date_value_WSW = start_index, end_index, start_date, end_date, end_date_value
        secondary_vals_WSW = secondary_vals
        delta_WSW, times_WSW_sec, times_WSW_bb = delta, count_unique_enh_times_sec, count_unique_enh_times_bb
    elif sectors[s] == 'WNW':
        BrightBands_w_NARR_WNW = BrightBands_w_NARR
        start_index_WNW, end_index_WNW, start_date_WNW, end_date_WNW, end_date_value_WNW = start_index, end_index, start_date, end_date, end_date_value
        secondary_vals_WNW = secondary_vals
        delta_WNW, times_WNW_sec, times_WNW_bb = delta, count_unique_enh_times_sec, count_unique_enh_times_bb
    elif sectors[s] == 'NW':
        BrightBands_w_NARR_NW = BrightBands_w_NARR
        start_index_NW, end_index_NW, start_date_NW, end_date_NW, end_date_value_NW = start_index, end_index, start_date, end_date, end_date_value
        secondary_vals_NW = secondary_vals
        delta_NW, times_NW_sec, times_NW_bb = delta, count_unique_enh_times_sec, count_unique_enh_times_bb

    print(''.join([sectors[s],' finished computing']))


for s in range(0,len(sectors)):
    #reference the corresponding data
    if sectors[s] == 'NE':
        BrightBands_w_NARR = BrightBands_w_NARR_NE
        start_index, end_index, start_date, end_date, end_date_value = start_index_NE, end_index_NE, start_date_NE, end_date_NE, end_date_value_NE
        xdatesBB = np.empty(BrightBands_w_NARR_NE.shape[0])
        xdatesNARR = np.empty(BrightBands_w_NARR_NE.shape[0])
        xdatesNPOL = np.empty(BrightBands_w_NARR_NE.shape[0])
        xdatesSecondary = np.empty(BrightBands_w_NARR_NE.shape[0])
    elif sectors[s] == 'SW':
        BrightBands_w_NARR = BrightBands_w_NARR_SW
        start_index, end_index, start_date, end_date, end_date_value = start_index_SW, end_index_SW, start_date_SW, end_date_SW, end_date_value_SW
        xdatesBB = np.empty(BrightBands_w_NARR_SW.shape[0])
        xdatesNARR = np.empty(BrightBands_w_NARR_SW.shape[0])
        xdatesNPOL = np.empty(BrightBands_w_NARR_SW.shape[0])
        xdatesSecondary = np.empty(BrightBands_w_NARR_SW.shape[0])
    elif sectors[s] == 'WSW':
        BrightBands_w_NARR = BrightBands_w_NARR_WSW
        start_index, end_index, start_date, end_date, end_date_value = start_index_WSW, end_index_WSW, start_date_WSW, end_date_WSW, end_date_value_WSW
        xdatesBB = np.empty(BrightBands_w_NARR_WSW.shape[0])
        xdatesNARR = np.empty(BrightBands_w_NARR_WSW.shape[0])
        xdatesNPOL = np.empty(BrightBands_w_NARR_WSW.shape[0])
        xdatesSecondary = np.empty(BrightBands_w_NARR_WSW.shape[0])
    elif sectors[s] == 'WNW':
        BrightBands_w_NARR = BrightBands_w_NARR_WNW
        start_index, end_index, start_date, end_date, end_date_value = start_index_WNW, end_index_WNW, start_date_WNW, end_date_WNW, end_date_value_WNW
        xdatesBB = np.empty(BrightBands_w_NARR_WNW.shape[0])
        xdatesNARR = np.empty(BrightBands_w_NARR_WNW.shape[0])
        xdatesNPOL = np.empty(BrightBands_w_NARR_WNW.shape[0])
        xdatesSecondary = np.empty(BrightBands_w_NARR_WNW.shape[0])
    elif sectors[s] == 'NW':
        BrightBands_w_NARR = BrightBands_w_NARR_NW
        start_index, end_index, start_date, end_date, end_date_value = start_index_NW, end_index_NW, start_date_NW, end_date_NW, end_date_value_NW
        xdatesBB = np.empty(BrightBands_w_NARR_NW.shape[0])
        xdatesNARR = np.empty(BrightBands_w_NARR_NW.shape[0])
        xdatesNPOL = np.empty(BrightBands_w_NARR_NW.shape[0])
        xdatesSecondary = np.empty(BrightBands_w_NARR_NW.shape[0])


    """
    BUILD X DATA FOR PLOTTING OF EACH VALUE - DATES TO NUMBERS
    """

    for xi in range(0,BrightBands_w_NARR.shape[0]):
        xdatesBB[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR[xi,0], '%m/%d/%y %H:%M:%S')])

    for xi in range(0,BrightBands_w_NARR.shape[0]):
        xdatesNARR[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR[xi,2], '%m/%d/%y %H:%M:%S')])
        if xdatesNARR[xi]>end_date_value:
            xdatesNARR[xi] = end_date_value
            BrightBands_w_NARR[xi,3] = float('NaN')

    for xi in range(0,BrightBands_w_NARR.shape[0]):
        xdatesNPOL[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR[xi,4], '%m/%d/%y %H:%M:%S')])
        if xdatesNPOL[xi]>end_date_value:
            xdatesNPOL[xi] = end_date_value
            BrightBands_w_NARR[xi,5] = float('NaN')

    for xi in range(0,BrightBands_w_NARR.shape[0]):
        xdatesSecondary[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR[xi,6], '%m/%d/%y %H:%M:%S')])

    if sectors[s] == 'NE':
        BrightBands_w_NARR_NE = BrightBands_w_NARR
        xdatesBB_NE, xdatesNARR_NE, xdatesNPOL_NE, xdatesSecondary_NE = xdatesBB, xdatesNARR, xdatesNPOL, xdatesSecondary
    elif sectors[s] == 'SW':
        BrightBands_w_NARR_SW = BrightBands_w_NARR
        xdatesBB_SW, xdatesNARR_SW, xdatesNPOL_SW, xdatesSecondary_SW = xdatesBB, xdatesNARR, xdatesNPOL, xdatesSecondary
    elif sectors[s] == 'WSW':
        BrightBands_w_NARR_WSW = BrightBands_w_NARR
        xdatesBB_WSW, xdatesNARR_WSW, xdatesNPOL_WSW, xdatesSecondary_WSW = xdatesBB, xdatesNARR, xdatesNPOL, xdatesSecondary
    elif sectors[s] == 'WNW':
        BrightBands_w_NARR_WNW = BrightBands_w_NARR
        xdatesBB_WNW, xdatesNARR_WNW, xdatesNPOL_WNW, xdatesSecondary_WNW = xdatesBB, xdatesNARR, xdatesNPOL, xdatesSecondary
    elif sectors[s] == 'NW':
        BrightBands_w_NARR_NW = BrightBands_w_NARR
        xdatesBB_NW, xdatesNARR_NW, xdatesNPOL_NW, xdatesSecondary_NW = xdatesBB, xdatesNARR, xdatesNPOL, xdatesSecondary

upper_limit_SW = np.array(BrightBands_w_NARR_SW[:,8], dtype = float)
lower_limit_SW = np.array(BrightBands_w_NARR_SW[:,7], dtype = float)
upper_limit_WSW = np.array(BrightBands_w_NARR_WSW[:,8], dtype = float)
lower_limit_WSW = np.array(BrightBands_w_NARR_WSW[:,7], dtype = float)
upper_limit_WNW = np.array(BrightBands_w_NARR_WNW[:,8], dtype = float)
lower_limit_WNW = np.array(BrightBands_w_NARR_WNW[:,7], dtype = float)
upper_limit_NW = np.array(BrightBands_w_NARR_NW[:,8], dtype = float)
lower_limit_NW = np.array(BrightBands_w_NARR_NW[:,7], dtype = float)
upper_limit_NE = np.array(BrightBands_w_NARR_NE[:,8], dtype = float)
lower_limit_NE = np.array(BrightBands_w_NARR_NE[:,7], dtype = float)

"""
PLOT TIME SERIES
"""

#sector_colors = ['black', 'green', 'grey', 'blue', 'red', 'purple'] #shifted 1 from sector list, enh-general, NE, SW, WSW, WNW, NW
#sector_colors_dark = ['darkgreen', 'dimgrey', 'darkblue', 'darkred', 'indigo'] #pairs with sector list, #NE, SW, WSW, WNW, NW

sector_colors = ['black', 'green', 'purple', 'red', 'blue', 'grey'] #shifted 1 from sector list, enh-general, ['NE', 'NW', 'WNW', 'WSW', 'SW']
sector_colors_dark = ['darkgreen', 'indigo', 'darkred', 'darkblue', 'dimgrey'] #pairs with sector list, ['NE', 'NW', 'WNW', 'WSW', 'SW']

min_x = np.nanmin([xdatesBB_NE, xdatesNARR_SW, xdatesBB_WSW, xdatesBB_WNW, xdatesBB_NW])
max_x = np.nanmax([xdatesBB_NE, xdatesNARR_SW, xdatesBB_WSW, xdatesBB_WNW, xdatesBB_NW])
#plt.rcParams["figure.figsize"] = (20,5)
#fig, ax = plt.subplots()

days = dates.DayLocator(interval = 2)
#hours_1 = dates.HourLocator(interval = 1)
#hours_2 = dates.HourLocator(interval = 2)
d_fmt = dates.DateFormatter('%m/%d/%y')
#h_fmt = dates.DateFormatter('%m/%d/%y %H:%M')
h_fmt = dates.DateFormatter(''.join(['%m/%d %H',' Z']))
h_fmt_short = dates.DateFormatter(''.join(['%H']))

custom_lines = []
custom_labels = []

delta = delta_NE if dir == 'NE' else min([delta_NE, delta_SW, delta_WSW, delta_WNW, delta_NW])

if plot_all_sectors:
    #plot_list = ['SW', 'WSW', 'WNW', 'NW', 'NE']
    plot_list = ['NE', 'NW', 'WNW', 'WSW', 'SW']
    nrows, ncols = 5,1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    ax_big = fig.add_subplot(111, frameon = False)
    ax_big.axis('off')

else:
    plot_list = dir
    nrows, ncols = len(dir),1
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)
    ax_big = fig.add_subplot(111, frameon = False)
    ax_big.axis('off')

if plot_BB:
    for row in range(nrows):
        if plot_list[row] == 'NE':
            if nrows == 1:
                ax.plot(xdatesBB_NE,BrightBands_w_NARR_NE[:,1], label = 'NPOL Radar Bright Band - NE', color = 'dimgray',linestyle = '-', linewidth = 2.0)
            else:
                ax[row].plot(xdatesBB_NE,BrightBands_w_NARR_NE[:,1], label = 'NPOL Radar Bright Band - NE', color = 'dimgray',linestyle = '-', linewidth = 2.0)
            for d in range(1,len(xdatesBB_NE)-1):
                if np.isnan(float(BrightBands_w_NARR_NE[d-1,1])) and np.isnan(float(BrightBands_w_NARR_NE[d+1,1])) and ~np.isnan(float(BrightBands_w_NARR_NE[d,1])):
                    dt = datetime.datetime.strptime(BrightBands_w_NARR_NE[d,0], '%m/%d/%y %H:%M:%S')
                    d1 = dates.date2num([dt-timedelta(minutes = 5)])
                    d2 = dates.date2num([dt+timedelta(minutes = 5)])
                    if nrows == 1:
                        ax.plot([d1,d2],[float(BrightBands_w_NARR_NE[d,1]), float(BrightBands_w_NARR_NE[d,1])], color = 'dimgray',linestyle = '-', linewidth = 2.0)
                    else:
                        ax[row].plot([d1,d2],[float(BrightBands_w_NARR_NE[d,1]), float(BrightBands_w_NARR_NE[d,1])], color = 'dimgray',linestyle = '-', linewidth = 2.0)
        elif plot_list[row] == 'SW':
            ax[row].plot(xdatesBB_SW,BrightBands_w_NARR_SW[:,1], label = 'NPOL Radar Bright Band - SW', color = 'dimgray',linestyle = '-', linewidth = 2.0)
            for d in range(1,len(xdatesBB_SW)-1):
                if np.isnan(float(BrightBands_w_NARR_SW[d-1,1])) and np.isnan(float(BrightBands_w_NARR_SW[d+1,1])) and ~np.isnan(float(BrightBands_w_NARR_SW[d,1])):
                    dt = datetime.datetime.strptime(BrightBands_w_NARR_SW[d,0], '%m/%d/%y %H:%M:%S')
                    d1 = dates.date2num([dt-timedelta(minutes = 5)])
                    d2 = dates.date2num([dt+timedelta(minutes = 5)])
                    if nrows == 1:
                        ax.plot([d1,d2],[float(BrightBands_w_NARR_SW[d,1]), float(BrightBands_w_NARR_SW[d,1])], color = 'dimgray',linestyle = '-', linewidth = 2.0)
                    else:
                        ax[row].plot([d1,d2],[float(BrightBands_w_NARR_SW[d,1]), float(BrightBands_w_NARR_SW[d,1])], color = 'dimgray',linestyle = '-', linewidth = 2.0)
        elif plot_list[row] == 'WSW':
            ax[row].plot(xdatesBB_WSW,BrightBands_w_NARR_WSW[:,1], label = 'NPOL Radar Bright Band - WSW', color = 'dimgray',linestyle = '-', linewidth = 2.0)
            for d in range(1,len(xdatesBB_WSW)-1):
                if np.isnan(float(BrightBands_w_NARR_WSW[d-1,1])) and np.isnan(float(BrightBands_w_NARR_WSW[d+1,1])) and ~np.isnan(float(BrightBands_w_NARR_WSW[d,1])):
                    dt = datetime.datetime.strptime(BrightBands_w_NARR_WSW[d,0], '%m/%d/%y %H:%M:%S')
                    d1 = dates.date2num([dt-timedelta(minutes = 5)])
                    d2 = dates.date2num([dt+timedelta(minutes = 5)])
                    if nrows == 1:
                        ax.plot([d1,d2],[float(BrightBands_w_NARR_WSW[d,1]), float(BrightBands_w_NARR_WSW[d,1])], color = 'dimgray',linestyle = '-', linewidth = 2.0)
                    else:
                        ax[row].plot([d1,d2],[float(BrightBands_w_NARR_WSW[d,1]), float(BrightBands_w_NARR_WSW[d,1])], color = 'dimgray',linestyle = '-', linewidth = 2.0)
        elif plot_list[row] == 'WNW':
            ax[row].plot(xdatesBB_WNW,BrightBands_w_NARR_WNW[:,1], label = 'NPOL Radar Bright Band - WNW', color = 'dimgray',linestyle = '-', linewidth = 2.0)
            for d in range(1,len(xdatesBB_WNW)-1):
                if np.isnan(float(BrightBands_w_NARR_WNW[d-1,1])) and np.isnan(float(BrightBands_w_NARR_WNW[d+1,1])) and ~np.isnan(float(BrightBands_w_NARR_WNW[d,1])):
                    dt = datetime.datetime.strptime(BrightBands_w_NARR_WNW[d,0], '%m/%d/%y %H:%M:%S')
                    d1 = dates.date2num([dt-timedelta(minutes = 5)])
                    d2 = dates.date2num([dt+timedelta(minutes = 5)])
                    if nrows == 1:
                        ax.plot([d1,d2],[float(BrightBands_w_NARR_WNW[d,1]), float(BrightBands_w_NARR_WNW[d,1])], color = 'dimgray',linestyle = '-', linewidth = 2.0)
                    else:
                        ax[row].plot([d1,d2],[float(BrightBands_w_NARR_WNW[d,1]), float(BrightBands_w_NARR_WNW[d,1])], color = 'dimgray',linestyle = '-', linewidth = 2.0)
        elif plot_list[row] == 'NW':
            ax[row].plot(xdatesBB_NW,BrightBands_w_NARR_NW[:,1], label = 'NPOL Radar Bright Band - NW', color = 'dimgray',linestyle = '-', linewidth = 2.0)
            for d in range(1,len(xdatesBB_NW)-1):
                if np.isnan(float(BrightBands_w_NARR_NW[d-1,1])) and np.isnan(float(BrightBands_w_NARR_NW[d+1,1])) and ~np.isnan(float(BrightBands_w_NARR_NW[d,1])):
                    dt = datetime.datetime.strptime(BrightBands_w_NARR_NW[d,0], '%m/%d/%y %H:%M:%S')
                    d1 = dates.date2num([dt-timedelta(minutes = 5)])
                    d2 = dates.date2num([dt+timedelta(minutes = 5)])
                    if nrows == 1:
                        ax.plot([d1,d2],[float(BrightBands_w_NARR_NW[d,1]), float(BrightBands_w_NARR_NW[d,1])], color = 'dimgray',linestyle = '-', linewidth = 2.0)
                    else:
                        ax[row].plot([d1,d2],[float(BrightBands_w_NARR_NW[d,1]), float(BrightBands_w_NARR_NW[d,1])], color = 'dimgray',linestyle = '-', linewidth = 2.0)

    custom_lines.append(Line2D([0], [0], color='dimgray', linestyle='-', linewidth = 2.0))
    custom_labels.append('NPOL Radar Bright Band')

if plot_NARR:
    if nrows == 1:
        ax.scatter(xdatesNARR_SW,BrightBands_w_NARR_SW[:,3], label = 'NARR Melt Level', color = '#e66101',marker = 'o', s = 6, alpha = 0.6)
    else:
        for row in range(nrows):
            ax[row].scatter(xdatesNARR_SW,BrightBands_w_NARR_SW[:,3], label = 'NARR Melt Level', color = '#e66101',marker = 'o', s = 6, alpha = 0.6)
    custom_lines.append(Line2D([0], [0], marker = 'o',color = 'w', markerfacecolor = '#e66101', markersize = 6))
    custom_labels.append('NARR Melt Level')

if plot_NPOL_0C:
    if nrows == 1:
        ax.scatter(xdatesNPOL_SW,BrightBands_w_NARR_SW[:,5], label = 'NPOL Sounding 0'+ '\u00b0'+ 'C',color = "mediumblue",marker = '^', s = 6, alpha = 0.6)
    else:
        for row in range(nrows):
            ax[row].scatter(xdatesNPOL_SW,BrightBands_w_NARR_SW[:,5], label = 'NPOL Sounding 0'+ '\u00b0'+ 'C',color = "mediumblue",marker = '^', s = 6, alpha = 0.6)
    custom_lines.append(Line2D([0], [0], marker = '^', color = 'w' , markerfacecolor = 'mediumblue', markersize = 6))
    custom_labels.append('NPOL Sounding 0'+ '\u00b0'+ 'C')

if plot_Secondary:
    n = 0
    plot_colors = []
    plot_colors_dark = []
    plot_names = []
    plot_abr = []

    if 'NE' in plot_list:
        row = [i for i in range(nrows) if plot_list[i] == 'NE'][0]
        #print("NE", row)
        if n == 0:
            xdatesSecondary = xdatesSecondary_NE[:]
            upper_limit = upper_limit_NE[:]
            lower_limit = lower_limit_NE[:]
            n += 1
        else:
            xdatesSecondary = np.vstack([xdatesSecondary,xdatesSecondary_NE])
            upper_limit = np.vstack([upper_limit, upper_limit_NE])
            lower_limit = np.vstack([lower_limit, lower_limit_NE])
            n += 1
        i_s = [i for i in range(len(sectors)) if sectors[i] == 'NE']
        plot_colors.append(sector_colors[i_s[0]+1])
        plot_colors_dark.append(sector_colors_dark[i_s[0]])
        plot_names.append(sector_names[i_s[0]])
        plot_abr.append(sectors[i_s[0]])

        for d in range(1,len(BrightBands_w_NARR_NE[:,7])-1):
            if np.isnan(float(BrightBands_w_NARR_NE[d-1,7])) and np.isnan(float(BrightBands_w_NARR_NE[d+1,7])) and ~np.isnan(float(BrightBands_w_NARR_NE[d,7])):
                dt = datetime.datetime.strptime(BrightBands_w_NARR_NE[d,6], '%m/%d/%y %H:%M:%S')
                d1 = dates.date2num([dt-timedelta(minutes = 5)])[0]
                d2 = dates.date2num([dt+timedelta(minutes = 5)])[0]
                if nrows == 1:
                    ax.fill_between(np.array([d1,d2]), float(BrightBands_w_NARR_NE[d,7]), float(BrightBands_w_NARR_NE[d,8]), facecolor=sector_colors[i_s[0]+1], edgecolor = sector_colors[i_s[0]+1], alpha=0.6)
                    ax.plot([d1,d2],[float(BrightBands_w_NARR_NE[d,7]), float(BrightBands_w_NARR_NE[d,7])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                    ax.plot([d1,d2],[float(BrightBands_w_NARR_NE[d,8]), float(BrightBands_w_NARR_NE[d,8])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                else:
                    ax[row].fill_between(np.array([d1,d2]), float(BrightBands_w_NARR_NE[d,7]), float(BrightBands_w_NARR_NE[d,8]), facecolor=sector_colors[i_s[0]+1], edgecolor = sector_colors[i_s[0]+1], alpha=0.6)
                    ax[row].plot([d1,d2],[float(BrightBands_w_NARR_NE[d,7]), float(BrightBands_w_NARR_NE[d,7])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                    ax[row].plot([d1,d2],[float(BrightBands_w_NARR_NE[d,8]), float(BrightBands_w_NARR_NE[d,8])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)

    if 'NW' in plot_list:
        row = [i for i in range(nrows) if plot_list[i] == 'NW'][0]
        #print("NW", row)
        if n == 0:
            xdatesSecondary = xdatesSecondary_NW[:]
            upper_limit = upper_limit_NW[:]
            lower_limit = lower_limit_NW[:]
            n += 1
        else:
            xdatesSecondary = np.vstack([xdatesSecondary,xdatesSecondary_NW])
            upper_limit = np.vstack([upper_limit, upper_limit_NW])
            lower_limit = np.vstack([lower_limit, lower_limit_NW])
            n += 1
        i_s = [i for i in range(len(sectors)) if sectors[i] == 'NW']
        plot_colors.append(sector_colors[i_s[0]+1])
        plot_colors_dark.append(sector_colors_dark[i_s[0]])
        plot_names.append(sector_names[i_s[0]])
        plot_abr.append(sectors[i_s[0]])

        for d in range(1,len(BrightBands_w_NARR_NW[:,7])-1):
            if np.isnan(float(BrightBands_w_NARR_NW[d-1,7])) and np.isnan(float(BrightBands_w_NARR_NW[d+1,7])) and ~np.isnan(float(BrightBands_w_NARR_NW[d,7])):
                dt = datetime.datetime.strptime(BrightBands_w_NARR_NW[d,6], '%m/%d/%y %H:%M:%S')
                d1 = dates.date2num([dt-timedelta(minutes = 5)])[0]
                d2 = dates.date2num([dt+timedelta(minutes = 5)])[0]
                if nrows == 1:
                    ax.fill_between(np.array([d1,d2]), float(BrightBands_w_NARR_NW[d,7]), float(BrightBands_w_NARR_NW[d,8]), facecolor=sector_colors[i_s[0]+1], edgecolor = sector_colors[i_s[0]+1], alpha=0.6)
                    ax.plot([d1,d2],[float(BrightBands_w_NARR_NW[d,7]), float(BrightBands_w_NARR_NW[d,7])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                    ax.plot([d1,d2],[float(BrightBands_w_NARR_NW[d,8]), float(BrightBands_w_NARR_NW[d,8])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                else:
                    ax[row].fill_between(np.array([d1,d2]), float(BrightBands_w_NARR_NW[d,7]), float(BrightBands_w_NARR_NW[d,8]), facecolor=sector_colors[i_s[0]+1], edgecolor = sector_colors[i_s[0]+1], alpha=0.6)
                    ax[row].plot([d1,d2],[float(BrightBands_w_NARR_NW[d,7]), float(BrightBands_w_NARR_NW[d,7])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                    ax[row].plot([d1,d2],[float(BrightBands_w_NARR_NW[d,8]), float(BrightBands_w_NARR_NW[d,8])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
    if 'WNW' in plot_list:
        row = [i for i in range(nrows) if plot_list[i] == 'WNW'][0]
        #print("WNW", row)
        if n == 0:
            xdatesSecondary = xdatesSecondary_WNW[:]
            upper_limit = upper_limit_WNW[:]
            lower_limit = lower_limit_WNW[:]
            n += 1
        else:
            xdatesSecondary = np.vstack([xdatesSecondary,xdatesSecondary_WNW])
            upper_limit = np.vstack([upper_limit, upper_limit_WNW])
            lower_limit = np.vstack([lower_limit, lower_limit_WNW])
            n += 1
        i_s = [i for i in range(len(sectors)) if sectors[i] == 'WNW']
        plot_colors.append(sector_colors[i_s[0]+1])
        plot_colors_dark.append(sector_colors_dark[i_s[0]])
        plot_names.append(sector_names[i_s[0]])
        plot_abr.append(sectors[i_s[0]])

        for d in range(1,len(BrightBands_w_NARR_WNW[:,7])-1):
            if np.isnan(float(BrightBands_w_NARR_WNW[d-1,7])) and np.isnan(float(BrightBands_w_NARR_WNW[d+1,7])) and ~np.isnan(float(BrightBands_w_NARR_WNW[d,7])):
                dt = datetime.datetime.strptime(BrightBands_w_NARR_WNW[d,6], '%m/%d/%y %H:%M:%S')
                d1 = dates.date2num([dt-timedelta(minutes = 5)])[0]
                d2 = dates.date2num([dt+timedelta(minutes = 5)])[0]
                if nrows == 1:
                    ax.fill_between(np.array([d1,d2]), float(BrightBands_w_NARR_WNW[d,7]), float(BrightBands_w_NARR_WNW[d,8]), facecolor=sector_colors[i_s[0]+1], edgecolor = sector_colors[i_s[0]+1], alpha=0.6)
                    ax.plot([d1,d2],[float(BrightBands_w_NARR_WNW[d,7]), float(BrightBands_w_NARR_WNW[d,7])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                    ax.plot([d1,d2],[float(BrightBands_w_NARR_WNW[d,8]), float(BrightBands_w_NARR_WNW[d,8])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                else:
                    ax[row].fill_between(np.array([d1,d2]), float(BrightBands_w_NARR_WNW[d,7]), float(BrightBands_w_NARR_WNW[d,8]), facecolor=sector_colors[i_s[0]+1], edgecolor = sector_colors[i_s[0]+1], alpha=0.6)
                    ax[row].plot([d1,d2],[float(BrightBands_w_NARR_WNW[d,7]), float(BrightBands_w_NARR_WNW[d,7])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                    ax[row].plot([d1,d2],[float(BrightBands_w_NARR_WNW[d,8]), float(BrightBands_w_NARR_WNW[d,8])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)

    if 'WSW' in plot_list:
        row = [i for i in range(nrows) if plot_list[i] == 'WSW'][0]
        #print("WSW", row)
        if n == 0:
            xdatesSecondary = xdatesSecondary_WSW[:]
            upper_limit = upper_limit_WSW[:]
            lower_limit = lower_limit_WSW[:]
            n += 1
        else:
            xdatesSecondary = np.vstack([xdatesSecondary,xdatesSecondary_WSW])
            upper_limit = np.vstack([upper_limit, upper_limit_WSW])
            lower_limit = np.vstack([lower_limit, lower_limit_WSW])
            n += 1

        i_s = [i for i in range(len(sectors)) if sectors[i] == 'WSW']
        plot_colors.append(sector_colors[i_s[0]+1])
        plot_colors_dark.append(sector_colors_dark[i_s[0]])
        plot_names.append(sector_names[i_s[0]])
        plot_abr.append(sectors[i_s[0]])

        for d in range(1,len(BrightBands_w_NARR_WSW[:,7])-1):
            if np.isnan(float(BrightBands_w_NARR_WSW[d-1,7])) and np.isnan(float(BrightBands_w_NARR_WSW[d+1,7])) and ~np.isnan(float(BrightBands_w_NARR_WSW[d,7])):
                dt = datetime.datetime.strptime(BrightBands_w_NARR_WSW[d,6], '%m/%d/%y %H:%M:%S')
                d1 = dates.date2num([dt-timedelta(minutes = 5)])[0]
                d2 = dates.date2num([dt+timedelta(minutes = 5)])[0]
                if nrows == 1:
                    ax.fill_between(np.array([d1,d2]), float(BrightBands_w_NARR_WSW[d,7]), float(BrightBands_w_NARR_WSW[d,8]), facecolor=sector_colors[i_s[0]+1], edgecolor = sector_colors[i_s[0]+1], alpha=0.6)
                    ax.plot([d1,d2],[float(BrightBands_w_NARR_WSW[d,7]), float(BrightBands_w_NARR_WSW[d,7])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                    ax.plot([d1,d2],[float(BrightBands_w_NARR_WSW[d,8]), float(BrightBands_w_NARR_WSW[d,8])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                else:
                    ax[row].fill_between(np.array([d1,d2]), float(BrightBands_w_NARR_WSW[d,7]), float(BrightBands_w_NARR_WSW[d,8]), facecolor=sector_colors[i_s[0]+1], edgecolor = sector_colors[i_s[0]+1], alpha=0.6)
                    ax[row].plot([d1,d2],[float(BrightBands_w_NARR_WSW[d,7]), float(BrightBands_w_NARR_WSW[d,7])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                    ax[row].plot([d1,d2],[float(BrightBands_w_NARR_WSW[d,8]), float(BrightBands_w_NARR_WSW[d,8])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)

    if 'SW' in plot_list:
        row = [i for i in range(nrows) if plot_list[i] == 'SW'][0]
        #print("SW", row)
        if n == 0:
            xdatesSecondary = xdatesSecondary_SW[:]
            upper_limit = upper_limit_SW[:]
            lower_limit = lower_limit_SW[:]
            n += 1
        else:
            xdatesSecondary = np.vstack([xdatesSecondary,xdatesSecondary_SW])
            upper_limit = np.vstack([upper_limit, upper_limit_SW])
            lower_limit = np.vstack([lower_limit, lower_limit_SW])
            n += 1

        i_s = [i for i in range(len(sectors)) if sectors[i] == 'SW']
        plot_colors.append(sector_colors[i_s[0]+1])
        plot_colors_dark.append(sector_colors_dark[i_s[0]])
        plot_names.append(sector_names[i_s[0]])
        plot_abr.append(sectors[i_s[0]])

        for d in range(1,len(BrightBands_w_NARR_SW[:,7])-1):
            if np.isnan(float(BrightBands_w_NARR_SW[d-1,7])) and np.isnan(float(BrightBands_w_NARR_SW[d+1,7])) and ~np.isnan(float(BrightBands_w_NARR_SW[d,7])):
                dt = datetime.datetime.strptime(BrightBands_w_NARR_SW[d,6], '%m/%d/%y %H:%M:%S')
                d1 = dates.date2num([dt-timedelta(minutes = 5)])[0]
                d2 = dates.date2num([dt+timedelta(minutes = 5)])[0]
                if nrows == 1:
                    ax.fill_between(np.array([d1,d2]), float(BrightBands_w_NARR_SW[d,7]), float(BrightBands_w_NARR_SW[d,8]), facecolor=sector_colors[i_s[0]+1], edgecolor = sector_colors[i_s[0]+1], alpha=0.6)
                    ax.plot([d1,d2],[float(BrightBands_w_NARR_SW[d,7]), float(BrightBands_w_NARR_SW[d,7])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                    ax.plot([d1,d2],[float(BrightBands_w_NARR_SW[d,8]), float(BrightBands_w_NARR_SW[d,8])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                else:
                    ax[row].fill_between(np.array([d1,d2]), float(BrightBands_w_NARR_SW[d,7]), float(BrightBands_w_NARR_SW[d,8]), facecolor=sector_colors[i_s[0]+1], edgecolor = sector_colors[i_s[0]+1], alpha=0.6)
                    ax[row].plot([d1,d2],[float(BrightBands_w_NARR_SW[d,7]), float(BrightBands_w_NARR_SW[d,7])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)
                    ax[row].plot([d1,d2],[float(BrightBands_w_NARR_SW[d,8]), float(BrightBands_w_NARR_SW[d,8])], color = sector_colors_dark[i_s[0]], linestyle = '-', linewidth = 1.0)


    if nrows == 1:
        ax.fill_between(xdatesSecondary, upper_limit,lower_limit, facecolor=plot_colors[0], edgecolor = plot_colors[0], alpha=0.6)
        ax.plot(xdatesSecondary,lower_limit, color = plot_colors_dark[0], linestyle = '-', linewidth = 1.0)
        ax.plot(xdatesSecondary,upper_limit, color = plot_colors_dark[0],linestyle = '-', linewidth = 1.0)
        #ax.text(0.99, 0.93, str(plot_names[0]), verticalalignment='top', horizontalalignment='right',transform=ax.transAxes, color='black', fontsize=8,bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
        ax.set_xlim([min_x, max_x])
        custom_lines.append(Line2D([0], [0], color=plot_colors[0], linestyle='-', linewidth = 2.0))
        custom_labels.append(''.join(['Secondary Enhancement']))
    else:
        for r in range(0,nrows):
            ax[r].fill_between(xdatesSecondary[r], upper_limit[r],lower_limit[r], facecolor=plot_colors[r], edgecolor = plot_colors[r], alpha=0.6)
            ax[r].plot(xdatesSecondary[r],lower_limit[r], color = plot_colors_dark[r], linestyle = '-', linewidth = 1.0)
            ax[r].plot(xdatesSecondary[r],upper_limit[r], color = plot_colors_dark[r],linestyle = '-', linewidth = 1.0)
            ax[r].text(0.99, 0.93, str(plot_names[r]), verticalalignment='top', horizontalalignment='right',transform=ax[r].transAxes, color='black', fontsize=8,bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
            ax[r].set_xlim([min_x, max_x])
            custom_lines.append(Line2D([0], [0], color=plot_colors[r], linestyle='-', linewidth = 2.0))
            custom_labels.append(''.join(['Secondary Enhancement - ', str(plot_abr[r])]))

if plot_Citation:
    for i in range(0,len(flight_starts)):
        if flight_starts[i] >= start_date and flight_ends[i] <= end_date:
            x1 = dates.date2num(flight_starts[i])
            x2 = dates.date2num(flight_ends[i])
            if nrows == 1:
                ax.axvspan(x1, x2, alpha=0.4, color='lightgrey')
            else:
                for row in range(nrows):
                    ax[row].axvspan(x1, x2, alpha=0.4, color='lightgrey')
        elif flight_starts[i] < start_date and flight_ends[i] >= start_date and flight_ends[i] <= end_date:
            x1 = dates.date2num(start_date)
            x2 = dates.date2num(flight_ends[i])
            if nrows == 1:
                ax.axvspan(x1, x2, alpha=0.4, color='lightgrey')
            else:
                for row in range(nrows):
                    ax[row].axvspan(x1, x2, alpha=0.4, color='lightgrey')
        elif flight_starts[i] >= start_date and flight_starts[i] <= end_date and flight_ends[i] > end_date:
            x1 = dates.date2num(flight_starts[i])
            x2 = dates.date2num(end_date)
            if nrows == 1:
                ax.axvspan(x1, x2, alpha=0.4, color='lightgrey')
            else:
                for row in range(nrows):
                    ax[row].axvspan(x1, x2, alpha=0.4, color='lightgrey')


#x_zero = dates.date2num(nov13_zero)

x_st = dates.date2num(date_start)
x_end = dates.date2num(date_end)

#ax.xticks(xdatesNPOL,BrightBands_w_NARR[:,0])
date_start_fixed = ''.join([str(day_start),'/',str(mon_start),'/2015 00:00:00'])
date_start_fixed = datetime.datetime.strptime(date_start_fixed, "%d/%m/%Y %H:%M:%S")
x_zero = []
for d in range(0,delta.days):
    if d == 0:
        date_plus = date_start_fixed + timedelta(days=1)
    else:
        date_plus = date_plus + timedelta(days = 1)
    x_zero.append(dates.date2num(date_plus))

if delta.days < 1:
    hours = dates.HourLocator(interval = 1)
    if nrows == 1:
        ax.set_xlim([x_st, x_end])
        for v in range(0,len(x_zero)):
            ax.vlines(x = x_zero[v], ymin = 0, ymax = 6.5, color = 'darkgrey')
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt_short)
    else:
        for row in range(nrows):
            ax[row].xaxis.set_major_locator(hours)
            ax[row].xaxis.set_major_formatter(h_fmt)
elif delta.days < 2:
    hours = dates.HourLocator(interval = 2)
    if nrows == 1:
        ax.set_xlim([x_st, x_end])
        for v in range(0,len(x_zero)):
            ax.vlines(x = x_zero[v], ymin = 0, ymax = 6.5, color = 'darkgrey')
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt_short)
    else:
        for row in range(nrows):
            ax[row].xaxis.set_major_locator(hours)
            ax[row].xaxis.set_major_formatter(h_fmt)
elif 2 <= delta.days <= 10:
    days = dates.DayLocator(interval = 1)
    if nrows == 1:
        ax.set_xlim([x_st, x_end])
        for v in range(0,len(x_zero)):
            ax.vlines(x = x_zero[v], ymin = 0, ymax = 6.5, color = 'darkgrey')
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(d_fmt)
    else:
        for row in range(nrows):
            ax[row].xaxis.set_major_locator(days)
            ax[row].xaxis.set_major_formatter(d_fmt)
else:
    if nrows == 1:
        ax.set_xlim([x_st, x_end])
        for v in range(0,len(x_zero)):
            ax.vlines(x = x_zero[v], ymin = 0, ymax = 6.5, color = 'darkgrey')
        ax.xaxis.set_major_locator(days)
        ax.xaxis.set_major_formatter(d_fmt)
    else:
        for row in range(nrows):
            ax[row].xaxis.set_major_locator(days)
            ax[row].xaxis.set_major_formatter(d_fmt)

if plot_Secondary:
    if nrows == 1:
        ax.set_ylim([0.25,6.5])
        ax.set_yticks(np.arange(1.0, 7.0, step=1.0))
        ax.set_yticklabels(['1.0', '2.0', '3.0', '4.0', '5.0', '6.0'], fontsize=8)
    else:
        for row in range(nrows):
            ax[row].set_ylim([0.25,6.5])
            ax[row].set_yticks(np.arange(1.0, 7.0, step=2.0))
            ax[row].set_yticklabels(['1.0', '3.0', '5.0'], fontsize=8)
else:
    if nrows == 1:
        ax.set_ylim([0.25,3.5])
    else:
        for row in range(nrows):
            ax[row].set_ylim([0.25,3.5])



for row in range(nrows):
    if nrows == 1:
        ax.grid(True, linestyle = '--', linewidth = 0.5)
        ax.tick_params(axis="x",direction="in")
        ax.tick_params(axis="y",direction="in")
    else:
        ax[row].grid(True, linestyle = '--', linewidth = 0.5)
        ax[row].tick_params(axis="x",direction="in")
        ax[row].tick_params(axis="y",direction="in")
    if row != nrows-1 and nrows != 1:
        labels = [item.get_text() for item in ax[row].get_xticklabels()]
        empty_string_labels = ['']*len(labels)
        ax[row].set_xticklabels(empty_string_labels)

#ax[2].set_ylabel('Height (km)')
#ax[0].set_ylabel(' ')

ylab = fig.text(0.04, 0.57, 'Height (km)', va='center', ha = 'left', rotation='vertical', transform=ax_big.transAxes)
#xlab_1 = fig.text(0.22, 0.04, 'Nov 12', va='center', ha = 'center', rotation='horizontal', transform=ax_big.transAxes)
#xlab_2 = fig.text(0.65, 0.04, 'Nov 13', va='center', ha = 'center', rotation='horizontal', transform=ax_big.transAxes)

title = ''.join([str(mon_start),'-',str(day_start),' to ',str(mon_end),'-',str(day_end)])
if nrows == 1:
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.xaxis.set_tick_params(labelsize=8)
    #ax.set_title(title)
else:
    plt.setp(ax[nrows-1].get_xticklabels(), rotation=60, ha='right')
    ax[nrows-1].xaxis.set_tick_params(labelsize=8)
    ax[0].set_title(title)

#plt.legend(custom_lines, custom_labels,loc = 'lower right', ncol = 1, fontsize = 10, borderaxespad = 0.6)
lgd = fig.legend(custom_lines, custom_labels, bbox_to_anchor=(1.04,0.57), bbox_transform = ax_big.transAxes, loc="center left", frameon = False)

#lgd = ax[0].legend(custom_lines, custom_labels, bbox_to_anchor=(1.04,0.5), loc="center left", frameon = False)
fig.tight_layout()
fig.subplots_adjust(hspace = 0.05)

#plt.show()
plt.savefig(save_fn_fig,bbox_extra_artists=(lgd,ylab), bbox_inches='tight', dpi = 300) #use this one for normal save
#plt.savefig(save_fn_fig,bbox_extra_artists=(ylab,xlab_1,xlab_2), bbox_inches='tight', dpi = 300) #specific to the Nov12-13 case for Lynn
plt.close()


"""
PLOT HISTOGRAMS
"""

secondary_vals_NE = [float(i) for i in secondary_vals_NE if ~np.isnan(np.float64(i))]
secondary_vals_SW = [float(i) for i in secondary_vals_SW if ~np.isnan(np.float64(i))]
secondary_vals_WSW = [float(i) for i in secondary_vals_WSW if ~np.isnan(np.float64(i))]
secondary_vals_WNW = [float(i) for i in secondary_vals_WNW if ~np.isnan(np.float64(i))]
secondary_vals_NW = [float(i) for i in secondary_vals_NW if ~np.isnan(np.float64(i))]

mu_NE = np.nanmean(secondary_vals_NE)
#std_east = np.nanstd(secondary_vals_east)
n_NE = len(secondary_vals_NE)
mu_SW = np.nanmean(secondary_vals_SW)
#std_west = np.nanstd(secondary_vals_west)
n_SW = len(secondary_vals_SW)
mu_WSW = np.nanmean(secondary_vals_WSW)
n_WSW = len(secondary_vals_WSW)
mu_WNW = np.nanmean(secondary_vals_WNW)
n_WNW = len(secondary_vals_WNW)
mu_NW = np.nanmean(secondary_vals_NW)
n_NW = len(secondary_vals_NW)

fig, ax = plt.subplots()

if 'NE' in plot_list:
    ax.hist(secondary_vals_NE, num_bins, normed=True, facecolor=sector_colors[0+1], alpha=0.2)
    fit_shape_NE, fit_loc_NE, fit_scale_NE = skewnorm.fit(secondary_vals_NE)
    xmin_NE, xmax_NE = np.nanmin(secondary_vals_NE), np.nanmax(secondary_vals_NE)
    x_NE = np.linspace(xmin_NE, xmax_NE, 100)
    p_NE = skewnorm.pdf(x_NE, fit_shape_NE, fit_loc_NE, fit_scale_NE)
    ax.plot(x_NE, p_NE, sector_colors[0+1], linewidth=3, linestyle = '-', alpha = 0.8, label = ''.join(['NORTHEAST, n = ',str(times_NE_sec),' (',str(times_NE_bb),')']))

if 'NW' in plot_list:
    ax.hist(secondary_vals_NW, num_bins, normed=True, facecolor=sector_colors[4+1], alpha=0.2)
    fit_shape_NW, fit_loc_NW, fit_scale_NW = skewnorm.fit(secondary_vals_NW)
    xmin_NW, xmax_NW = np.nanmin(secondary_vals_NW), np.nanmax(secondary_vals_NW)
    x_NW = np.linspace(xmin_NW, xmax_NW, 100)
    p_NW = skewnorm.pdf(x_NW, fit_shape_NW, fit_loc_NW, fit_scale_NW)
    ax.plot(x_NW, p_NW, sector_colors[1+1], linewidth=3, linestyle = '-', alpha = 0.8, label = ''.join(['NORTHWEST, n = ',str(times_NW_sec),' (',str(times_NW_bb),')']))

if 'WNW' in plot_list:
    ax.hist(secondary_vals_WNW, num_bins, normed=True, facecolor=sector_colors[3+1], alpha=0.2)
    fit_shape_WNW, fit_loc_WNW, fit_scale_WNW = skewnorm.fit(secondary_vals_WNW)
    xmin_WNW, xmax_WNW = np.nanmin(secondary_vals_WNW), np.nanmax(secondary_vals_WNW)
    x_WNW = np.linspace(xmin_WNW, xmax_WNW, 100)
    p_WNW = skewnorm.pdf(x_WNW, fit_shape_WNW, fit_loc_WNW, fit_scale_WNW)
    ax.plot(x_WNW, p_WNW, sector_colors[2+1], linewidth=3, linestyle = '-', alpha = 0.8, label = ''.join(['WEST-NORTHWEST, n = ',str(times_WNW_sec),' (',str(times_WNW_bb),')']))

if 'WSW' in plot_list:
    ax.hist(secondary_vals_WSW, num_bins, normed=True, facecolor=sector_colors[2+1], alpha=0.2)
    fit_shape_WSW, fit_loc_WSW, fit_scale_WSW = skewnorm.fit(secondary_vals_WSW)
    xmin_WSW, xmax_WSW = np.nanmin(secondary_vals_WSW), np.nanmax(secondary_vals_WSW)
    x_WSW = np.linspace(xmin_WSW, xmax_WSW, 100)
    p_WSW = skewnorm.pdf(x_WSW, fit_shape_WSW, fit_loc_WSW, fit_scale_WSW)
    ax.plot(x_WSW, p_WSW, sector_colors[3+1], linewidth=3, linestyle = '-', alpha = 0.8, label = ''.join(['WEST-SOUTHWEST, n = ',str(times_WSW_sec),' (',str(times_WSW_bb),')']))

if 'SW' in plot_list:
    ax.hist(secondary_vals_SW, num_bins, normed=True, facecolor=sector_colors[1+1], alpha=0.2)
    fit_shape_SW, fit_loc_SW, fit_scale_SW = skewnorm.fit(secondary_vals_SW)
    xmin_SW, xmax_SW = np.nanmin(secondary_vals_SW), np.nanmax(secondary_vals_SW)
    x_SW = np.linspace(xmin_SW, xmax_SW, 100)
    p_SW = skewnorm.pdf(x_SW, fit_shape_SW, fit_loc_SW, fit_scale_SW)
    ax.plot(x_SW, p_SW, sector_colors[4+1], linewidth=3, linestyle = '-', alpha = 0.8, label = ''.join(['SOUTHWEST, n = ',str(times_SW_sec),' (',str(times_SW_bb),')']))


ax.set_xlim((0,45))
ax.set_xlabel('Reflectivity (dBZ)')
ax.set_ylabel('Probability')

title = ''.join([str(mon_start),'-',str(day_start),' to ',str(mon_end),'-',str(day_end)])
plt.title(title)
lgd = ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", frameon = False, labelspacing=.3)
#lgd = ax.legend(bbox_to_anchor=(0.02,0.98), loc="upper left", frameon = False, fontsize = 8)
#plt.legend()
#plt.show()
plt.savefig(save_fn_dist_fig, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi = 300)
plt.close()
