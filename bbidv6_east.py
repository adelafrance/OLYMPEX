#Andrew DeLaFrance
#10 Apr 2019
#Finding bright bands
#departs partly from methods of G&C used in bbidv2.py
#attempts to use dBZ, ZDR?, and rhohv incorporating some thresholds from Giagrande et al. 08

#June 18
#Works from bbidv5 but implements parallel procsessing

from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats
import datetime
import os
import pandas as pd
import multiprocessing as mp
from multiprocessing import Queue
import warnings
import sys
#np.set_printoptions(threshold=sys.maxsize)
startTime = datetime.datetime.now()

#######

'''
Setup input
'''

nodes = 16 #how many processors to run (623 computers seem to work well on 16, 24 was slower due to communication between computers)

dir = 'east' #look east or west (lowercase)

use_rhohv = True
use_ZDR = False
use_both = False
require_time_cont = True #require a bright band to be present for a continuous amount of time
require_height_cont = True

#spatial domain in radius from the radar
small_rad_dim = 10.0 #radius to restrict to away from the radar (changing these requires recalculating n_total cells)
big_rad_dim = 60.0 #outer bounds of the radar scan, beyond 60 beam smoothing becomes an issue

dBZ_exceed_val = 25.0 #threshold value that any vertical column for given x,y to be considered
min_ave_dBZ = 15.0 #threshold for whether or not to use second mode layer
bb_crit_1 =35.0 #percentage of cells that need to have a value above the exceed level within rhohv range
n_levels_allowed = 1 #number of levels allowed to select from above or below the mode (each level is 0.5km)

time_cont = 0 #hours of temporal continuity needed for a bright band to be stratiform

num_stds = 2.0 #standard deviations away from the mean for any time period of consecutive bbs
ht_exc = 0.75 #additional requirement on top of standard deviation,distance away from mean required to be removed
ht_max = 4 #maximum height in kilometers that a bright band can exist in
level_max = np.int(ht_max*2)
check_dBZ = 20.0

#rhohv and ZDR bounds
rhohv_min = 0.91
rhohv_max = 0.97
ZDR_min = 0.8
ZDR_max = 2.5

'''
File input/output organization/structure
'''

rhi_dir = '/home/disk/bob/olympex/zebra/moments/npol_qc2/rhi/' #base directory for gridded RHI's
save_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/BrightBands/' #output directory for saved images
data_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Data/' #directory for local data
if use_rhohv:
    save_name_fig = ''.join(['brightbandsfound_v6_r_6_time',str(time_cont),'x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withrhohv_',str(rhohv_min),str(rhohv_max),'_',dir,'.png'])
    save_name_data = ''.join(['brightbandsfound_v6_r_6_time',str(time_cont),'x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withrhohv_',str(rhohv_min),str(rhohv_max),'_',dir,'.npy'])
    save_name_data_csv = ''.join(['brightbandsfound_v6_r_6_time',str(time_cont),'x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withrhohv_',str(rhohv_min),str(rhohv_max),'_',dir,'.csv'])
elif use_ZDR:
    save_name_fig = ''.join(['brightbandsfound_v6_r_6_time',str(time_cont),'x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withZDR_',str(ZDR_min),'_',dir,'.png'])
    save_name_data = ''.join(['brightbandsfound_v6_r_6_time',str(time_cont),'x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withZDR_',str(ZDR_min),'_',dir,'.npy'])
    save_name_data_csv = ''.join(['brightbandsfound_v6_r_6_time',str(time_cont),'x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withZDR_',str(ZDR_min),'_',dir,'.csv'])
else:
    save_name_fig = ''.join(['brightbandsfound_v6_r_6_time',str(time_cont),'x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_',dir,'.png'])
    save_name_data = ''.join(['brightbandsfound_v6_r_6_time',str(time_cont),'x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_',dir,'.npy'])
    save_name_data_csv = ''.join(['brightbandsfound_v6_r_6_time',str(time_cont),'x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_',dir,'.csv'])

NARR_data = 'NARR_at_NPOL.csv'
save_fn_fig = ''.join([save_dir,save_name_fig])
save_fn_data = ''.join([save_dir,save_name_data])
save_fn_data_csv = ''.join([save_dir,save_name_data_csv])
NARR_fn = ''.join([data_dir,NARR_data])

#########
#create a list of dates Nov 12 - Dec 19
nov_start = '20151112/'
date_list = [nov_start]
fig_date_1 = 'Nov 12'
fig_date_list = [fig_date_1]
for d in range(13,27):
    s = ''
    date_to_append = s.join(['201511',str(d),'/'])
    fig_date_to_append = s.join(['Nov ',str(d)])
    date_list.append(date_to_append)
    fig_date_list.append(fig_date_to_append)
date_list.append('20151130/')
fig_date_list.append('Nov 30')
for d in range(1,20):
    s = ''
    if d < 10:
        date_to_append = s.join(['2015120',str(d),'/'])
        fig_date_to_append = s.join(['Dec 0',str(d)])
    else:
        date_to_append = s.join(['201512',str(d),'/'])
        fig_date_to_append = s.join(['Dec ',str(d)])
    date_list.append(date_to_append)
    fig_date_list.append(fig_date_to_append)

days_in_series = len(date_list)
days_out = np.concatenate((np.arange(12,31),np.arange(1,20)))

#initalize numpy array for bright band data
bright_bands = np.array([1,2,3,4,5,6,7,8])
#Columns => date/time,bright band found?, bright band melt level, NARR date, NARR melt level,
#percent above dBZ threshold, percent of cells met with polarmetric criteria

filelist = []
#create a list of files for parallel computing
for date in date_list:
    date_folder = ''.join([rhi_dir,date])
    for file in os.listdir(date_folder):
        if file.split(".")[3] == dir:
            filelist.append(file)

numfiles = len(filelist)

#bring in NARR data
df=pd.read_csv(NARR_fn, sep=',',header=None)
NARR_data = np.array(df) #NARR Time,IVT,Melting Level (m),925speed (kt),925dir,Nd,Nm
n_bbs = days_in_series
n_NARRs = NARR_data.shape[0]
items = []
for h in range(0,n_NARRs-1):
    items.append(datetime.datetime.strptime(NARR_data[h+1,0], "%Y-%m-%d %H:%M:%S"))
###############

'''
Script functions
'''

def mode1(x): #most commonly occurring level
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]

def mode2(x): #second most commonly occurring level
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    counts[m] = 0
    n = counts.argmax()
    return values[n], counts[n]
#set up for parallel loop through date folders
def main_func(i):
    date = ''.join([filelist[i].split(".")[1],'/'])
    file = filelist[i].split(".")[2]
    #date = date_list[i]
    date_folder = ''.join([rhi_dir,date])
    file_path = ''.join([date_folder,filelist[i]])
    outname = ''.join([date,file])
    print(''.join(['Worker starting: ',outname]))

    #pull out NetCDF data
    nc_fid = Dataset(file_path, 'r') #r means read only, cant modify data
    nc_attrs = nc_fid.ncattrs()
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    x_dim = nc_fid.dimensions['x'].size
    y_dim = nc_fid.dimensions['y'].size
    z_dim = nc_fid.dimensions['z'].size
    x_spacing = nc_fid.variables['x_spacing'][:]
    y_spacing = nc_fid.variables['y_spacing'][:]
    z_spacing = nc_fid.variables['z_spacing'][:]
    lon = nc_fid.variables['lon'][:]
    lat = nc_fid.variables['lat'][:]

    #set up elevation profile
    elev = np.arange(0,(z_dim*z_spacing),z_spacing)

    #get the time of scan
    date = num2date(nc_fid.variables['base_time'][:], units =nc_fid.variables['base_time'].units , calendar = 'standard')
    #Date broken down
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    hour = date.strftime("%H")
    minute = date.strftime("%M")
    second = date.strftime("%S")
    day_out = np.where(days_out == int(day))[0][0]
    #Reflectivity
    dBZ = np.array(nc_fid.variables['DBZ'][:,:,:,:])#dimesions [time, z, y, x] = [1,25,300,300]
    #Correlation Coefficient
    rhohv = np.array(nc_fid.variables['RHOHV'][:,:,:,:])#dimesions [time, z, y, x] = [1,25,300,300]
    #Differential Reflectivity
    ZDR = np.array(nc_fid.variables['ZDR'][:,:,:,:])#dimesions [time, z, y, x] = [1,25,300,300]
    #missing values is -9999.0, set to NaN
    dBZ[dBZ == -9999.0] = float('NaN')
    rhohv[rhohv == -9999.0] = float('NaN')
    ZDR[ZDR == -9999.0] = float('NaN')

    #set up empty array of grid y, x for top and bottom values of bb
    bb_melt_levs = np.full([y_dim, x_dim],float('NaN')) #empty array to hold levels that meet all specified polarmetric criteria
    rhohv_vals = np.full([y_dim, x_dim],float('NaN')) #values of rhohv if satisfied by bounds
    ZDR_vals = np.full([y_dim, x_dim],float('NaN')) #values of ZDR if satisfied by bounds
    min_rhohv_lev_for_mode = np.full([y_dim, x_dim],float('NaN'))

    #First, limit to values within small to big radius ranges
    for z_ind in range(z_dim):
        for x_ind in range(x_dim):
            x_dist = abs(x_ind - 150) #shifting over to the center point of 150 km
            for y_ind in range(y_dim):
                y_dist = abs(y_ind - 150) #shifting over to the center point of 150 km
                #if within the small radius or outside the big radius, set to NaN
                dist = ((x_dist**2)+(y_dist**2))**(0.5) #distance
                if dist < small_rad_dim or dist > big_rad_dim:
                    dBZ[0,z_ind, y_ind, x_ind] = float('NaN')
                    rhohv[0,z_ind, y_ind, x_ind] = float('NaN')
                    ZDR[0,z_ind, y_ind, x_ind] = float('NaN')

    if dir == 'east':
        n_total = 965 #applicable for 10-60km domain
    elif dir == 'west':
        n_total = 3525 #applicable for 10-60km domain

    #Look through to find out where the maximum level is occurring at each grid point
    for x_ind in range(x_dim):
        for y_ind in range(y_dim):
            if np.isnan(rhohv[0,:,y_ind,x_ind]).all():
                min_rhohv_lev_for_mode[y_ind,x_ind] = float('NaN')
            else:
                if np.nanmin(rhohv[0,:,y_ind,x_ind]) <= rhohv_max:
                    #make a copy of this column
                    rhohv_copy = rhohv[0,:,y_ind,x_ind]
                    rhohv_copy[rhohv_copy<rhohv_min] = float('NaN')
                    rhohv_copy[rhohv_copy>rhohv_max] = float('NaN')
                    if np.isnan(rhohv_copy).all():
                        min_rhohv_lev_for_mode[y_ind,x_ind] = float('NaN')
                    elif (np.where(rhohv_copy == np.nanmin(rhohv_copy))[0][0]).size > 1:
                        min_rhohv_lev_for_mode[y_ind,x_ind] = np.nanmax(np.where(rhohv_copy == np.nanmin(rhohv_copy))[0][0])
                    else:
                        min_rhohv_lev_for_mode[y_ind,x_ind] = np.where(rhohv_copy == np.nanmin(rhohv_copy))[0][0]
                else:
                    min_rhohv_lev_for_mode[y_ind,x_ind] = float('NaN')
    #..., and calculate the modes (most occurring level)
    period_mode = mode1(min_rhohv_lev_for_mode)[0]
    if (period_mode.size > 1):
        period_mode = period_mode[period_mode.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode):
        period_mode = np.int(period_mode)

    period_mode_2 = mode2(min_rhohv_lev_for_mode)[0]
    if (period_mode_2.size > 1):
        period_mode_2 = period_mode_2[period_mode_2.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode_2):
        period_mode_2 = np.int(period_mode_2)

    dBZ_means = np.full(z_dim, float('NaN'))
    for i in range(z_dim):
        if ~np.isnan(dBZ[0,i,:,:]).all():
            dBZ_means[i] = np.nanmean(dBZ[0,i,:,:])

    if period_mode_2 > period_mode and dBZ_means[period_mode_2] >= min_ave_dBZ and (period_mode_2*0.5) < ht_max:
        period_mode = period_mode_2

    n_above_dBZ = 0 #for calculating percentages of cells meeting criteria
    n_matched = 0

    for x_ind in range(x_dim):
        for y_ind in range(y_dim):
            if np.isnan(dBZ[0,:,y_ind,x_ind]).all():
                max_col_dBZ = float('NaN')
            else:
                max_col_dBZ = np.nanmax(dBZ[0,:,y_ind,x_ind])

            isrhohv = True
            if np.isnan(rhohv[0,:,y_ind,x_ind]).all(): #check that there is data for each variable at each time
                isrhohv = False
            isZDR = True
            if np.isnan(ZDR[0,:,y_ind,x_ind]).all():
                isZDR = False

            if max_col_dBZ >= dBZ_exceed_val and isrhohv and isZDR and period_mode >= 0:
                n_above_dBZ = n_above_dBZ+1
                #restrict identification of max dBZ layer to be within bounds of rhohv and ZDR criteria
                dBZ_met1 = np.where(dBZ[0,:,y_ind,x_ind] >= dBZ_exceed_val)[0]
                if period_mode == 0:
                    dBZ_met = [x for x in dBZ_met1 if x in range(period_mode,(period_mode+(2*n_levels_allowed)+1))]
                else:
                    dBZ_met = [x for x in dBZ_met1 if x in range((period_mode-n_levels_allowed),(period_mode+(2*n_levels_allowed)+1))]
                matched_layer = [x for x in dBZ_met]
                bb_layer = float('NaN') #initializing nothing found yet

                if len(matched_layer) == 0:
                    pass
                elif len(matched_layer) == 1:
                    bb_layer = np.float64(matched_layer[0])
                    bb_melt_levs[y_ind,x_ind] = np.float64(bb_layer*0.5)#0.5km vertical resolution
                    n_matched = n_matched + 1
                else:
                    #sort out which of the matched layers is maximum in dBZ
                    if np.isnan(dBZ[0,matched_layer,y_ind,x_ind]).all():
                        bb_layer = float('NaN')
                    else:
                        bb_layer = np.float64(np.where(dBZ[0,:,y_ind,x_ind] == np.nanmax(dBZ[0,matched_layer,y_ind,x_ind]))[0][0])#the layer that has the max reflectivity
                    bb_melt_levs[y_ind,x_ind] = np.float64(bb_layer*0.5)#0.5km vertical resolution
                    n_matched = n_matched + 1

            #something in the reflectivity scan but none that exceed the reflectivity values
            elif max_col_dBZ < dBZ_exceed_val and max_col_dBZ > 0:
                pass
            else:
                dBZ[0,:,y_ind,x_ind] = float('NaN')
            #end the x,y loop

    #clean up numpy array of levels
    bb_melt_levs= bb_melt_levs[~np.isnan(bb_melt_levs)]
    ##########################
    if np.isnan(dBZ).all():#empty slice, clear sky conditions
        clear_sky = True
    else:
        clear_sky = False
        if len(bb_melt_levs)==0:
            bb_melt_levs_std = float('NaN')
            bb_melting_height = float('NaN')
        else:
            bb_melt_levs_std = np.float64(format(np.nanstd(bb_melt_levs), '.2f'))
            bb_melting_height = np.float(format(np.nanmean(bb_melt_levs), '.2f'))
    prcnt_cells_met = float(format(((n_matched/n_total)*100), '.2f'))
    prcnt_above_dBZ = float(format(((n_above_dBZ/n_total)*100), '.2f'))
    hour_out = int(hour)

    #####
    #find nearest NARR time and melt level
    pivot = date
    timedeltas = []
    for j in range(0,len(items)):
        timedeltas.append(np.abs(pivot-items[j]))
    min_index = timedeltas.index(np.min(timedeltas)) + 1 #closest time step
    closest_NARR_date = NARR_data[min_index,0]
    melt_layer = NARR_data[min_index,2]

    #does this satisfy the criterion for a bright band
    if clear_sky:
        bb_data_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S"),99,float('NaN'),closest_NARR_date,melt_layer,0,0,0])
    else:
        if prcnt_cells_met >= bb_crit_1: #does this meet bright band criteria with polarmetric criteria
            bb_data_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S"),1,bb_melting_height,closest_NARR_date,melt_layer,prcnt_above_dBZ, prcnt_cells_met,period_mode])
        elif prcnt_cells_met >= check_dBZ: #does this meet bright band criteria with dBZ only
            bb_data_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S"),2,bb_melting_height,closest_NARR_date,melt_layer,prcnt_above_dBZ, prcnt_cells_met,period_mode])
        else:#("layer does not meet either criteria")
            bb_data_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S"),0,bb_melting_height,closest_NARR_date,melt_layer,prcnt_above_dBZ, prcnt_cells_met,period_mode])
        del bb_melt_levs_std, bb_melt_levs
    #print(bb_data_to_append)
    print(''.join(['Worker finished: ',outname]))
    return(bb_data_to_append)


#####
'''
Set up the parallel processing environment
'''

pool = mp.Pool(processes=nodes)
results = pool.map_async(main_func, range(numfiles))
bright_bands = np.vstack((bright_bands,results.get()))

######

#sort by NPOL date/time
bright_bands= bright_bands[bright_bands[:,0].argsort()]

#create a day_time_array for plotting by hour
#empty array to hold output for whether bright band or not
day_time_array = np.zeros([days_in_series,24])
day_time_array[:,:] = -1#default to clear sky
ntimes = np.shape(bright_bands)[0]-1 #number of times evaluated
for i_day in range(0,days_in_series):
    date = date_list[i_day]
    month = int(date[-5:-3])
    day = int(date[-3:-1])
    for i_hour in range(24):
        bb_vals = []
        for bbtime in range(1,ntimes):#find all times that fall within that hour
            time = datetime.datetime.strptime(bright_bands[bbtime,0], "%m/%d/%y %H:%M:%S")
            hour = (int(time.strftime("%H")))
            if (int(time.strftime("%m")) == month and int(time.strftime("%d")) == day and hour == i_hour):
                bb_vals.append(bright_bands[bbtime,1])
        if '1' in bb_vals:
            day_time_array[i_day,i_hour] = 1
        elif all(x == 'nan' or x == '99' for x in bb_vals):
            day_time_array[i_day,i_hour] = -1
        elif any(y == '0' or y == '2' for y in bb_vals):
            day_time_array[i_day,i_hour] = 0
        else:
            day_time_array[i_day,i_hour] = -1

'''
If temporal continuity is required
'''

if require_time_cont:
    #assess temporal continuity > x hours for stratiform
    i_begin = 1 #start at 1 since row 0 is just placeholders for columns
    while (i_begin <= ntimes):
        if bright_bands[i_begin,1] == '1' or bright_bands[i_begin,1] == '2':#is a bright band layer
            #look for the end of contiuous bright bands found
            bb_remaining = bright_bands[i_begin:ntimes,1]
            i_end = i_begin + next((i for i, v in enumerate(bb_remaining) if v not in ['1','2']), ntimes)
            start_time = datetime.datetime.strptime(bright_bands[i_begin,0], "%m/%d/%y %H:%M:%S")
            if i_end < ntimes:
                end_time = datetime.datetime.strptime(bright_bands[i_end-1,0], "%m/%d/%y %H:%M:%S")
            else:
                end_time = datetime.datetime.strptime(bright_bands[ntimes,0], "%m/%d/%y %H:%M:%S")
            tdelta = end_time - start_time #outputs difference in seconds
            tdelta_hours = tdelta.seconds/3600 #3600 seconds in an hour
            if tdelta_hours > np.abs(time_cont):
                period_kept = True
                if time_cont > 0:
                    bright_bands[i_begin:i_end,1] = 1
            else:
                bright_bands[i_begin:i_end,1] = 3
                period_kept = False
            #have a period of consecutive bbs, check height continuity
            if require_height_cont and period_kept:
                vals = [float(x) for x in bright_bands[i_begin:i_end,2]]
                height_std = np.nanstd(vals)
                height_mean = np.nanmean(vals)
                for i_ht in range(i_begin,i_end):
                    #try local mean window for tossing out stray values
                    if i_ht<(ntimes-5) and i_ht >= 5:
                        local_set = [i for i, v in enumerate(bright_bands[i_ht-5:i_ht+5,2]) if v in ['1','2']]
                        local_mean = np.nanmean(local_set)
                        ht_diff = local_mean - float(bright_bands[i_ht,2])
                        if ht_diff > ht_exc:
                            bright_bands[i_ht,1] = 4
                        elif ht_diff > ht_exc:
                            bright_bands[i_ht,1] = 4
                print(start_time,end_time,height_std,height_mean)
            i_begin = i_end + 1
        else: #check the next one
            i_begin = i_begin + 1

    #update plotting array after temporal continuity assesment
    for i_day in range(0,days_in_series):
        date = date_list[i_day]
        month = int(date[-5:-3])
        day = int(date[-3:-1])
        for i_hour in range(0,24):
            if day_time_array[i_day,i_hour] == 1: #if a bright band layer was previously found, see if it should still be there
                #check against bright_bands array that has been filtered for temporal continuity
                bb_vals = []
                for bbtime in range(1,ntimes):#find all times that fall within that hour
                    time = datetime.datetime.strptime(bright_bands[bbtime,0], "%m/%d/%y %H:%M:%S")
                    hour = (int(time.strftime("%H")))
                    if (int(time.strftime("%m")) == month and int(time.strftime("%d")) == day and hour == i_hour):
                        bb_vals.append(bright_bands[bbtime,1])
                if '1' in bb_vals:
                    day_time_array[i_day,i_hour] = 1
                elif all(x == 'nan' or x == '99' for x in bb_vals):
                    day_time_array[i_day,i_hour] = -1
                elif any(y == '0' or y == '2' or y == '3' or y == '4' for y in bb_vals):
                    day_time_array[i_day,i_hour] = 0
                else:
                    day_time_array[i_day,i_hour] = -1

'''
Plotting and saving
'''
colors = ['#f0f0f0','#636363','#b3e2cd'] #,'#fdcdac'
cmap = matplotlib.colors.ListedColormap(colors)
fig, ax = plt.subplots()
im = ax.imshow(day_time_array.T, origin = 'lower',cmap=cmap)
ax.set_title(''.join(['OLYMPEX Bright Band Identification\nNPOL-',dir]))
ax.set_ylabel('Hour (UTC)')
ax.set_xticks(range(0,len(date_list)))
ax.set_yticks(range(0,24))
ax.set_xticklabels(fig_date_list,fontsize=8, rotation=90)
ax.set_yticklabels(range(0,24),fontsize=8)
ax.grid(True, linestyle = ':', linewidth = 0.5)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(im,cax = cax, cmap = cmap, ticks=[-1, 0, 1])
im.set_clim(-1.5,1.5)
cbar.ax.set_yticklabels(['OFF\nClear', 'No BB\nFound', 'BB\nFound'], fontsize = 8)  # vertically oriented colorbar
plt.savefig(save_fn_fig)
np.save(save_fn_data,bright_bands)
pd.DataFrame(bright_bands).to_csv(save_fn_data_csv)

print("Total time:", datetime.datetime.now() - startTime)
