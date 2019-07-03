#Andrew DeLaFrance
#June 28

#algorithm aimed at identifying the secondary enhancement above the melging layer
#ingests melting layer data identified by BBIDv6

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
import sys
import math
from scipy import stats

startTime = datetime.datetime.now()

'''
Thresholds and variables
'''

nodes = 4 #how many processors to run from

dir = 'east' #lowercase

excd_prcnt = 25 #percent value that must be exceeded over the slope fit of the grid cells above the bright band

secondary_crit = 15 #percentage of cells that must meet criteria in order to say a secondary enhancement is found

n_levels_allowed = 2 #number of levels allowed to select from above the mode (each level is 0.5km) fixed at 1 level below

rhohv_min = 0.91
rhohv_max = 0.97

max_ht = 7 #km - max height allowd for a mode layer to be selected

#spatial domain in radius from the radar
small_rad_dim = 10.0 #radius to restrict to away from the radar (changing these requires recalculating n_total cells)
big_rad_dim = 60.0 #outer bounds of the radar scan, beyond 60 beam smoothing becomes an issue

'''
File input/ouput - data organization
'''

rhi_dir = '/home/disk/bob/olympex/zebra/moments/npol_qc2/rhi/' #base directory for gridded RHI's
bb_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/BrightBands/' #output directory for saved images
save_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/Secondary/' #output directory for saved images
data_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Data/' #directory for local data

save_fn_data = ''.join(['secondary_',str(secondary_crit),'X',str(excd_prcnt),'excd_',dir,'.npy'])
save_fn_data_csv = ''.join(['secondary_',str(secondary_crit),'X',str(excd_prcnt),'excd_',dir,'.csv'])

#load latest bright band data from BBIDv6
if dir == 'east':
    bb_data = ''.join(['brightbandsfound_v6_r_6_time0x35.0pcntx25.0_withrhohv_0.910.97_',dir,'.npy'])
elif dir == 'west':
    bb_data = ''.join(['brightbandsfound_v6_r_6_time0x15.0pcntx25.0_withrhohv_0.910.97_',dir,'.npy'])

bb_fn = ''.join([bb_dir,bb_data])
bright_bands = np.load(bb_fn)#time,bbfound,level, ...

#load in date_list and file_list from BBIDv6
date_list_fn = ''.join([data_dir,'date_list.npy'])
filelist_fn = ''.join([data_dir,'filelist.npy'])
date_list = np.load(date_list_fn)
filelist = np.load(filelist_fn)
numfiles = len(filelist)
days_in_series = len(date_list)
days_out = np.concatenate((np.arange(12,31),np.arange(1,20)))

day_time_array = np.zeros([days_in_series,24])
day_time_array[:,:] = -1#default to clear sky

secondary = np.array([1,2,3,4,5,6,7,8,9,10]) #columns = date, anything above?, lower level of enhancement,
#upper level of enhancement, mean height of enhancement,pecent cells met, period mode

'''
Main functions
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

def main_func(i):
    date = ''.join([filelist[i].split(".")[1],'/'])
    file = filelist[i].split(".")[2]
    date_folder = ''.join([rhi_dir,date])
    file_path = ''.join([date_folder,filelist[i]])
    outname = ''.join([date,file])
    print(''.join(['Worker starting: ',outname]))

    #pull out NetCDF data
    nc_fid = Dataset(file_path, 'r') #r means read only, cant modify data
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

    bb_index = np.where(bright_bands[:,0]==date.strftime("%m/%d/%y %H:%M:%S"))[0][0]

    if bright_bands[bb_index,1] ==  '1': #there was a bright band found from bbidv6

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

        min_rhohv_lev_for_mode = np.full([y_dim, x_dim],float('NaN'))
        max_dBZ_lev_for_mode = np.full([y_dim, x_dim],float('NaN'))

        bb_ht = (bright_bands[bb_index,2])
        bb_lev = np.int(np.round(np.float64(bb_ht) * 2,0)) #nearest level
        bb_lev_up = np.int(math.ceil(np.float64(bb_ht) * 2))+1# Rounds up to next level, then adds an additional layer above the bright band.

        #Look through to find out where the maximum level is occurring at each grid point
        for x_ind in range(x_dim):
            for y_ind in range(y_dim):

                if np.isnan(rhohv[0,:,y_ind,x_ind]).all():
                    min_rhohv_lev_for_mode[y_ind,x_ind] = float('NaN')
                else:
                    if ~np.isnan(rhohv[0,bb_lev_up:z_dim+1,y_ind,x_ind]).all():
                        #make a copy of this column
                        rhohv_copy = np.copy(rhohv[0,:,y_ind,x_ind])
                        for i in range(0,z_dim): #set values outside allowed range to NaN, the long way to avoid warnings when computing NaN in bool statement
                            if ~np.isnan(rhohv_copy[i]):
                                if rhohv_copy[i] < rhohv_min or rhohv_copy[i] > rhohv_max:
                                    rhohv_copy[i] = float('NaN')
                        #rhohv_copy[rhohv_copy<rhohv_min] = float('NaN')
                        #rhohv_copy[rhohv_copy>rhohv_max] = float('NaN')
                        rhohv_copy[0:bb_lev_up+1] = float('NaN')
                        if np.isnan(rhohv_copy).all():
                            min_rhohv_lev_for_mode[y_ind,x_ind] = float('NaN')
                        elif (np.where(rhohv_copy == np.nanmin(rhohv_copy))[0][0]).size > 1:
                            min_rhohv_lev_for_mode[y_ind,x_ind] = np.nanmax(np.where(rhohv_copy == np.nanmin(rhohv_copy))[0][0])
                        else:
                            min_rhohv_lev_for_mode[y_ind,x_ind] = np.where(rhohv_copy == np.nanmin(rhohv_copy))[0][0]
                    else:
                        min_rhohv_lev_for_mode[y_ind,x_ind] = float('NaN')

                if np.isnan(dBZ[0,:,y_ind,x_ind]).all():
                    max_dBZ_lev_for_mode[y_ind,x_ind] = float('NaN')
                else:
                    if ~np.isnan(dBZ[0,bb_lev_up:z_dim+1,y_ind,x_ind]).all():
                        dBZ_copy = np.copy(dBZ[0,:,y_ind,x_ind])
                        dBZ_copy[0:bb_lev_up+1] = float('NaN')
                        if np.isnan(dBZ_copy).all():
                            max_dBZ_lev_for_mode[y_ind,x_ind] = float('NaN')
                        elif (np.where(dBZ_copy == np.nanmax(dBZ_copy))[0][0]).size > 1:
                            max_dBZ_lev_for_mode[y_ind,x_ind] = np.nanmax(np.where(dBZ_copy == np.nanmax(dBZ_copy))[0][0])
                        else:
                            max_dBZ_lev_for_mode[y_ind,x_ind] = np.where(dBZ_copy == np.nanmax(dBZ_copy))[0][0]
                    else:
                        max_dBZ_lev_for_mode[y_ind,x_ind] = float('NaN')

        #..., and calculate the modes (most occurring level)
        period_mode_rhohv = mode1(min_rhohv_lev_for_mode)[0]
        if (period_mode_rhohv.size > 1):
            period_mode_rhohv = period_mode_rhohv[period_mode_rhohv.size-1]#use the highest layer when more than one meets criteria
        if ~np.isnan(period_mode_rhohv):
            period_mode_rhohv = np.int(period_mode_rhohv)

        period_mode_dBZ = mode1(max_dBZ_lev_for_mode)[0]
        if (period_mode_dBZ.size > 1):
            period_mode_dBZ = period_mode_dBZ[period_mode_dBZ.size-1]#use the highest layer when more than one meets criteria
        if ~np.isnan(period_mode_dBZ):
            period_mode_dBZ = np.int(period_mode_dBZ)

        #when to use which mode??
        if (period_mode_rhohv*0.5) >= max_ht:
            period_mode = period_mode_dBZ
        elif (period_mode_dBZ*0.5) >= max_ht:
            period_mode = period_mode_rhohv
        else:
            period_mode = period_mode_rhohv

        dBZ_means = np.full(z_dim, float('NaN'))
        for i in range(z_dim):
            if ~np.isnan(dBZ[0,i,:,:]).all():
                dBZ_means[i] = np.nanmean(dBZ[0,i,:,:])

        #sort out slope to mean values
        if ~np.isnan(dBZ_means[bb_lev_up:z_dim+1]).all():
            where_nan = np.argwhere(~np.isnan(dBZ_means[:]))
            top_lev = np.max(where_nan)
            zabovebb = (np.arange(bb_lev_up, top_lev+1))*z_spacing
            dbzs2fit = dBZ_means[bb_lev_up:top_lev+1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(zabovebb,dbzs2fit)
            fit_dBZ_line = np.concatenate(((np.full(bb_lev_up,float('NaN'))),np.asarray(slope*zabovebb+intercept)),axis = None)

        n_found = 0
        prcnt_cells_met = 0
        enhancement_found = 0
        peak_enhance_lev = []
        low_enhance_lev = []
        high_enhance_lev = []
        #search through each column looking for an enhancement aloft near the mode layer
        for x_ind in range(x_dim):
            for y_ind in range(y_dim):
                #which layers exceed the slope fit above the bright band layer?
                if period_mode == bb_lev_up:
                    exceed_levs = [z for z in range((period_mode),(period_mode+(n_levels_allowed)+1)) if dBZ[0,z,y_ind,x_ind]>(fit_dBZ_line[z]*(1+(excd_prcnt/100)))]
                else:
                    exceed_levs = [z for z in range((period_mode-1),(period_mode+(n_levels_allowed)+1)) if dBZ[0,z,y_ind,x_ind]>(fit_dBZ_line[z]*(1+(excd_prcnt/100)))]
                if len(exceed_levs) == 0:
                    low_enhance_lev.append(float('NaN'))
                    high_enhance_lev.append(float('NaN'))
                    peak_enhance_lev.append(float('NaN'))
                elif len(exceed_levs) == 1:
                    low_enhance_lev.append(exceed_levs[0])
                    high_enhance_lev.append(exceed_levs[0])
                    peak_enhance_lev.append(exceed_levs[0])
                    peak_diff = dBZ[0,exceed_levs,y_ind,x_ind]-fit_dBZ_line[exceed_levs]
                    n_found = n_found+1
                elif len(exceed_levs) > 1:
                    low_enhance_lev.append(np.nanmin(exceed_levs))
                    high_enhance_lev.append(np.nanmax(exceed_levs))
                    diffs = dBZ[0,0:top_lev+1,y_ind,x_ind]-fit_dBZ_line
                    peak_enhance_lev.append(exceed_levs[np.nanargmax(diffs[exceed_levs])])
                    peak_diff = np.nanmax(diffs[exceed_levs])
                    n_found = n_found+1

        if n_found>0:
            prcnt_cells_met = np.float64(format((n_found/n_total)*100,'.2f'))
            mean_low_enhance_lev = np.nanmean(low_enhance_lev)*0.5
            mean_high_enhance_lev = np.nanmean(high_enhance_lev)*0.5
            mean_peak_enhance_lev = np.nanmean(peak_enhance_lev)*0.5
        else:
            prcnt_cells_met = 0
            mean_low_enhance_lev = float('NaN')
            mean_high_enhance_lev = float('NaN')
            mean_peak_enhance_lev = float('NaN')
        if prcnt_cells_met>secondary_crit:
            enhancement_found = 1
        row_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S"),enhancement_found,mean_low_enhance_lev,mean_high_enhance_lev,mean_peak_enhance_lev,prcnt_cells_met,period_mode,period_mode_rhohv,period_mode_dBZ,bb_lev])
    else:
        enhancement_found = 0
        row_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S"),enhancement_found,float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN'),float('NaN')])
    #print(row_to_append)
    print(''.join(['Worker finished: ',outname]))
    return(row_to_append)


'''
Set up the parallel processing environment
'''

pool = mp.Pool(processes=nodes)
results = pool.map_async(main_func, range(numfiles))
secondary = np.vstack((secondary,results.get()))


np.save(save_fn_data,secondary)
pd.DataFrame(secondary).to_csv(save_fn_data_csv)


print("Total time:", datetime.datetime.now() - startTime)
