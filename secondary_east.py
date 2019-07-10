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

np.set_printoptions(threshold=sys.maxsize)

startTime = datetime.datetime.now()

'''
Thresholds and variables
'''

nodes = 4 #how many processors to run from

dir = 'east' #lowercase

excd_val = 0 #dBZ value to exceed

min_dBZ = 15

overlap = 20 # % overlap for cell in beam spread

grid_step = 20 #km
n_grids_needed = 1

##try to keep this one same as BBIDV6? 15/35
secondary_crit = 15 #percentage of cells that must meet criteria in order to say a secondary enhancement is found

min_sep = 0 #number of levels needed between bright band and region to look within

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

save_fn_data = ''.join([save_dir,'secondary_D_',str(secondary_crit),'X',str(excd_val),'excd_',dir,'.npy'])
save_fn_data_csv = ''.join([save_dir,'secondary_D_',str(secondary_crit),'X',str(excd_val),'excd_',dir,'.csv'])

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

secondary = np.array([1,2,3,4,5,6]) #columns = date, anything above?,
#mean height of enhancement,pecent cells met, bb height

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

        bb_ht = (bright_bands[bb_index,2])
        bb_lev = np.int(np.round(np.float64(bb_ht) * 2,0)) #nearest level
        bb_diff = np.float64(bb_ht) - (bb_lev * 0.5)

        #want a minimum amount of space between bright band and where to look to avoid noise and variability of bright band level
        if bb_diff > 0: #bright band is above nearest level
            bb_lev_up = np.int(bb_lev + 1 + min_sep)
        elif bb_diff <= 0:
            bb_lev_up = np.int(bb_lev + min_sep)

        #bb_lev_up = np.int(math.ceil(np.float64(bb_ht) * 2))+1# Rounds up to next level, then adds an additional layer above the bright band.

        dBZ_means = np.full(z_dim, float('NaN'))
        for i in range(z_dim):
            if ~np.isnan(dBZ[0,i,:,:]).all():
                dBZ_means[i] = np.nanmean(dBZ[0,i,:,:])

        #print(dBZ_means)
        where_nan = np.argwhere(~np.isnan(dBZ_means[:]))
        top_lev = np.max(where_nan)

        n_found = 0
        prcnt_cells_met = 0
        enhancement_found = 0
        low_enhance_lev = []
        high_enhance_lev = []
        peak_enhance_lev = []

        enhancement = np.full((y_dim,x_dim,2),float('NaN'))

        #search all of x, y cell by cell and put low and high enhancement levels in an array
        for x_ind in range(x_dim):
            for y_ind in range(y_dim):
                deltas = np.full(top_lev,float('NaN'))
                if ~np.isnan(dBZ[0,bb_lev:top_lev+1,y_ind,x_ind]).all():
                    for z in range(1,top_lev):
                        deltas[z] = ((dBZ[0,z,y_ind,x_ind]-dBZ[0,z-1,y_ind,x_ind]))
                    #above bright band, where does delta become positive
                    low_enhance = next((i for i, v in enumerate(deltas) if v > excd_val and i > bb_lev), float('NaN'))
                    high_enhance = next((i-1 for i, v in enumerate(deltas) if v < (-excd_val) and i > low_enhance), low_enhance)
                    enhancement[y_ind,x_ind,0] = low_enhance
                    enhancement[y_ind,x_ind,1] = high_enhance

        low_enhance_mode = np.int(mode1(enhancement[:,:,0])[0])
        high_enhance_mode = np.int(mode1(enhancement[:,:,1])[0])
        low_enhance_mean = np.float64(format(np.nanmean(enhancement[:,:,0])*0.5,'.2f'))
        high_enhance_mean = np.float64(format(np.nanmean(enhancement[:,:,1])*0.5,'.2f'))

        #do any grid boxes have an enhancement above threshold dBZ
        grid_list = np.arange(0,300,grid_step)
        for x in grid_list:
            for y in grid_list:
                dBZ_subset = dBZ[0,:,x:x+grid_step,y:y+grid_step]
                n_data = np.count_nonzero(~np.isnan(np.nanmax(dBZ_subset,axis = 0)))
                if n_data >= (grid_step*grid_step*(overlap/100)):
                    dBZ_subset_means = np.full(top_lev,float('NaN'))
                    deltas = np.full(top_lev,float('NaN'))
                    for z in range(0,top_lev):
                        dBZ_subset_means[z] = np.nanmean(dBZ_subset[z,:,:])
                        if z > 0:
                            deltas[z] = (dBZ_subset_means[z]-dBZ_subset_means[z-1])
                        #at this grid box, does a secondary enhancement exist
                        if low_enhance_mode <= z <= high_enhance_mode:
                            if deltas[z] > excd_val:
                                n_found += 1
                                enhancement_found = 1



        row_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S"),enhancement_found,low_enhance_mean,high_enhance_mean,n_found,bb_lev])
    else:
        enhancement_found = 0
        row_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S"),enhancement_found,float('NaN'),float('NaN'),float('NaN'),float('NaN')])

    #print(row_to_append)
    print(''.join(['Worker finished: ',outname]))
    return(row_to_append)


'''
Set up the parallel processing environment
'''

pool = mp.Pool(processes=nodes)
results = pool.map_async(main_func, range(numfiles))
secondary = np.vstack((secondary,results.get()))

#sort by NPOL date/time
secondary= secondary[secondary[:,0].argsort()]

np.save(save_fn_data,secondary)
pd.DataFrame(secondary).to_csv(save_fn_data_csv)


print("Total time:", datetime.datetime.now() - startTime)
