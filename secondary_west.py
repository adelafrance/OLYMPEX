#Andrew DeLaFrance
#June 28

#algorithm aimed at identifying the secondary enhancement above the melging layer
#ingests melting layer data identified by BBIDv6 and pulls in -15c height from NPOL - for plotting reference

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

nodes = 16 #how many processors to run from

plot_figs = False #diagnostic plots a plan view of cells that met criteria as well as a vertical plot of scan averaged reflectivity
plot_xy = True #plot the horizonal view of cells that showed an enhancement
plot_z = True #plot the vertial scan averaged figure  with secondary boundaries
print_results = False #print the results to the screen for each time - turn off for multiprocessing computing
run_100 = False #only run the first 100 times = Nov 12/13
save_data = True

dir = 'west' #lowercase


excd_val_low = 4 #dBZ value to exceed to define the lower level
excd_val_high = -4  #dBZ value to exceed to define the upper level

grid_size = 16 #km x km grid box for looking for secondary enhancement
grid_density = 15 # percentage of cells that must have an enhancement found

min_sep = 2 #number of levels above or below a mode layer that can be searched

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

sounding_data = 'NPOL_sounding_15_levs.npy'
sounding_fn = ''.join([bb_dir,sounding_data])
NPOL_data = np.load(sounding_fn)

save_fn_data = ''.join([save_dir,'secondary_D_',str(grid_density),'X',str(excd_val_low),'excd_',dir,'.npy'])
save_fn_data_csv = ''.join([save_dir,'secondary_D_',str(grid_density),'X',str(excd_val_low),'excd_',dir,'.csv'])

#load latest bright band data from BBIDv6
if dir == 'east':
    bb_data = ''.join(['brightbandsfound_v6_r_6_time0x35.0pcntx25.0_withrhohv_0.910.97_',dir,'.npy'])
elif dir == 'west':
    bb_data = ''.join(['brightbandsfound_v6_r_6_time0x15.0pcntx25.0_withrhohv_0.910.97_',dir,'.npy'])

bb_fn = ''.join([bb_dir,bb_data])
bright_bands = np.load(bb_fn)#time,bbfound,level, ...

#load in date_list and file_list from BBIDv6
date_list_fn = ''.join([data_dir,'date_list.npy'])
filelist_fn = ''.join([data_dir,'filelist_west.npy'])
date_list = np.load(date_list_fn)
filelist = np.load(filelist_fn)
filelist.sort()
numfiles = len(filelist)
if run_100:
    numfiles = 100
days_in_series = len(date_list)
days_out = np.concatenate((np.arange(12,31),np.arange(1,20)))

day_time_array = np.zeros([days_in_series,24])
day_time_array[:,:] = -1#default to clear sky

secondary = np.array([1,2,3,4,5,6,7,8]) #columns = date, enhancement_found,mean_low_enhance_lev,mean_high_enhance_lev,
#mean_low_enhance_lev_2,mean_high_enhance_lev_2,prcnt_cells_met,bb_lev]

n_NPOLs = NPOL_data.shape[0]
items_NPOL = []
for h in range(0,n_NPOLs-1):
    items_NPOL.append(datetime.datetime.strptime(NPOL_data[h+1,0], "%m/%d/%y %H:%M:%S:"))

total_grid_cells = grid_size*grid_size
half_width = int(grid_size/2)

'''
Main functions
'''
def compare_nan_array(func, a, thresh): #counts number of cells that are greater/less than threshold
    out = ~np.isnan(a)
    out[out] = func(a[out] , thresh)
    return out

def mode1(x): #most commonly occurring level
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]

def mode2(x): #second most commonly occurring level
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    counts[m] = 0
    n = counts.argmax()
    if values[n] > 0:
        return values[n], counts[n]
    else:
        return values[m], counts[m]

def main_func(i):
    date = ''.join([filelist[i].split(".")[1],'/'])
    outdate = filelist[i].split(".")[1]
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
    plot_array = np.full((y_dim,x_dim),float('NaN'))
    plot_array_1 = np.full((y_dim,x_dim),float('NaN'))
    plot_array_2 = np.full((y_dim,x_dim),float('NaN'))
    elev_array = np.full((y_dim,x_dim),float('NaN'))
    dBZ_means = np.full(z_dim, float('NaN'))
    ZDR_means = np.full(z_dim, float('NaN'))
    enhancement_true = False
    if bright_bands[bb_index,1] == '1': #there was a bright band found from bbidv6

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

        levels = np.full((y_dim,x_dim,6),float('NaN')) #store low and high levels for seccondary enhancement at each grid cell
        #low, high, low2, high2, low 1 enhancement associated with low mode or high mode (0/1), low 2 enhancement associated with high or low mode

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
            bb_lev_up = np.int(bb_lev + 1)
        elif bb_diff <= 0:
            bb_lev_up = np.int(bb_lev)
        for z_i in range(z_dim):
            if ~np.isnan(dBZ[0,z_i,:,:]).all():
                dBZ_means[z_i] = np.nanmean(dBZ[0,z_i,:,:])
                ZDR_means[z_i] = np.nanmean(ZDR[0,z_i,:,:])

        where_nan = np.argwhere(~np.isnan(dBZ_means[:]))
        top_lev = np.max(where_nan)

        plot_z = range(15)
        plot_z_hts = [z*0.5 for z in plot_z]

        n_found = 0
        n_1_found = 0
        n_2_found = 0
        #n_found_rhohv = 0
        prcnt_cells_met = 0
        enhancement_found = 0
        low_enhance_lev = []
        high_enhance_lev = []
        low_enhance_2_lev = []
        high_enhance_2_lev = []

        #search through each column looking for an enhancement aloft
        for x_ind in range(x_dim):
            for y_ind in range(y_dim):
                deltas = np.full(top_lev,float('NaN'))
                dBZ_column = np.full(top_lev,float('NaN'))
                if ~np.isnan(dBZ[0,bb_lev:top_lev+1,y_ind,x_ind]).all():
                    for z in range(1,top_lev):
                        deltas[z] = ((dBZ[0,z,y_ind,x_ind]-dBZ[0,z-1,y_ind,x_ind])) #change in dBZ from lower level to current level
                        dBZ_column[z] = dBZ[0,z,y_ind,x_ind]
                    #first look for a decrease above the bright band
                    decrease_lev = next((z_ind for z_ind, v in enumerate(deltas) if v < 0 and z_ind > bb_lev), z_dim)
                    #above bright band, where does delta become positive and exceed threshold.
                    low_enhance = next((z_ind for z_ind, v in enumerate(deltas) if v > excd_val_low and z_ind > decrease_lev), float('NaN'))
                    levels[y_ind,x_ind,0] = low_enhance
                    if ~np.isnan(low_enhance):
                        high_enhance = next((z_ind-1 for z_ind, v in enumerate(deltas) if v < excd_val_high and z_ind > low_enhance), low_enhance)
                        levels[y_ind,x_ind,1] = high_enhance
                        low_enhance_2 = next((z_ind for z_ind, v in enumerate(deltas) if v > excd_val_low and z_ind > high_enhance), float('NaN'))
                        if ~np.isnan(low_enhance_2):
                            levels[y_ind,x_ind,2] = low_enhance_2
                            high_enhance_2 = next((z_ind-1 for z_ind, v in enumerate(deltas) if v < excd_val_high and z_ind > low_enhance_2), low_enhance_2)
                            levels[y_ind,x_ind,3] = high_enhance_2
                    else:
                        high_enhance = float('NaN')

                else:
                    low_enhance = float('NaN')
                    high_enhance = float('NaN')

                if ~np.isnan(levels[y_ind,x_ind,0]): #found at least one enhanced layer
                    plot_array[y_ind,x_ind] = 1
                    n_found += 1

        if n_found>0:
            prcnt_cells_met = np.float64(format((n_found/n_total)*100,'.2f'))
            low_mode = int(mode1(levels[:,:,[0,2]])[0])
            low_mode_2 = int(mode2(levels[:,:,[0,2]])[0])

            modesOK = True

            if low_mode_2 < low_mode:
                low_mode, low_mode_2 = low_mode_2, low_mode
            elif low_mode_2 == low_mode:
                #low_mode_2 = top_lev
                modesOK = False

            #reset to empty arrays to pick out levels
            low_enhance_lev = []
            low_enhance_2_lev = []

            #loop back through to find high levels paired with each low mode, to get percentages and average heights
            #also determine whoch layer is enhanced.
            for x_ind in range(x_dim):
                for y_ind in range(y_dim):
                    if modesOK:
                        if low_mode-min_sep <= levels[y_ind,x_ind,0] <= low_mode+min_sep:
                            levels[y_ind,x_ind,4] = 0 #grid cell's low enhancement is in the low mode
                            low_enhance_lev.append(levels[y_ind,x_ind,0])
                            high_enhance_lev.append(levels[y_ind,x_ind,1])
                            n_1_found += 1
                            plot_array_1[y_ind,x_ind] =  1 #mark a mode 1 enhancement
                        elif low_mode_2-min_sep <= levels[y_ind,x_ind,0] <= low_mode_2+min_sep:
                            levels[y_ind,x_ind,4] = 1 #grid cell's low enhancement is in the upper mode
                            low_enhance_2_lev.append(levels[y_ind,x_ind,0])
                            high_enhance_2_lev.append(levels[y_ind,x_ind,1])
                            n_2_found += 1
                            plot_array_2[y_ind,x_ind] = 1 #mark a mode 2 enhancement
                        if low_mode_2-min_sep <= levels[y_ind,x_ind,2] <= low_mode_2+min_sep:
                            levels[y_ind,x_ind,5] = 1 #grid cell's high enhancement is in the upper mode
                            low_enhance_2_lev.append(levels[y_ind,x_ind,2])
                            high_enhance_2_lev.append(levels[y_ind,x_ind,3])
                            n_2_found += 1
                            plot_array_2[y_ind,x_ind] = 1
                        elif low_mode-min_sep <= levels[y_ind,x_ind,2] <= low_mode+min_sep:
                            levels[y_ind,x_ind,5] = 0 #grid cell's high enhancement is in the low mode
                            low_enhance_lev.append(levels[y_ind,x_ind,2])
                            high_enhance_lev.append(levels[y_ind,x_ind,3])
                            n_1_found += 1
                            plot_array_1[y_ind,x_ind] = 1

            prcnt_cells_met_1 = np.float64(format((n_1_found/n_total)*100,'.2f'))
            prcnt_cells_met_2 = np.float64(format((n_2_found/n_total)*100,'.2f'))
        else:
            prcnt_cells_met = 0
            prcnt_cells_met_1 = 0
            prcnt_cells_met_2 = 0
            mean_low_enhance_lev = float('NaN')
            mean_high_enhance_lev = float('NaN')
            mean_low_enhace_2_lev = float('NaN')
            mean_high_enhace_2_lev = float('NaN')

        #search through plotting array to find high density regions of secondary enhancement
        low_lev_enh = False
        high_lev_enh = False
        two_lev_enh = False
        tick_range = 1
        for x_ind in range(0+half_width,x_dim-half_width):
            for y_ind in range(0+half_width,y_dim-half_width):
                subset = plot_array[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width]
                subset_1 = plot_array_1[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width]
                subset_2 = plot_array_2[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width]
                total_met = np.nansum(compare_nan_array(np.greater, subset, 0))
                #need to find out which enhanced level satisfies the grid box criteria
                total_met_1 = np.nansum(compare_nan_array(np.greater, subset_1, 0))
                total_met_2 = np.nansum(compare_nan_array(np.greater, subset_2, 0))
                if total_met_1 >= ((grid_density/100)*total_grid_cells): #set cells that met criteria = 2
                    enhancement_true = True
                    enhancement_found = 1
                    for a in range(grid_size):
                        for b in range(grid_size):
                            if subset_1[a,b] > 0: #a grid cell that had originally found an enhancement layer in layer 1
                                subset_1[a,b] = 2
                                low_lev_enh = True
                    plot_array[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width] = subset_1
                    plot_array_1[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width] = subset_1
                if total_met_2 >= ((grid_density/100)*total_grid_cells): #set cells that met criteria = 2
                    enhancement_true = True
                    enhancement_found = 1
                    for a in range(grid_size):
                        for b in range(grid_size):
                            if subset_1[a,b] == 2 and subset_2[a,b] > 0:  #grid cell that had an enhancement in layer 2 that was already identified by layer 1
                                subset_2[a,b] = 4
                                two_lev_enh = True
                            elif subset_2[a,b] > 0 and subset_1[a,b] != 2: #grid cell that had an enhancement in layer 2 only
                                subset_2[a,b] = 3
                                high_lev_enh = True
                    plot_array[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width] = subset_2
                    plot_array_2[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width] = subset_2

        #collect all enhancement levels after grid assesment
        grid_low_enhance_lev = []
        grid_high_enhance_lev = []
        grid_low_enhance_2_lev = []
        grid_high_enhance_2_lev = []

        for x_ind in range(x_dim):
            for y_ind in range(y_dim):
                if plot_array_1[y_ind,x_ind] > 1 and levels[y_ind,x_ind,4] == 0: #enhancement found in grid approach and the enhancement is in the lower mode
                    grid_low_enhance_lev.append(levels[y_ind,x_ind,0])
                    grid_high_enhance_lev.append(levels[y_ind,x_ind,1])
                elif plot_array_1[y_ind,x_ind] > 1 and levels[y_ind,x_ind,5] == 0: #enhancement found in grid approach and the enhancement is in the upper mode
                    grid_low_enhance_lev.append(levels[y_ind,x_ind,2])
                    grid_high_enhance_lev.append(levels[y_ind,x_ind,3])
                if plot_array_2[y_ind,x_ind] > 1 and levels[y_ind,x_ind,4] == 1:
                    grid_low_enhance_2_lev.append(levels[y_ind,x_ind,0])
                    grid_high_enhance_2_lev.append(levels[y_ind,x_ind,1])
                elif plot_array_2[y_ind,x_ind] > 1 and levels[y_ind,x_ind,5] == 1:
                    grid_low_enhance_2_lev.append(levels[y_ind,x_ind,2])
                    grid_high_enhance_2_lev.append(levels[y_ind,x_ind,3])

        if two_lev_enh or (low_lev_enh and high_lev_enh): #two distinct layers
            two_lev_enh = True
            mean_low_enhance_lev = np.float64(format(np.nanmean(grid_low_enhance_lev)*0.5, '.2f'))
            mean_high_enhance_lev = np.float64(format(np.nanmean(grid_high_enhance_lev)*0.5, '.2f'))
            mean_low_enhance_2_lev = np.float64(format(np.nanmean(grid_low_enhance_2_lev)*0.5, '.2f'))
            mean_high_enhance_2_lev = np.float64(format(np.nanmean(grid_high_enhance_2_lev)*0.5, '.2f'))
        elif low_lev_enh: #layer 1 is the enhancement
            mean_low_enhance_lev = np.float64(format(np.nanmean(grid_low_enhance_lev)*0.5, '.2f'))
            mean_high_enhance_lev = np.float64(format(np.nanmean(grid_high_enhance_lev)*0.5, '.2f'))
            mean_low_enhance_2_lev = float('NaN')
            mean_high_enhance_2_lev = float('NaN')
        elif high_lev_enh: #layer 2 is the enhancement
            mean_low_enhance_2_lev = np.float64(format(np.nanmean(grid_low_enhance_2_lev)*0.5, '.2f'))
            mean_high_enhance_2_lev = np.float64(format(np.nanmean(grid_high_enhance_2_lev)*0.5, '.2f'))
            mean_low_enhance_lev = float('NaN')
            mean_high_enhance_lev = float('NaN')
        else:
            if prcnt_cells_met_1 > prcnt_cells_met_2:
                mean_low_enhance_lev = np.float64(format(np.nanmean(low_enhance_lev)*0.5, '.2f'))
                mean_high_enhance_lev = np.float64(format(np.nanmean(high_enhance_lev)*0.5, '.2f'))
                #mean_low_enhance_2_lev = np.float64(format(np.nanmean(low_enhance_2_lev)*0.5, '.2f'))
                #mean_high_enhance_2_lev = np.float64(format(np.nanmean(high_enhance_2_lev)*0.5, '.2f'))
            else:
                mean_low_enhance_lev = np.float64(format(np.nanmean(low_enhance_2_lev)*0.5, '.2f'))
                mean_high_enhance_lev = np.float64(format(np.nanmean(high_enhance_2_lev)*0.5, '.2f'))
                #mean_low_enhance_2_lev = np.float64(format(np.nanmean(low_enhance_lev)*0.5, '.2f'))
                #mean_high_enhance_2_lev = np.float64(format(np.nanmean(high_enhance_lev)*0.5, '.2f'))
            mean_low_enhance_2_lev = float('NaN')
            mean_high_enhance_2_lev = float('NaN')

        if high_lev_enh:
            row_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S"),enhancement_found,mean_low_enhance_lev,mean_high_enhance_lev,mean_low_enhance_2_lev,mean_high_enhance_2_lev,prcnt_cells_met,bb_lev])
        else:
            row_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S"),enhancement_found,mean_low_enhance_lev,mean_high_enhance_lev,mean_low_enhance_2_lev,mean_high_enhance_2_lev,prcnt_cells_met,bb_lev])

    else:
        prcnt_cells_met = float('NaN')
        enhancement_found = 0
        enhancement_true = False
        row_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S"),enhancement_found,float('NaN'),float('NaN'), float('NaN'),float('NaN'), float('NaN'),float('NaN')])

    if print_results:
        print(row_to_append)

    if plot_figs:
        datetime_object = datetime.datetime.strptime(row_to_append[0], "%m/%d/%y %H:%M:%S")
        pivot = datetime_object

        timedeltas = []
        for j in range(0,len(items_NPOL)):
            timedeltas.append(np.abs(pivot-items_NPOL[j]))
        min_index = timedeltas.index(np.min(timedeltas)) +1
        d2 = datetime.datetime.strptime(NPOL_data[min_index,0], '%m/%d/%y %H:%M:%S:').strftime('%m/%d/%y %H:%M:%S')
        dend_ht = float(NPOL_data[min_index,1])/1000

        smallRadius = plt.Circle((150,150), small_rad_dim, color = 'black', fill = False)
        Radius20 = plt.Circle((150,150), 20, color = 'grey', linestyle = '--', fill = False, alpha = 0.5)
        Radius30 = plt.Circle((150,150), 30, color = 'grey', linestyle = '--', fill = False, alpha = 0.5)
        Radius40 = plt.Circle((150,150), 40, color = 'grey', linestyle = '--', fill = False, alpha = 0.5)
        Radius50 = plt.Circle((150,150), 50, color = 'grey', linestyle = '--', fill = False, alpha = 0.5)
        bigRadius = plt.Circle((150,150), big_rad_dim, color = 'black', fill = False)
        cmap = matplotlib.colors.ListedColormap(['dimgrey','#1b9e77','darkviolet','#e66101'])
        fig, (ax1,ax2) = plt.subplots(1,2)
        if plot_xy:
            fig, (ax1) = plt.subplots(1,1)
            im = ax1.imshow(plot_array, origin = 'Lower', cmap = cmap)
            ax1.add_artist(smallRadius)
            ax1.add_artist(Radius20)
            ax1.add_artist(Radius30)
            ax1.add_artist(Radius40)
            ax1.add_artist(Radius50)
            ax1.add_artist(bigRadius)
            ax1.set_xlim([85,150])
            ax1.set_ylim([85,215])
            fig.canvas.draw()
            #shift labels over 150 km to center on NPOL
            ax1.set_xticks(np.arange(90,170,20)) # choose which x locations to have ticks
            ax1.set_xticklabels(np.arange(60,-20, -20)) # set the labels to display at those ticks
            ax1.set_yticks(np.arange(90,230,20)) # choose which y locations to have ticks
            ax1.set_yticklabels(np.arange(-60,80, 20)) # set the labels to display at those ticks
            ax1.yaxis.tick_right()
            ax1.yaxis.set_label_position("right")
            ax1.set_xlabel('km')
            ax1.set_ylabel('km')
            im.set_clim(0,5) #set the colorbar limits
            cbar = fig.colorbar(im, ax = ax1, orientation = 'vertical', fraction=0.046, pad=0.15, boundaries=np.linspace(0.5, 4.5, 5))
            cbar.set_ticks([1,2,3,4])
            labels = ['Enh.','Lower\nEnh.','Upper\nEnh.','2-Layer\nEnh.']
            cbar.set_ticklabels(labels)
            ax1.set_title(date.strftime(''.join(['%m/%d/%y %H:%M:%S\n'])))
            plt.tight_layout()
            fig.subplots_adjust(wspace=1.0)
            plt.savefig(''.join(['/home/disk/meso-home/adelaf/OLYMPEX/olytestfigs/',outdate,'_',file,'_',dir,'_xy.png']), bbox_inches='tight',dpi=300)
            plt.close()

        if bright_bands[bb_index,1] ==  '1' and plot_z: #there was a bright band found from bbidv6
            fig, (ax2) = plt.subplots(1,1) #single plot
            bbline = plt.axhline(y=np.float64(bb_ht)*2, color='gray', linestyle='--', label = 'Bright Band')
            dendline = plt.axhline(y=np.float64(dend_ht)*2, color='mediumblue', linestyle='--', label = ''.join(['NPOL Sounding -15'+ '\u00b0'+ 'C\n',str(d2)]))
            dbzs = ax2.plot(dBZ_means[0:15],plot_z, color = 'black')
            if low_lev_enh:
                lowline = plt.axhline(y=mean_low_enhance_lev*2, color='#1b9e77', linestyle='-', label = 'Secondary Enhancement')
                highline = plt.axhline(y=mean_high_enhance_lev*2, color='#1b9e77', linestyle='-')
                ax2.add_artist(lowline)
                ax2.add_artist(highline)
            if high_lev_enh:
                lowline = plt.axhline(y=mean_low_enhance_2_lev*2, color='darkviolet', linestyle='-', label = 'Secondary Enhancement')
                highline = plt.axhline(y=mean_high_enhance_2_lev*2, color='darkviolet', linestyle='-')
                ax2.add_artist(lowline)
                ax2.add_artist(highline)
            if two_lev_enh and not low_lev_enh and not high_lev_enh:
                lowline = plt.axhline(y=mean_low_enhance_lev*2, color='#1b9e77', linestyle='-', label = 'Secondary Enhancement')
                highline = plt.axhline(y=mean_high_enhance_lev*2, color='#1b9e77', linestyle='-')
                ax2.add_artist(lowline)
                ax2.add_artist(highline)
                lowline2 = plt.axhline(y=mean_low_enhance_2_lev*2, color='darkviolet', linestyle='-', label = 'Secondary Enhancement')
                highline2 = plt.axhline(y=mean_high_enhance_2_lev*2, color='darkviolet', linestyle='-')
                ax2.add_artist(lowline2)
                ax2.add_artist(highline2)

            if not low_lev_enh and not high_lev_enh and not two_lev_enh:
                lowline = plt.axhline(y=mean_low_enhance_lev*2, color='#1b9e77', linestyle=':', label = 'Secondary Enhancement',alpha=0.5)
                highline = plt.axhline(y=mean_high_enhance_lev*2, color='#1b9e77', linestyle=':',alpha = 0.5)
                #lowline2 = plt.axhline(y=mean_low_enhance_2_lev*2, color='#1b9e77', linestyle=':', label = 'Secondary Enhancement [2]', alpha = 0.5)
                #highline2 = plt.axhline(y=mean_high_enhance_2_lev*2, color='#1b9e77', linestyle=':', alpha = 0.5)
                ax2.add_artist(lowline)
                ax2.add_artist(highline)
                #ax2.add_artist(lowline2)
                #ax2.add_artist(highline2)

            ax2.add_artist(bbline)
            ax2.add_artist(dendline)
            ax2.set_yticks(plot_z)
            ax2.set_yticklabels(plot_z_hts)
            ax2.set_xlim((10,45))
            ax2.grid(True, linestyle = '--', linewidth = 0.5)
            ax2.set_xlabel(''.join(['Reflectivity (dBZ)']))
            ax2.set_ylabel('Height (km)')
            lgd = ax2.legend(bbox_to_anchor=(1.04,0.5), loc="center left", frameon = False)
            ax2.text(0.97, 0.95, date.strftime(''.join(['%m/%d/%y %H:%M:%S','\n',dir.upper(),''])), verticalalignment='top', horizontalalignment='right',transform=ax2.transAxes, color='black', fontsize=10,bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
            #ax3 = ax2.twiny()
            #ZDRs = ax3.plot(ZDR_means[0:15],plot_z, color = 'grey')
            if not plot_xy:
                ax2.set_title(date.strftime(''.join(['%m/%d/%y %H:%M:%S\n'])))
                fig.text(0.25, 0.05, ''.join(['closest sounding ',str(d2),]), horizontalalignment='center',fontsize=10)
            plt.tight_layout()
            fig.subplots_adjust(wspace=1.0)
            plt.savefig(''.join(['/home/disk/meso-home/adelaf/OLYMPEX/olytestfigs/',outdate,'_',file,'_',dir,'_z.png']), bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)
            plt.close()
    plt.close()
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

if save_data:
    np.save(save_fn_data,secondary)
    pd.DataFrame(secondary).to_csv(save_fn_data_csv)


print("Total time:", datetime.datetime.now() - startTime)
