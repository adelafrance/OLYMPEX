#Andrew DeLaFrance
#June 28

#algorithm aimed at identifying the secondary enhancement above the melging layer
#ingests melting layer data identified by BBIDv6 and pulls in -15c height from NPOL

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

nodes = 1 #how many processors to run from

plot_figs = True #diagnostic plots a plan view of cells that met criteria as well as a vertical plot of scan averaged reflectivity
print_results = True #print the results for each time - turn off for multiprocessing computing
run_100 = True #only run the first 100 times = Nov 12/13
save_data = False

dir = 'east' #lowercase

excd_val_low = 4 #dBZ value to exceed to define the lower level
excd_val_high = -4  #dBZ value to exceed to define the upper level

grid_size = 10 #km x km grid box for looking for secondary enhancement
grid_density = 50 # percentage of cells that must have an enhancement found

##try to keep this one same as BBIDV6? 15-east/35-west
secondary_crit = 15 #percentage of cells that must meet criteria in order to say a secondary enhancement is found

min_sep = 2 #number of levels needed between bright band and region to look within

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

save_fn_data = ''.join([save_dir,'secondary_C_',str(secondary_crit),'X',str(excd_val_low),'excd_',dir,'.npy'])
save_fn_data_csv = ''.join([save_dir,'secondary_C_',str(secondary_crit),'X',str(excd_val_low),'excd_',dir,'.csv'])

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


'''
Main functions
'''
def compare_nan_array(func, a, thresh):
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
    return values[n], counts[n]

def mode_addtl(x): #layers that occur outside 'min_sep' levels of the first mode
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    counts[m-min_sep:m+min_sep+1] = 0 #doesnt work
    n = counts.argmax()
    return values[n], counts[n]

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
    dBZ_means = np.full(z_dim, float('NaN'))
    ZDR_means = np.full(z_dim, float('NaN'))
    enhancement_true = False
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

        levels = np.full((y_dim,x_dim,5),float('NaN')) #store low and high levels for seccondary enhancement at each grid cell
        #low, high, low2, high2, counted (0/1)

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
        two_mode = False
        low_enhance_lev = []
        high_enhance_lev = []
        #peak_enhance_lev = []
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
                    #above bright band, where does delta become positive and exceed threshold.
                    low_enhance = next((z_ind for z_ind, v in enumerate(deltas) if v > excd_val_low and z_ind > bb_lev_up), float('NaN'))
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
                    #low_enhance_2 = float('NaN')
                    #high_enhance_2 = float('NaN')

                if np.isnan(low_enhance): #didnt find any enhancement
                    pass
                elif low_enhance == high_enhance: #found a single-layer enhancement
                    low_enhance_lev.append(low_enhance)
                    #high_enhance_lev.append(high_enhance)
                    #peak_enhance_lev.append(low_enhance)
                    n_found += 1
                    plot_array[y_ind,x_ind] = 1
                else: #found a multi-layer enhancement
                    low_enhance_lev.append(low_enhance)
                    #high_enhance_lev.append(high_enhance)
                    #peak_enhance = np.argmax(dBZ[0,low_enhance:high_enhance+1,y_ind,x_ind])+low_enhance
                    #peak_enhance_lev.append(peak_enhance)
                    n_found += 1
                    plot_array[y_ind,x_ind] = 1

        #search through plotting array to find high density regions of secondary enhancement
        total_grid_cells = grid_size*grid_size
        half_width = int(grid_size/2)
        for x_ind in range(0+half_width,x_dim-half_width):
            for y_ind in range(0+half_width,y_dim-half_width):
                subset = plot_array[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width]
                total_met = np.nansum(compare_nan_array(np.greater, subset, 0))
                if total_met >= ((grid_density/100)*total_grid_cells): #set cells that met criteria = 2
                    enhancement_true = True
                    enhancement_found = 1
                    for a in range(grid_size):
                        for b in range(grid_size):
                            if subset[a,b] > 0:
                                subset[a,b] = 2
                    plot_array[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width] = subset

        if n_found>0:
            prcnt_cells_met = np.float64(format((n_found/n_total)*100,'.2f'))
            #prcnt_2_cells_met = np.float64(format((n_2_found/n_total)*100,'.2f'))
            low_mode = int(mode1(low_enhance_lev)[0])
            #high_mode = int(mode1(high_enhance_lev)[0])
            low_mode_2 = int(mode2(low_enhance_lev)[0])
            #high_mode_2 = int(mode_addtl(high_enhance_lev)[0])

            if low_mode_2 < low_mode:
                low_mode, low_mode_2 = low_mode_2, low_mode
            #if high_mode_2 < high_mode:
                #high_mode, high_mode_2 = high_mode_2, high_mode

            print(low_mode,low_mode_2)

            #reset to empty arrays to pick out levels
            low_enhance_lev = []
            low_enhance_2_lev = []

            #loop back through to find high levels paired with each low mode
            for x_ind in range(x_dim):
                for y_ind in range(y_dim):
                    if low_mode-min_sep <= levels[y_ind,x_ind,0] <= low_mode+min_sep:
                        levels[y_ind,x_ind,4] = 1
                        low_enhance_lev.append(levels[y_ind,x_ind,0])
                        high_enhance_lev.append(levels[y_ind,x_ind,1])
                        n_1_found += 1
                    elif low_mode_2-min_sep <= levels[y_ind,x_ind,0] <= low_mode_2+min_sep:
                        levels[y_ind,x_ind,4] = 1
                        low_enhance_2_lev.append(levels[y_ind,x_ind,0])
                        high_enhance_2_lev.append(levels[y_ind,x_ind,1])
                        n_2_found += 1
                    if low_mode-min_sep <= levels[y_ind,x_ind,3] <= low_mode+min_sep:
                        low_enhance_lev.append(levels[y_ind,x_ind,0])
                        high_enhance_lev.append(levels[y_ind,x_ind,1])
                        if levels[y_ind,x_ind,4] != 1:
                            n_1_found += 1
                    elif low_mode_2-min_sep <= levels[y_ind,x_ind,3] <= low_mode_2+min_sep:
                        low_enhance_2_lev.append(levels[y_ind,x_ind,0])
                        high_enhance_2_lev.append(levels[y_ind,x_ind,1])
                        if levels[y_ind,x_ind,4] != 1:
                            n_2_found += 1

            prcnt_cells_met_1 = np.float64(format((n_1_found/n_total)*100,'.2f'))
            prcnt_cells_met_2 = np.float64(format((n_2_found/n_total)*100,'.2f'))

            if (low_mode_2 - low_mode) > min_sep and prcnt_cells_met_1 >= secondary_crit and prcnt_cells_met_2 >= secondary_crit: #two distinct layers
                two_mode = True
                print("2-fer")
                mean_low_enhance_lev = np.float64(format(np.nanmean(low_enhance_lev)*0.5, '.2f'))
                mean_high_enhance_lev = np.float64(format(np.nanmean(high_enhance_lev)*0.5, '.2f'))
                mean_low_enhance_2_lev = np.float64(format(np.nanmean(low_enhance_2_lev)*0.5, '.2f'))
                mean_high_enhance_2_lev = np.float64(format(np.nanmean(high_enhance_2_lev)*0.5, '.2f'))
            elif (low_mode_2 - low_mode) > min_sep and prcnt_cells_met_1 >= secondary_crit and prcnt_cells_met_2 < secondary_crit: #layer 1 is the enhancement
                two_mode = False
                mean_low_enhance_lev = np.float64(format(np.nanmean(low_enhance_lev)*0.5, '.2f'))
                mean_high_enhance_lev = np.float64(format(np.nanmean(high_enhance_lev)*0.5, '.2f'))
                mean_low_enhance_2_lev = float('NaN')
                mean_high_enhance_2_lev = float('NaN')
            elif (low_mode_2 - low_mode) > min_sep and prcnt_cells_met_1 < secondary_crit and prcnt_cells_met_2 >= secondary_crit: #layer 2 is the enhancement
                two_mode = False
                mean_low_enhance_lev = np.float64(format(np.nanmean(low_enhance_2_lev)*0.5, '.2f'))
                mean_high_enhance_lev = np.float64(format(np.nanmean(high_enhance_2_lev)*0.5, '.2f'))
                mean_low_enhance_2_lev = float('NaN')
                mean_high_enhance_2_lev = float('NaN')
            else:
                two_mode = False
                low_enhance_lev.extend(low_enhance_2_lev)
                high_enhance_lev.extend(high_enhance_2_lev)
                mean_low_enhance_lev = np.float64(format(np.nanmean(low_enhance_lev)*0.5, '.2f'))
                mean_high_enhance_lev = np.float64(format(np.nanmean(high_enhance_lev)*0.5, '.2f'))
                mean_low_enhance_2_lev = float('NaN')
                mean_high_enhance_2_lev = float('NaN')

        else:
            prcnt_cells_met = 0
            mean_low_enhance_lev = float('NaN')
            mean_high_enhance_lev = float('NaN')
            #mean_peak_enhance_lev = float('NaN')
            mean_low_enhace_2_lev = float('NaN')
            mean_high_enhace_2_lev = float('NaN')
        if prcnt_cells_met>secondary_crit:
            #enhancement_found = 1
            #enhancement_true = True
            pass

        #search through plotting array to find high density regions of secondary enhancement
        total_grid_cells = grid_size*grid_size
        half_width = int(grid_size/2)
        for x_ind in range(0+half_width,x_dim-half_width):
            for y_ind in range(0+half_width,y_dim-half_width):
                subset = plot_array[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width]
                total_met = np.nansum(compare_nan_array(np.greater, subset, 0))
                if total_met >= ((grid_density/100)*total_grid_cells): #set cells that met criteria = 2
                    enhancement_true = True
                    enhancement_found = 1
                    for a in range(grid_size):
                        for b in range(grid_size):
                            if subset[a,b] > 0:
                                subset[a,b] = 2
                    plot_array[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width] = subset

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
        Radius20 = plt.Circle((150,150), 20, color = 'grey', linestyle = '--', fill = False)
        Radius30 = plt.Circle((150,150), 30, color = 'grey', linestyle = '--', fill = False)
        Radius40 = plt.Circle((150,150), 40, color = 'grey', linestyle = '--', fill = False)
        Radius50 = plt.Circle((150,150), 50, color = 'grey', linestyle = '--', fill = False)
        bigRadius = plt.Circle((150,150), big_rad_dim, color = 'black', fill = False)
        fig, (ax1,ax2) = plt.subplots(1,2)
        cmap = matplotlib.colors.ListedColormap(['#7570b3','#d95f02'])
        im = ax1.imshow(plot_array.T, origin = 'Lower', cmap = cmap)
        ax1.add_artist(smallRadius)
        ax1.add_artist(Radius20)
        ax1.add_artist(Radius30)
        ax1.add_artist(Radius40)
        ax1.add_artist(Radius50)
        ax1.add_artist(bigRadius)
        ax1.set_xlim([85,215])
        ax1.set_ylim([85,215])
        ax1.axis('off')
        ax1.set_title(date.strftime(''.join(['%m/%d/%y %H:%M:%S\n\n',str(prcnt_cells_met_1),'% layer 1\n',str(prcnt_cells_met_2),'% layer 2'])))
        fig.text(0.27, 0.05, ''.join(['10 km inner, 60 km outer\ndashed rings every 10 km\n\n\nclosest sounding ',str(d2),]), horizontalalignment='center',fontsize=12)


        if bright_bands[bb_index,1] ==  '1': #there was a bright band found from bbidv6

            bbline = plt.axhline(y=np.float64(bb_ht)*2, color='black', linestyle='--', label = 'Bright Band')
            dendline = plt.axhline(y=np.float64(dend_ht)*2, color='blue', linestyle='--', label = 'Sounding -15'+ '\u00b0'+ 'C')
            dbzs = ax2.plot(dBZ_means[0:15],plot_z, color = 'black')
            if enhancement_true:
                lowline = plt.axhline(y=mean_low_enhance_lev*2, color='green', linestyle='-', label = 'Secondary Enhancement')
                highline = plt.axhline(y=mean_high_enhance_lev*2, color='green', linestyle='-')
            else:
                lowline = plt.axhline(y=mean_low_enhance_lev*2, color='r', linestyle='-', label = 'Secondary Enhancement')
                highline = plt.axhline(y=mean_high_enhance_lev*2, color='r', linestyle='-')
            if enhancement_true:
                ax2.add_artist(lowline)
                ax2.add_artist(highline)
            lowline2 = plt.axhline(y=mean_low_enhance_2_lev*2, color='purple', linestyle='-', label = 'Secondary Enhancement [2]')
            highline2 = plt.axhline(y=mean_high_enhance_2_lev*2, color='purple', linestyle='-')
            if two_mode:
                ax2.add_artist(lowline2)
                ax2.add_artist(highline2)
            ax2.add_artist(bbline)
            ax2.add_artist(dendline)
            ax2.set_yticks(plot_z)
            ax2.set_yticklabels(plot_z_hts)

            ax2.set_xlabel(''.join(['Reflectivity [dBZ]']))
            #ax3.set_xlabel('ZDR [dB]', color = 'grey')
            ax2.set_ylabel('height (km)')
            lgd = ax2.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
            #ax3 = ax2.twiny()
            #ZDRs = ax3.plot(ZDR_means[0:15],plot_z, color = 'grey')
            #ax3.set_title('scan averaged reflectivity')
            plt.tight_layout()
            plt.savefig(''.join(['/home/disk/meso-home/adelaf/OLYMPEX/olytestfigs/',outdate,file,'.png']), bbox_extra_artists=(lgd,), bbox_inches='tight')
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
