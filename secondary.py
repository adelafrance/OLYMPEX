#Andrew DeLaFrance
#June 28 '19

#algorithm aimed at identifying the secondary enhancement above the melging layer
#ingests melting layer data identified by BBIDv6 and pulls in -15c height from NPOL - for plotting reference

##requires a direction input when called from terminal (i.e. python3 secondary.py east) - east/west in lowercase

##update Aug 10 to include sector mapping

from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  #raster rendering backend capable of writing png image to file
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
from scipy import stats
import datetime
import os
import pandas as pd
import multiprocessing as mp
from multiprocessing import Queue
import sys
import math
from scipy import stats
from copy import copy


startTime = datetime.datetime.now()


"""
THRESHOLDS AND VARIABLES
"""
nodes = 4 #how many processors to run from

plot_map = False #plot an image of the sectors on a geographic background
plot_sectors = False #plot an image of the sectors with no background
plot_figs = False #diagnostic plots a plan view of cells that met criteria as well as a vertical plot of scan averaged reflectivity
plot_xy = True #plot the horizonal view of cells that showed an enhancement
plot_z_fig = True #plot the vertial scan averaged figure  with secondary boundaries
print_results = False #print the results to the screen for each time - turn off for multiprocessing computing
run_100 = False #only run the first 100 times = Nov 12/13
save_data = True

excd_val_low = 4 #dBZ value to exceed to define the lower level

grid_size = 15 #km x km grid box for looking for secondary enhancement
grid_density = 15 #percentage of cells that must have an enhancement found
enhance_threshold = 8 # % cells needed to have an enhancement found in any given sector

min_sep = 2 #number of levels above or below a mode layer that can be searched
modes_within = 2 #modes within N layers, consider one larger layer
min_dBZ = 15 #minimum dBZ needed in order to be condidered an enhancement

dBZ_exceed_val = 25.0 #threshold value that must exist within +/- min_sep of a bright band for given x,y to be considered

#spatial domain in radius from the radar
small_rad_dim = 10.0 #radius to restrict to away from the radar (changing these requires recalculating n_total cells)
big_rad_dim = 60.0 #outer bounds of the radar scan, beyond 60 beam smoothing becomes an issue

#dir = sys.argv[1] #input from script call, east or west after script name, or all

plot_z = range(15) #plotting range
plot_z_hts = [z*0.5 for z in plot_z]

total_grid_cells = grid_size*grid_size
half_width = int((grid_size-1)/2)

"""
FILE INPUT/OUTPUT - ORGANIZATION
"""
rhi_dir = '/home/disk/bob/olympex/zebra/moments/npol_qc2/rhi/' #base directory for gridded RHI's
bb_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/BrightBands/' #output directory for saved images
save_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/Secondary/' #output directory for saved images
data_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Data/' #directory for local data

sounding_data = 'NPOL_sounding_15_levs.npy'
sounding_fn = ''.join([bb_dir,sounding_data])
NPOL_data = np.load(sounding_fn)

save_fn_data_NE = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_NE.npy'])
save_fn_data_csv_NE = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_NE.csv'])
save_fn_data_SW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_SW.npy'])
save_fn_data_csv_SW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_SW.csv'])
save_fn_data_WSW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_WSW.npy'])
save_fn_data_csv_WSW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_WSW.csv'])
save_fn_data_WNW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_WNW.npy'])
save_fn_data_csv_WNW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_WNW.csv'])
save_fn_data_NW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_NW.npy'])
save_fn_data_csv_NW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_NW.csv'])

save_fn_vals_NE = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_vals_NE.npy'])
save_fn_vals_csv_NE = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_vals_NE.csv'])
save_fn_vals_SW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_vals_SW.npy'])
save_fn_vals_csv_SW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_vals_SW.csv'])
save_fn_vals_WSW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_vals_WSW.npy'])
save_fn_vals_csv_WSW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_vals_WSW.csv'])
save_fn_vals_WNW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_vals_WNW.npy'])
save_fn_vals_csv_WNW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_vals_WNW.csv'])
save_fn_vals_NW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_vals_NW.npy'])
save_fn_vals_csv_NW = ''.join([save_dir,'secondary_E_',str(enhance_threshold),'X',str(excd_val_low),'excd_vals_NW.csv'])


#load latest bright band data from BBIDv6

bb_data_east = ''.join(['brightbandsfound_v6_r_6_time0x35.0pcntx25.0_withrhohv_0.910.97_east.npy'])
filelist_fn_east = ''.join([data_dir,'filelist.npy']) #from BBIDv6

bb_data_west = ''.join(['brightbandsfound_v6_r_6_time0x15.0pcntx25.0_withrhohv_0.910.97_west.npy'])
filelist_fn_west = ''.join([data_dir,'filelist_west.npy']) #from BBIDv6

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

bright_bands_NE = np.load(bb_fn_NE)
bright_bands_SW = np.load(bb_fn_SW)
bright_bands_WSW = np.load(bb_fn_WSW)
bright_bands_WNW = np.load(bb_fn_WNW)
bright_bands_NW = np.load(bb_fn_NW)

bb_fn_east = ''.join([bb_dir,bb_data_east])
bright_bands_east = np.load(bb_fn_east)#time,bbfound,level, ...

bb_fn_west = ''.join([bb_dir,bb_data_west])
bright_bands_west = np.load(bb_fn_west)

date_list_fn = ''.join([data_dir,'date_list.npy']) #load in date_list from BBIDv6
date_list = np.load(date_list_fn)
filelist_east = np.load(filelist_fn_east)
filelist_east.sort()
filelist_west = np.load(filelist_fn_west)
filelist_west.sort()
numfiles_east = len(filelist_east)
numfiles_west = len(filelist_west)

if run_100:
    numfiles_east = 100
    numfiles_west = 100

#numfiles = numfiles_east if dir == 'east' else numfiles_west
numfiles = numfiles_west

days_in_series = len(date_list)
days_out = np.concatenate((np.arange(12,31),np.arange(1,20)))

n_NPOLs = NPOL_data.shape[0]
items_NPOL = []
for h in range(0,n_NPOLs-1):
    items_NPOL.append(datetime.datetime.strptime(NPOL_data[h+1,0], "%m/%d/%y %H:%M:%S:"))

#empty global arrays for storing output from main script
secondary_NE = np.array([1,2,3,4,5,6,7,8]) #columns = date, enhancement_found,mean_low_enhance_lev,mean_high_enhance_lev,
                                        #mean_low_enhance_lev_2,mean_high_enhance_lev_2,prcnt_cells_met,bb_lev]
secondary_vals_NE = np.array([1,2,3]) #columns = date, low-mean dBZ, high-mean dBZ
secondary_SW = np.array([1,2,3,4,5,6,7,8])
secondary_vals_SW = np.array([1,2,3])
secondary_WSW = np.array([1,2,3,4,5,6,7,8])
secondary_vals_WSW = np.array([1,2,3])
secondary_WNW = np.array([1,2,3,4,5,6,7,8])
secondary_vals_WNW = np.array([1,2,3])
secondary_NW = np.array([1,2,3,4,5,6,7,8])
secondary_vals_NW = np.array([1,2,3])

"""
SETUP FUNCTIONS
"""

def cell_within_angles(y, x, alpha_1, alpha_2): #determines if cell lies within alpha_1 and alpha_2, returns true/false
    xs = x - 150 #shift over 150km to put 0 at the center
    ys = y - 150
    if 0 <= alpha_1 < 90:
        ylim_1 = xs*math.tan(((90-alpha_2)*np.pi)/180.)
        ylim_2 = xs*math.tan(((90-alpha_1)*np.pi)/180.)
        return True if ylim_1 < ys <= ylim_2 else False
    elif 180 <= alpha_1 < 270:
        ylim_1 = xs*math.tan(((270-alpha_1)*np.pi)/180.)
        ylim_2 = xs*math.tan(((270-alpha_2)*np.pi)/180.)
        return True if ylim_1 < ys <= ylim_2 else False
    elif 270 <= alpha_1 < 360:
        ylim_1 = -xs*math.tan(((alpha_1-270)*np.pi)/180.)
        ylim_2 = -xs*math.tan(((alpha_2-270)*np.pi)/180.)
        return True if ylim_1 <= ys < ylim_2 else False


"""
SETUP AND BUILD 'MASKS' FOR EACH SECTOR - SPECIFIC TO NPOL RADAR SCANS
"""

sector_colors = ['black', 'green','grey', 'blue', 'red', 'purple']
sector_colors_dark = [ 'darkgreen','dimgrey', 'darkblue', 'darkred', 'indigo']

#angles for sectors to search within
NE = [30,60]
SW = [210,240]
WSW = [240,270]
WNW = [270,300]
NW = [296,326] #allow for overlap of the NW sector to make the number of grid cells the same

sectors = [NE,SW,WSW,WNW,NW]
sector_names = ['NE', 'SW', 'WSW', 'WNW', 'NW']

NE_mask = np.full((300,300), float('NaN'))
SW_mask = np.full((300,300), float('NaN'))
WSW_mask = np.full((300,300), float('NaN'))
WNW_mask = np.full((300,300), float('NaN'))
NW_mask = np.full((300,300), float('NaN'))

line_endpoints = np.full((len(sectors),2,2,2),float('NaN')) #sector, line1/line2, start/end, x/y
counts = np.zeros(len(sectors)) #count the number of grid points in each sector to be sure that each are the same size

for s in range(len(sectors)):
    alpha_1 = sectors[s][0]
    alpha_2 = sectors[s][1]
    n_counts = 0
    if 0 <= alpha_1 <= 180 and 0 <= alpha_2 <= 180:
        if 0 <= alpha_1 <= 90 and 0 <= alpha_2 <= 90: #NE quadrant
            for x_ind in range(0,300):
                for y_ind in range(0,300):
                    if cell_within_angles(y_ind,x_ind,alpha_1,alpha_2):
                        x_dist = abs(x_ind - 150) #shifting over to the center point of 150 km
                        y_dist = abs(y_ind - 150) #shifting over to the center point of 150 km
                        dist = ((x_dist**2)+(y_dist**2))**(0.5) #distance
                        if small_rad_dim < dist < big_rad_dim:
                            NE_mask[y_ind,x_ind] = 1
                            n_counts += 1
            counts[s] = n_counts
            line_endpoints[s,0,0,0] = (small_rad_dim*math.cos(((90-alpha_1)*np.pi)/180.)) + 150 #x point
            line_endpoints[s,0,0,1] = (small_rad_dim*math.sin(((90-alpha_1)*np.pi)/180.)) + 150 #y point
            line_endpoints[s,0,1,0] = (big_rad_dim*math.cos(((90-alpha_1)*np.pi)/180.)) + 150 #x point
            line_endpoints[s,0,1,1] = (big_rad_dim*math.sin(((90-alpha_1)*np.pi)/180.)) + 150 #y point
            line_endpoints[s,1,0,0] = (small_rad_dim*math.cos(((90-alpha_2)*np.pi)/180.)) + 150 #x point
            line_endpoints[s,1,0,1] = (small_rad_dim*math.sin(((90-alpha_2)*np.pi)/180.)) + 150 #y point
            line_endpoints[s,1,1,0] = (big_rad_dim*math.cos(((90-alpha_2)*np.pi)/180.)) + 150 #x point
            line_endpoints[s,1,1,1] = (big_rad_dim*math.sin(((90-alpha_2)*np.pi)/180.)) + 150 #y point

    elif 180 <= alpha_1 <= 360 and 180 <= alpha_2 <= 360:
        if 180 <= alpha_1 <= 270 and 180 <= alpha_2 <= 270: #SW quadrant
            if [alpha_1,alpha_2] == SW:
                for x_ind in range(0,300):
                    for y_ind in range(0,300):
                        if cell_within_angles(y_ind,x_ind,alpha_1,alpha_2):
                            x_dist = abs(x_ind - 150) #shifting over to the center point of 150 km
                            y_dist = abs(y_ind - 150) #shifting over to the center point of 150 km
                            dist = ((x_dist**2)+(y_dist**2))**(0.5) #distance
                            if small_rad_dim < dist < big_rad_dim:
                                SW_mask[y_ind,x_ind] = 2
                                n_counts += 1
                counts[s] = n_counts
                line_endpoints[s,0,0,0] = -(small_rad_dim*math.cos(((270-alpha_1)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,0,0,1] = -(small_rad_dim*math.sin(((270-alpha_1)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,0,1,0] = -(big_rad_dim*math.cos(((270-alpha_1)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,0,1,1] = -(big_rad_dim*math.sin(((270-alpha_1)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,1,0,0] = -(small_rad_dim*math.cos(((270-alpha_2)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,1,0,1] = -(small_rad_dim*math.sin(((270-alpha_2)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,1,1,0] = -(big_rad_dim*math.cos(((270-alpha_2)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,1,1,1] = -(big_rad_dim*math.sin(((270-alpha_2)*np.pi)/180.)) + 150 #y point
            elif [alpha_1,alpha_2] == WSW:
                for x_ind in range(0,300):
                    for y_ind in range(0,300):
                        if cell_within_angles(y_ind,x_ind,alpha_1,alpha_2):
                            x_dist = abs(x_ind - 150) #shifting over to the center point of 150 km
                            y_dist = abs(y_ind - 150) #shifting over to the center point of 150 km
                            dist = ((x_dist**2)+(y_dist**2))**(0.5) #distance
                            if small_rad_dim < dist < big_rad_dim:
                                if n_counts == 0:
                                    shortest = dist
                                    dont_use = [y_ind,x_ind]
                                elif dist < shortest:
                                    dont_use = [y_ind,x_ind]
                                if y_ind == 150 and x_ind % 2 == 0:
                                    pass
                                else:
                                    WSW_mask[y_ind,x_ind] = 3
                                    n_counts += 1
                #go back and find the one closest to NPOL and remove to make number of grid cells match all other segments (913)
                for x_ind in range(0,300):
                    for y_ind in range(0,300):
                        if [y_ind,x_ind] == dont_use:
                            WSW_mask[y_ind,x_ind] = float('NaN')
                            n_counts -= 1
                counts[s] = n_counts
                line_endpoints[s,0,0,0] = -(small_rad_dim*math.cos(((270-alpha_1)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,0,0,1] = -(small_rad_dim*math.sin(((270-alpha_1)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,0,1,0] = -(big_rad_dim*math.cos(((270-alpha_1)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,0,1,1] = -(big_rad_dim*math.sin(((270-alpha_1)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,1,0,0] = -(small_rad_dim*math.cos(((270-alpha_2)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,1,0,1] = -(small_rad_dim*math.sin(((270-alpha_2)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,1,1,0] = -(big_rad_dim*math.cos(((270-alpha_2)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,1,1,1] = -(big_rad_dim*math.sin(((270-alpha_2)*np.pi)/180.)) + 150 #y point
        elif 270 <= alpha_1 <= 360 and 270 <= alpha_2 <= 360: #NW quadrant
            if [alpha_1,alpha_2] == WNW:
                for x_ind in range(0,300):
                    for y_ind in range(0,300):
                        if cell_within_angles(y_ind,x_ind,alpha_1,alpha_2):
                            x_dist = abs(x_ind - 150) #shifting over to the center point of 150 km
                            y_dist = abs(y_ind - 150) #shifting over to the center point of 150 km
                            dist = ((x_dist**2)+(y_dist**2))**(0.5) #distance
                            if small_rad_dim < dist < big_rad_dim:
                                if y_ind == 150 and not x_ind % 2 == 0:
                                    pass
                                else:
                                    WNW_mask[y_ind,x_ind] = 4
                                    n_counts += 1
                counts[s] = n_counts
                line_endpoints[s,0,0,0] = -(small_rad_dim*math.cos(((alpha_1-270)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,0,0,1] = (small_rad_dim*math.sin(((alpha_1-270)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,0,1,0] = -(big_rad_dim*math.cos(((alpha_1-270)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,0,1,1] = (big_rad_dim*math.sin(((alpha_1-270)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,1,0,0] = -(small_rad_dim*math.cos(((alpha_2-270)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,1,0,1] = (small_rad_dim*math.sin(((alpha_2-270)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,1,1,0] = -(big_rad_dim*math.cos(((alpha_2-270)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,1,1,1] = (big_rad_dim*math.sin(((alpha_2-270)*np.pi)/180.)) + 150 #y point
            elif [alpha_1,alpha_2] == NW:
                for x_ind in range(0,300):
                    for y_ind in range(0,300):
                        if cell_within_angles(y_ind,x_ind,alpha_1,alpha_2):
                            x_dist = abs(x_ind - 150) #shifting over to the center point of 150 km
                            y_dist = abs(y_ind - 150) #shifting over to the center point of 150 km
                            dist = ((x_dist**2)+(y_dist**2))**(0.5) #distance
                            if small_rad_dim < dist < big_rad_dim:
                                NW_mask[y_ind,x_ind] = 5
                                n_counts += 1
                counts[s] = n_counts
                line_endpoints[s,0,0,0] = -(small_rad_dim*math.cos(((alpha_1-270)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,0,0,1] = (small_rad_dim*math.sin(((alpha_1-270)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,0,1,0] = -(big_rad_dim*math.cos(((alpha_1-270)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,0,1,1] = (big_rad_dim*math.sin(((alpha_1-270)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,1,0,0] = -(small_rad_dim*math.cos(((alpha_2-270)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,1,0,1] = (small_rad_dim*math.sin(((alpha_2-270)*np.pi)/180.)) + 150 #y point
                line_endpoints[s,1,1,0] = -(big_rad_dim*math.cos(((alpha_2-270)*np.pi)/180.)) + 150 #x point
                line_endpoints[s,1,1,1] = (big_rad_dim*math.sin(((alpha_2-270)*np.pi)/180.)) + 150 #y point

master_mask = np.full((300,300), float('NaN'))
for x_ind in range(300):
    for y_ind in range(300):
        if ~np.isnan(NE_mask[y_ind,x_ind]) or ~np.isnan(SW_mask[y_ind,x_ind]) or ~np.isnan(WSW_mask[y_ind,x_ind]) or ~np.isnan(WNW_mask[y_ind,x_ind]) or ~np.isnan(NW_mask[y_ind,x_ind]):
            master_mask[y_ind,x_ind] = 1

n_total = counts[0]

"""
PLOT A MAP OF THE SECTORED MASKS ON GEOGRAPHIC BACKGROUND
"""
if plot_map:
    lon = -124.211
    lat = 47.277

    map = Basemap(projection='ortho', lon_0=lon, lat_0=lat, resolution=None)
    map_lon, map_lat = map(lon,lat)

    #the 1km shift appears to place the imshow position correctly but I dont know why?? 150000,150000 shifts one grid cell up and right
    m = Basemap(projection='ortho', lon_0=lon, lat_0=lat, resolution='h', llcrnrx=-151000.,llcrnry=-151000.,urcrnrx=150000.,urcrnry=150000.)

    x, y = map(lon, lat) #NPOL marker
    m.plot(x, y, marker='s',markersize = 10, color='black')
    plt.annotate('NPOL', xy=(x+3000, y),  xycoords='data', xytext=(x+15000, y-15000), textcoords='data',arrowprops=dict(arrowstyle="-|>"),bbox=dict(pad=0, facecolor="none", edgecolor="none"))

    m.drawcoastlines(linewidth = 1.0)
    m.drawrivers(linewidth=1.5)
    m.fillcontinents(color = 'burlywood', alpha = 0.3)

    for s in range(len(sectors)): #sector, line1/line2, start/end, x/y
        for line in range(2):
            start_x, start_y = (line_endpoints[s,line,0,0]*1000)+x-150000, (line_endpoints[s,line,0,1]*1000)+y-150000
            end_x, end_y = (line_endpoints[s,line,1,0]*1000)+x-150000, (line_endpoints[s,line,1,1]*1000)+y-150000
            m.plot([start_x,end_x], [start_y,end_y], color = sector_colors_dark[s])
            flip = -1 if s in [1,2] else 1 #if south of NPOL flip y to plot negative values
            x1 = np.linspace((line_endpoints[s,0,0,0]*1000), (line_endpoints[s,1,0,0]*1000), 100)
            y1 = (flip*((small_rad_dim*1000)**2 - (x1-150000)**2)**(1/2))
            x1 = x1+x-150000
            y1 = y1+y
            m.plot(x1,y1, color = sector_colors_dark[s])
            x2 = np.linspace((line_endpoints[s,0,1,0]*1000), (line_endpoints[s,1,1,0]*1000), 100)
            y2 = (flip*((big_rad_dim*1000)**2 - (x2-150000)**2)**(1/2))
            x2 = x2+x - 150000
            y2 = y2+y
            m.plot(x2,y2, color = sector_colors_dark[s])

    cmap = matplotlib.colors.ListedColormap(sector_colors)

    m.imshow(NE_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)
    m.imshow(SW_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)
    m.imshow(WSW_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)
    m.imshow(WNW_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)
    m.imshow(NW_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)

    ax1 = plt.gca()
    ax1.set_xlim([x-70000,x+130000])
    ax1.set_ylim([y-70000,y+130000])

    #plt.show()
    plt.savefig(''.join(['/home/disk/meso-home/adelaf/OLYMPEX/olytestfigs/sectors_on_geo.png']), bbox_inches='tight',dpi=300)
    plt.close()

"""
PLOT A MAP OF THE SECTORED MASKS
"""
if plot_sectors:
    cmap = matplotlib.colors.ListedColormap(sector_colors)
    fig, (ax1) = plt.subplots(1,1)

    im = ax1.imshow(NE_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)
    im = ax1.imshow(SW_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)
    im = ax1.imshow(WSW_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)
    im = ax1.imshow(WNW_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)
    im = ax1.imshow(NW_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)

    for s in range(len(sectors)): #sector, line1/line2, start/end, x/y
        for line in range(2):
            ax1.plot([line_endpoints[s,line,0,0],line_endpoints[s,line,1,0]], [line_endpoints[s,line,0,1],line_endpoints[s,line,1,1]], color = sector_colors_dark[s])
        flip = -1 if s in [1,2] else 1 #if south of NPOL flip y to plot negative values
        x = np.linspace(line_endpoints[s,0,0,0], line_endpoints[s,1,0,0], 100)
        y = (flip*(small_rad_dim**2 - (x-150)**2)**(1/2))+150
        ax1.plot(x,y, color = sector_colors_dark[s])
        x = np.linspace(line_endpoints[s,0,1,0], line_endpoints[s,1,1,0], 100)
        y = (flip*(big_rad_dim**2 - (x-150)**2)**(1/2))+150
        ax1.plot(x,y, color = sector_colors_dark[s])

    fig.canvas.draw()
    #shift labels over 150 km to center on NPOL
    ax1.set_xticks(np.arange(90,230,20))
    ax1.set_xticklabels(np.arange(-60,80, 20))
    ax1.set_yticks(np.arange(90,230,20)) # choose which y locations to have ticks
    ax1.set_yticklabels(np.arange(-60,80, 20)) # set the labels to display at those ticks

    ax1.set_xlabel('km')
    ax1.set_ylabel('km')

    ax1.set_xlim([85,215])
    ax1.set_ylim([85,215])
    #plt.show()
    plt.savefig(''.join(['/home/disk/meso-home/adelaf/OLYMPEX/olytestfigs/sectors_only.png']), bbox_inches='tight',dpi=300)
    plt.close()



"""
SCRIPT FUNCTIONS
"""

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
    """
    OPEN UP AND GRAB DATA FROM THE FILE
    """
    date = ''.join([filelist_west[i].split(".")[1],'/'])
    outdate = filelist_west[i].split(".")[1]
    file_west = filelist_west[i].split(".")[2]
    file_east = filelist_east[i].split(".")[2]
    date_folder = ''.join([rhi_dir,date])
    file_path_west = ''.join([date_folder,filelist_west[i]])
    file_path_east = ''.join([date_folder,filelist_east[i]])
    outname_west = ''.join([date,file_west])
    outname_east = ''.join([date,file_east])

    #file = file_east if dir == 'east' else file_west

    print(''.join(['Worker starting: ',outname_west,' to ',outname_east]))

    #pull out NetCDF data for the west
    nc_fid = Dataset(file_path_west, 'r') #r means read only, cant modify data
    x_dim = nc_fid.dimensions['x'].size
    y_dim = nc_fid.dimensions['y'].size
    z_dim = nc_fid.dimensions['z'].size
    z_spacing = nc_fid.variables['z_spacing'][:]

    #get the time of scan
    date_west = num2date(nc_fid.variables['base_time'][:], units =nc_fid.variables['base_time'].units , calendar = 'standard')
    dBZ_west = np.array(nc_fid.variables['DBZ'][:,:,:,:])#dimesions [time, z, y, x] = [1,25,300,300]
    dBZ_west[dBZ_west == -9999.0] = float('NaN')
    nc_fid.close()

    #then get the east - should follow the west by about 20 minutes
    nc_fid = Dataset(file_path_east, 'r') #r means read only, cant modify data
    date_east = num2date(nc_fid.variables['base_time'][:], units =nc_fid.variables['base_time'].units , calendar = 'standard')
    dBZ_east = np.array(nc_fid.variables['DBZ'][:,:,:,:])
    dBZ_east[dBZ_east == -9999.0] = float('NaN')
    nc_fid.close()

    time_diff = (date_east-date_west).total_seconds()/60 #should output the time difference between the two scans in minutes
    dBZ_complete = np.full((1,z_dim,y_dim,x_dim),float('NaN'))
    if 10 < time_diff < 30: #put the two scans together for one scan
        dBZ_complete[:,:,:,0:int(x_dim/2)] = dBZ_west[:,:,:,0:int(x_dim/2)]
        dBZ_complete[:,:,:,int(x_dim/2):int(x_dim)] = dBZ_east[:,:,:,int(x_dim/2):int(x_dim)]

    bb_index_west = np.where(bright_bands_west[:,0]==date_west.strftime("%m/%d/%y %H:%M:%S"))[0][0]
    bb_index_east = np.where(bright_bands_east[:,0]==date_east.strftime("%m/%d/%y %H:%M:%S"))[0][0]

    bb_index_NE = np.where(bright_bands_NE[:,0]==date_east.strftime("%m/%d/%y %H:%M:%S"))[0][0]
    #bb_index_SW, bb_index_WSW, bb_index_WNW, bb_index_NW = bb_index_NE, bb_index_NE, bb_index_NE, bb_index_NE
    bb_index_SW = np.where(bright_bands_SW[:,0]==date_west.strftime("%m/%d/%y %H:%M:%S"))[0][0]
    bb_index_WSW = np.where(bright_bands_WSW[:,0]==date_west.strftime("%m/%d/%y %H:%M:%S"))[0][0]
    bb_index_WNW = np.where(bright_bands_WNW[:,0]==date_west.strftime("%m/%d/%y %H:%M:%S"))[0][0]
    bb_index_NW = np.where(bright_bands_NW[:,0]==date_west.strftime("%m/%d/%y %H:%M:%S"))[0][0]

    #set up elevation profile
    elev = np.arange(0,(z_dim*z_spacing),z_spacing)

    #if dir == 'east':
        #dBZ = dBZ_east[:,:,:,:]
        #date = date_east
    #elif dir == 'west':
        #dBZ = dBZ_west[:,:,:,:]
        #date = date_west
    #elif dir == 'all':
    dBZ = dBZ_complete[:,:,:,:]
    date = date_west
    """
    INITIALIZE OUTPUT/PLOTTING ARRAYS FOR USE LATER
    """
    plot_array = np.full((y_dim,x_dim),float('NaN'))
    elev_array = np.full((y_dim,x_dim),float('NaN'))
    dBZ_means = np.full(z_dim, float('NaN'))
    dBZ_means_west = np.full(z_dim, float('NaN'))
    dBZ_means_east = np.full(z_dim, float('NaN'))
    #ZDR_means = np.full(z_dim, float('NaN'))
    enhancement_true = False


    adj_NE = -1 if bb_index_NE == numfiles_east else 0
    adj_SW = -1 if bb_index_SW == numfiles_west else 0
    adj_WSW = -1 if bb_index_WSW == numfiles_west else 0
    adj_WNW = -1 if bb_index_WNW == numfiles_west else 0
    adj_NW = -1 if bb_index_NW == numfiles_west else 0

    if bb_index_east == numfiles_east: #adjust to avoid running past end of the file list
        adj_east = -1
    else:
        adj_east = 0
    if bb_index_west == numfiles_west: #adjust to avoid running past end of the file list
        adj_west = -1
    else:
        adj_west = 0

    enhanced_sectors = [False,False,False,False,False] #NE,SW,WSW,WNW,NW

    dBZ_NE = np.full([z_dim, y_dim, x_dim], float('NaN'))
    dBZ_SW = np.full([z_dim, y_dim, x_dim], float('NaN'))
    dBZ_WSW = np.full([z_dim, y_dim, x_dim], float('NaN'))
    dBZ_WNW = np.full([z_dim, y_dim, x_dim], float('NaN'))
    dBZ_NW = np.full([z_dim, y_dim, x_dim], float('NaN'))


    """
    REMOVE ALL DATA NOT BEING USED BY APPLYING MASTER MASK
    """
    for x_ind in range(x_dim):
        for y_ind in range(y_dim):
            #if it doesnt exist it the master mask, remove the data
            if master_mask[y_ind,x_ind] != 1:
                dBZ_east[0,:, y_ind, x_ind] = float('NaN')
                dBZ_west[0,:, y_ind, x_ind] = float('NaN')
                dBZ[0,:, y_ind, x_ind] = float('NaN')

            if ~np.isnan(NE_mask[y_ind,x_ind]):
                dBZ_NE[:,y_ind,x_ind] = dBZ_complete[0,:,y_ind,x_ind]
            if ~np.isnan(SW_mask[y_ind,x_ind]):
                dBZ_SW[:,y_ind,x_ind] = dBZ_complete[0,:,y_ind,x_ind]
            if ~np.isnan(WSW_mask[y_ind,x_ind]):
                dBZ_WSW[:,y_ind,x_ind] = dBZ_complete[0,:,y_ind,x_ind]
            if ~np.isnan(WNW_mask[y_ind,x_ind]):
                dBZ_WNW[:,y_ind,x_ind] = dBZ_complete[0,:,y_ind,x_ind]
            if ~np.isnan(NW_mask[y_ind,x_ind]):
                dBZ_NW[:,y_ind,x_ind] = dBZ_complete[0,:,y_ind,x_ind]


    #print(period_mode_NE)
    dBZ_means_NE = np.full(z_dim, float('NaN'))
    dBZ_means_SW = np.full(z_dim, float('NaN'))
    dBZ_means_WSW = np.full(z_dim, float('NaN'))
    dBZ_means_WNW = np.full(z_dim, float('NaN'))
    dBZ_means_NW = np.full(z_dim, float('NaN'))

    for i in range(z_dim):
        if ~np.isnan(dBZ_NE[i,:,:]).all():
            dBZ_means_NE[i] = np.nanmean(dBZ_NE[i,:,:])
        if ~np.isnan(dBZ_SW[i,:,:]).all():
            dBZ_means_SW[i] = np.nanmean(dBZ_SW[i,:,:])
        if ~np.isnan(dBZ_WSW[i,:,:]).all():
            dBZ_means_WSW[i] = np.nanmean(dBZ_WSW[i,:,:])
        if ~np.isnan(dBZ_WNW[i,:,:]).all():
            dBZ_means_WNW[i] = np.nanmean(dBZ_WNW[i,:,:])
        if ~np.isnan(dBZ_NW[i,:,:]).all():
            dBZ_means_NW[i] = np.nanmean(dBZ_NW[i,:,:])


    """
    IF BRIGHT BAND WAS FOUND, RUN THROUGH SEARCH ALGORITHM
    """

    if ('1' in bright_bands_SW[bb_index_SW-1:bb_index_SW+1+adj_SW,1]) or \
        ('1' in bright_bands_WSW[bb_index_WSW-1:bb_index_WSW+1+adj_WSW,1]) or \
        ('1' in bright_bands_WNW[bb_index_WNW-1:bb_index_WNW+1+adj_WNW,1]) or \
        ('1' in bright_bands_NW[bb_index_NW-1:bb_index_NW+1+adj_NW,1]) or \
        ('1' in bright_bands_NE[bb_index_NE-1:bb_index_NE+1+adj_NE,1]): #there was a bright band found from bbidv6

        #allow for selection of +/- 1 time period away from the bright band
        if bright_bands_SW[bb_index_SW,1] == '1':
            bb_ht_SW = (bright_bands_SW[bb_index_SW,2])
        elif bright_bands_west[bb_index_west-1,1] == '1':
            bb_ht_SW = (bright_bands_SW[bb_index_SW-1,2])
        elif bright_bands_SW[bb_index_SW+1,1] == '1':
            bb_ht_SW = (bright_bands_SW[bb_index_SW+1,2])
        else:
            bb_ht_SW = float('NaN')

        if bright_bands_WSW[bb_index_WSW,1] == '1':
            bb_ht_WSW = (bright_bands_WSW[bb_index_WSW,2])
        elif bright_bands_west[bb_index_west-1,1] == '1':
            bb_ht_WSW = (bright_bands_WSW[bb_index_WSW-1,2])
        elif bright_bands_WSW[bb_index_WSW+1,1] == '1':
            bb_ht_WSW = (bright_bands_WSW[bb_index_WSW+1,2])
        else:
            bb_ht_WSW = float('NaN')

        if bright_bands_WNW[bb_index_WNW,1] == '1':
            bb_ht_WNW = (bright_bands_WNW[bb_index_WNW,2])
        elif bright_bands_west[bb_index_west-1,1] == '1':
            bb_ht_WNW = (bright_bands_WNW[bb_index_WNW-1,2])
        elif bright_bands_WNW[bb_index_WNW+1,1] == '1':
            bb_ht_WNW = (bright_bands_WNW[bb_index_WNW+1,2])
        else:
            bb_ht_WNW = float('NaN')

        if bright_bands_NW[bb_index_NW,1] == '1':
            bb_ht_NW = (bright_bands_NW[bb_index_NW,2])
        elif bright_bands_west[bb_index_west-1,1] == '1':
            bb_ht_NW = (bright_bands_NW[bb_index_NW-1,2])
        elif bright_bands_NW[bb_index_NW+1,1] == '1':
            bb_ht_NW = (bright_bands_NW[bb_index_NW+1,2])
        else:
            bb_ht_NW = float('NaN')

        if bright_bands_NE[bb_index_NE,1] == '1':
            bb_ht_NE = (bright_bands_NE[bb_index_NE,2])
        elif bright_bands_NE[bb_index_NE-1,1] == '1':
            bb_ht_NE = (bright_bands_NE[bb_index_NE-1,2])
        elif bright_bands_NE[bb_index_NE+1,1] == '1':
            bb_ht_NE = (bright_bands_NE[bb_index_NE+1,2])
        else:
            bb_ht_NE = float('NaN')


        #nearest level to the bright band
        bb_lev_SW = np.int(np.round(np.float64(bb_ht_SW) * 2,0)) if ~np.isnan(np.float64(bb_ht_SW)) else float('NaN')
        bb_diff_SW = np.float64(bb_ht_SW) - (bb_lev_SW * 0.5) if ~np.isnan(np.float64(bb_ht_SW)) else float('NaN')

        bb_lev_WSW = np.int(np.round(np.float64(bb_ht_WSW) * 2,0)) if ~np.isnan(np.float64(bb_ht_WSW)) else float('NaN')
        bb_diff_WSW = np.float64(bb_ht_WSW) - (bb_lev_WSW * 0.5) if ~np.isnan(np.float64(bb_ht_WSW)) else float('NaN')

        bb_lev_WNW = np.int(np.round(np.float64(bb_ht_WNW) * 2,0)) if ~np.isnan(np.float64(bb_ht_WNW)) else float('NaN')
        bb_diff_WNW = np.float64(bb_ht_WNW) - (bb_lev_WNW * 0.5) if ~np.isnan(np.float64(bb_ht_WNW)) else float('NaN')

        bb_lev_NW = np.int(np.round(np.float64(bb_ht_NW) * 2,0)) if ~np.isnan(np.float64(bb_ht_NW)) else float('NaN')
        bb_diff_NW = np.float64(bb_ht_NW) - (bb_lev_NW * 0.5) if ~np.isnan(np.float64(bb_ht_NW)) else float('NaN')

        bb_lev_NE = np.int(np.round(np.float64(bb_ht_NE) * 2,0)) if ~np.isnan(np.float64(bb_ht_NE)) else float('NaN')
        bb_diff_NE = np.float64(bb_ht_NE) - (bb_lev_NE * 0.5) if ~np.isnan(np.float64(bb_ht_NE)) else float('NaN')

        for z_i in range(z_dim):
            if ~np.isnan(dBZ_west[0,z_i,:,:]).all():
                dBZ_means_west[z_i] = np.nanmean(dBZ_west[0,z_i,:,:])
            if ~np.isnan(dBZ_east[0,z_i,:,:]).all():
                dBZ_means_east[z_i] = np.nanmean(dBZ_east[0,z_i,:,:])
            if ~np.isnan(dBZ[0,z_i,:,:]).all():
                dBZ_means[z_i] = np.nanmean(dBZ[0,z_i,:,:])
                #ZDR_means[z_i] = np.nanmean(ZDR[0,z_i,:,:])

        where_nan_west = np.argwhere(~np.isnan(dBZ_means_west[:]))
        where_nan_east = np.argwhere(~np.isnan(dBZ_means_east[:]))
        if len(where_nan_west) > 0:
            top_lev_west = np.max(where_nan_west)
        else:
            top_lev_west = z_dim
        if len(where_nan_east) > 0:
            top_lev_east = np.max(where_nan_east)
        else:
            top_lev_east = z_dim


        n_found = 0 #total enhanced grid cells found

        n_NE_found = 0
        n_SW_found = 0
        n_WSW_found = 0
        n_WNW_found = 0
        n_NW_found = 0

        prcnt_cells_met = 0
        enhancement_found = 0

        low_enhance_lev = []
        high_enhance_lev = []
        low_enhance_2_lev = []
        high_enhance_2_lev = []

        levels = np.full((y_dim,x_dim,6),float('NaN')) #store low and high levels for seccondary enhancement at each grid cell
        #low, high, low2, high2, low 1 enhancement associated with low mode or high mode (0/1), low 2 enhancement associated with high or low mode
        NE_sector_levels = np.full((y_dim,x_dim,6),float('NaN'))
        SW_sector_levels = np.full((y_dim,x_dim,6),float('NaN'))
        WSW_sector_levels = np.full((y_dim,x_dim,6),float('NaN'))
        WNW_sector_levels = np.full((y_dim,x_dim,6),float('NaN'))
        NW_sector_levels = np.full((y_dim,x_dim,6),float('NaN'))

        NE_enhancement = np.full(4,float('NaN')) #mean low enhance,mean high enhance, mean low2 enahance, mean high2 enhance
        SW_enhancement = np.full(4,float('NaN'))
        WSW_enhancement = np.full(4,float('NaN'))
        WNW_enhancement = np.full(4,float('NaN'))
        NW_enhancement = np.full(4,float('NaN'))


        """
        INITIAL GRID BY GRID SEARCH FOR AN ENHANCEMENT IN EACH COLUMN - STORE VALUES TO RECALL LATER
        """
        for x_ind in range(x_dim):
            for y_ind in range(y_dim):

                if ~np.isnan(NE_mask[y_ind,x_ind]):
                    top_lev = top_lev_east
                    bb_lev = bb_lev_NE
                elif ~np.isnan(SW_mask[y_ind,x_ind]):
                    top_lev = top_lev_west
                    bb_lev = bb_lev_SW
                elif ~np.isnan(WSW_mask[y_ind,x_ind]):
                    top_lev = top_lev_west
                    bb_lev = bb_lev_WSW
                elif ~np.isnan(WNW_mask[y_ind,x_ind]):
                    top_lev = top_lev_west
                    bb_lev = bb_lev_WNW
                elif ~np.isnan(NW_mask[y_ind,x_ind]):
                    top_lev = top_lev_west
                    bb_lev = bb_lev_NW
                else:
                    top_lev = z_dim
                    bb_lev = float('NaN')


                #if x_ind < (x_dim/2): #left half of domain => west
                    #top_lev = top_lev_west
                    #bb_lev = bb_lev_west
                #else: #right half => east
                    #top_lev = top_lev_east
                    #bb_lev = bb_lev_east

                deltas = np.full(top_lev,float('NaN'))
                dBZ_column = np.full(top_lev,float('NaN'))
                if ~np.isnan(bb_lev):
                    if ~np.isnan(dBZ[0,bb_lev:top_lev+1,y_ind,x_ind]).all():
                        for z in range(1,top_lev):
                            deltas[z] = ((dBZ[0,z,y_ind,x_ind]-dBZ[0,z-1,y_ind,x_ind])) #change in dBZ from lower level to current level
                            dBZ_column[z] = dBZ[0,z,y_ind,x_ind]
                        #first look for a decrease above the bright band
                        decrease_lev = next((z_ind for z_ind, v in enumerate(deltas) if v < 0 and z_ind >= bb_lev), z_dim)
                        decrease_lev_2 = next((z_ind for z_ind, v in enumerate(deltas) if v < 0 and z_ind > decrease_lev), z_dim)
                        #above bright band, where does delta become positive and exceed threshold.
                        low_enhance = next((z_ind for z_ind, v in enumerate(deltas) if v >= excd_val_low and z_ind > decrease_lev_2 and min_dBZ < dBZ_column[z_ind]), float('NaN'))
                        levels[y_ind,x_ind,0] = low_enhance

                        if ~np.isnan(low_enhance):
                            high_enhance = next((z_ind-1 for z_ind, v in enumerate(dBZ_column) if (v < dBZ_column[low_enhance-1] or np.isnan(v)) and z_ind > low_enhance), low_enhance)
                            levels[y_ind,x_ind,1] = high_enhance
                            low_enhance_2 = next((z_ind for z_ind, v in enumerate(deltas) if v >= excd_val_low and z_ind > high_enhance and min_dBZ < dBZ_column[z_ind]), float('NaN'))
                            if ~np.isnan(low_enhance_2):
                                levels[y_ind,x_ind,2] = low_enhance_2
                                high_enhance_2 = next((z_ind-1 for z_ind, v in enumerate(dBZ_column) if (v < dBZ_column[low_enhance_2-1] or np.isnan(v)) and z_ind > low_enhance_2), low_enhance_2)
                                levels[y_ind,x_ind,3] = high_enhance_2
                        else:
                            high_enhance = float('NaN')
                    else:
                        low_enhance = float('NaN')
                        high_enhance = float('NaN')
                else:
                    low_enhance = float('NaN')
                    high_enhance = float('NaN')

                if ~np.isnan(levels[y_ind,x_ind,0]): #found at least one enhanced layer
                    plot_array[y_ind,x_ind] = 1
                    n_found += 1

                    """
                    BREAK UP INTO SECTORS (NE, SW, WSW, WNW, NW) TO ANALYZE EACH OF THEM INDIVIDUALLY IN ORDER TO DETERMINE ENHANCEMNET FOR EACH SECTOR
                    """
                    if ~np.isnan(NE_mask[y_ind,x_ind]):
                        NE_sector_levels[y_ind, x_ind, :] = levels[y_ind, x_ind, :]
                        n_NE_found += 1
                    elif ~np.isnan(SW_mask[y_ind,x_ind]):
                        SW_sector_levels[y_ind, x_ind, :] = levels[y_ind, x_ind, :]
                        n_SW_found += 1
                    elif ~np.isnan(WSW_mask[y_ind,x_ind]):
                        WSW_sector_levels[y_ind, x_ind, :] = levels[y_ind, x_ind, :]
                        n_WSW_found += 1
                    elif ~np.isnan(WNW_mask[y_ind,x_ind]):
                        WNW_sector_levels[y_ind, x_ind, :] = levels[y_ind, x_ind, :]
                        n_WNW_found += 1
                    elif ~np.isnan(NW_mask[y_ind,x_ind]):
                        NW_sector_levels[y_ind, x_ind, :] = levels[y_ind, x_ind, :]
                        n_NW_found += 1

        low_lev_enh = False
        high_lev_enh = False
        two_lev_enh = False



        """
        MODE SELECTION, LOOP THROUGH EACH SECTOR
        """

        for s in range(len(sectors)):
            alpha_1 = sectors[s][0]
            alpha_2 = sectors[s][1]
            if [alpha_1,alpha_2] == NE:
                mask,sector_levels = NE_mask[:,:], NE_sector_levels[:,:,:]
                n = n_NE_found
            elif [alpha_1,alpha_2] == SW:
                mask,sector_levels = SW_mask[:,:], SW_sector_levels[:,:,:]
                n = n_SW_found
            elif [alpha_1,alpha_2] == WSW:
                mask,sector_levels = WSW_mask[:,:], WSW_sector_levels[:,:,:]
                n = n_WSW_found
            elif [alpha_1,alpha_2] == WNW:
                mask,sector_levels = WNW_mask[:,:], WNW_sector_levels[:,:,:]
                n = n_WNW_found
            elif [alpha_1,alpha_2] == NW:
                mask,sector_levels = NW_mask[:,:], NW_sector_levels[:,:,:]
                n = n_NW_found

            n_1_found = 0 #total found in lower layer
            n_2_found = 0 #total found in upper layer

            #reset to empty arrays to pick out levels
            low_enhance_lev = []
            low_enhance_2_lev = []

            if n>0:
                prcnt_cells_met = np.float64(format((n/n_total)*100,'.2f'))
                low_mode = int(mode1(sector_levels[:,:,[0,2]])[0])
                low_mode_2 = int(mode2(sector_levels[:,:,[0,2]])[0])

                modesOK = True
                if low_mode_2 < low_mode:
                    low_mode, low_mode_2 = low_mode_2, low_mode
                elif low_mode_2 == low_mode: #returns the first mode when it cant find a second mode, i.e. only one layer shows enhancement
                    modesOK = False

                big_layer = False #if the two modes are close enough together, use one larger layer
                if abs(low_mode_2 - low_mode) <= modes_within:
                    big_layer = True


                """
                SORT ENHANCEMENT TO MODE LAYERS - HIGH, LOW, OR ALL & BUILD ENHANCEMENT ARRAYS FOR AVERAGING/PERCENTAGES
                """

                for x_ind in range(x_dim):
                    for y_ind in range(y_dim):
                        if not big_layer: #look for the first enhanced layer
                            if low_mode-min_sep <= sector_levels[y_ind,x_ind,0] <= low_mode+min_sep:
                                sector_levels[y_ind,x_ind,4] = 0 #grid cell's low enhancement is in the low mode
                                low_enhance_lev.append(sector_levels[y_ind,x_ind,0])
                                high_enhance_lev.append(sector_levels[y_ind,x_ind,1])
                                n_1_found += 1
                            elif low_mode_2-min_sep <= sector_levels[y_ind,x_ind,0] <= low_mode_2+min_sep:
                                sector_levels[y_ind,x_ind,4] = 1 #grid cell's low enhancement is in the upper mode
                                low_enhance_2_lev.append(sector_levels[y_ind,x_ind,0])
                                high_enhance_2_lev.append(sector_levels[y_ind,x_ind,1])
                                n_2_found += 1
                            if modesOK: #look for the second enhanced layer
                                if low_mode_2-min_sep <= sector_levels[y_ind,x_ind,2] <= low_mode_2+min_sep:
                                    sector_levels[y_ind,x_ind,5] = 1 #grid cell's high enhancement is in the upper mode
                                    low_enhance_2_lev.append(sector_levels[y_ind,x_ind,2])
                                    high_enhance_2_lev.append(sector_levels[y_ind,x_ind,3])
                                    n_2_found += 1
                                elif low_mode-min_sep <= sector_levels[y_ind,x_ind,2] <= low_mode+min_sep:
                                    sector_levels[y_ind,x_ind,5] = 0 #grid cell's high enhancement is in the low mode
                                    low_enhance_lev.append(sector_levels[y_ind,x_ind,2])
                                    high_enhance_lev.append(sector_levels[y_ind,x_ind,3])
                                    n_1_found += 1
                        elif big_layer: #look for a single big enhanced layer
                            if low_mode-min_sep <= sector_levels[y_ind,x_ind,0] <= low_mode_2+min_sep:
                                sector_levels[y_ind,x_ind,4] = 2 #grid cell's lower enhancement falls within the big mode
                                low_enhance_lev.append(sector_levels[y_ind,x_ind,0])
                                high_enhance_lev.append(sector_levels[y_ind,x_ind,1])
                                n_1_found += 1
                            elif low_mode-min_sep <= sector_levels[y_ind,x_ind,2] <= low_mode_2+min_sep:
                                sector_levels[y_ind,x_ind,5] = 2 #grid cell's upper enhancement falls within the big mode
                                low_enhance_lev.append(sector_levels[y_ind,x_ind,2])
                                high_enhance_lev.append(sector_levels[y_ind,x_ind,3])
                                n_1_found += 1

                prcnt_cells_met_1 = np.float64(format((n_1_found/n_total)*100,'.2f'))
                prcnt_cells_met_2 = np.float64(format((n_2_found/n_total)*100,'.2f'))

                if prcnt_cells_met_1 > enhance_threshold:
                    mean_low_enhance_lev = np.float64(format(np.nanmean(low_enhance_lev)*0.5, '.2f'))
                    mean_high_enhance_lev = np.float64(format(np.nanmean(high_enhance_lev)*0.5, '.2f'))
                else:
                    mean_low_enhance_lev = float('NaN')
                    mean_high_enhance_lev = float('NaN')
                if prcnt_cells_met_2 > enhance_threshold:
                    mean_low_enhance_2_lev = np.float64(format(np.nanmean(low_enhance_2_lev)*0.5, '.2f'))
                    mean_high_enhance_2_lev =  np.float64(format(np.nanmean(high_enhance_2_lev)*0.5, '.2f'))
                else:
                    mean_low_enhance_2_lev = float('NaN')
                    mean_high_enhance_2_lev = float('NaN')

                """
                ASSIGN RESULTS TO APPROPRIATE ARRAY
                """

                if [alpha_1,alpha_2] == NE:
                    NE_enhancement[0] = mean_low_enhance_lev
                    NE_enhancement[1] = mean_high_enhance_lev
                    NE_enhancement[2] = mean_low_enhance_2_lev
                    NE_enhancement[3] = mean_high_enhance_2_lev
                elif [alpha_1,alpha_2] == SW:
                    SW_enhancement[0] = mean_low_enhance_lev
                    SW_enhancement[1] = mean_high_enhance_lev
                    SW_enhancement[2] = mean_low_enhance_2_lev
                    SW_enhancement[3] = mean_high_enhance_2_lev
                elif [alpha_1,alpha_2] == WSW:
                    WSW_enhancement[0] = mean_low_enhance_lev
                    WSW_enhancement[1] = mean_high_enhance_lev
                    WSW_enhancement[2] = mean_low_enhance_2_lev
                    WSW_enhancement[3] = mean_high_enhance_2_lev
                elif [alpha_1,alpha_2] == WNW:
                    WNW_enhancement[0] = mean_low_enhance_lev
                    WNW_enhancement[1] = mean_high_enhance_lev
                    WNW_enhancement[2] = mean_low_enhance_2_lev
                    WNW_enhancement[3] = mean_high_enhance_2_lev
                elif [alpha_1,alpha_2] == NW:
                    NW_enhancement[0] = mean_low_enhance_lev
                    NW_enhancement[1] = mean_high_enhance_lev
                    NW_enhancement[2] = mean_low_enhance_2_lev
                    NW_enhancement[3] = mean_high_enhance_2_lev
            else:
                pass
                """
                prcnt_cells_met = 0
                prcnt_cells_met_1 = 0
                prcnt_cells_met_2 = 0
                mean_low_enhance_lev = float('NaN')
                mean_high_enhance_lev = float('NaN')
                mean_low_enhance_2_lev = float('NaN')
                mean_high_enhance_2_lev = float('NaN')
                """

        low_lines = [NE_enhancement[0],SW_enhancement[0],WSW_enhancement[0],WNW_enhancement[0],NW_enhancement[0]]
        high_lines = [NE_enhancement[1],SW_enhancement[1],WSW_enhancement[1],WNW_enhancement[1],NW_enhancement[1]]
        low_2_lines = [NE_enhancement[2],SW_enhancement[2],WSW_enhancement[2],WNW_enhancement[2],NW_enhancement[2]]
        high_2_lines = [NE_enhancement[3],SW_enhancement[3],WSW_enhancement[3],WNW_enhancement[3],NW_enhancement[3]]

        """
        UPDATE PLOTTING ARRAY
        """

        for x_ind in range(x_dim):
            for y_ind in range(y_dim):
                if plot_array[y_ind,x_ind] == 1: #enhancement found at that grid cell
                    #does if fall within the NE mask and was an enhancement found in that sector for a low-level mode
                    """
                    NE SECTOR
                    """
                    if ~np.isnan(NE_mask[y_ind,x_ind]) and ~np.isnan(NE_enhancement[0]):
                        plot_array[y_ind,x_ind] = 2
                        low_lev_enh = True
                        enhanced_sectors[0] = True
                    #does if fall within the NE mask and was an enhancement found in that sector for a upper-level mode
                    if ~np.isnan(NE_mask[y_ind,x_ind]) and ~np.isnan(NE_enhancement[2]):
                        if plot_array[y_ind,x_ind] == 2: #enhancement already found in the upper lower level => two level enhanced cell
                            plot_array[y_ind,x_ind] = 4
                            two_lev_enh = True
                            enhanced_sectors[0] = True
                        else:
                            plot_array[y_ind,x_ind] == 3
                            high_lev_enh = True
                            enhanced_sectors[0] = True
                    """
                    SW SECTOR
                    """
                    if ~np.isnan(SW_mask[y_ind,x_ind]) and ~np.isnan(SW_enhancement[0]):
                        plot_array[y_ind,x_ind] = 2
                        low_lev_enh = True
                        enhanced_sectors[1] = True
                    if ~np.isnan(SW_mask[y_ind,x_ind]) and ~np.isnan(SW_enhancement[2]):
                        if plot_array[y_ind,x_ind] == 2:
                            plot_array[y_ind,x_ind] = 4
                            two_lev_enh = True
                            enhanced_sectors[1] = True
                        else:
                            plot_array[y_ind,x_ind] == 3
                            high_lev_enh = True
                            enhanced_sectors[1] = True
                    """
                    WSW SECTOR
                    """
                    if ~np.isnan(WSW_mask[y_ind,x_ind]) and ~np.isnan(WSW_enhancement[0]):
                        plot_array[y_ind,x_ind] = 2
                        low_lev_enh = True
                        enhanced_sectors[2] = True
                    if ~np.isnan(WSW_mask[y_ind,x_ind]) and ~np.isnan(WSW_enhancement[2]):
                        if plot_array[y_ind,x_ind] == 2:
                            plot_array[y_ind,x_ind] = 4
                            two_lev_enh = True
                            enhanced_sectors[2] = True
                        else:
                            plot_array[y_ind,x_ind] == 3
                            high_lev_enh = True
                            enhanced_sectors[2] = True
                    """
                    WNW SECTOR
                    """
                    if ~np.isnan(WNW_mask[y_ind,x_ind]) and ~np.isnan(WNW_enhancement[0]):
                        plot_array[y_ind,x_ind] = 2
                        low_lev_enh = True
                        enhanced_sectors[3] = True
                    if ~np.isnan(WNW_mask[y_ind,x_ind]) and ~np.isnan(WNW_enhancement[2]):
                        if plot_array[y_ind,x_ind] == 2:
                            plot_array[y_ind,x_ind] = 4
                            two_lev_enh = True
                            enhanced_sectors[3] = True
                        else:
                            plot_array[y_ind,x_ind] == 3
                            high_lev_enh = True
                            enhanced_sectors[3] = True
                    """
                    NW SECTOR
                    """
                    if ~np.isnan(NW_mask[y_ind,x_ind]) and ~np.isnan(NW_enhancement[0]):
                        plot_array[y_ind,x_ind] = 2
                        low_lev_enh = True
                        enhanced_sectors[4] = True
                    if ~np.isnan(NW_mask[y_ind,x_ind]) and ~np.isnan(NW_enhancement[2]):
                        if plot_array[y_ind,x_ind] == 2:
                            plot_array[y_ind,x_ind] = 4
                            two_lev_enh = True
                            enhanced_sectors[4] = True
                        else:
                            plot_array[y_ind,x_ind] == 3
                            high_lev_enh = True
                            enhanced_sectors[4] = True


        """
        TRY TO NOT USE THIS SECTION -- LEAVING TILL EVALUATION OF WHETHER THE SECTORED METHOD SEEMS TO WORK WELL ENOUGH
        DENSITY ASSESMENT - EVALUATE GRID BY GRID FOR HIGH DENSITY REGIONS OF SECONDARY ENHANCEMENT
        IDNETIFY REGIONS OF ENHANCEMENT IN (X,Y) AND Z, BUILD PLOTTING ARRAY
        """

        """
        for x_ind in range(0+half_width,x_dim-half_width):
            for y_ind in range(0+half_width,y_dim-half_width):
                subset = plot_array[y_ind-half_width:y_ind+half_width+1,x_ind-half_width:x_ind+half_width+1]
                subset_1 = plot_array_1[y_ind-half_width:y_ind+half_width+1,x_ind-half_width:x_ind+half_width+1]
                subset_2 = plot_array_2[y_ind-half_width:y_ind+half_width+1,x_ind-half_width:x_ind+half_width+1]
                total_met = np.nansum(compare_nan_array(np.greater, subset, 0))
                #which enhanced level satisfies the grid box criteria
                total_met_1 = np.nansum(compare_nan_array(np.greater, subset_1, 0))
                total_met_2 = np.nansum(compare_nan_array(np.greater, subset_2, 0))

                if total_met_1 >= ((grid_density/100)*total_grid_cells): #set cells that met criteria = 2
                    enhancement_true = True
                    enhancement_found = 1
                    for a in range(grid_size):
                        for b in range(grid_size):
                            if subset_1[a,b] == 1: #a grid cell that had originally found an enhancement layer in layer 1
                                subset_1[a,b] = 2
                            if subset[a,b] == 3 or subset[a,b] == 4:
                                subset[a,b] = 4
                            elif subset[a,b] == 1:
                                subset[a,b] = 2
                            low_lev_enh = True
                    #plot_array[y_ind-half_width:y_ind+half_width,x_ind-half_width:x_ind+half_width] = subset_1

                if total_met_2 >= ((grid_density/100)*total_grid_cells): #set cells that met criteria = 2
                    enhancement_true = True
                    enhancement_found = 1
                    for a in range(grid_size):
                        for b in range(grid_size):
                            if subset_2[a,b] == 1:  #grid cell that had an enhancement in layer 2 that was already identified by layer 1
                                subset_2[a,b] = 3
                            if subset[a,b] == 2 or subset[a,b] == 4:
                                subset[a,b] = 4
                                two_lev_enh = True
                            elif subset[a,b] == 1: #grid cell that had an enhancement in layer 2 only
                                subset[a,b] = 3
                                high_lev_enh = True

                #update plotting arrays with new values
                plot_array_1[y_ind-half_width:y_ind+half_width+1,x_ind-half_width:x_ind+half_width+1] = subset_1[:,:]
                plot_array_2[y_ind-half_width:y_ind+half_width+1,x_ind-half_width:x_ind+half_width+1] = subset_2[:,:]
                plot_array[y_ind-half_width:y_ind+half_width+1,x_ind-half_width:x_ind+half_width+1] = subset[:,:]

        """


        """
        COLLECT ENHANCEMENT LAYERS AND VALUES FROM THE HIGH DENSITY REGIONS - (ONLY USES CELLS THAT FALL WITHIN SECTORS THAT FOUND ENHANCEMENT)
        COLLECT BY SECTOR FOR STATISTICAL ANALYSIS LATER
        """

        for s in range(len(sectors)):
            alpha_1 = sectors[s][0]
            alpha_2 = sectors[s][1]
            if [alpha_1,alpha_2] == NE:
                mask,sector_levels = NE_mask[:,:], NE_sector_levels[:,:,:]
                date = date_east
            elif [alpha_1,alpha_2] == SW:
                mask,sector_levels = SW_mask[:,:], SW_sector_levels[:,:,:]
                date = date_west
            elif [alpha_1,alpha_2] == WSW:
                mask,sector_levels = WSW_mask[:,:], WSW_sector_levels[:,:,:]
                date = date_west
            elif [alpha_1,alpha_2] == WNW:
                mask,sector_levels = WNW_mask[:,:], WNW_sector_levels[:,:,:]
                date = date_west
            elif [alpha_1,alpha_2] == NW:
                mask,sector_levels = NW_mask[:,:], NW_sector_levels[:,:,:]
                date = date_west

            #enhancement levels
            grid_low_enhance_lev = []
            grid_high_enhance_lev = []
            grid_low_enhance_2_lev = []
            grid_high_enhance_2_lev = []

            #dBZ values
            grid_enhance_vals = []
            grid_enhance_2_vals = []

            for x_ind in range(x_dim):
                for y_ind in range(y_dim):
                    if (plot_array[y_ind,x_ind] == 2 or plot_array[y_ind,x_ind] == 4) and (sector_levels[y_ind,x_ind,4] == 0 or sector_levels[y_ind,x_ind,4] == 2): #enhancement found in grid approach and the enhancement is in the lower mode
                        grid_low_enhance_lev.append(sector_levels[y_ind,x_ind,0])
                        grid_high_enhance_lev.append(sector_levels[y_ind,x_ind,1])
                        grid_enhance_vals.append(np.nanmean(dBZ[0,int(sector_levels[y_ind,x_ind,0]):int(sector_levels[y_ind,x_ind,1]+1), y_ind, x_ind]))
                    elif (plot_array[y_ind,x_ind] == 2 or plot_array[y_ind,x_ind] == 4) and (sector_levels[y_ind,x_ind,5] == 0 or sector_levels[y_ind,x_ind,5]) == 2: #enhancement found in grid approach and the enhancement is in the upper mode
                        grid_low_enhance_lev.append(sector_levels[y_ind,x_ind,2])
                        grid_high_enhance_lev.append(sector_levels[y_ind,x_ind,3])
                        grid_enhance_vals.append(np.nanmean(dBZ[0,int(sector_levels[y_ind,x_ind,2]):int(sector_levels[y_ind,x_ind,3]+1), y_ind, x_ind]))

                    if (plot_array[y_ind,x_ind] == 3 or plot_array[y_ind,x_ind] == 4) and sector_levels[y_ind,x_ind,4] == 1:
                        grid_low_enhance_2_lev.append(sector_levels[y_ind,x_ind,0])
                        grid_high_enhance_2_lev.append(sector_levels[y_ind,x_ind,1])
                        grid_enhance_2_vals.append(np.nanmean(dBZ[0,int(sector_levels[y_ind,x_ind,0]):int(sector_levels[y_ind,x_ind,1]+1), y_ind, x_ind]))
                    elif (plot_array[y_ind,x_ind] == 3 or plot_array[y_ind,x_ind] == 4) and sector_levels[y_ind,x_ind,5] == 1:
                        grid_low_enhance_2_lev.append(sector_levels[y_ind,x_ind,2])
                        grid_high_enhance_2_lev.append(sector_levels[y_ind,x_ind,3])
                        grid_enhance_2_vals.append(np.nanmean(dBZ[0,int(sector_levels[y_ind,x_ind,2]):int(sector_levels[y_ind,x_ind,3]+1), y_ind, x_ind]))

            """
            ORGANIZE REFLECTIVITY VALUES OF SECONDARY ENHANCEMENT
            """
            date_secondary_vals = np.array([1,2,3])
            for n in range(len(grid_enhance_vals)):
                array_to_add = np.array([date.strftime("%m/%d/%y %H:%M:%S"),grid_enhance_vals[n],float('NaN')])
                date_secondary_vals = np.vstack((date_secondary_vals,array_to_add))
            for n in range(len(grid_enhance_2_vals)):
                array_to_add = np.array([date.strftime("%m/%d/%y %H:%M:%S"),float('NaN'),grid_enhance_2_vals[n]])
                date_secondary_vals = np.vstack((date_secondary_vals,array_to_add))
            if date_secondary_vals.shape[0] > 3:
                date_secondary_vals = date_secondary_vals[1:date_secondary_vals.shape[0],:]
            else:
                date_secondary_vals = np.array([date.strftime("%m/%d/%y %H:%M:%S"),float('NaN'),float('NaN')])

            """
            OUTPUT DATA FOR EACH SECTOR
            """
            if [alpha_1,alpha_2] == NE:
                row_to_append_NE = np.array([date.strftime("%m/%d/%y %H:%M:%S"),int(enhanced_sectors[0]),NE_enhancement[0],NE_enhancement[1],NE_enhancement[2],NE_enhancement[3],np.float64(format((n_NE_found/n_total)*100,'.2f')),bb_lev_NE])
                date_secondary_vals_NE = date_secondary_vals[:]
            elif [alpha_1,alpha_2] == SW:
                row_to_append_SW = np.array([date.strftime("%m/%d/%y %H:%M:%S"),int(enhanced_sectors[1]),SW_enhancement[0],SW_enhancement[1],SW_enhancement[2],SW_enhancement[3],np.float64(format((n_SW_found/n_total)*100,'.2f')),bb_lev_SW])
                date_secondary_vals_SW = date_secondary_vals[:]
            elif [alpha_1,alpha_2] == WSW:
                row_to_append_WSW = np.array([date.strftime("%m/%d/%y %H:%M:%S"),int(enhanced_sectors[2]),WSW_enhancement[0],WSW_enhancement[1],WSW_enhancement[2],WSW_enhancement[3],np.float64(format((n_WSW_found/n_total)*100,'.2f')),bb_lev_WSW])
                date_secondary_vals_WSW = date_secondary_vals[:]
            elif [alpha_1,alpha_2] == WNW:
                row_to_append_WNW = np.array([date.strftime("%m/%d/%y %H:%M:%S"),int(enhanced_sectors[3]),WNW_enhancement[0],WNW_enhancement[1],WNW_enhancement[2],WNW_enhancement[3],np.float64(format((n_WNW_found/n_total)*100,'.2f')),bb_lev_WNW])
                date_secondary_vals_WNW = date_secondary_vals[:]
            elif [alpha_1,alpha_2] == NW:
                row_to_append_NW = np.array([date.strftime("%m/%d/%y %H:%M:%S"),int(enhanced_sectors[4]),NW_enhancement[0],NW_enhancement[1],NW_enhancement[2],NW_enhancement[3],np.float64(format((n_NW_found/n_total)*100,'.2f')),bb_lev_NW])
                date_secondary_vals_NW = date_secondary_vals[:]

            """
            OUTPUT REFLECTIVITY VALUES OF SECONDARY ENHANCEMENT
            """
            date_secondary_vals = np.array([1,2,3])
            for n in range(len(grid_enhance_vals)):
                array_to_add = np.array([date.strftime("%m/%d/%y %H:%M:%S"),grid_enhance_vals[n],float('NaN')])
                date_secondary_vals = np.vstack((date_secondary_vals,array_to_add))
            for n in range(len(grid_enhance_2_vals)):
                array_to_add = np.array([date.strftime("%m/%d/%y %H:%M:%S"),float('NaN'),grid_enhance_2_vals[n]])
                date_secondary_vals = np.vstack((date_secondary_vals,array_to_add))
            if date_secondary_vals.shape[0] > 3:
                date_secondary_vals = date_secondary_vals[1:date_secondary_vals.shape[0],:]
            else:
                date_secondary_vals = np.array([date.strftime("%m/%d/%y %H:%M:%S"),float('NaN'),float('NaN')])

    else:
        row_to_append_NE = np.array([date_east.strftime("%m/%d/%y %H:%M:%S"),0,float('NaN'),float('NaN'), float('NaN'),float('NaN'), float('NaN'),float('NaN')])
        date_secondary_vals_NE = np.array([date_east.strftime("%m/%d/%y %H:%M:%S"),float('NaN'),float('NaN')])
        row_to_append_SW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),0,float('NaN'),float('NaN'), float('NaN'),float('NaN'), float('NaN'),float('NaN')])
        date_secondary_vals_SW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),float('NaN'),float('NaN')])
        row_to_append_WSW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),0,float('NaN'),float('NaN'), float('NaN'),float('NaN'), float('NaN'),float('NaN')])
        date_secondary_vals_WSW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),float('NaN'),float('NaN')])
        row_to_append_WNW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),0,float('NaN'),float('NaN'), float('NaN'),float('NaN'), float('NaN'),float('NaN')])
        date_secondary_vals_WNW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),float('NaN'),float('NaN')])
        row_to_append_NW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),0,float('NaN'),float('NaN'), float('NaN'),float('NaN'), float('NaN'),float('NaN')])
        date_secondary_vals_NW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),float('NaN'),float('NaN')])

    if print_results:
        print('NE', row_to_append_NE)
        print('SW', row_to_append_SW)
        print('WSW', row_to_append_WSW)
        print('WNW', row_to_append_WNW)
        print('NW', row_to_append_NW)


    """
    PLOTTING
    """

    plot_list = ['SW', 'WSW', 'WNW', 'NW', 'NE']

    if plot_figs:
        datetime_object = datetime.datetime.strptime(row_to_append_SW[0], "%m/%d/%y %H:%M:%S")
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
        #if dir == 'east':
            #gridbox = matplotlib.patches.Rectangle((210-grid_size,90), grid_size,grid_size, linewidth=1, edgecolor='r', facecolor='none')
        #elif dir == 'west':
            #gridbox = matplotlib.patches.Rectangle((90,90), grid_size,grid_size, linewidth=1, edgecolor='r', facecolor='none')
        bigRadius = plt.Circle((150,150), big_rad_dim, color = 'black', fill = False)
        cmap = matplotlib.colors.ListedColormap(['dimgrey','#1b9e77','darkviolet','#e66101'])
        fig, (ax1,ax2) = plt.subplots(1,2)
        """
        PLOT (X,Y) FIGURE FOR CELLS MEETING CRITEREIA AND WHICH LEVEL THEY MET CRITERIA
        """
        if plot_xy and ('1' in bright_bands_SW[bb_index_SW-1:bb_index_SW+1+adj_SW,1] or \
            '1' in bright_bands_WSW[bb_index_WSW-1:bb_index_WSW+1+adj_WSW,1] or \
            '1' in bright_bands_WNW[bb_index_WNW-1:bb_index_WNW+1+adj_WNW,1] or \
            '1' in bright_bands_NW[bb_index_NW-1:bb_index_NW+1+adj_NW,1] or \
            '1' in bright_bands_NE[bb_index_NE-1:bb_index_NE+1+adj_NE,1]):

            fig, (ax1) = plt.subplots(1,1)
            im = ax1.imshow(plot_array, origin = 'Lower', cmap = cmap)
            ax1.add_artist(smallRadius)
            ax1.add_artist(Radius20)
            ax1.add_artist(Radius30)
            ax1.add_artist(Radius40)
            ax1.add_artist(Radius50)
            ax1.add_artist(bigRadius)
            #ax1.add_artist(gridbox)
            #if dir == 'east':
                #ax1.set_xlim([150,215])
            #elif dir == 'west':
                #ax1.set_xlim([85,150])
            #elif dir =='all':
            ax1.set_xlim([85,215])
            ax1.set_ylim([85,215])

            #light up regions that are enhanced
            for s in range(len(sectors)): #sector, line1/line2, start/end, x/y
                if enhanced_sectors[s]:
                    for line in range(2):
                        ax1.plot([line_endpoints[s,line,0,0],line_endpoints[s,line,1,0]], [line_endpoints[s,line,0,1],line_endpoints[s,line,1,1]], color = sector_colors_dark[s],linewidth=1.5)
                    flip = -1 if s in [1,2] else 1 #if south of NPOL flip y to plot negative values
                    x = np.linspace(line_endpoints[s,0,0,0], line_endpoints[s,1,0,0], 100)
                    y = (flip*(small_rad_dim**2 - (x-150)**2)**(1/2))+150
                    ax1.plot(x,y, color = sector_colors_dark[s],linewidth=1.5)
                    x = np.linspace(line_endpoints[s,0,1,0], line_endpoints[s,1,1,0], 100)
                    y = (flip*(big_rad_dim**2 - (x-150)**2)**(1/2))+150
                    ax1.plot(x,y, color = sector_colors_dark[s],linewidth=1.5)

            fig.canvas.draw()
            #shift labels over 150 km to center on NPOL
            #dir == 'east':
                #ax1.set_xticks(np.arange(150,230,20))  # choose which x locations to have ticks
                #ax1.set_xticklabels(np.arange(0,80, 20))  # set the labels to display at those ticks
                #date = date_east
            #elif dir == 'west':
                #ax1.set_xticks(np.arange(90,170,20))
                #ax1.set_xticklabels(np.arange(60,-20, -20))
                #date = date_west
            #elif dir == 'all':
            ax1.set_xticks(np.arange(90,230,20))
            ax1.set_xticklabels(np.arange(-60,80, 20))
            date = date_west
            ax1.set_yticks(np.arange(90,230,20)) # choose which y locations to have ticks
            ax1.set_yticklabels(np.arange(-60,80, 20)) # set the labels to display at those ticks
            #if dir == 'west':
                #ax1.yaxis.tick_right()
                #ax1.yaxis.set_label_position("right")
            ax1.set_xlabel('km')
            ax1.set_ylabel('km')
            im.set_clim(0,5) #set the colorbar limits
            cbar = fig.colorbar(im, ax = ax1, orientation = 'vertical', fraction=0.046, pad=0.12, boundaries=np.linspace(0.5, 4.5, 5))
            cbar.set_ticks([1,2,3,4])
            labels = ['Enh.','Lower\nEnh.','Upper\nEnh.','2-Layer\nEnh.']
            cbar.set_ticklabels(labels)
            ax1.set_title(date.strftime(''.join(['%m/%d/%y %H:%M:%S\n'])))
            plt.tight_layout()
            fig.subplots_adjust(wspace=1.0)
            #plt.show()
            plt.savefig(''.join(['/home/disk/meso-home/adelaf/OLYMPEX/olytestfigs/',outdate,'_',file_west,'_xy.png']), bbox_inches='tight',dpi=300)
            plt.close()

        """
        PLOT VERTICAL AVERAGE OF THE SCAN REFLECTIVITY AND ENHANCEMENT LAYER TOP/BOTTOM
        """
        if  plot_z_fig and ('1' in bright_bands_SW[bb_index_SW-1:bb_index_SW+1+adj_SW,1] or \
            '1' in bright_bands_WSW[bb_index_WSW-1:bb_index_WSW+1+adj_WSW,1] or \
            '1' in bright_bands_WNW[bb_index_WNW-1:bb_index_WNW+1+adj_WNW,1] or \
            '1' in bright_bands_NW[bb_index_NW-1:bb_index_NW+1+adj_NW,1] or \
            '1' in bright_bands_NE[bb_index_NE-1:bb_index_NE+1+adj_NE,1]): #there was a bright band found from bbidv6
            custom_lines = [Line2D([0], [0], color='gray', linestyle='--'), Line2D([0], [0], color='gray', linestyle=':')]
            custom_labels = ['Bright Band', ''.join(['NPOL Sounding -15'+ '\u00b0'+ 'C\n',str(d2)])]

            nrows, ncols = 1, 5
            fig, ax = plt.subplots(nrows,ncols) #single plot, n panels

            for c in range(ncols):
                if plot_list[c] == 'NE':
                    dBZ_means = dBZ_means_NE
                    bb_ht = bb_ht_NE
                    s = 0
                elif plot_list[c] == 'SW':
                    dBZ_means = dBZ_means_SW
                    bb_ht = bb_ht_SW
                    s = 1
                elif plot_list[c] == 'WSW':
                    dBZ_means = dBZ_means_WSW
                    bb_ht = bb_ht_WSW
                    s = 2
                elif plot_list[c] == 'WNW':
                    dBZ_means = dBZ_means_WNW
                    bb_ht = bb_ht_WNW
                    s = 3
                elif plot_list[c] == 'NW':
                    dBZ_means = dBZ_means_NW
                    bb_ht = bb_ht_NW
                    s = 4

                low_line = low_lines[s]
                high_line = high_lines[s]
                low_2_line = low_2_lines[s]
                high_2_line = high_2_lines[s]
                sector_color = sector_colors[s+1]
                sector_color_dark = sector_colors_dark[s]

                #plot the west figure
                dbzs = ax[c].plot(dBZ_means[0:15],plot_z, color = 'black')
                ax[c].hlines(y=np.float64(bb_ht)*2, xmin = 10, xmax = 45, color='gray', linestyle='--', label = 'Bright Band')
                ax[c].hlines(y=np.float64(dend_ht)*2, xmin = 10, xmax = 45, color='gray', linestyle=':', label = ''.join(['NPOL Sounding -15'+ '\u00b0'+ 'C\n',str(d2)]))

                if two_lev_enh or (low_lev_enh and high_lev_enh):
                    ax[c].hlines(y=low_line*2, xmin = 10, xmax = 45, color=sector_color, linestyle='-', label = 'Secondary Enhancement')
                    ax[c].hlines(y=high_line*2, xmin = 10, xmax = 45, color=sector_color, linestyle='-')
                    if low_line > 0:
                        custom_lines.append(Line2D([0], [0], color=sector_color, linestyle='-'))
                        custom_labels.append(''.join(['Secondary Enhancement ',plot_list[c]]))
                    ax[c].hlines(y=low_2_line*2, xmin = 10, xmax = 45, color=sector_color_dark, linestyle='-', label = 'Secondary Enhancement')
                    ax[c].hlines(y=high_2_line*2, xmin = 10, xmax = 45, color=sector_color_dark, linestyle='-')
                    if low_2_line > 0:
                        custom_lines.append(Line2D([0], [0], color=sector_color, linestyle='-'))
                        custom_labels.append(''.join(['Secondary Enhancement [2] ',plot_list[c]]))
                elif low_lev_enh:
                    ax[c].hlines(y=low_line*2, xmin = 10, xmax = 45, color=sector_color, linestyle='-', label = 'Secondary Enhancement')
                    ax[c].hlines(y=high_line*2, xmin = 10, xmax = 45, color=sector_color, linestyle='-')
                    if low_line > 0:
                        custom_lines.append(Line2D([0], [0], color=sector_color, linestyle='-'))
                        custom_labels.append(''.join(['Secondary Enhancement ',plot_list[c]]))
                elif high_lev_enh:
                    ax[c].hlines(y=low_2_line*2, xmin = 10, xmax = 45, color=sector_color_dark, linestyle='-', label = 'Secondary Enhancement')
                    ax[c].hlines(y=high_2_line*2, xmin = 10, xmax = 45, color=sector_color_dark, linestyle='-')
                    if low_2_line > 0:
                        custom_lines.append(Line2D([0], [0], color=sector_color, linestyle='-'))
                        custom_labels.append(''.join(['Secondary Enhancement [2] ',plot_list[c]]))

                if not low_lev_enh and not high_lev_enh and not two_lev_enh:
                    ax[c].hlines(y=low_line*2, xmin = 10, xmax = 45, color=sector_color, linestyle=':', label = 'Secondary Enhancement', alpha = 0.5)
                    ax[c].hlines(y=high_line*2, xmin = 10, xmax = 45, color=sector_color, linestyle=':', alpha = 0.5)
                    if low_line > 0:
                        custom_lines.append(Line2D([0], [0], color=sector_color, linestyle=':'))
                        custom_labels.append(''.join(['Non-Enhanced Layer ',plot_list[c]]))

                ax[c].set_xlim((10,45))
                ax[c].set_yticks(plot_z)
                ax[c].tick_params(axis="y",direction="in")
                labels = [item.get_text() for item in ax[c].get_yticklabels()]
                empty_string_labels = ['']*len(labels)
                if c != 0:
                    ax[c].set_yticklabels(empty_string_labels)
                else:
                    ax[0].set_yticklabels(plot_z_hts)
                    ax[0].set_ylabel('Height (km)')

                ax[2].set_xlabel(''.join(['Reflectivity (dBZ)']))


                ax[c].grid(True, linestyle = '--', linewidth = 0.5)
                lgd = ax[ncols-1].legend(custom_lines, custom_labels, bbox_to_anchor=(1.04,0.5), loc="center left", frameon = False)
                if s == 0:
                    ax[c].text(0.94, 0.96, date_east.strftime(''.join(['%m/%d/%y\n%H:%M:%S','\n',plot_list[c].upper(),''])), verticalalignment='top', horizontalalignment='right',transform=ax[c].transAxes, color='black', fontsize=8,bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
                else:
                    ax[c].text(0.94, 0.96, date_west.strftime(''.join(['%m/%d/%y\n%H:%M:%S','\n',plot_list[c].upper(),''])), verticalalignment='top', horizontalalignment='right',transform=ax[c].transAxes, color='black', fontsize=8,bbox=dict(facecolor='white', edgecolor='none', boxstyle='round'))
                #ax3 = ax2.twiny()
                #ZDRs = ax3.plot(ZDR_means[0:15],plot_z, color = 'grey')
                if not plot_xy:
                    ax[c].set_title(date.strftime(''.join(['%m/%d/%y %H:%M:%S\n'])))
                    fig.text(0.25, 0.05, ''.join(['closest sounding ',str(d2),]), horizontalalignment='center',fontsize=10)
            plt.tight_layout()
            fig.subplots_adjust(wspace=0.05)
            plt.savefig(''.join(['/home/disk/meso-home/adelaf/OLYMPEX/olytestfigs/',outdate,'_',file_west,'_z.png']), bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)
            #plt.show()
            plt.close()
    plt.close()
    print(''.join(['Worker finished: ',outname_west,' to ',outname_east]))
    return(row_to_append_NE, date_secondary_vals_NE, row_to_append_SW, date_secondary_vals_SW, row_to_append_WSW, date_secondary_vals_WSW, row_to_append_WNW, date_secondary_vals_WNW, row_to_append_NW, date_secondary_vals_NW)


"""
PARALLEL PROCESSING SETUP
"""

pool = mp.Pool(processes=nodes)
results = pool.map_async(main_func, range(numfiles))  #all runs
#results = pool.map_async(main_func, np.arange(35,38))  #testing run
for result in results.get():
    secondary_NE = np.vstack((secondary_NE,result[0]))
    secondary_vals_NE = np.vstack((secondary_vals_NE,result[1]))
    secondary_SW = np.vstack((secondary_SW,result[2]))
    secondary_vals_SW = np.vstack((secondary_vals_SW,result[3]))
    secondary_WSW = np.vstack((secondary_WSW,result[4]))
    secondary_vals_WSW = np.vstack((secondary_vals_WSW,result[5]))
    secondary_WNW = np.vstack((secondary_WNW,result[6]))
    secondary_vals_WNW = np.vstack((secondary_vals_WNW,result[7]))
    secondary_NW = np.vstack((secondary_NW,result[8]))
    secondary_vals_NW = np.vstack((secondary_vals_NW,result[9]))
    #secondary = np.vstack((secondary,results.get()))  #original method for extracting only one return from the function

#sort by NPOL date/time
secondary_NE = secondary_NE[secondary_NE[:,0].argsort()]
secondary_SW = secondary_SW[secondary_SW[:,0].argsort()]
secondary_WSW = secondary_WSW[secondary_WSW[:,0].argsort()]
secondary_WNW = secondary_WNW[secondary_WNW[:,0].argsort()]
secondary_NW = secondary_NW[secondary_NW[:,0].argsort()]

"""
SAVING
"""
if save_data:
    np.save(save_fn_data_NE, secondary_NE)
    np.save(save_fn_vals_NE, secondary_vals_NE)
    pd.DataFrame(secondary_NE).to_csv(save_fn_data_csv_NE)
    pd.DataFrame(secondary_vals_NE).to_csv(save_fn_vals_csv_NE)

    np.save(save_fn_data_SW, secondary_SW)
    np.save(save_fn_vals_SW, secondary_vals_SW)
    pd.DataFrame(secondary_SW).to_csv(save_fn_data_csv_SW)
    pd.DataFrame(secondary_vals_SW).to_csv(save_fn_vals_csv_SW)

    np.save(save_fn_data_WSW, secondary_WSW)
    np.save(save_fn_vals_WSW, secondary_vals_WSW)
    pd.DataFrame(secondary_WSW).to_csv(save_fn_data_csv_WSW)
    pd.DataFrame(secondary_vals_WSW).to_csv(save_fn_vals_csv_WSW)

    np.save(save_fn_data_WNW, secondary_WNW)
    np.save(save_fn_vals_WNW, secondary_vals_WNW)
    pd.DataFrame(secondary_WNW).to_csv(save_fn_data_csv_WNW)
    pd.DataFrame(secondary_vals_WNW).to_csv(save_fn_vals_csv_WNW)

    np.save(save_fn_data_NW, secondary_NW)
    np.save(save_fn_vals_NW, secondary_vals_NW)
    pd.DataFrame(secondary_NW).to_csv(save_fn_data_csv_NW)
    pd.DataFrame(secondary_vals_NW).to_csv(save_fn_vals_csv_NW)

print("Total time:", datetime.datetime.now() - startTime)
