#Andrew DeLaFrance
#Aug 15 2019

#updates bbidv6 algorithm to include sector search approach implemented into secondary algorithm

from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import numpy as np
import matplotlib
#matplotlib.use('Agg')  #raster rendering backend capable of writing png image to file
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
#np.set_printoptions(linewidth=np.nan, threshold=sys.maxsize)
startTime = datetime.datetime.now()

"""
THRESHOLDS AND VARIABLES
"""

nodes = 4 #how many processors to run (623 computers seem to work well on 16, 24 was slower due to communication between computers)

#dir = sys.argv[1] #input from script call, east or west after script name, or all

plot_map = False #plot an image of the sectors on a geographic background
plot_sectors = False #plot an image of the sectors with no background

print_results = False

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
n_levels_allowed = 2 #number of levels allowed to select from above the mode (each level is 0.5km) fixed at 1 level below

time_cont = 0 #hours of temporal continuity needed for a bright band to be stratiform

ht_exc = 0.5 #additional requirement on top of standard deviation,distance away from mean required to be removed
ht_max = 4 #maximum height in kilometers that a bright band can exist in
level_max = np.int(ht_max*2)
check_dBZ = 20.0

#rhohv and ZDR bounds
rhohv_min = 0.91
rhohv_max = 0.97
ZDR_min = 0.8
ZDR_max = 2.5

"""
FILE INPUT/OUTPUT - ORGANIZATION
"""

rhi_dir = '/home/disk/bob/olympex/zebra/moments/npol_qc2/rhi/' #base directory for gridded RHI's
save_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/BrightBands/' #output directory for saved images
data_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Data/' #directory for local data

NARR_data = 'NARR_at_NPOL.csv'
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
bright_bands_NE = np.array([1,2,3,4,5,6,7,8])
bright_bands_SW = np.array([1,2,3,4,5,6,7,8])
bright_bands_WSW = np.array([1,2,3,4,5,6,7,8])
bright_bands_WNW = np.array([1,2,3,4,5,6,7,8])
bright_bands_NW = np.array([1,2,3,4,5,6,7,8])

#Columns => date/time,bright band found?, bright band melt level, NARR date, NARR melt level,
#percent above dBZ threshold, percent of cells met with polarmetric criteria

filelist_east = []
filelist_west = []
#create a list of files for parallel computing
for date in date_list:
    date_folder = ''.join([rhi_dir,date])
    for file in os.listdir(date_folder):
        if file.split(".")[3] == 'east':
            filelist_east.append(file)
        elif file.split(".")[3] == 'west':
            filelist_west.append(file)

numfiles_east = len(filelist_east)
numfiles_west = len(filelist_west)

filelist_east.sort()
filelist_west.sort()

numfiles = numfiles_west


"""
SAVE OPTION FOR FILELIST TO USE IN SECONDARY.PY
"""
#np.save('/home/disk/meso-home/adelaf/OLYMPEX/Data/filelist.npy', np.asarray(filelist))
#np.save('/home/disk/meso-home/adelaf/OLYMPEX/Data/date_list.npy', np.asarray(date_list))

#bring in NARR data
df=pd.read_csv(NARR_fn, sep=',',header=None)
NARR_data = np.array(df) #NARR Time,IVT,Melting Level (m),925speed (kt),925dir,Nd,Nm
n_bbs = days_in_series
n_NARRs = NARR_data.shape[0]
items = []
for h in range(0,n_NARRs-1):
    items.append(datetime.datetime.strptime(NARR_data[h+1,0], "%Y-%m-%d %H:%M:%S"))
###############


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
NW = [296,326] #in the case of bright bands, not concerned with the number and dont want to be using values twice

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
print(counts)

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

    plt.show()
    #plt.savefig(''.join(['/home/disk/meso-home/adelaf/OLYMPEX/olytestfigs/sectors_on_geo.png']), bbox_inches='tight',dpi=300)
    plt.close()

"""
PLOT A MAP OF THE SECTORED MASKS
"""
if plot_sectors:
    cmap = matplotlib.colors.ListedColormap(sector_colors)
    fig, (ax1) = plt.subplots(1,1)

    im = ax1.imshow(NE_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)
    im = ax1.imshow(SW_mask, origin = 'Lower', alpha = 0.5, cmap = cmap, vmin=0, vmax = 5)
    im = ax1.imshow(WSW_mask, origin = 'Lower', alpha = 0.3, cmap = cmap, vmin=0, vmax = 5)
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
    plt.show()
    #plt.savefig(''.join(['/home/disk/meso-home/adelaf/OLYMPEX/olytestfigs/sectors_only.png']), bbox_inches='tight',dpi=300)
    plt.close()


"""
SCRIPT FUNCTIONS
"""

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

"""

"""
def main_func(i):
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
    rhohv_west = np.array(nc_fid.variables['RHOHV'][:,:,:,:])#dimesions [time, z, y, x] = [1,25,300,300]
    ZDR_west = np.array(nc_fid.variables['ZDR'][:,:,:,:])#dimesions [time, z, y, x] = [1,25,300,300]
    dBZ_west[dBZ_west == -9999.0] = float('NaN')
    rhohv_west[rhohv_west == -9999.0] = float('NaN')
    ZDR_west[ZDR_west == -9999.0] = float('NaN')
    day_west = date_west.strftime("%d")
    hour_west = date_west.strftime("%H")
    day_out = np.where(days_out == int(day_west))[0][0]
    nc_fid.close()

    #then get the east - should follow the west by about 20 minutes
    nc_fid = Dataset(file_path_east, 'r') #r means read only, cant modify data
    date_east = num2date(nc_fid.variables['base_time'][:], units =nc_fid.variables['base_time'].units , calendar = 'standard')
    dBZ_east = np.array(nc_fid.variables['DBZ'][:,:,:,:])
    rhohv_east = np.array(nc_fid.variables['RHOHV'][:,:,:,:])#dimesions [time, z, y, x] = [1,25,300,300]
    ZDR_east = np.array(nc_fid.variables['ZDR'][:,:,:,:])#dimesions [time, z, y, x] = [1,25,300,300]
    dBZ_east[dBZ_east == -9999.0] = float('NaN')
    rhohv_east[rhohv_east == -9999.0] = float('NaN')
    ZDR_east[ZDR_east == -9999.0] = float('NaN')
    day_east = date_east.strftime("%d")
    hour_east = date_east.strftime("%H")
    day_out = np.where(days_out == int(day_east))[0][0]
    nc_fid.close()

    time_diff = (date_east-date_west).total_seconds()/60 #should output the time difference between the two scans in minutes
    dBZ_complete = np.full((1,z_dim,y_dim,x_dim),float('NaN'))
    rhohv_complete = np.full((1,z_dim,y_dim,x_dim),float('NaN'))
    ZDR_complete = np.full((1,z_dim,y_dim,x_dim),float('NaN'))

    if 10 < time_diff < 30: #put the two scans together for one scan
        dBZ_complete[:,:,:,0:int(x_dim/2)] = dBZ_west[:,:,:,0:int(x_dim/2)]
        dBZ_complete[:,:,:,int(x_dim/2):int(x_dim)] = dBZ_east[:,:,:,int(x_dim/2):int(x_dim)]
        rhohv_complete[:,:,:,0:int(x_dim/2)] = rhohv_west[:,:,:,0:int(x_dim/2)]
        rhohv_complete[:,:,:,int(x_dim/2):int(x_dim)] = rhohv_east[:,:,:,int(x_dim/2):int(x_dim)]
        ZDR_complete[:,:,:,0:int(x_dim/2)] = ZDR_west[:,:,:,0:int(x_dim/2)]
        ZDR_complete[:,:,:,int(x_dim/2):int(x_dim)] = ZDR_east[:,:,:,int(x_dim/2):int(x_dim)]
    else:
        print('time indexing problem')

    dBZ = np.copy(dBZ_complete)
    rhohv = np.copy(rhohv_complete)
    ZDR = np.copy(ZDR_complete)

    #set up elevation profile
    elev = np.arange(0,(z_dim*z_spacing),z_spacing)

    n_found = 0 #total enhanced grid cells found

    n_NE_found = 0
    n_SW_found = 0
    n_WSW_found = 0
    n_WNW_found = 0
    n_NW_found = 0

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
                rhohv_east[0,:, y_ind, x_ind] = float('NaN')
                ZDR_east[0,:, y_ind, x_ind] = float('NaN')

                dBZ_west[0,:, y_ind, x_ind] = float('NaN')
                rhohv_west[0,:, y_ind, x_ind] = float('NaN')
                ZDR_west[0,:, y_ind, x_ind] = float('NaN')

                dBZ[0,:, y_ind, x_ind] = float('NaN')
                rhohv[0,:, y_ind, x_ind] = float('NaN')
                ZDR[0,:, y_ind, x_ind] = float('NaN')

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

    min_rhohv_lev_for_mode_NE = np.full([y_dim, x_dim],float('NaN'))
    min_rhohv_lev_for_mode_SW = np.full([y_dim, x_dim],float('NaN'))
    min_rhohv_lev_for_mode_WSW = np.full([y_dim, x_dim],float('NaN'))
    min_rhohv_lev_for_mode_WNW = np.full([y_dim, x_dim],float('NaN'))
    min_rhohv_lev_for_mode_NW = np.full([y_dim, x_dim],float('NaN'))

    """
    INITIAL SEARCH FOR THE LEVEL OF MINIMUM IN RHOHV
    """
    #Look through to find out where the minimum rhohv level is occurring at each grid point
    for x_ind in range(x_dim):
        for y_ind in range(y_dim):
            temp_rhohv_lev_for_mode = float('NaN')
            if np.isnan(rhohv[0,:,y_ind,x_ind]).all():
                #min_rhohv_lev_for_mode[y_ind,x_ind] = float('NaN')
                temp_rhohv_lev_for_mode = float('NaN')
            else:
                if np.nanmin(rhohv[0,:,y_ind,x_ind]) <= rhohv_max:
                    #make a copy of this column
                    rhohv_copy = rhohv[0,:,y_ind,x_ind]
                    rhohv_copy[rhohv_copy<rhohv_min] = float('NaN')
                    rhohv_copy[rhohv_copy>rhohv_max] = float('NaN')
                    if np.isnan(rhohv_copy).all():
                        #min_rhohv_lev_for_mode[y_ind,x_ind] = float('NaN')
                        temp_rhohv_lev_for_mode = float('NaN')
                    elif (np.where(rhohv_copy == np.nanmin(rhohv_copy))[0][0]).size > 1:
                        #min_rhohv_lev_for_mode[y_ind,x_ind] = np.nanmax(np.where(rhohv_copy == np.nanmin(rhohv_copy))[0][0])
                        temp_rhohv_lev_for_mode = np.nanmax(np.where(rhohv_copy == np.nanmin(rhohv_copy))[0][0])
                    else:
                        #min_rhohv_lev_for_mode[y_ind,x_ind] = np.where(rhohv_copy == np.nanmin(rhohv_copy))[0][0]
                        temp_rhohv_lev_for_mode = np.where(rhohv_copy == np.nanmin(rhohv_copy))[0][0]
                else:
                    #min_rhohv_lev_for_mode[y_ind,x_ind] = float('NaN')
                    temp_rhohv_lev_for_mode = float('NaN')
            #print(temp_rhohv_lev_for_mode)
            """
            BREAK UP INTO SECTORS (NE, SW, WSW, WNW, NW) TO ANALYZE EACH OF THEM INDIVIDUALLY IN ORDER TO DETERMINE ENHANCEMNET FOR EACH SECTOR
            """
            if ~np.isnan(NE_mask[y_ind,x_ind]):
                min_rhohv_lev_for_mode_NE[y_ind, x_ind] = temp_rhohv_lev_for_mode
                n_NE_found += 1
            elif ~np.isnan(SW_mask[y_ind,x_ind]):
                min_rhohv_lev_for_mode_SW[y_ind, x_ind] = temp_rhohv_lev_for_mode
                n_SW_found += 1
            elif ~np.isnan(WSW_mask[y_ind,x_ind]):
                min_rhohv_lev_for_mode_WSW[y_ind, x_ind] = temp_rhohv_lev_for_mode
                n_WSW_found += 1
            elif ~np.isnan(WNW_mask[y_ind,x_ind]):
                min_rhohv_lev_for_mode_WNW[y_ind, x_ind] = temp_rhohv_lev_for_mode
                n_WNW_found += 1
            elif ~np.isnan(NW_mask[y_ind,x_ind]):
                min_rhohv_lev_for_mode_NW[y_ind, x_ind] = temp_rhohv_lev_for_mode
                n_NW_found += 1

    """
    MODES CALCULATION (MOST OCCURRING LEVEL)
    """
    period_mode_NE = mode1(min_rhohv_lev_for_mode_NE)[0]
    period_mode_SW = mode1(min_rhohv_lev_for_mode_SW)[0]
    period_mode_WSW = mode1(min_rhohv_lev_for_mode_WSW)[0]
    period_mode_WNW = mode1(min_rhohv_lev_for_mode_WNW)[0]
    period_mode_NW = mode1(min_rhohv_lev_for_mode_NW)[0]
    #print(period_mode_NE)
    if (period_mode_NE.size > 1):
        period_mode_NE = period_mode_NE[period_mode_NE.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode_NE):
        period_mode_NE = np.int(period_mode_NE)
    if (period_mode_SW.size > 1):
        period_mode_SW = period_mode_SW[period_mode_SW.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode_SW):
        period_mode_SW = np.int(period_mode_SW)
    if (period_mode_WSW.size > 1):
        period_mode_WSW = period_mode_WSW[period_mode_WSW.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode_WSW):
        period_mode_WSW = np.int(period_mode_WSW)
    if (period_mode_WNW.size > 1):
        period_mode_WNW = period_mode_WNW[period_mode_WNW.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode_WNW):
        period_mode_WNW = np.int(period_mode_WNW)
    if (period_mode_NW.size > 1):
        period_mode_NW = period_mode_NW[period_mode_NW.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode_NW):
        period_mode_NW = np.int(period_mode_NW)

    period_mode_2_NE = mode2(min_rhohv_lev_for_mode_NE)[0]
    period_mode_2_SW = mode2(min_rhohv_lev_for_mode_SW)[0]
    period_mode_2_WSW = mode2(min_rhohv_lev_for_mode_WSW)[0]
    period_mode_2_WNW = mode2(min_rhohv_lev_for_mode_WNW)[0]
    period_mode_2_NW = mode2(min_rhohv_lev_for_mode_NW)[0]

    if (period_mode_2_NE.size > 1):
        period_mode_2_NE = period_mode_2_NE[period_mode_2_NE.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode_2_NE):
        period_mode_2_NE = np.int(period_mode_2_NE)
    if (period_mode_2_SW.size > 1):
        period_mode_2_SW = period_mode_2_SW[period_mode_2_SW.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode_2_SW):
        period_mode_2_SW = np.int(period_mode_2_SW)
    if (period_mode_2_WSW.size > 1):
        period_mode_2_WSW = period_mode_2_WSW[period_mode_2_WSW.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode_2_WSW):
        period_mode_2_WSW = np.int(period_mode_2_WSW)
    if (period_mode_2_WNW.size > 1):
        period_mode_2_WNW = period_mode_2_WNW[period_mode_2_WNW.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode_2_WNW):
        period_mode_2_WNW = np.int(period_mode_2_WNW)
    if (period_mode_2_NW.size > 1):
        period_mode_2_NW = period_mode_2_NW[period_mode_2_NW.size-1]#use the highest layer when more than one meets criteria
    if ~np.isnan(period_mode_2_NW):
        period_mode_2_NW = np.int(period_mode_2_NW)

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

    if period_mode_2_NE > period_mode_NE and dBZ_means_NE[period_mode_2_NE] >= min_ave_dBZ and (period_mode_2_NE*0.5) < ht_max:
        period_mode_NE = period_mode_2_NE
    if period_mode_2_SW > period_mode_SW and dBZ_means_SW[period_mode_2_SW] >= min_ave_dBZ and (period_mode_2_SW*0.5) < ht_max:
        period_mode_SW = period_mode_2_SW
    if period_mode_2_WSW > period_mode_WSW and dBZ_means_WSW[period_mode_2_WSW] >= min_ave_dBZ and (period_mode_2_WSW*0.5) < ht_max:
        period_mode_WSW = period_mode_2_WSW
    if period_mode_2_WNW > period_mode_WNW and dBZ_means_WNW[period_mode_2_WNW] >= min_ave_dBZ and (period_mode_2_WNW*0.5) < ht_max:
        period_mode_WNW = period_mode_2_WNW
    if period_mode_2_NW > period_mode_NW and dBZ_means_NW[period_mode_2_NW] >= min_ave_dBZ and (period_mode_2_NW*0.5) < ht_max:
        period_mode_NW = period_mode_2_NW


    n_above_dBZ_NE = 0 #for calculating percentages of cells meeting criteria
    n_matched_NE = 0
    n_above_dBZ_SW = 0
    n_matched_SW = 0
    n_above_dBZ_WSW = 0
    n_matched_WSW = 0
    n_above_dBZ_WNW = 0
    n_matched_WNW = 0
    n_above_dBZ_NW = 0
    n_matched_NW = 0

    #set up empty array of grid sector, y, x for top and bottom values of bb
    bb_melt_levs_NE = np.full([y_dim,x_dim], float('NaN'))#empty array to hold levels that meet all specified polarmetric criteria
    bb_melt_levs_SW = np.full([y_dim,x_dim], float('NaN'))
    bb_melt_levs_WSW = np.full([y_dim,x_dim], float('NaN'))
    bb_melt_levs_WNW = np.full([y_dim,x_dim], float('NaN'))
    bb_melt_levs_NW = np.full([y_dim,x_dim], float('NaN'))

    """
    RE-EVALUATE WITHIN FOUND MODES
    """
    for x_ind in range(x_dim):
        for y_ind in range(y_dim):
            n_matched = 0
            n_above_dBZ = 0
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

            if ~np.isnan(NE_mask[y_ind,x_ind]):
                period_mode = period_mode_NE
            elif ~np.isnan(SW_mask[y_ind,x_ind]):
                period_mode = period_mode_SW
            elif ~np.isnan(WSW_mask[y_ind,x_ind]):
                period_mode = period_mode_WSW
            elif ~np.isnan(WNW_mask[y_ind,x_ind]):
                period_mode = period_mode_WNW
            elif ~np.isnan(NW_mask[y_ind,x_ind]):
                period_mode = period_mode_NW

            if max_col_dBZ >= dBZ_exceed_val and isrhohv and isZDR and period_mode >= 0:
                #n_above_dBZ = n_above_dBZ+1
                n_above_dBZ = 1
                #restrict identification of max dBZ layer to be within bounds of rhohv and ZDR criteria
                dBZ_met1 = np.where(dBZ[0,:,y_ind,x_ind] >= dBZ_exceed_val)[0]
                if period_mode == 0:
                    dBZ_met = [x for x in dBZ_met1 if x in range(period_mode,(period_mode+(n_levels_allowed)+1))]
                else:
                    dBZ_met = [x for x in dBZ_met1 if x in range((period_mode-1),(period_mode+(n_levels_allowed)+1))]

                matched_layer = [x for x in dBZ_met]
                bb_layer = float('NaN') #initializing nothing found yet
                temp_bb_melt_lev = float('NaN')

                if len(matched_layer) == 0:
                    pass
                elif len(matched_layer) == 1:
                    bb_layer = np.float64(matched_layer[0])
                    #bb_melt_levs[y_ind,x_ind] = np.float64(bb_layer*0.5)#0.5km vertical resolution
                    temp_bb_melt_lev = np.float64(bb_layer*0.5)#0.5km vertical resolution
                    #n_matched = n_matched + 1
                    n_matched = 1
                else:
                    #sort out which of the matched layers is maximum in dBZ
                    if np.isnan(dBZ[0,matched_layer,y_ind,x_ind]).all():
                        bb_layer = float('NaN')
                    else:
                        bb_layer = np.float64(np.where(dBZ[0,:,y_ind,x_ind] == np.nanmax(dBZ[0,matched_layer,y_ind,x_ind]))[0][0])#the layer that has the max reflectivity
                        #n_matched = n_matched + 1
                        n_matched = 1
                    #bb_melt_levs[y_ind,x_ind] = np.float64(bb_layer*0.5)#0.5km vertical resolution
                    temp_bb_melt_lev = np.float64(bb_layer*0.5)#0.5km vertical resolution

                if ~np.isnan(NE_mask[y_ind,x_ind]):
                    bb_melt_levs_NE[y_ind, x_ind] = temp_bb_melt_lev
                    n_matched_NE = n_matched_NE + n_matched
                    n_above_dBZ_NE = n_above_dBZ_NE + n_above_dBZ
                elif ~np.isnan(SW_mask[y_ind,x_ind]):
                    bb_melt_levs_SW[y_ind, x_ind] = temp_bb_melt_lev
                    n_matched_SW = n_matched_SW + n_matched
                    n_above_dBZ_SW = n_above_dBZ_SW + n_above_dBZ
                elif ~np.isnan(WSW_mask[y_ind,x_ind]):
                    bb_melt_levs_WSW[y_ind, x_ind] = temp_bb_melt_lev
                    n_matched_WSW = n_matched_WSW + n_matched
                    n_above_dBZ_WSW = n_above_dBZ_WSW + n_above_dBZ
                elif ~np.isnan(WNW_mask[y_ind,x_ind]):
                    bb_melt_levs_WNW[y_ind, x_ind] = temp_bb_melt_lev
                    n_matched_WNW = n_matched_WNW + n_matched
                    n_above_dBZ_WNW = n_above_dBZ_WNW + n_above_dBZ
                elif ~np.isnan(NW_mask[y_ind,x_ind]):
                    bb_melt_levs_NW[y_ind, x_ind] = temp_bb_melt_lev
                    n_matched_NW = n_matched_NW + n_matched
                    n_above_dBZ_NW = n_above_dBZ_NW + n_above_dBZ

            #something in the reflectivity scan but none that exceed the reflectivity values
            elif max_col_dBZ < dBZ_exceed_val and max_col_dBZ > 0:
                pass
            else:
                dBZ[0,:,y_ind,x_ind] = float('NaN')
            #end the x,y loop

    """
    CLEAN UP NUMPY ARRAY OF LEVELS - REMOVE NANS
    """
    #clean up numpy array of levels
    bb_melt_levs_NE= bb_melt_levs_NE[~np.isnan(bb_melt_levs_NE)]
    bb_melt_levs_SW= bb_melt_levs_SW[~np.isnan(bb_melt_levs_SW)]
    bb_melt_levs_WSW= bb_melt_levs_WSW[~np.isnan(bb_melt_levs_WSW)]
    bb_melt_levs_WNW= bb_melt_levs_WNW[~np.isnan(bb_melt_levs_WNW)]
    bb_melt_levs_NW= bb_melt_levs_NW[~np.isnan(bb_melt_levs_NW)]

    ##########################
    if np.isnan(dBZ_NE).all():#empty slice, clear sky conditions
        clear_sky_NE = True
    else:
        clear_sky_NE = False
        if len(bb_melt_levs_NE)==0:
            bb_melting_height_NE = float('NaN')
        else:
            bb_melting_height_NE = np.float(format(np.nanmean(bb_melt_levs_NE), '.2f'))

    if np.isnan(dBZ_SW).all():#empty slice, clear sky conditions
        clear_sky_SW = True
    else:
        clear_sky_SW = False
        if len(bb_melt_levs_SW)==0:
            bb_melting_height_SW = float('NaN')
        else:
            bb_melting_height_SW = np.float(format(np.nanmean(bb_melt_levs_SW), '.2f'))

    if np.isnan(dBZ_WSW).all():#empty slice, clear sky conditions
        clear_sky_WSW = True
    else:
        clear_sky_WSW = False
        if len(bb_melt_levs_WSW)==0:
            bb_melting_height_WSW = float('NaN')
        else:
            bb_melting_height_WSW = np.float(format(np.nanmean(bb_melt_levs_WSW), '.2f'))

    if np.isnan(dBZ_WNW).all():#empty slice, clear sky conditions
        clear_sky_WNW = True
    else:
        clear_sky_WNW = False
        if len(bb_melt_levs_WNW)==0:
            bb_melting_height_WNW = float('NaN')
        else:
            bb_melting_height_WNW = np.float(format(np.nanmean(bb_melt_levs_WNW), '.2f'))

    if np.isnan(dBZ_NW).all():#empty slice, clear sky conditions
        clear_sky_NW = True
    else:
        clear_sky_NW = False
        if len(bb_melt_levs_NW)==0:
            bb_melting_height_NW = float('NaN')
        else:
            bb_melting_height_NW = np.float(format(np.nanmean(bb_melt_levs_NW), '.2f'))


    prcnt_cells_met_NE = float(format(((n_matched_NE/counts[0])*100), '.2f'))
    prcnt_above_dBZ_NE = float(format(((n_above_dBZ_NE/counts[0])*100), '.2f'))
    prcnt_cells_met_SW = float(format(((n_matched_SW/counts[1])*100), '.2f'))
    prcnt_above_dBZ_SW = float(format(((n_above_dBZ_SW/counts[1])*100), '.2f'))
    prcnt_cells_met_WSW = float(format(((n_matched_WSW/counts[2])*100), '.2f'))
    prcnt_above_dBZ_WSW = float(format(((n_above_dBZ_WSW/counts[2])*100), '.2f'))
    prcnt_cells_met_WNW = float(format(((n_matched_WNW/counts[3])*100), '.2f'))
    prcnt_above_dBZ_WNW = float(format(((n_above_dBZ_WNW/counts[3])*100), '.2f'))
    prcnt_cells_met_NW = float(format(((n_matched_NW/counts[4])*100), '.2f'))
    prcnt_above_dBZ_NW = float(format(((n_above_dBZ_NW/counts[4])*100), '.2f'))

    hour_out_west = int(hour_west)
    hour_out_east = int(hour_east)

    """
    FIND THE NEAREST NARR TIME AND MELTING LEVEL
    """

    pivot = date_west
    timedeltas = []
    for j in range(0,len(items)):
        timedeltas.append(np.abs(pivot-items[j]))
    min_index = timedeltas.index(np.min(timedeltas)) + 1 #closest time step
    closest_NARR_date = NARR_data[min_index,0]
    melt_layer = NARR_data[min_index,2]


    """
    APPLY THE EVALUATION CRITERIA FOR BRIGHT BAND FOUND YES/NO
    """

    if clear_sky_NE:
        bb_data_to_append_NE = np.array([date_east.strftime("%m/%d/%y %H:%M:%S"),99,float('NaN'),closest_NARR_date,melt_layer,0,0,0])
    else:
        if prcnt_cells_met_NE >= bb_crit_1: #does this meet bright band criteria with polarmetric criteria
            bb_data_to_append_NE = np.array([date_east.strftime("%m/%d/%y %H:%M:%S"),1,bb_melting_height_NE,closest_NARR_date,melt_layer,prcnt_above_dBZ_NE, prcnt_cells_met_NE,period_mode_NE])
        elif prcnt_cells_met_NE >= check_dBZ: #does this meet bright band criteria with dBZ only
            bb_data_to_append_NE = np.array([date_east.strftime("%m/%d/%y %H:%M:%S"),2,bb_melting_height_NE,closest_NARR_date,melt_layer,prcnt_above_dBZ_NE, prcnt_cells_met_NE,period_mode_NE])
        else:#("layer does not meet either criteria")
            bb_data_to_append_NE = np.array([date_east.strftime("%m/%d/%y %H:%M:%S"),0,bb_melting_height_NE,closest_NARR_date,melt_layer,prcnt_above_dBZ_NE, prcnt_cells_met_NE,period_mode_NE])
        del bb_melt_levs_NE

    if clear_sky_SW:
        bb_data_to_append_SW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),99,float('NaN'),closest_NARR_date,melt_layer,0,0,0])
    else:
        if prcnt_cells_met_SW >= bb_crit_1: #does this meet bright band criteria with polarmetric criteria
            bb_data_to_append_SW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),1,bb_melting_height_SW,closest_NARR_date,melt_layer,prcnt_above_dBZ_SW, prcnt_cells_met_SW,period_mode_SW])
        elif prcnt_cells_met_SW >= check_dBZ: #does this meet bright band criteria with dBZ only
            bb_data_to_append_SW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),2,bb_melting_height_SW,closest_NARR_date,melt_layer,prcnt_above_dBZ_SW, prcnt_cells_met_SW,period_mode_SW])
        else:#("layer does not meet either criteria")
            bb_data_to_append_SW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),0,bb_melting_height_SW,closest_NARR_date,melt_layer,prcnt_above_dBZ_SW, prcnt_cells_met_SW,period_mode_SW])
        del bb_melt_levs_SW

    if clear_sky_WSW:
        bb_data_to_append_WSW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),99,float('NaN'),closest_NARR_date,melt_layer,0,0,0])
    else:
        if prcnt_cells_met_WSW >= bb_crit_1: #does this meet bright band criteria with polarmetric criteria
            bb_data_to_append_WSW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),1,bb_melting_height_WSW,closest_NARR_date,melt_layer,prcnt_above_dBZ_WSW, prcnt_cells_met_WSW,period_mode_WSW])
        elif prcnt_cells_met_WSW >= check_dBZ: #does this meet bright band criteria with dBZ only
            bb_data_to_append_WSW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),2,bb_melting_height_WSW,closest_NARR_date,melt_layer,prcnt_above_dBZ_WSW, prcnt_cells_met_WSW,period_mode_WSW])
        else:#("layer does not meet either criteria")
            bb_data_to_append_WSW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),0,bb_melting_height_WSW,closest_NARR_date,melt_layer,prcnt_above_dBZ_WSW, prcnt_cells_met_WSW,period_mode_WSW])
        del bb_melt_levs_WSW

    if clear_sky_WNW:
        bb_data_to_append_WNW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),99,float('NaN'),closest_NARR_date,melt_layer,0,0,0])
    else:
        if prcnt_cells_met_WNW >= bb_crit_1: #does this meet bright band criteria with polarmetric criteria
            bb_data_to_append_WNW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),1,bb_melting_height_WNW,closest_NARR_date,melt_layer,prcnt_above_dBZ_WNW, prcnt_cells_met_WNW,period_mode_WNW])
        elif prcnt_cells_met_WNW >= check_dBZ: #does this meet bright band criteria with dBZ only
            bb_data_to_append_WNW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),2,bb_melting_height_WNW,closest_NARR_date,melt_layer,prcnt_above_dBZ_WNW, prcnt_cells_met_WNW,period_mode_WNW])
        else:#("layer does not meet either criteria")
            bb_data_to_append_WNW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),0,bb_melting_height_WNW,closest_NARR_date,melt_layer,prcnt_above_dBZ_WNW, prcnt_cells_met_WNW,period_mode_WNW])
        del bb_melt_levs_WNW

    if clear_sky_NW:
        bb_data_to_append_NW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),99,float('NaN'),closest_NARR_date,melt_layer,0,0,0])
    else:
        if prcnt_cells_met_NW >= bb_crit_1: #does this meet bright band criteria with polarmetric criteria
            bb_data_to_append_NW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),1,bb_melting_height_NW,closest_NARR_date,melt_layer,prcnt_above_dBZ_NW, prcnt_cells_met_NW,period_mode_NW])
        elif prcnt_cells_met_NW >= check_dBZ: #does this meet bright band criteria with dBZ only
            bb_data_to_append_NW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),2,bb_melting_height_NW,closest_NARR_date,melt_layer,prcnt_above_dBZ_NW, prcnt_cells_met_NW,period_mode_NW])
        else:#("layer does not meet either criteria")
            bb_data_to_append_NW = np.array([date_west.strftime("%m/%d/%y %H:%M:%S"),0,bb_melting_height_NW,closest_NARR_date,melt_layer,prcnt_above_dBZ_NW, prcnt_cells_met_NW,period_mode_NW])
        del bb_melt_levs_NW

    if print_results:
        print('NE', bb_data_to_append_NE)
        print('SW', bb_data_to_append_SW)
        print('WSW', bb_data_to_append_WSW)
        print('WNW', bb_data_to_append_WNW)
        print('NW', bb_data_to_append_NW)

    print(''.join(['Worker finished: ',outname_west,' to ',outname_east]))
    return(bb_data_to_append_NE, bb_data_to_append_SW, bb_data_to_append_WSW, bb_data_to_append_WNW, bb_data_to_append_NW)

'''
Set up the parallel processing environment
'''

pool = mp.Pool(processes=nodes)
results = pool.map_async(main_func, range(numfiles))
#results = pool.map_async(main_func, range(100))
#bright_bands = np.vstack((bright_bands,results.get()))

for result in results.get():
    bright_bands_NE = np.vstack((bright_bands_NE,result[0]))
    bright_bands_SW = np.vstack((bright_bands_SW,result[1]))
    bright_bands_WSW = np.vstack((bright_bands_WSW,result[2]))
    bright_bands_WNW = np.vstack((bright_bands_WNW,result[3]))
    bright_bands_NW = np.vstack((bright_bands_NW,result[4]))


#sort by NPOL date/time
bright_bands_NE = bright_bands_NE[bright_bands_NE[:,0].argsort()]
bright_bands_SW = bright_bands_SW[bright_bands_SW[:,0].argsort()]
bright_bands_WSW = bright_bands_WSW[bright_bands_WSW[:,0].argsort()]
bright_bands_WNW = bright_bands_WNW[bright_bands_WNW[:,0].argsort()]
bright_bands_NW = bright_bands_NW[bright_bands_NW[:,0].argsort()]


"""
SETUP FOR LOOP FOR OUTPUT AND PLOTTING EACH SECTOR
"""

for s in range(0,len(sectors)):

    if sector_names[s] == 'NE':
        bright_bands = np.copy(bright_bands_NE)
    if sector_names[s] == 'SW':
        bright_bands = np.copy(bright_bands_SW)
    if sector_names[s] == 'WSW':
        bright_bands = np.copy(bright_bands_WSW)
    if sector_names[s] == 'WNW':
        bright_bands = np.copy(bright_bands_WNW)
    if sector_names[s] == 'NW':
        bright_bands = np.copy(bright_bands_NW)

    if use_rhohv:
        save_name_fig = ''.join(['brightbandsfound_V7_x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withrhohv_',str(rhohv_min),str(rhohv_max),'_',sector_names[s],'.png'])
        save_name_data = ''.join(['brightbandsfound_V7_x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withrhohv_',str(rhohv_min),str(rhohv_max),'_',sector_names[s],'.npy'])
        save_name_data_csv = ''.join(['brightbandsfound_V7_x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withrhohv_',str(rhohv_min),str(rhohv_max),'_',sector_names[s],'.csv'])
    elif use_ZDR:
        save_name_fig = ''.join(['brightbandsfound_V7_x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withZDR_',str(ZDR_min),'_',sector_names[s],'.png'])
        save_name_data = ''.join(['brightbandsfound_V7_x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withZDR_',str(ZDR_min),'_',sector_names[s],'.npy'])
        save_name_data_csv = ''.join(['brightbandsfound_V7_x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_withZDR_',str(ZDR_min),'_',sector_names[s],'.csv'])
    else:
        save_name_fig = ''.join(['brightbandsfound_V7_x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_',sector_names[s],'.png'])
        save_name_data = ''.join(['brightbandsfound_V7_x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_',sector_names[s],'.npy'])
        save_name_data_csv = ''.join(['brightbandsfound_V7_x',str(bb_crit_1),'pcntx',str(dBZ_exceed_val),'_',sector_names[s],'.csv'])

    save_fn_fig = ''.join([save_dir,save_name_fig])
    save_fn_data = ''.join([save_dir,save_name_data])
    save_fn_data_csv = ''.join([save_dir,save_name_data_csv])

    """
    SETUP PLOTTING ARRAY
    """

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

    """
    TEMPORAL CONTINUITY/HEIGHT CONTINUITY
    """

    if require_time_cont:
        #assess temporal continuity > x hours for stratiform
        i_begin = 1 #start at 1 since row 0 is just placeholders for columns
        while (i_begin <= ntimes):
            if bright_bands[i_begin,1] == '1':#is a bright band layer
                #look for the end of contiuous bright bands found
                bb_remaining = bright_bands[i_begin:ntimes,1]
                i_end = i_begin + next((i for i, v in enumerate(bb_remaining) if v not in ['1']), ntimes)
                start_time = datetime.datetime.strptime(bright_bands[i_begin,0], "%m/%d/%y %H:%M:%S")
                if i_end < ntimes:
                    end_time = datetime.datetime.strptime(bright_bands[i_end-1,0], "%m/%d/%y %H:%M:%S")
                else:
                    end_time = datetime.datetime.strptime(bright_bands[ntimes,0], "%m/%d/%y %H:%M:%S")
                tdelta = end_time - start_time #outputs difference in seconds
                tdelta_hours = tdelta.seconds/3600 #3600 seconds in an hour
                if tdelta_hours >= np.abs(time_cont):
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
                        if i_ht<(ntimes-7) and i_ht > 5:
                            subset = bright_bands[(i_ht-5):(i_ht+5),2]
                            local_set = [float(v) for u,v in enumerate(subset)]
                            n = 0
                            for i_u in range((i_ht-5),(i_ht+5)):
                                if bright_bands[i_u,1] not in ['1']:
                                    local_set[n] =  float('NaN')
                                n = n+1
                            local_mean = np.nanmean(local_set)
                            ht_diff = local_mean - float(bright_bands[i_ht,2])
                            if ht_diff > ht_exc:
                                bright_bands[i_ht,1] = 4
                    print(sector_names[s], start_time,end_time,height_std,height_mean)
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

    """
    PLOTTING AND SAVING
    """
    colors = ['#f0f0f0','#636363','#b3e2cd'] #,'#fdcdac'
    cmap = matplotlib.colors.ListedColormap(colors)
    fig, ax = plt.subplots()
    im = ax.imshow(day_time_array.T, origin = 'lower',cmap=cmap)
    ax.set_title(''.join(['OLYMPEX Bright Band Identification\nNPOL-',sector_names[s]]))
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
