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
import matplotlib.pyplot as plt
from matplotlib import dates


#########

'''
setup stuff
'''

dir = 'east' #look east or west (lowercase)
plot_NARR = True
plot_NPOL_0C = True
plot_NPOL_15C = False
plot_BB = True
plot_Secondary = True

#dates to start and end plot
mon_start = 12
day_start = 3
mon_end = 12
day_end = 3

output_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/BrightBands/' #location of previous output
secondary_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/Secondary/'
data_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Data/' #directory for local data
bb_data = ''.join(['brightbandsfound_v6_r_6_time0x35.0pcntx25.0_withrhohv_0.910.97_',dir,'.npy'])
secondary_data = ''.join(['secondary_C_15X4excd_',dir,'.npy'])
sounding_data = 'NPOL_sounding_melt_levs.npy'
NARR_data = 'NARR_at_NPOL.csv'
save_name_data_csv = ''.join(['BrightBandsXNARRXNPOL_v6_r_6_time0x35.0pcntx25.0_withrhohv_0.910.97_',dir,'.csv'])
save_name_fig = ''.join(['BrightBandsXNARRXNPOL_v6_r_6_time0x35.0pcntx25.0_withrhohv_0.910.97_',dir,'.png'])
bb_fn = ''.join([output_dir,bb_data])
secondary_fn = ''.join([secondary_dir,secondary_data])
sounding_fn = ''.join([output_dir,sounding_data])
NARR_fn = ''.join([data_dir,NARR_data])
save_fn_data_csv = ''.join([output_dir,save_name_data_csv])
save_fn_fig = ''.join([output_dir,save_name_fig])

bright_bands = np.load(bb_fn)#time,bbfound, top, btm, bbcrit1, bbcrit2
secondary = np.load(secondary_fn)
NPOL_data = np.load(sounding_fn)
df=pd.read_csv(NARR_fn, sep=',',header=None)
NARR_data = np.array(df) #NARR Time,IVT,Melting Level (m),925speed (kt),925dir,Nd,Nm
n_bbs = bright_bands.shape[0]
n_NARRs = NARR_data.shape[0]
n_NPOLs = NPOL_data.shape[0]

#restrict plotting to only use times that a bright band occurred
for i in range(0,n_bbs):
    if bright_bands[i,1] != '1':
        bright_bands[i,2] = np.float('NaN')

BrightBands_w_NARR = bright_bands[:,[0,2]] #assign bright bands found in bbidv6 to the array
secondary_levs = secondary[:,[0,3,4]] #grab dates, low enhance lev, high enhnace lev
NARR_melt_levs = bright_bands[:,[0,2]] #place holder for data just to build array, will be replaced later
NPOL_melt_levs = bright_bands[:,[0,2]]
#array structure = NPOL date, BB top found in algorithm, NARR date, melting layer
BrightBands_w_NARR = np.hstack((BrightBands_w_NARR,NARR_melt_levs,NPOL_melt_levs,secondary_levs))

#print(BrightBands_w_NARR[2,:])
#NARR_dates = NARR_data[0:n_NARRs,0]
items = []
for h in range(0,n_NARRs-1):
    items.append(datetime.datetime.strptime(NARR_data[h+1,0], "%Y-%m-%d %H:%M:%S"))

items_NPOL = []
for h in range(0,n_NPOLs-1):
    items_NPOL.append(datetime.datetime.strptime(NPOL_data[h+1,0], "%m/%d/%y %H:%M:%S:"))
#datetime_object_bb = datetime.strptime(bright_bands[1,0], "%m/%d - %H:%M:%S")

start_index_found = False
end_index_found = False
start_index = 0
end_index = n_bbs-1

end_list = []

BrightBands_w_NARR = BrightBands_w_NARR[1:BrightBands_w_NARR.shape[0],:] #just remove first row of index values
BrightBands_w_NARR = BrightBands_w_NARR[BrightBands_w_NARR[:,0].argsort()]

#loop through all bb times to find nearest NARR melting level
for i in range(0,n_bbs-1):
    datetime_object = datetime.datetime.strptime(BrightBands_w_NARR[i,0], "%m/%d/%y %H:%M:%S")
    #datetime_object = datetime_object.replace(year = 2015)
    pivot = datetime_object

    timedeltas = []
    for j in range(0,len(items)):
        timedeltas.append(np.abs(pivot-items[j]))
    min_index = timedeltas.index(np.min(timedeltas)) +1
    d = datetime.datetime.strptime(NARR_data[min_index,0], "%Y-%m-%d %H:%M:%S").strftime('%m/%d/%y %H:%M:%S')
    melt_layer = NARR_data[min_index,2]
    #BrightBands_w_NARR[i,2] = NARR_data[min_index,0] #assign the NARR date to my array
    BrightBands_w_NARR[i,2] = d
    BrightBands_w_NARR[i,3] = float(NARR_data[min_index,2].replace(',',''))/1000 #assign the NARR melting layer to my array

    timedeltas = []
    for j in range(0,len(items_NPOL)):
        timedeltas.append(np.abs(pivot-items_NPOL[j]))
    min_index = timedeltas.index(np.min(timedeltas)) +1
    d2 = datetime.datetime.strptime(NPOL_data[min_index,0], '%m/%d/%y %H:%M:%S:').strftime('%m/%d/%y %H:%M:%S')
    melt_layer2 = NPOL_data[min_index,1]
    BrightBands_w_NARR[i,4] = d2
    BrightBands_w_NARR[i,5] = float(NPOL_data[min_index,1])/1000 #assign the NPOL melting layer to my array

    secondary_datetime_object = datetime.datetime.strptime(BrightBands_w_NARR[i,6], "%m/%d/%y %H:%M:%S")
    if secondary_datetime_object != datetime_object:
        print('index mismatch')
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

delta = end_date - start_date

#limit the plotting to days within defined start end periods.
BrightBands_w_NARR = BrightBands_w_NARR[start_index:end_index+1]

#save the data
pd.DataFrame(BrightBands_w_NARR).to_csv(save_fn_data_csv)

xdatesBB = np.empty(BrightBands_w_NARR.shape[0])
for xi in range(0,BrightBands_w_NARR.shape[0]):
    xdatesBB[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR[xi,0], '%m/%d/%y %H:%M:%S')])
xdatesNARR = np.empty(BrightBands_w_NARR.shape[0])
for xi in range(0,BrightBands_w_NARR.shape[0]):
    xdatesNARR[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR[xi,2], '%m/%d/%y %H:%M:%S')])
    if xdatesNARR[xi]>end_date_value:
        xdatesNARR[xi] = end_date_value
        BrightBands_w_NARR[xi,3] = float('NaN')
xdatesNPOL = np.empty(BrightBands_w_NARR.shape[0])
for xi in range(0,BrightBands_w_NARR.shape[0]):
    xdatesNPOL[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR[xi,4], '%m/%d/%y %H:%M:%S')])
    if xdatesNPOL[xi]>end_date_value:
        xdatesNPOL[xi] = end_date_value
        BrightBands_w_NARR[xi,5] = float('NaN')
xdatesSecondary = np.empty(BrightBands_w_NARR.shape[0])
for xi in range(0,BrightBands_w_NARR.shape[0]):
    xdatesSecondary[xi] = dates.date2num([datetime.datetime.strptime(BrightBands_w_NARR[xi,6], '%m/%d/%y %H:%M:%S')])


#plot the data
fig, ax = plt.subplots()
days = dates.DayLocator(interval = 2)
hours = dates.HourLocator(interval = 2)
d_fmt = dates.DateFormatter('%m/%d/%y')
h_fmt = dates.DateFormatter('%m/%d/%y %H:%M')
if plot_NARR:
    ax.scatter(xdatesNARR,BrightBands_w_NARR[:,3], label = 'NARR Melt Level', color = 'gray',marker = 'o', s = 10)
if plot_NPOL_0C:
    ax.scatter(xdatesNPOL,BrightBands_w_NARR[:,5], label = 'NPOL Sounding 0C',color = "#1b9e77",marker = '^', s = 10)
if plot_BB:
    ax.plot(xdatesBB,BrightBands_w_NARR[:,1], label = 'NPOL Algorithm BB', color = 'blue',linestyle = '--', linewidth = 1.25)
if plot_Secondary:
    upper_limit = np.array(BrightBands_w_NARR[:,8], dtype = float)
    lower_limit = np.array(BrightBands_w_NARR[:,7], dtype = float)
    ax.fill_between(xdatesSecondary, upper_limit,lower_limit, color="#9a0bad", alpha=0.4)
    ax.plot(xdatesSecondary,lower_limit, label = 'Secondary Enhancement', color = 'purple', linestyle = '-', linewidth = 0.75)
    ax.plot(xdatesSecondary,upper_limit, color = 'purple',linestyle = '-', linewidth = 0.75)

#ax.xticks(xdatesNPOL,BrightBands_w_NARR[:,0])
if delta.days < 2:
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
ax.set_ylim([0.25,7.5])
ax.set_title(''.join(['OLYMPEX Bright Band Identification - ',dir]))
ax.set_ylabel('Height (km)')
plt.setp(ax.get_xticklabels(), rotation=90)
plt.legend(loc = 'upper right')
fig.tight_layout()
plt.show()
#plt.savefig(save_fn_fig)
