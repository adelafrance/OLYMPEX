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

dir = 'west' #look east or west (lowercase)
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

output_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/BrightBands/' #location of previous output
secondary_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/Secondary/'
data_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Data/' #directory for local data
if dir == 'east':
    bb_data = ''.join(['brightbandsfound_v6_r_6_time0x35.0pcntx25.0_withrhohv_0.910.97_',dir,'.npy'])
elif dir == 'west':
    bb_data = ''.join(['brightbandsfound_v6_r_6_time0x15.0pcntx25.0_withrhohv_0.910.97_',dir,'.npy'])
#secondary_data = ''.join(['secondary_C_15X4excd_',dir,'.npy'])
secondary_data = ''.join(['secondary_D_15X4excd_',dir,'.npy'])
sounding_data = 'NPOL_sounding_0_levs.npy'
NARR_data = 'NARR_at_NPOL.csv'
save_name_data_csv = ''.join(['BB_w_Secondary_',dir,'_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.csv'])
save_name_fig = ''.join(['BB_w_Secondary_',dir,'_',str(mon_start),'-',str(day_start),'to',str(mon_end),'-',str(day_end),'.png'])
bb_fn = ''.join([output_dir,bb_data])
secondary_fn = ''.join([secondary_dir,secondary_data])
sounding_fn = ''.join([output_dir,sounding_data])
NARR_fn = ''.join([data_dir,NARR_data])
save_fn_data_csv = ''.join([secondary_dir,save_name_data_csv])
save_fn_fig = ''.join([secondary_dir,save_name_fig])

#list of citation flights
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

bright_bands = np.load(bb_fn)#time,bbfound, top, btm, bbcrit1, bbcrit2
secondary = np.load(secondary_fn)
NPOL_data = np.load(sounding_fn)
df=pd.read_csv(NARR_fn, sep=',',header=None)
NARR_data = np.array(df) #NARR Time,IVT,Melting Level (m),925speed (kt),925dir,Nd,Nm
n_bbs = bright_bands.shape[0]
n_NARRs = NARR_data.shape[0]
n_NPOLs = NPOL_data.shape[0]
n_secondary = secondary.shape[0]

#restrict plotting to only use times that a bright band occurred
for i in range(0,n_bbs):
    if bright_bands[i,1] != '1':
        bright_bands[i,2] = np.float('NaN')

#same for secondary enhancement
for i in range(0,n_secondary):
    if secondary[i,1] != '1':
        secondary[i,2] = np.float('NaN')
        secondary[i,3] = np.float('NaN')

BrightBands_w_NARR = bright_bands[:,[0,2]] #assign bright bands found in bbidv6 to the array
secondary_levs = secondary[:,[0,2,3]] #grab dates, low enhance lev, high enhnace lev
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
    ax.scatter(xdatesNARR,BrightBands_w_NARR[:,3], label = 'NARR Melt Level', color = '#e66101',marker = 'o', s = 10)
if plot_NPOL_0C:
    ax.scatter(xdatesNPOL,BrightBands_w_NARR[:,5], label = 'NPOL Sounding 0'+ '\u00b0'+ 'C',color = "mediumblue",marker = '^', s = 10)
if plot_BB:
    ax.plot(xdatesBB,BrightBands_w_NARR[:,1], label = 'NPOL Radar Bright Band', color = 'dimgray',linestyle = '--', linewidth = 1.25)
if plot_Secondary:
    upper_limit = np.array(BrightBands_w_NARR[:,8], dtype = float)
    lower_limit = np.array(BrightBands_w_NARR[:,7], dtype = float)

    #ax.fill_between(xdatesSecondary, upper_limit,lower_limit, color="#9a0bad", alpha=0.4)
    #ax.plot(xdatesSecondary,lower_limit, label = 'Secondary Enhancement', color = 'purple', linestyle = '-', linewidth = 0.75)
    #ax.plot(xdatesSecondary,upper_limit, color = 'purple',linestyle = '-', linewidth = 0.75)
    ax.fill_between(xdatesSecondary, upper_limit,lower_limit, color="#1b9e77", alpha=0.4)
    ax.plot(xdatesSecondary,lower_limit, label = 'Secondary Enhancement', color = 'darkgreen', linestyle = '-', linewidth = 0.75)
    ax.plot(xdatesSecondary,upper_limit, color = 'darkgreen',linestyle = '-', linewidth = 0.75)

if plot_Citation:
    for i in range(0,len(flight_starts)):
        if flight_starts[i] >= start_date and flight_ends[i] <= end_date:
            x1 = dates.date2num(flight_starts[i])
            x2 = dates.date2num(flight_ends[i])
            ax.axvspan(x1, x2, alpha=0.5, color='lightgrey')
        elif flight_starts[i] < start_date and flight_ends[i] >= start_date and flight_ends[i] <= end_date:
            x1 = dates.date2num(start_date)
            x2 = dates.date2num(flight_ends[i])
            ax.axvspan(x1, x2, alpha=0.5, color='lightgrey')
        elif flight_starts[i] >= start_date and flight_starts[i] <= end_date and flight_ends[i] > end_date:
            x1 = dates.date2num(flight_starts[i])
            x2 = dates.date2num(end_date)
            ax.axvspan(x1, x2, alpha=0.5, color='lightgrey')

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
if plot_Secondary:
    ax.set_ylim([0.25,7.5])
else:
    ax.set_ylim([0.25,3.5])
#ax.set_title(''.join(['OLYMPEX Bright Band Identification - ',dir]))
ax.set_ylabel('Height (km)')
plt.setp(ax.get_xticklabels(), rotation=90)
plt.legend(loc = 'upper right', ncol = 1)
fig.tight_layout()
#plt.show()
plt.savefig(save_fn_fig, dpi = 300)
