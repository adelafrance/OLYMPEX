#get the sounding data from NPOL to compare to algorithm output

import numpy as np
from netCDF4 import Dataset, num2date, date2num
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#sounding directory
sound_dir = '/home/disk/bob/olympex/soundings/QC_Level4/netcdf/'
save_dir = '/home/disk/meso-home/adelaf/OLYMPEX/Output/BrightBands/'
save_fn = ''.join([save_dir,"NPOL_sounding_melt_levs"])

fn = 'OLYMPEX_upaL4.0_npol.nc'
#missing values are -999.0
file_path = ''.join([sound_dir,fn])
nc_fid = Dataset(file_path, 'r')

sounding = nc_fid.variables["sounding"][:]
n_soundings = len(nc_fid.variables["sounding"][:])
level = nc_fid.variables["level"][:]
n_levels = len(nc_fid.variables["level"][:])

launch_time = nc_fid.variables["launch_time"][:]
time_units = nc_fid.variables['launch_time'].units

date_start = num2date(launch_time[0], units = time_units , calendar = 'standard')

alt = nc_fid.variables["alt"][:]
temp = nc_fid.variables["T"][:]

sounding_ml = np.array([1,2]) #date, elevation

for time in range(0,n_soundings):
    date = num2date(launch_time[time], units = time_units , calendar = 'standard')
    ml = np.where(temp[time,:] < 0)[0][0]-1 #lowest level above 0 C
    ml_ht = alt[time,ml]
    data_to_append = np.array([date.strftime("%m/%d/%y %H:%M:%S:"),ml_ht])
    sounding_ml = np.vstack((sounding_ml,data_to_append))

np.save(save_fn,sounding_ml)
