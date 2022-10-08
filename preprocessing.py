###############################################################################
### Run_Forecast.py
### 2DCNN:执行常规预报模块
### 下载部分需要在linux系统下进行
### Author: Aoxing Zhang et al.
### Sep.27, 2022
###############################################################################


# Main
from datetime import datetime, timedelta
import os
import numpy as np
from datetime import datetime, timedelta, date
import os
import pandas as pd
import netCDF4 as nc
import pygrib
from skimage import transform
import tensorflow.keras as keras
from copy import copy

# 初始化
# Initialization

# GEFS气象场中集合预报成员的数量（30个）
# The number of ensemble menbers in GEFS (30)
n_of_ensembles = 30

# Get today's date (00z).
# 设定预报日期（默认为运行程序的当天，可以手动修改）
today = date.today()
todayYMD = today.strftime('%Y%m%d')

print("=========================================")
print("===== CNN ensemble ozone forecast =======")
print("== SUSTech Atmospheric Chemistry Group ==")
print("======== Aoxing Zhang, Feb 2022 =========")
print("=========================================")
print("Today is " + today.strftime('%Y-%m-%d'))

# 预报小时数：从today变量所在天的UTC 00：00开始计算
# 9 day (216 hours) forecast from UTC 00:00 of today.
casthours = 216

# 忽略下载的时间：因为关注的预报肯定是从发布预报的当天（一般是执行预报的下一天）开始，、
#                所以（当地时间）第一天的气象场就不需要参与下载和预报了
# We dont need today's 'forecast', so the first 16 hours'
# results are neglected, so that the 'actual' forecast starts
# at 'tomorrow' 00:00 UTC+8
# for the 3-hr GEFS dataset, the starting time is at 'tomorrow' 02:00 UTC+8
jumpedhours = 18

print("Forecast will start for local time " + (datetime.strptime(todayYMD,'%Y%m%d')+timedelta(hours=jumpedhours+8)).strftime('%Y-%m-%d %H:%M:%S'))
print("Forecast will be calculated until local time " + (datetime.strptime(todayYMD,'%Y%m%d')+timedelta(hours=casthours+8)).strftime('%Y-%m-%d %H:%M:%S'))

# GEFS forecast dataset temporal resolution is 3 hours.
# GEFS 时间分辨率
timeres = 3
print("Temporal resolution: "+str(timeres)+ " hours")

# training data 2-D size:
xn = 24
yn = 19
zn = 7

# lat and lon values for training region around Shenzhen
lat0 = 19
lat1 = 26
lon0 = 111
lon1 = 117

max_nframe = int((casthours-jumpedhours)/timeres+1)

# The target training set meteorological field:
X = np.empty((n_of_ensembles, max_nframe, xn, yn, zn))
X[:] = np.NaN

ind = {'u10':0, 'v10':1, 't2m':2, 'rh2m':3, 'ps':4, 'prec6h':5, 'hr':6}

# Path of the GEFS
# GEFS预计的文件存放路径
path = './data/'+ todayYMD + '/'

##########################################################
# 数据预处理
filenames = os.listdir(path)
for fn in filenames:
    # nens: the ensemble number of the file
    # castdate_UTC: the forecast date in UTC (datetime class)
    # ftype: 'surface' or 'profile'. Vars in both files will be used.
    print(fn)
    #print(int(fn[3:5]))
    #print(fn[-27:-17])
    #print(fn[-12:-5])
    #print(int(fn[-16:-13]))
    try:
        nens = int(fn[3:5])
        castdate_UTC = datetime.strptime(fn[-27:-17], '%Y%m%d%H')
        ftype = fn[-12:-5]
        fcstduation = int(fn[-16:-13])
        
        if castdate_UTC < datetime.strptime(todayYMD,'%Y%m%d') + timedelta(hours = jumpedhours):
            print('met too early')
            continue
        else:
            nframe = (castdate_UTC - datetime.strptime(todayYMD,'%Y%m%d') - timedelta(hours = jumpedhours)).total_seconds()/3600/timeres
            nframe = int(nframe)
            if nframe>=max_nframe:
                print('met too late')
                continue
        print("# of ensembles: " + str(nens))
        print("Castdate in UTC: " + str(castdate_UTC))
        print("File type processing: " + ftype)
        print("# of frame: " + str(nframe))
    except:
        continue

    # get met field from GEFS files

    if ftype == 'surface':

        grbs = pygrib.open(path+fn)
        grb = grbs.select(name = '2 metre temperature')[0]

        if not ('lats_sfc' in globals()):

            lats_sfc,lons_sfc = grb.latlons()

            latind0_sfc = np.argmin(np.abs(lat0-lats_sfc[:,0]))
            latind1_sfc = np.argmin(np.abs(lat1-lats_sfc[:,0]))
            lonind0_sfc = np.argmin(np.abs(lon0-lons_sfc[0,:]))
            lonind1_sfc = np.argmin(np.abs(lon1-lons_sfc[0,:]))

            print(latind0_sfc)
            print(latind1_sfc)
            print(lonind0_sfc)
            print(lonind1_sfc)

        t2m = np.array(grb.values)[latind0_sfc:latind1_sfc:-1,lonind0_sfc:lonind1_sfc]
        t2m = transform.resize(t2m, (24,19), order=3)
        X[nens-1, nframe, :, :, ind['t2m']] = t2m
        grb = grbs.select(name = '10 metre U wind component')[0]
        u10 = np.array(grb.values)[latind0_sfc:latind1_sfc:-1,lonind0_sfc:lonind1_sfc]
        u10 = transform.resize(u10, (24,19), order=3)
        X[nens-1, nframe, :, :, ind['u10']] = u10
        
        grb = grbs.select(name = 'Surface pressure')[0]
        ps = np.array(grb.values)[latind0_sfc:latind1_sfc:-1,lonind0_sfc:lonind1_sfc]
        ps = transform.resize(ps, (24,19), order=3)
        X[nens-1, nframe, :, :, ind['ps']] = ps
        
        grb = grbs.select(name = '10 metre V wind component')[0]
        v10 = np.array(grb.values)[latind0_sfc:latind1_sfc:-1,lonind0_sfc:lonind1_sfc]
        v10 = transform.resize(v10, (24,19), order=3)
        X[nens-1, nframe, :, :, ind['v10']] = v10
        
        grb = grbs.select(name = '2 metre dewpoint temperature')[0]
        dt2m = np.array(grb.values)[latind0_sfc:latind1_sfc:-1,lonind0_sfc:lonind1_sfc]
        dt2m = np.array(grb.values)[latind0_sfc:latind1_sfc:-1,lonind0_sfc:lonind1_sfc]
        dt2m = transform.resize(dt2m, (24,19), order=3)

        # Precipitation is the accumulation precipitation starting from the initial condition till the forecast time.
        grb = grbs.select(name = 'Total Precipitation')[0]
        prec = np.array(grb.values)[latind0_sfc:latind1_sfc:-1,lonind0_sfc:lonind1_sfc]
        prec = transform.resize(prec, (24,19), order=3)
        X[nens-1, nframe, :, :, ind['prec6h']] = prec*2 # unit mm
        
        # Calculate RH from dt and temperature
        B = (dt2m-273.15)/(237.3+dt2m-273.15)
        rh2m = np.exp(17.27 * B - 17.27 * (t2m-273.15) / (237.3 + t2m - 273.15))*100
        hr = np.zeros(rh2m.shape) + castdate_UTC.hour
        X[nens-1, nframe, :, :, ind['hr']] = hr
        X[nens-1, nframe, :, :, ind['rh2m']] = rh2m

np.save('GEFS_ens_test_sfc_' + today.strftime('%Y%m%d') + '.npy', np.array(X))

 
