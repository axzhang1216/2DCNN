###############################################################################
### Run_Forecast.py
### 2DCNN:执行常规预报模块
### 下载部分需要在linux系统下进行
### Author: Aoxing Zhang et al.
### Sep.27, 2022
###############################################################################


# 下载气象集合预报数据
def download_meteorology(todayYMD, casthours, jumpedhours, timeres):
    for i in range((casthours-jumpedhours)/timeres):
        for nens in range(30):
            #target = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod/gefs."+todayYMD+"/00/atmos/pgrb2sp25/gep" + str(nens+1).zfill(2) + ".t00z.pgrb2s.0p25.f" + str(i*timeres+jumpedhours).zfill(3)
            target = "https://ftp.ncep.noaa.gov/data/nccf/com/gens/prod/gefs."+todayYMD+"/00/atmos/pgrb2sp25/gep" + str(nens+1).zfill(2) + ".t00z.pgrb2s.0p25.f" + str(i*timeres+jumpedhours).zfill(3)
            command = "wget " + target
            print(command)
            os.system(command)
    return


# 修改已下载的气象数据的文件名
def change_metfile_names(todayYMD, path):
    dt_dl = datetime.strptime(todayYMD + "00",'%Y%m%d%H')
    filenames = os.listdir(path)
    for fn in filenames:
        sysfunc = "cp " + path + fn + " " + path + fn[0:5] + '_' + dt_dl.strftime("%Y%m%d%H")+'-'+\
            (dt_dl+timedelta(hours=int(fn[-3:]))).strftime("%Y%m%d%H") + '_' + fn[-3:] + ".surface.grb2"
        print(sysfunc)
        print(int(fn[-3:]))
        os.system(sysfunc)
    return

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

# 下载气象集合预报数据
download_meteorology(todayYMD)

# 修改已下载的气象数据的文件名
change_metfile_names(path)

 
