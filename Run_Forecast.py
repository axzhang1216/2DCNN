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
path = './GEFS/'+ todayYMD + '/'

# 下载气象集合预报数据
download_meteorology(todayYMD)

# 修改已下载的气象数据的文件名
change_metfile_names(path)

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

# 数据预测
X = np.load('GEFS_ens_test_sfc_' + today.strftime('%Y%m%d') + '.npy')
#today = datetime.strptime('20220307', '%Y%m%d')
#X = np.load('GEFS_ens_test_20220307.npy')
Xshape0 = X.shape[0]
Xshape1 = X.shape[1]
X = X.reshape((Xshape0*Xshape1, X.shape[2], X.shape[3], X.shape[4]))
vername = '20220415'
def predict_20220415(X):
    Xcopy = copy(X)
    XScale1 = np.load('/work/ese-zhangax/1_Codes/1_DNN/XScale1_ver20220415_SZ_urban.npy')
    XScale2 = np.load('/work/ese-zhangax/1_Codes/1_DNN/XScale2_ver20220415_SZ_urban.npy')
    print(Xcopy.shape)
    # Normalization
    X_use = Xcopy[:,:,:,[0,1,2,3,4,-2,-1]]
    # surface only
    X_use = Xcopy[:,:,:,:]
    for c in range(X_use.shape[3]):
        # Normalize to 0~1:
        X_use[:,:,:,c] = (X_use[:,:,:,c]-XScale1[c])/XScale2[c]
    
    model = keras.models.load_model('/work/ese-zhangax/1_Codes/1_DNN/models/CNNver20220415_SZ_urban')
    
    y_pred = model.predict(X_use)
    y_pred = np.array(y_pred).flatten()
    return y_pred
y_pred = predict_20220415(X)
y_pred = y_pred.reshape(Xshape0, Xshape1)
print(y_pred.shape)


# make datelist
datelist = []
for i in range(int((casthours-jumpedhours)/timeres+1)):
    datelist = datelist + [(datetime.strptime(todayYMD,'%Y%m%d') + timedelta(hours = jumpedhours + i*timeres + 8))]
y_pred_mean = np.mean(y_pred, axis=0)
y_pred_std = np.std(y_pred, axis=0)
print(y_pred_mean.shape)
print(y_pred_mean)

# 预报结果可视化
from matplotlib import pyplot as plt
fig = plt.figure(figsize=(12, 16), dpi=100)
plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['font.size'] = '15'
fig.set_facecolor('white')
ax = fig.add_subplot(3,1,1)
ax.set_title('CNN深圳臭氧预报（基于GEFS气象）', fontsize = 20)
ax.plot(datelist, y_pred[0,:], '--', label = 'CNN 臭氧集合预报成员值', linewidth=1,alpha=0.7)
ax.plot(datelist, y_pred[1:].T, '--', linewidth=1,alpha=0.5)
ax.errorbar(datelist, y_pred_mean,yerr=y_pred_std, capsize=5,c='black',label = 'CNN 臭氧集合预报均值及标准差', linewidth=2)
ax.set_ylabel('臭氧浓度 (微克每立方米)')
ax.set_xlabel('北京时间')
ax.set_xticks(datelist[::8])
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m月%d日"))
ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m月%d日"))
#ax.tick_params(axis='x', rotation=60)
ax.set_ylim(0,250)
ax.legend(loc='upper left',frameon=False, fontsize=15)
#ax1 = ax.twinx()

def iAQI(o3list,datelist):  
    d = 0
    o3list_copy = copy(o3list)
    datelist_daily = []
    o3_8hrmean = copy(o3list)
    iaqi_MDA8 = []
    iaqi_maxhr = []
    risk = []
    for i in range(1, o3list.shape[0]-1):
        o3_8hrmean[i] = np.nanmean([o3list_copy[i-1],o3list_copy[i],o3list_copy[i+1]])
    ndays = int(len(datelist)/8)
    for d in range(ndays):
        #print(d)
        o3_hrmax = np.max(o3list[d*8:d*8+8])
        o3_MDA8 = np.max(o3_8hrmean[d*8:d*8+8])
        iaqihr = iaqihr_cal(o3_hrmax)
        iaqiMDA8 = iaqiMDA8_cal(o3_MDA8)
        iaqi_maxhr = iaqi_maxhr + [iaqihr]
        iaqi_MDA8 = iaqi_MDA8 + [iaqiMDA8]
        datelist_daily = datelist_daily + [datelist[d*8]-timedelta(hours=2)]
        risk = risk + [0]
        if o3_hrmax>=180:
            risk[-1] = risk[-1] + 1
        if o3_MDA8>=130:
            risk[-1] = risk[-1] + 2
    #print('datelist_daily = ',datelist_daily) 
    return iaqi_maxhr,iaqi_MDA8, datelist_daily, risk

def iaqihr_cal(o3_hrmax):
    aqithres_hr = np.array([0,160,200,300,400,800,1000])
    aqibin_hr = np.array([0,50,100,150,200,300,400,500])
    k = 0
    while o3_hrmax > aqithres_hr[k]:
        k += 1
    iaqi = aqibin_hr[k-1] + (aqibin_hr[k]-aqibin_hr[k-1])/(aqithres_hr[k]-aqithres_hr[k-1])*(o3_hrmax-aqithres_hr[k-1])
    #print(o3_hrmax)
    #print(iaqi)
    return iaqi
def iaqiMDA8_cal(o3_MDA8):
    aqithres_MDA8 = np.array([0,100,160,215,265,800])
    aqibin_MDA8 = np.array([0,50,100,150,200,300])
    k = 0
    while o3_MDA8 > aqithres_MDA8[k]:
        k += 1
    iaqi = aqibin_MDA8[k-1] + (aqibin_MDA8[k]-aqibin_MDA8[k-1])/(aqithres_MDA8[k]-aqithres_MDA8[k-1])*(o3_MDA8-aqithres_MDA8[k-1])
    return iaqi
iaqilist_maxhr = []
iaqilist_MDA8 = []
risklist = []
for i in range(y_pred.shape[0]):
    iaqi_maxhr,iaqi_MDA8, datelist_daily,risk = iAQI(y_pred[i,:],datelist)
    iaqilist_maxhr = iaqilist_maxhr + [iaqi_maxhr]
    iaqilist_MDA8 = iaqilist_MDA8 + [iaqi_MDA8]
    risklist = risklist + [risk]
iaqilist_maxhr = np.array(iaqilist_maxhr)
iaqilist_MDA8 = np.array(iaqilist_MDA8)
risklist = np.max(risklist,axis=0)
iaqi_mean_maxhr = np.nanmean(iaqilist_maxhr, axis=0)
iaqi_std_maxhr = np.nanstd(iaqilist_maxhr, axis=0)
p_episode_maxhr = np.count_nonzero(iaqilist_maxhr>100,axis=0)/30.0*100
p_episode_maxhr = p_episode_maxhr + iaqi_mean_maxhr - iaqi_mean_maxhr # 如果iaqi是nan，让p_episode也是nan
iaqi_mean_MDA8 = np.nanmean(iaqilist_MDA8, axis=0)
iaqi_std_MDA8 = np.nanstd(iaqilist_MDA8, axis=0)
p_episode_MDA8 = np.count_nonzero(iaqilist_MDA8>100,axis=0)/30.0*100
p_episode_MDA8 = p_episode_MDA8 + iaqi_mean_MDA8 - iaqi_mean_MDA8 # 如果iaqi是nan，让p_episode也是nan


# 超标概率与风险
ax2 = fig.add_subplot(3,1,2)
ax2.set_title('CNN深圳臭氧iAQI和超标概率', fontsize = 20)
ax2.scatter(datelist_daily*30,iaqilist_maxhr,marker='d',c='None',edgecolor='c',
            label = 'CNN 各预报成员臭氧iAQI-日最大小时值',alpha=0.7)
ax2.scatter(datelist_daily*30,iaqilist_MDA8,marker='d',c='None',edgecolor='salmon',
            label = 'CNN 各预报成员臭氧iAQI-日最大8小时平均',alpha=0.7)
ax2.errorbar(datelist_daily, iaqi_mean_maxhr,yerr=iaqi_std_maxhr, capsize=5,color = 'blue', label = 'CNN 臭氧iAQI-日最大小时值', linewidth=2)
ax2.errorbar(datelist_daily, iaqi_mean_MDA8,yerr=iaqi_std_MDA8, capsize=5,color = 'red', label = 'CNN 臭氧iAQI-日最大8小时平均', linewidth=2)

#ax2.scatter([],[],marker='o',color='orange',label = '臭氧超标概率 （%）')
n=0
for a,b in zip(datelist_daily, iaqi_mean_maxhr):
    ax2.text(a+timedelta(hours=7),b+1,"%.1f"%iaqi_mean_maxhr[n]+r'$\pm$'+"%.1f"%iaqi_std_MDA8[n],ha="center", va="bottom",fontsize=12)
    n = n+1
n=0
for a,b in zip(datelist_daily, iaqi_mean_MDA8):
    ax2.text(a+timedelta(hours=7),b+1,"%.1f"%iaqi_mean_MDA8[n]+r'$\pm$'+"%.1f"%iaqi_std_MDA8[n],ha="center", va="bottom",fontsize=12)
    n = n+1
ax2.set_ylabel('CNN 臭氧iAQI')
ax2.set_xlabel('北京时间')
import matplotlib.dates as mdates
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m月%d日"))
ax2.xaxis.set_minor_formatter(mdates.DateFormatter("%m月%d日"))
#plt.xticks(datelist_daily,[x.strftime('%m月%d日') for x in datelist_daily])
#ax2.set_xticklabels([x.strftime('%m月%d日') for x in datelist_daily[::2]])
#ax2.tick_params(axis='x', rotation=60)
#ax2.tick_params(axis='x')
ax2.legend(loc='upper left',frameon=False, fontsize=15)
ax2.set_ylim(0,150)
ax3 = ax2.twinx()
datelist_daily_L = [x-timedelta(hours=1) for x in datelist_daily]
datelist_daily_R = [x+timedelta(hours=1) for x in datelist_daily]

ax3.scatter(datelist_daily_L,p_episode_maxhr,marker='o',color='orange',label = '基于日最大小时值的臭氧超标概率 （%）')
ax3.scatter(datelist_daily_R,p_episode_MDA8,marker='o',color='c',label = '基于MDA8的臭氧超标概率 （%）')

for a,b in zip(datelist_daily, p_episode_maxhr):
    ax3.text(a-timedelta(hours=3),b+1,"%.1f"%b+'%',ha="center", va="bottom",fontsize=12)
for a,b in zip(datelist_daily, p_episode_MDA8):
    ax3.text(a+timedelta(hours=3),b+1,"%.1f"%b+'%',ha="center", va="bottom",fontsize=12)

ax3.set_ylim(-5,105)
ax3.set_ylabel('臭氧超标概率 （%）')
import matplotlib.dates as mdates
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m月%d日"))
ax3.xaxis.set_minor_formatter(mdates.DateFormatter("%m月%d日"))
ax3.legend(loc='upper right',frameon=False, fontsize=15)

axtext=fig.add_subplot(3,1,3,frameon=False)
axtext.axis('off')
axtext.text(0, 1.1,'臭氧风险评估：', ha='left', va='center')
k=1
for n in range(len(datelist_daily)):
    if p_episode_maxhr[n]>0:
        if p_episode_MDA8[n]>0:
            axtext.text(0.1, 1.1-k*0.1,'基于CNN集合预报，深圳市'+datelist_daily[n].strftime('%m月%d日') + '存在臭氧小时浓度和MDA8超标风险。', ha='left', va='center')
            k = k+1
        else:
            axtext.text(0.1, 1.1-k*0.1,'基于CNN集合预报，深圳市'+datelist_daily[n].strftime('%m月%d日') + '存在臭氧小时浓度超标风险。', ha='left', va='center')
            k = k+1
    elif p_episode_MDA8[n]>0:
        axtext.text(0.1, 1.1-k*0.1,'基于CNN集合预报，深圳市'+datelist_daily[n].strftime('%m月%d日') + '存在臭氧MDA8超标风险。', ha='left', va='center')
        k = k+1
if k==1:
    axtext.text(0.1, 1.1-k*0.1,'基于CNN集合预报，深圳市在近日预报结果中无显著超标风险。', ha='left', va='center')
fig.tight_layout()
plt.show()
#plt.savefig('CNN_GFES_Spring' + vername + '_' + today.strftime('%Y%m%d') + '.png',bbox_inches='tight')
 