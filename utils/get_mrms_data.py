# %%
from datetime import datetime, time
import os
import urllib.request
from urllib.request import HTTPError
from datetime import timedelta

# source for code: https://github.com/HydrologicEngineeringCenter/data-retrieval-scripts/blob/master/retrieve_qpe_gagecorr_01h.py
# 
# https://www.hec.usace.army.mil/confluence/hmsdocs/hmsguides/working-with-gridded-boundary-condition-data/downloading-multi-radar-multi-sensor-mrms-precipitation-data
# 

# 
# Unzip files after download

# change dates and sample interval

start = datetime(2023, 6, 1, 1, 0)
end = datetime(2023, 6, 30, 23, 58)
minute = timedelta(minutes=2)
#minute = timedelta(hours=1)
# only download certain months
mo1 = 6
mo2 = 6

# indicate which product to download (rate,RQI,QPE_multi,QPE_radar)
product = 'RQI'

# tell program where to download files ==> /data/raw
# Create a path to the code file
codeDir = os.path.dirname(os.path.abspath(os.getcwd()))
# Create a path to the data folder
# destination = os.path.join(codeDir,"data","processed")
destination = 'Z:\\working code\\MRMS-eval-with-gages-in-CO\\temp'
# Change to data folder
os.chdir(destination)

missing_dates = []
fallback_to_radaronly = True #Enables a post-processing step that will go through the list of missing dates for gage-corrected
############################# and tries to go get the radar-only values if they exist.

date = start

while date <= end:
    if date.month>=mo1 and date.month<=mo2:
        if product == 'rate':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/PrecipRate/PrecipRate_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(
            date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        
        elif product == 'RQI':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/RadarQualityIndex/RadarQualityIndex_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        elif product == 'QPE_multi':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/MultiSensor_QPE_01H_Pass2/MultiSensor_QPE_01H_Pass2_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        elif product == 'QPE_multi':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/RadarOnly_QPE_01H/RadarOnly_QPE_01H_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        
        filename = url.split("/")[-1]
        try:
            fetched_request = urllib.request.urlopen(url)
        except HTTPError as e:
            missing_dates.append(date)
        else:
            with open(destination + os.sep + filename, 'wb') as f:
                f.write(fetched_request.read())
        finally:
            date += minute
    else:
        date += minute

if fallback_to_radaronly:
    radar_also_missing = []
    for date in missing_dates:
        if product == 'rate':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/PrecipRate/PrecipRate_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(
            date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        
        elif product == 'RQI':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/RadarQualityIndex/RadarQualityIndex_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        elif product == 'QPE_multi':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/MultiSensor_QPE_01H_Pass2/MultiSensor_QPE_01H_Pass2_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        elif product == 'QPE_multi':
            url = "http://mtarchive.geol.iastate.edu/{:04d}/{:02d}/{:02d}/mrms/ncep/RadarOnly_QPE_01H/RadarOnly_QPE_01H_00.00_{:04d}{:02d}{:02d}-{:02d}{:02d}00.grib2.gz".format(date.year, date.month, date.day, date.year, date.month, date.day, date.hour,date.minute)
        
        
        filename = url.split("/")[-1]
        try:
            fetched_request = urllib.request.urlopen(url)
        except HTTPError as e:
            radar_also_missing.append(date)
        else:
            with open(destination + os.sep + filename, 'wb') as f:
                f.write(fetched_request.read())

print(radar_also_missing)






# %%
