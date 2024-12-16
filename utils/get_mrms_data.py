#%%
import os
import requests
from datetime import datetime, timedelta

# Function to download a file
def download_file(url, save_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"Downloaded: {save_path}")
        else:
            print(f"Failed to download: {url} - Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

# Define the years and months for JJA
years = range(2016, 2023)  # 2016 to 2022 inclusive
months = ['06', '07', '08']
SAVE_DIR = '../data/MRMS'
# Loop through each year, month, and day
for year in years:
    for month in months:
        # Determine the number of days in the month
        num_days = 30 if month in ['06'] else 31
        for day in range(1, num_days + 1):
            date_str = f"{year}{month}{day:02d}"
            base_url = f"https://mtarchive.geol.iastate.edu/{year}/{month}/{day:02d}/mrms/ncep/RadarOnly_QPE_01H/"
            
            for hour in range(24):
                hour_str = f"{hour:02d}0000"
                filename = f"RadarOnly_QPE_01H_00.00_{date_str}-{hour_str}.grib2.gz"
                file_url = f"{base_url}{filename}"
                save_path = os.path.join(SAVE_DIR,filename)
                
                # Download the file
                download_file(file_url, save_path)
