#%%
import os
import requests

# Directory to save PRISM data
save_dir = "../data/prism"
os.makedirs(save_dir, exist_ok=True)

# Base URL for PRISM data (adjust as necessary)
base_url = "https://ftp.prism.oregonstate.edu/daily/"

# Years and months for JJA (June, July, August) from 2016 to 2022
years = range(2016, 2023)
months = ['06', '07', '08']  # June, July, August
data_type = "ppt"  # Set to 'ppt' for precipitation; use 'tmax' or 'tmin' for temperatures

for year in years:
    for month in months:
        # Determine the number of days in the month
        if month == '06' or month == '08':  # June and August have 30 and 31 days
            days_in_month = 30 if month == '06' else 31
        elif month == '07':  # July has 31 days
            days_in_month = 31

        for day in range(1, days_in_month + 1):
            # Format the day with leading zero if needed
            day_str = f"{day:02d}"
            
            # Construct the file URL (adjust if necessary based on data type or source)
            file_name = f"PRISM_{data_type}_stable_4kmD2_{year}{month}{day_str}_bil.zip"
            url = f"{base_url}{data_type}/{year}/{file_name}"
            save_path = os.path.join(save_dir, file_name)
            
            # Download the file
            try:
                print(f"Downloading {file_name} for {year}-{month}-{day_str}...")
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(save_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    print(f"Downloaded {file_name}")
                else:
                    print(f"Failed to download {file_name} (Status Code: {response.status_code})")
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")
# %%
