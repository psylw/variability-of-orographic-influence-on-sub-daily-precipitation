#%%
import os
import requests

# Directory to save PRISM data
save_dir = "../data/prism"
os.makedirs(save_dir, exist_ok=True)

# Base URL for PRISM data (adjust as necessary)
base_url = "https://ftp.prism.oregonstate.edu/daily/"

# Years and months from March 1981 to August 2022
start_year, start_month = 1996, 12
end_year, end_month = 1997, 2
data_type = "ppt"  # Set to 'ppt' for precipitation; use 'tmax' or 'tmin' for temperatures

for year in range(start_year, end_year + 1):
    for month in range(1, 13):  # Loop through all months
        if (year == start_year and month < start_month) or (year == end_year and month > end_month):
            continue  # Skip months outside the March 1981 - August 2022 range
        
        # Determine the number of days in the month
        if month in [1, 3, 5, 7, 8, 10, 12]:  # Months with 31 days
            days_in_month = 31
        elif month in [4, 6, 9, 11]:  # Months with 30 days
            days_in_month = 30
        elif month == 2:  # February, check for leap year
            days_in_month = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28

        for day in range(1, days_in_month + 1):
            # Format the month and day with leading zeros if needed
            month_str = f"{month:02d}"
            day_str = f"{day:02d}"
            
            # Construct the file URL (adjust if necessary based on data type or source)
            file_name = f"PRISM_{data_type}_stable_4kmD2_{year}{month_str}{day_str}_bil.zip"
            url = f"{base_url}{data_type}/{year}/{file_name}"
            save_path = os.path.join(save_dir, file_name)
            
            # Download the file
            try:
                print(f"Downloading {file_name} for {year}-{month_str}-{day_str}...")
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
