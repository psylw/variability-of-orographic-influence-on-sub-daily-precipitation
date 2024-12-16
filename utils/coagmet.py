#%%
import requests

# URL of the CSV file
url = "https://coagmet.colostate.edu/data/metadata.csv?header=yes&units=m"

# Path to save the downloaded file
output_path = "metadata.csv"

# Download the file
response = requests.get(url)
if response.status_code == 200:
    with open(output_path, 'wb') as file:
        file.write(response.content)
    print(f"File downloaded successfully and saved as {output_path}")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")


#%%
import os
import requests

# Base URL for data download
base_url = "https://coagmet.colostate.edu/data/hourly.csv?header=yes&units=m&fields=precip"

# Define the output directory and create it if it doesn't exist
output_dir = "../output/coagment"
os.makedirs(output_dir, exist_ok=True)

# Loop over each year and download the data
for year in range(1991, 2016):
    # Construct the URL with the appropriate date range for each year
    url = f"{base_url}&from={year}-06-01&to={year}-08-31"
    
    # Define the output file path
    output_file = os.path.join(output_dir, f"precip_data_{year}.csv")
    
    # Download the data
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_file, 'wb') as file:
            file.write(response.content)
        print(f"Data for {year} downloaded successfully and saved as {output_file}")
    else:
        print(f"Failed to download data for {year}. Status code: {response.status_code}")


# %%
