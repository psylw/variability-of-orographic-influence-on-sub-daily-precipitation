import ftplib
import os
import zipfile
import xarray as xr
import numpy as np

# FTP information
ftp_server = 'ftp.prism.oregonstate.edu'
base_ftp_path = '/daily/ppt/'  # Base path for the FTP
local_download_path = './prism_data/'  # Local directory to save files

# Ensure local directory exists
os.makedirs(local_download_path, exist_ok=True)

# Define years and months of interest
years = range(2002, 2024)
months_of_interest = ['06', '07', '08']  # June, July, August

# Connect to FTP
ftp = ftplib.FTP(ftp_server)

# Loop through each year
for year in years:
    ftp_path = f"{base_ftp_path}{year}/"  # Path for the specific year
    ftp.cwd(ftp_path)
    files = ftp.nlst()  # List all files in the directory
    
    # Filter files for June, July, and August
    for file_name in files:
        if file_name.endswith('.zip'):
            # Extract the year and month from the filename (assuming format YYYYMMDD in the filename)
            year_month = file_name.split('_')[4][:6]  # Get YYYYMM part from the filename
            year_str, month_str = year_month[:4], year_month[4:]
            
            # Check if the month is June, July, or August
            if month_str in months_of_interest:
                local_file = os.path.join(local_download_path, file_name)
                
                # Download the zipped file
                print(f"Downloading {file_name}...")
                with open(local_file, 'wb') as f:
                    ftp.retrbinary(f'RETR {file_name}', f.write)
                
                # Unzip the file
                with zipfile.ZipFile(local_file, 'r') as zip_ref:
                    zip_ref.extractall(local_download_path)
                
                # Find the .bil file extracted
                bil_files = [f for f in os.listdir(local_download_path) if f.endswith('.bil')]
                prj_files = [f for f in os.listdir(local_download_path) if f.endswith('.prj')]
                # just use first prj
                prj_file_path = prj_files[0]
                with open(prj_file_path, 'r') as prj_file:
                    prj_string = prj_file.read()
                # Create a CRS object from the .prj file string
                crs = CRS.from_wkt(prj_string)


                for bil_file in bil_files:
                    bil_path = os.path.join(local_download_path, bil_file)
                    # Step 4: Use rasterio to open the .bil file
                    with rasterio.open(bil_path) as dataset:
                        # Read the data as a numpy array
                        bil_data = dataset.read(1)  # Read the first band (assuming single band)
                        
                        # Get the affine transform from the dataset
                        transform = dataset.transform
                        height, width = bil_data.shape
                        
                        # Create arrays for x and y coordinates using numpy and the transform
                        x_indices = np.arange(width)
                        y_indices = np.arange(height)
                        
                        # Create 2D arrays of x and y coordinates
                        x_coords, y_coords = np.meshgrid(x_indices, y_indices)
                        
                        # Apply the affine transformation to get actual geographic coordinates
                        x_geo, y_geo = rasterio.transform.xy(transform, y_coords, x_coords, offset='center')
                        
                        # Convert the results to 2D arrays
                        x_geo = np.array(x_geo).reshape(height, width)
                        y_geo = np.array(y_geo).reshape(height, width)

                        # Convert to an xarray DataArray with the correct CRS and coordinates
                        da = xr.DataArray(
                            bil_data,
                            dims=['y', 'x'],
                            coords={'y': y_geo[:, 0], 'x': x_geo[0, :]},  # Take first row/col for coords
                            attrs={'crs': crs.to_string()}  # Apply the CRS from the .prj file
                        )

                # Delete the original zip file after extraction
                os.remove(local_file)

# Close FTP connection
ftp.quit()
