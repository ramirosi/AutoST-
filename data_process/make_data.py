from rasterio.transform import from_origin
from datetime import datetime, timedelta
from pyproj import CRS, Transformer
from tqdm import tqdm
import xarray as xr
import pandas as pd
import rioxarray
import glob
import numpy as np


def retrieve_index_point_tiff(img, lat, lon):
    # Define the CRS for WGS84 and EPSG:3035
    wgs84 = CRS.from_epsg(4326)
    epsg3035 = CRS.from_epsg(3035)
    
    # Transform the coordinates to EPSG:3035
    transformer = Transformer.from_crs(wgs84, epsg3035)
    y, x = transformer.transform(lat, lon)

    ds = img.sel(x=x, y=y, method="nearest")

    ind_x, ind_y = float(ds.x), float(ds.y)

    index_x = np.where(img.x == ind_x)[0][0]
    index_y = np.where(img.y == ind_y)[0][0]

    return index_x, index_y

def retrieve_image(img, index_x, index_y, grid_size):
    
    radius = int((grid_size-1)/2)

    min_index_x = index_x - radius
    max_index_x = index_x + radius + 1
    min_index_y = index_y - radius
    max_index_y = index_y + radius + 1

    # Select the grid
    sliced_img = img[min_index_y:max_index_y, min_index_x:max_index_x]

    # Fill na's as 0
    fill_value = 65535 
    sliced_img = xr.where(sliced_img == fill_value, 0, sliced_img)

    # Get numpy array and rescale values back
    numpy_array = sliced_img.values*0.001

    return numpy_array

def process_groundstations():
    # Read grounstation locations:
    locations = pd.read_csv("data/groundstations/pm25_locations.csv")

    # Bounding box for the Netherlands
    netherlands_bbox = {
        'min_lat': 50.75,
        'max_lat': 53.5,
        'min_lon': 3.2,
        'max_lon': 7.2
    }
    
    # Filter coordinates within the bounding box
    loc_df = locations[
        (locations['latitude'] >= netherlands_bbox['min_lat']) &
        (locations['latitude'] <= netherlands_bbox['max_lat']) &
        (locations['longitude'] >= netherlands_bbox['min_lon']) &
        (locations['longitude'] <= netherlands_bbox['max_lon'])
    ].reset_index(drop=True)
    loc_df.columns = ["Latitude", "Longitude"]

    # Read grounstation measurement:
    measurements = pd.read_csv("data/groundstations/pm25_europe.csv")

    # Merge the measurements DataFrame with the filtered_df based on latitude and longitude
    merged_df = pd.merge(measurements, loc_df, on=['Latitude', 'Longitude'], how='inner')

    merged_df = merged_df[(merged_df["average"] >= "2019-01-01") & (merged_df["average"] < "2021-01-01")].reset_index(drop=True)
    merged_df = merged_df.sort_values(by='type')  # Sort by 'type' to prioritize 'daily' over 'hourly'
    merged_df = merged_df.drop_duplicates(subset=['average', 'Longitude', 'Latitude'], keep='first').reset_index(drop=True).drop(["type", "AveragingTime"], axis=1).reset_index(drop=True)

    # Create grouped_df with row counts
    grouped_df = merged_df.groupby(['Latitude', 'Longitude']).size().reset_index(name='row_count')

    # Merge the row count information back into merged_df
    merged_df = pd.merge(merged_df, grouped_df, on=['Latitude', 'Longitude'], how='left')

    # Filter rows with row_count greater than 500
    filtered_df = merged_df[merged_df['row_count'] > 700].reset_index(drop=True)

    # Drop the additional row_count column if needed
    filtered_df = filtered_df.drop(columns=['row_count'])

    return loc_df, filtered_df

def make_time_series(directory, grid_size=45):

    loc_df, filtered_df = process_groundstations()

    tiff_files = glob.glob(f"{directory}/*.tif")
    xarr = rioxarray.open_rasterio(tiff_files[0])
    img = xarr[0, :, :]

    # Pre calculate the indices for all points, to use them in later tiff files.
    indices = [retrieve_index_point_tiff(img, lat, lon) for lat, lon in zip(loc_df['Latitude'], loc_df['Longitude'])]


    start_date = datetime.strptime('2019-01-01', '%Y-%m-%d')
    end_date = datetime.strptime('2020-12-31', '%Y-%m-%d')
    date_strings = [(start_date + timedelta(days=day)).strftime('%Y-%m-%d') for day in range((end_date - start_date).days + 1)]

    num_days = len(date_strings)
    num_locations = len(loc_df)

    time_series = np.zeros((num_locations, num_days, grid_size, grid_size))
    target = np.full((num_locations,num_days), np.nan)  # Initialize with NaN values


    for day, current_date_str in tqdm(enumerate(date_strings), desc="Processing days"):

        # If the current date exceeds '2020-12-31', break the loop
        if datetime.strptime(current_date_str, '%Y-%m-%d') > end_date:
            break

        xarr = rioxarray.open_rasterio(tiff_files[day])
        img = xarr[0, :, :]

        # Subset the measurement for a certain day
        day_subset = filtered_df[filtered_df["average"] == current_date_str]

        for loc_idx, (row, (ind_x, ind_y)) in enumerate(zip(loc_df.itertuples(index=False), indices)):

            lat = row.Latitude
            lon = row.Longitude

            # Use boolean indexing to filter day_subset for the specific location
            loc_subset = day_subset[(day_subset["Latitude"] == lat) & (day_subset["Longitude"] == lon)]

            if not loc_subset.empty:
                target[loc_idx, day] = float(loc_subset["Concentration"])

            daily_data = retrieve_image(img, ind_x, ind_y, grid_size)
            time_series[loc_idx, day, :, :] = daily_data

    return time_series, target

if __name__ == "__main__":
    directory_tif = "C:/Users/rxkro/Desktop/st-darts/data/GHaod_2019_v1"
    grid_size = 45

    time_series, target = make_time_series(directory_tif, grid_size)