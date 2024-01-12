import pandas as pd
from scipy.spatial import cKDTree
import os
import re
from tqdm import tqdm

class get_aod():
    def __init__(self, grid, points_path):
        self.grid = grid
        self.points_path = points_path

        self.grid_points = None
        self.locations = None
        self.closest_points = None

    def load_data(self):
        # Load grid points
        df = pd.read_feather(f"metadata/{self.grid}.feather")
        self.grid_points = df[df.is_land == True].drop("is_land", axis=1).reset_index(drop=True)
        # Load needed points
        self.locations = pd.read_csv(self.points_path)

    def find_closest_points(self):
        # Determine the bounds of the grid DataFrame
        min_lat, max_lat = self.grid_points['lat'].min(), self.grid_points['lat'].max()
        min_lon, max_lon = self.grid_points['lon'].min(), self.grid_points['lon'].max()

        # Filter points to be inside the bounds of the grid
        points = self.locations[(self.locations['latitude'] >= min_lat) & (self.locations['latitude'] <= max_lat) &
                        (self.locations['longitude'] >= min_lon) & (self.locations['longitude'] <= max_lon)].reset_index(drop=True)
        points.columns = ["point_latitude", "point_longitude"]

        # Create a KDTree from the filtered grid DataFrame
        grid_kdtree = cKDTree(self.grid_points[['lat', 'lon']])

        # Query the KDTree for the closest points in grid for each point in filtered points
        _, indices = grid_kdtree.query(points[['point_latitude', 'point_longitude']], k=1)

        # Retrieve the closest points from the grid DataFrame
        closest_points = self.grid_points.iloc[indices].reset_index(drop=True)
        closest_points.columns = ["grid_latitude", "grid_longitude"]

        closest_points = pd.concat([points, closest_points], axis=1)
        self.closest_points = closest_points.round({'grid_latitude': 4, 'grid_longitude': 4})

        return
    
    def process_dir(self):

        self.load_data()
        self.find_closest_points()
        
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        directory = f"processed/{self.grid}"
        dfs_to_concat = []  # List to store DataFrames for concatenation

        for filename in tqdm(os.listdir(directory), desc="Processing files", unit="file"):
            if filename.endswith(".feather"):
                values = pd.read_feather(os.path.join(directory, filename))
                values = values.round({'lat': 4, 'lon': 4})

                merged_df = pd.merge(values, self.closest_points, left_on=['lat', 'lon'], right_on=['grid_latitude', 'grid_longitude'], how='inner')
                merged_df = merged_df.drop(['grid_latitude', 'grid_longitude'], axis=1).drop(["lat", "lon"], axis=1)
                merged_df.columns = ["Optical_Depth_055", "latitude", "longitude"]
                
                date_match = re.search(date_pattern, filename).group(0)
                merged_df["date"] = date_match

                # Append to the list for concatenation
                dfs_to_concat.append(merged_df)

        # Concatenate all DataFrames in the list
        total_df = pd.concat(dfs_to_concat, ignore_index=True)

        return total_df


if __name__ == "__main__":
    aod = get_aod("18v04", "sampled.csv")
    total_df = aod.process_dir()
    total_df.to_csv("C:/Users/MarjolijnStam/Desktop/test_folder/18v04_total_2019_2023_amstelveen.csv", index=False)
