import pandas as pd
from hdf_proces import hdf_aod_to_csv, get_coords_tile
from global_land_mask import globe
from pyhdf import SD
from glob import glob as gg
from multiprocessing import Pool
from datetime import datetime
import numpy as np
import time
import feather
import os
import re


class process_aod_dir:
    def __init__(self, path):
        self.path = path

    def make_grid_meta(self, filepath):
        """
        Function that gets all the possible coordinates within a certain tile, furthermore it determines which cells are on land or sea.
        The output is a dataframe with lat, lon and is_land (True or False). This will later be used to get the percentage of available AOD values.
        """

        hdf = SD.SD(filepath)
        coords = get_coords_tile(hdf)
        coords['is_land'] = coords.apply(lambda row: globe.is_land(row['lat'], row['lon']), axis=1)

        return coords
    
    def check_directory_tile(self, tile):
        """
        Input: string MODIS tile
        Create directories to store processed files for a certain tile
        """

        processed_path = "processed"
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
            print(f"Folder {processed_path} created")

        tile_processed_path = f"{processed_path}/{tile}"

        if not os.path.exists(tile_processed_path):
            os.makedirs(tile_processed_path)
            print(f"Folder {tile_processed_path} created")

    def check_tile_metadata(self, filepath):

        meta_path = "metadata"
        if not os.path.exists(meta_path):
            os.makedirs(meta_path)
            print(f"Folder '{meta_path}' created.")

        # Get tile string
        pattern = r'\.h(\d{2}v\d{2})\.'
        tile = re.search(pattern, filepath).group(1)

        # Check if there is a processing map with specified tile map inside
        self.check_directory_tile(tile)

        grid_meta_path = f"metadata/{tile}.feather"
        total_meta_path = f"metadata/meta_total.feather"
        
        if not os.path.isfile(grid_meta_path):
            print(f"Making meta grid file for grid: {tile}")
            coords = self.make_grid_meta(filepath)
            print(f"Done making meta grid file for grid: {tile}")
            feather.write_dataframe(coords, grid_meta_path)

            if not os.path.isfile(total_meta_path):
                meta_data = pd.DataFrame({'grid': [tile], 'length': [len(coords[coords["is_land"] == True])]})
                meta_data.to_feather(total_meta_path)
            else:
                meta_data = pd.read_feather(total_meta_path)
                row = pd.DataFrame({'grid': [tile], 'length': [len(coords[coords["is_land"] == True])]})
                meta_data = pd.concat([meta_data, row], ignore_index=True).reset_index(drop=True)
                meta_data.to_feather(total_meta_path)

            return len(coords[coords["is_land"] == True]), tile
        
        else:
            meta_data = pd.read_feather(total_meta_path)
            total_len_points = meta_data[meta_data["grid"] == tile]["length"]

        return total_len_points.iloc[0,], tile

    def process_chunk(self, filelist):
        
        count = 0
        chunk_data = pd.DataFrame()

        for file in filelist:
            count += 1
            if count % 10 == 0: print(f"{count}/{len(filelist)}")

            print(file)

            hdf = SD.SD(file)
            features=["Optical_Depth_055", "AOD_QA"]
            modis = hdf_aod_to_csv(hdf, file, features, AOD_QA=True, time=True).reset_index(drop=True)

            if len(modis) == 0:
                    continue
            else:
                modis = modis[(modis.land_type == "Land") & ((modis.cloud_mask == "Clear") | (modis.cloud_mask == "Possibly cloudy"))]
                modis["time"] = pd.to_datetime(modis['time'].dt.strftime('%Y-%m-%d'))
                modis = modis.groupby(["lat", "lon", "time"], as_index=False).agg({'Optical_Depth_055': "mean"}) 
                date = str(modis["time"].dt.strftime('%Y-%m-%d')[0]) # Save date for save name
                modis = modis.drop("time", axis=1)
                modis = modis.round(6)

                total_len_points, tile = self.check_tile_metadata(file)
                percent_coverage = np.round((len(modis) / total_len_points) * 100, 0)
                percent_coverage_int = int(percent_coverage)
                feather.write_dataframe(modis, f"processed/{tile}/{tile}_{str(date)}_{percent_coverage_int}.feather")
        
        return chunk_data.reset_index(drop=True)

    def process_aod_dir_parallel(self, num_processes=8):

        files = [f for f in gg(self.path + "/*.hdf")]
        chunks = [files[i::8] for i in range(8)]

        with Pool(processes=8) as pool:
            results = []

            for chunk in chunks:
                results.append(pool.apply_async(self.process_chunk, (chunk,)))

                # Small delay because of the creation of metadata
                time.sleep(1)
            
            for result in results:
                result.get()
    
    def split_files_weeks(self, grid):
        # Directory containing your files
        directory = f"processed/{grid}"

        # Regular expression pattern to extract dates
        date_pattern = r'\d{4}-\d{2}-\d{2}'

        files_by_week = {}

        for filename in os.listdir(directory):
            if filename.endswith(".feather"):
                date_match = re.search(date_pattern, filename)
                if date_match:
                    date_str = date_match.group(0)
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    week_number = date_obj.strftime("%U-%Y")

                    if week_number not in files_by_week:
                        files_by_week[week_number] = []

                    files_by_week[week_number].append(os.path.join(directory, filename))
        
        count = 0
        for week, file_list in files_by_week.items():
            if count != 2 :
                count += 1
                continue

            print(file_list)
            merged_data = None

            for file_path in file_list:
                count += 1
                print(f"{count}/{len(file_list)}")
                data = pd.read_feather(file_path)
                print(len(data))
                if merged_data is None:
                    merged_data = data
                else:
                    merged_data = pd.concat([merged_data, data])

            # For days minimum and take weakly mean per cel
            filtered_df = merged_data.groupby(['lat', 'lon']).filter(lambda x: len(x) > 1)
            result_df = filtered_df.groupby(['lat', 'lon'], as_index=False)['Optical_Depth_055'].mean()

            break
        
        return result_df.reset_index(drop=True)

if __name__ == "__main__":
    # Call the parallelized function with the desired number of processes
    proc = process_aod_dir(path="C:/Users/MarjolijnStam/Desktop/test_folder/unprocessed/h18v04")
    result = proc.split_files_weeks(grid="18v04")
    print(result)

    # import matplotlib.pyplot as plt
    # plt.scatter(x=result["lat"], y=result["lon"], c=result["Optical_Depth_055"])
    # plt.show()
    # proc.process_aod_dir_parallel()

