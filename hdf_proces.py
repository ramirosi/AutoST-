from pyhdf import SD
import pyproj
from pyproj import CRS, Proj
import numpy as np
import pandas as pd
import geopandas as gpd
from pyhdf.SD import SD, SDC, SDS
import random
from typing import Dict, List, Union
#Functie om de data te calibreren
#Ongeldige metingen worden veranderd in NA's
def calibrate_data(dataset: SDS, shape: List[int], calibration_dict: Dict, AOD_QA):
    #Lege array maken met de vorm van de dataset v/d bijbehorende attribuut
    corrected_AOD = np.ma.empty(shape, dtype=np.double)
    #Voor elke orbit zijn er metingen op elke locatie
    for orbit in range(shape[0]):
        data = dataset[orbit, :, :].astype(np.double)
        #Een meting is ongeldig mits de waarde gelijk is aan de _FillValue
        invalid_condition = (
            (data == calibration_dict["_FillValue"])
        )
        #De ongeldige waarnemingen worden NA's
        data[invalid_condition] = np.nan
        #De waardes zijn geschaald om opslag groote te verminderen
        #De orginele AOD metingen kunnen hersteld worden m.b.v de add_offset en de scale_factor
        if AOD_QA == 0:
          data = (
              (data - calibration_dict["add_offset"]) *
              calibration_dict["scale_factor"]
              )
        data = np.ma.masked_array(data, np.isnan(data))
        #Per orbit wordt de data toegevoegt aan corrected_AOD
        corrected_AOD[orbit, : :] = data
    corrected_AOD.fill_value = np.nan
    return corrected_AOD

#Met behulp van de allignment_dict wordt een meshgrid gemaakt
def create_meshgrid(alignment_dict: Dict, shape: List[int]):

    # Determine grid bounds using two coordinates
    x0, y0 = alignment_dict["upper_left"]
    x1, y1 = alignment_dict["lower_right"]
    
    # Interpolate points between corners, inclusive of bounds
    x = np.linspace(x0, x1, shape[2], endpoint=True)
    y = np.linspace(y0, y1, shape[1], endpoint=True)
    
    # Return two 2D arrays representing X & Y coordinates of all points
    xv, yv = np.meshgrid(x, y)
    return xv, yv

#Omzetten van coordinaten projectie
def transform_arrays(
    xv: Union[np.array, float],
    yv: Union[np.array, float],
    crs_from: CRS,
    crs_to: CRS
):
    transformer = pyproj.Transformer.from_crs(
        crs_from,
        crs_to,
        always_xy=True,
    )
    lon, lat = transformer.transform(xv, yv)
    return lon, lat

#Een geodataframe maken
def convert_array_to_df(
    calibratie_samen: list,
    variables: list,
    lat:np.ndarray,
    lon: np.ndarray,
    granule_id: str,
    crs: CRS,
    total_bounds: np.ndarray = None
):
    lats = lat.ravel()
    lons = lon.ravel()
    #n_orbits is hetzelfde voor elke feature, dus vast op eerste element van de lijst
    n_orbits = len(calibratie_samen[0])
    size = lats.size
    values = {
        "lat": np.tile(lats, n_orbits),
        "lon": np.tile(lons, n_orbits),
        "orbit": np.arange(n_orbits).repeat(size),
        "granule_id": [granule_id] * size * n_orbits
    }
    #for i in range(len(calibratie_samen[0])):
    for i in range(len(variables)):
        values[variables[i]] = np.concatenate([d.data.ravel() for d in calibratie_samen[i]])
    
    df = pd.DataFrame(values)
    df = df[pd.notnull(df['Optical_Depth_055'])]

    return df

#Uitleg werking QA bitmask link hieronder
##https://spatialthoughts.com/2021/08/19/qa-bands-bitmasks-gee/##

def decrypt_QA_MCD19A2_manier2(gdf):
  cloud_mask_dict = {"000": "Undefined",
                     "001": "Clear",
                     "010": "Possibly cloudy",
                     "011": "Cloudy",
                     "101": "Cloud shadow",
                     "110": "Fire hot spot",
                     "111": "Water sediments"
                     }
  land_type_dict = {"00": "Land",
                    "01": "Water",
                    "10": "Snow",
                    "11": "Ice"
                    }
  adjacency_mask_dict = {"000": "Normal condition/Clear",
                         "001": "Adjacent to clouds",
                         "010": "Surrounded by more than 8 cloudy pixels",
                         "011": "Adjacent to a single cloudy pixel",
                         "100": "Adjacent to snow",
                         "101": "Snow was previously detected in this pixel"
                    }
  qa_for_aod_dict = {"0000": "Best quality",
                     "0010": "NA",
                     "0001": "Water Sediments are detected (water)",
                     "0011": "There is 1 neighbor cloud",
                     "0100": "There is >1 neighbor clouds",
                     "0101": "No retrieval (cloudy, or whatever)",
                     "0110": "No retrievals near detected or previously detected snow",
                     "0111": "Climatology AOD: altitude above 3.5km (water) and 4.2km (land)",
                     "1000": "No retrieval due to sun glint (water)",
                     "1001": "Retrieved AOD is very low (<0.05) due to glint (water)",
                     "1010": "AOD within +-2km from the coastline (may be unreliable)",
                     "1011": "Land, research quality: AOD retrieved but CM is possibly cloudy",
                     }
  
  # Selecteer de kwaliteits feature uit het gdf
  x = np.array(gdf["AOD_QA"]).astype(int)
  
  # Functie omzetten int naar binair getal (16bit)
  def int_to_bin(x):
    return np.binary_repr(x, width=16)
    
  # Functie moet voor elk element in np.array individueel uitgevoerd worden
  int_to_bin_v = np.vectorize(int_to_bin)
  x_new = int_to_bin_v(x)
  
  # Lege np arrays maken om verschillende QA variabelen op te slaan
  cloud_mask = np.empty((len(x_new),), dtype=object)
  land_type = np.empty((len(x_new),), dtype=object)
  adjacency_mask = np.empty((len(x_new),), dtype=object)
  qa_for_aod = np.empty((len(x_new),), dtype=object)
  
  # Binare code ontleden en terugkoppelen aan de dictionaires die hierboven zijn gemaakt
  # Om te bepalen welke categorie hoort bij dat deel van het binaire getal
  for i in range(len(x_new)):
    cloud_mask[i] = cloud_mask_dict[x_new[i][-3:]]
    land_type[i] = land_type_dict[x_new[i][-5:-3]]
    adjacency_mask[i] = adjacency_mask_dict[x_new[i][-8:-5]]
    qa_for_aod[i] = qa_for_aod_dict[x_new[i][-12:-8]]

  # De kwaliteits features toevoegen aan het orginele dataframe  
  gdf["cloud_mask"] = cloud_mask
  gdf["land_type"] = land_type
  gdf["adjacency_mask"] = adjacency_mask
  gdf["qa_for_aod"] = qa_for_aod
  return gdf

# Functie om julianday om te zetten naar maand en dag
def JulianDate_to_MMDDYYY(y,jd):
  import calendar
  month = 1
  day = 0
  while jd - calendar.monthrange(y,month)[1] > 0 and month <= 12:
      jd = jd - calendar.monthrange(y,month)[1]
      month = month + 1
  return month, jd, y

def get_time_orbits_MCD19A2(gdf, orbit_time):

  import datetime
  
  orbit_datums = np.empty((len(orbit_time),), dtype=object)

  for i in range(len(orbit_time)):
    time_str = orbit_time[i]

    year = time_str[:4]
    julian_day = time_str[4:7]
    hour = time_str[7:9]
    min = time_str[9:11]
    
    maand, dag, jaar = JulianDate_to_MMDDYYY(int(year), int(julian_day))
    orbit_datums[i] = datetime.datetime(jaar, maand, dag, int(hour), int(min))
  
  orbit = np.array(gdf["orbit"])

  orbit_date_vector = np.empty((len(orbit),), dtype=object)
  
  for i in range(len(orbit)):
    orbit_date_vector[i] = orbit_datums[orbit[i]-1]

  gdf["time"] = orbit_date_vector
  return gdf

# De shape van de attributen binnen de HDF zijn hetzelfde
# Dat betekent dat het aantal orbits voor elke attributen hetzelfde zijn
# De meshgrid en de transformatie van coordinaten is hetzelfde voor elke attribuut binnen een hdf
# De calibratie van een attribuut moet individueel uitgevoerd worden.

# Function to transform MODIS MAIAC AOD hdf file into a csv

def get_coords_tile(hdf):
  
  variable = hdf.select("Optical_Depth_055")
  shape = variable.info()[2]

  raw_attr = hdf.attributes()["StructMetadata.0"]
  group_1 = raw_attr.split("END_GROUP=GRID_1")[0]
  hdf_metadata = dict([x.split("=") for x in group_1.split() if "=" in x])

  #Parse metadata
  for key, val in hdf_metadata.items():
      try:
          hdf_metadata[key] = eval(val)
      except:
          pass
  #Nu de metadata geparsed is kunnen we de attributen die we nodig hebben om de data te allignen en repojecten.
  # Note that coordinates are provided in meters
  alignment_dict = {
      "upper_left": hdf_metadata["UpperLeftPointMtrs"],
      "lower_right": hdf_metadata["LowerRightMtrs"],
      "crs": hdf_metadata["Projection"],
      "crs_params": hdf_metadata["ProjParams"]
      }

  #Meshgrid maken
  xv, yv = create_meshgrid(alignment_dict, shape)
  #Omzetten van sinusoidal projection naar normale WGS84 projectie
  sinu_crs = Proj(f"+proj=sinu +R={alignment_dict['crs_params'][0]} +nadgrids=@null +wktext").crs
  wgs84_crs = CRS.from_epsg("4326")
  lon, lat = transform_arrays(xv, yv, sinu_crs, wgs84_crs)
  lats = lat.ravel()
  lons = lon.ravel()

  values = {
      "lat": np.tile(lats, 1),
      "lon": np.tile(lons, 1),
  }

  return pd.DataFrame(values)  

def hdf_aod_to_csv(hdf, file, variables, AOD_QA, time):
    #Load the hdf file

    ### FOR-LOOP ###
    calibratie_samen = []
    for i in range(len(variables)):

      #Meegeven of het om de QA_AOD feature gaat
      #Deze heeft namelijk geen add_offset / scale_factor dus hoeft niet getransformeerd te worden.
      if variables[i] == "AOD_QA":
        AOD_QA = 1
      else:
        AOD_QA = 0
      
      variable = hdf.select(variables[i])
      shape = variable.info()[2]
      calibration_dict = variable.attributes()
      #Data calibreren
      globals()['corrected_AOD'+str(i)] = calibrate_data(variable, shape, calibration_dict, AOD_QA)
      calibratie_samen.append(globals()['corrected_AOD'+str(i)])
    ###############

    #Opslaan van StructMetadata.0 hieraan staat informatie die later gebruikt wordt bij het omzetten van coordinaten.
    raw_attr = hdf.attributes()["StructMetadata.0"]
    #We kiezen de 1km resolutie, niet de 5km resolutie
    group_1 = raw_attr.split("END_GROUP=GRID_1")[0]
    #Ordenen
    hdf_metadata = dict([x.split("=") for x in group_1.split() if "=" in x])
    #Parse metadata
    for key, val in hdf_metadata.items():
      try:
        hdf_metadata[key] = eval(val)
      except:
        pass
    #Nu de metadata geparsed is kunnen we de attributen die we nodig hebben om de data te allignen en repojecten.
    # Note that coordinates are provided in meters
    alignment_dict = {
        "upper_left": hdf_metadata["UpperLeftPointMtrs"],
        "lower_right": hdf_metadata["LowerRightMtrs"],
        "crs": hdf_metadata["Projection"],
        "crs_params": hdf_metadata["ProjParams"]
        }
    
    #Meshgrid maken
    xv, yv = create_meshgrid(alignment_dict, shape)
    #Omzetten van sinusoidal projection naar normale WGS84 projectie
    sinu_crs = Proj(f"+proj=sinu +R={alignment_dict['crs_params'][0]} +nadgrids=@null +wktext").crs
    wgs84_crs = CRS.from_epsg("4326")
    lon, lat = transform_arrays(xv, yv, sinu_crs, wgs84_crs)

    #Omzetten naar geodataframe
    gdf = convert_array_to_df(calibratie_samen, variables, lat, lon, file, wgs84_crs)

    if len(gdf) == 0:
      return gdf

    #AOD_QA omzetten
    if AOD_QA == True:
      gdf = decrypt_QA_MCD19A2_manier2(gdf)
    
    #Tijden per orbit toevoegen
    if time == True:
      orbit_time = str.split(hdf.attributes()["Orbit_time_stamp"])
      gdf = get_time_orbits_MCD19A2(gdf, orbit_time)

    return gdf.reset_index(drop=True)