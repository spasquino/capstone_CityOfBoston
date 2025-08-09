# Packages
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
import re
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
tqdm.pandas()
from sklearn.preprocessing import StandardScaler 
from datetime import datetime
from get_data_db import get_data_db

# Declaring filepath and shared variables

# Data path
DATA_PATH = r"..\data"

# Coordinate reference systems
CRS_2249 = "EPSG:2249"
CRS_4326 = "EPSG:4326"  # lat-lon
CRS_32619 = "EPSG:32619"  # projected for northern hemisphere utm zone 19

# Census block group geometries, pulled directly from Civis using get_data_db helper function.

df_block_groups = get_data_db(
    query='census_block_groups.sql',
    col_names = ['gid','objectid','statefp20','countyfp20','tractce20','blkgrpce20','geoid20','namelsad20','mtfcc20','funcstat20','aland20','awater20','intptlat20','intptlon20','shape_star','shape_stle','geom','st_astext'],
    save_file=False
)

df_block_groups.insert(0, "geometry", df_block_groups['st_astext'].apply(wkt.loads))
df_block_groups = gpd.GeoDataFrame(df_block_groups, geometry='geometry', crs=CRS_2249)
df_block_groups.to_crs(CRS_4326, inplace=True)
df_block_groups_proj = df_block_groups.to_crs(CRS_32619)

# Helper function to help merge different geometries
def assign_id_on_largest_intersection(df1, df2, id1, id2):
    '''
    Assign to all entries of df1 an id in df2, where the assigned id shares the largest intersection with the entry
    IN:
    - df1: dataframe of objects to assign an id to
    - df2: dataframe of objects that have an id
    - id_col1: id column name for df1
    - id_col2: id column name for df2
    OUT:
    - df1 with new id2 column added
    '''
    
    # If id2 is already in df1 (or vice versa), return error
    if id2 in df1.columns:
        raise ValueError("ID column of df2 already exists in df1. Please rename id2.")
    if id1 in df2.columns:
        raise ValueError("ID column of df1 already exists in df2. Please rename id1.")

    # Get intersections between df1 and df2
    df_overlay = gpd.overlay(df1, df2, keep_geom_type=False)
    
    # Get area of intersections
    df_overlay['intersection_area'] = df_overlay.geometry.to_crs(CRS_32619).area
    
    # For each id1, get id2 of largest intersection
    df1_id2 = df_overlay.groupby(id1).apply(lambda group: group.loc[group.intersection_area.idxmax(), id2], include_groups=False).reset_index(name=id2)
    
    # Merge back to df1
    return df1.merge(df1_id2[[id1, id2]], on=id1, how='left')

# Data
df_open = gpd.read_file(DATA_PATH + r"\sidewalk_park_waste\PRK_Open_Space.shp")
df_open.to_crs(CRS_4326, inplace=True)
df_open_proj = df_open.to_crs(CRS_32619)

# Park area within block group
# Get areas of park-block group intersections
df_block_park_intersections = gpd.overlay(df_block_groups, df_open[['geometry']], keep_geom_type=False)
df_block_park_intersections['park_area'] = df_block_park_intersections.to_crs(CRS_32619).geometry.area

df_block_park_intersections.head(3)

# Get total park area per block group
df_park_areas = df_block_park_intersections.groupby('gid').park_area.sum().reset_index()

# Merge to df_block_groups
df_block_groups = df_block_groups.merge(df_park_areas, on='gid', how='left')
df_block_groups['park_area'] = df_block_groups['park_area'].fillna(0)
# df_block_groups['park_area'] = df_block_groups['park_area'] / df_block_groups_proj.geometry.area  # Scale by area of block

# Distance to nearest park (0 if there is a park within the block group)
df_block_groups_proj = df_block_groups.to_crs(CRS_32619)
park_dists = df_block_groups.progress_apply(lambda row: 0 if row.park_area>0 else min(df_open_proj.distance(row.geometry)), axis=1)  # 0 if has park area, else minimum distance to nearest park
df_block_groups['park_dist'] = park_dists

# Data
# Waste
df_waste = gpd.read_file(DATA_PATH + r"\sidewalk_park_waste\WasteReceptacles.shp")
df_waste.to_crs(CRS_4326, inplace=True)
df_waste_proj = df_waste.to_crs(CRS_32619)

# Neighborhoods
df_nbr = get_data_db(
    query='neighborhood_data.sql',
    col_names = ['OBJECTID','name','acres','neighborhood_id','sqmiles','geom_multipolygon_2249','geom_multipolygon_4326','_ingest_datetime','st_astext'],
    save_file=False
)

df_nbr.insert(0, "geometry", df_nbr['st_astext'].apply(wkt.loads))
df_nbr = gpd.GeoDataFrame(df_nbr, geometry='geometry', crs=CRS_2249)
df_nbr.to_crs(CRS_4326, inplace=True)
# Fix invalid geometries
df_nbr['geometry'] = df_nbr['geometry'].buffer(0)

# Determine which block groups have missing data, according to which neighborhood they lie in

# Find names of neighborhoods with 0 receptacles
nbr_no_intersections = [not any(df_waste.intersects(row.geometry)) for i, row in df_nbr.iterrows()]  # True if neighborhood has no intersections with waste receptacle data
nbrs_missing = list(df_nbr.name[nbr_no_intersections])  # Get names of missing neighborhoods
nbrs_missing += ['Hyde Park', 'South Boston', 'West Roxbury']  # Add Hyde Park, South Boston, West Roxbury (not enough data)

# Assign each block group to a neighborhood
df_block_groups = assign_id_on_largest_intersection(df_block_groups, df_nbr, 'gid', 'name')

# df_block_groups[pd.isna(df_block_groups.Name)]
# # Drop block groups without a neighborhood
# df_block_groups = df_block_groups[~pd.isna(df_block_groups.Name)].reset_index(drop=True)
df_block_groups_proj = df_block_groups.to_crs(CRS_32619)

# Determine if block group is missing waste data (based on neighborhood)
block_group_missing = df_block_groups.name.apply(lambda n: n in nbrs_missing)
print(f"Data missing for {round(np.mean(block_group_missing)*100, 2)}% of census blocks")

# Get number of waste receptacles in each block group

# Number of intersections with waste receptacle locations
df_n_waste = df_block_groups[['geometry', 'gid']].sjoin(df_waste[['geometry']]).groupby('gid').size().reset_index(name='n_waste_bins')
df_block_groups = df_block_groups.merge(df_n_waste, on='gid', how='left')
df_block_groups['n_waste_bins'] = df_block_groups['n_waste_bins'].fillna(0)  # 0 if no intersections
df_block_groups['n_waste_bins'] = [None if block_group_missing[i] else df_block_groups.n_waste_bins[i] for i in range(df_block_groups.shape[0])]  # None if block is missing

# Get distance to nearest waste receptacle
df_block_groups_proj = df_block_groups.to_crs(CRS_32619)
waste_dists = df_block_groups_proj.progress_apply(lambda row: 0 if row.n_waste_bins>0 else min(df_waste_proj.distance(row.geometry)), axis=1)
df_block_groups['waste_dist'] = waste_dists
df_block_groups['waste_dist'] = [None if block_group_missing[i] else df_block_groups.waste_dist[i] for i in range(df_block_groups.shape[0])]  # None if block group is missing

# df_census_block_group_population: Extract of total population and housing units by census block group, from the 2020 census. 
# Collected from https://mcdc.missouri.edu/

df_census_block_group_population = pd.read_csv(
    DATA_PATH + r"\raw\dexter_2427100782_extract.csv",
    header=1,
    dtype={'Geographic Code Identifier ':str,
           'Total population ':int,
           'Total housing units':int},
    thousands=','
    )

df_block_groups = df_block_groups.merge(df_census_block_group_population[['Geographic Code Identifier ', 'Total population ','Total housing units']], left_on='geoid20', right_on='Geographic Code Identifier ', how='left')

# Get population and building density (# objects / land area per census block group)
df_block_groups["pop_density"] = df_block_groups['Total population '] / df_block_groups.aland20
df_block_groups["bldg_density"] = df_block_groups['Total housing units']  / df_block_groups.aland20

##########################################################################################################################################################
# Read data
df_parcel_raw = gpd.read_file(DATA_PATH + r"\feature_engineering_inputs\parcel_data\parcel_data.shp")
df_parcel_raw.to_crs(CRS_4326, inplace=True)


##########################################################################################################################################################
# Subset to relevant columns & rows

# Remove rows that are condo main (because all condo mains are made up of RESIDENTIAL CONDOs)
df_parcel = df_parcel_raw[df_parcel_raw.LU_DESC != "CONDO MAIN"].reset_index(drop=True)

df_parcel = df_parcel[["PID", "NUM_BLDGS", "LU", "RES_FLOOR", "GROSS_AREA", "LAND_SF", "BLDG_VALUE", "TOTAL_VALU", "YR_BUILT", "YR_REMODEL", "STRUCTURE_", "INT_WALL", 
           "EXT_FNISHE", "OVERALL_CO", "HEAT_TYPE", "geometry"]]

df_parcel.columns = ["PID", "NUM_BLDGS", "LU", "RES_FLOOR", "GROSS_AREA", "LAND_SF", "BLDG_VALUE", "TOTAL_VALUE", "YR_BUILT", "YR_REMODEL", "STRUCTURE_CLASS", "INT_WALL", 
           "EXT_FINISHED", "OVERALL_COND", "HEAT_TYPE", "geometry"]


##########################################################################################################################################################
# Clean and create columns

# Get parcel area
df_parcel["AREA"] = df_parcel.to_crs(CRS_32619).geometry.area

# BLDG_VALUE
# Make numeric
df_parcel.BLDG_VALUE = pd.to_numeric([re.sub(r"[^\d]", "", x) for x in df_parcel.BLDG_VALUE])

# YR_BUILT
# Fix typos
idx_built = np.where(df_parcel.YR_BUILT > 2024)[0]
df_parcel.loc[idx_built[0], "YR_BUILT"] = 2019

# YR_REMODEL
# Fix typos
idx_remodel_big = np.where(df_parcel.YR_REMODEL > 2024)[0]  # After 2024
df_parcel.loc[idx_remodel_big[0], "YR_REMODEL"] = 2021
df_parcel.loc[idx_remodel_big[1], "YR_REMODEL"] = 2021
df_parcel.loc[idx_remodel_big[2], "YR_REMODEL"] = 2022

idx_remodel_small = np.where((df_parcel.YR_REMODEL <1700) & (df_parcel.YR_REMODEL > 0))[0]  # Before 1700 (first house built)
df_parcel.loc[idx_remodel_small[0], "YR_REMODEL"] = 2010
df_parcel.loc[idx_remodel_small[1], "YR_REMODEL"] = 2021
df_parcel.loc[idx_remodel_small[2], "YR_REMODEL"] = 2010

# OVERALL_COND
# Convert some types to standard format
df_parcel.OVERALL_COND = [None if c is None else re.sub("EX - Excellent", "E - Excellent", c) for c in df_parcel.OVERALL_COND]
df_parcel.OVERALL_COND = [None if c is None else re.sub("AVG - Default - Average", "A - Average", c) for c in df_parcel.OVERALL_COND]

# Make Boston College building value None (to remove outlier)
idx = df_parcel.index[df_parcel.PID=='2102473001'][0]
df_parcel.loc[idx, 'BLDG_VALUE'] = None

# LU (Land Use): Sorting into fewer categories

# Group into smaller categories (residential, commercial, r/c, etc.)

mapping = {'A': 'R',    # R - Residential
            'R1': 'R', 
            'R2': 'R',
            'R3': 'R',
            'R4': 'R', 
            'RL - RL': 'R',
            'CD': 'CO',  # CO - Condos
            'CM': 'CO',
            'CP': 'CO',
            'C': 'C',    # C - Commercial
            'CC': 'C',
            'CL': 'C',
            'RC': 'RC',  # RC - Mixed residential/commercial
            'I': 'I',    # I - Industrial
            'AH': 'AH',  # AH - Agricultural
            'E': 'E',    # E - Tax Exempt
            'EA': 'E'}

# Assign new categories
LU_new = df_parcel['LU'].replace(mapping)
df_parcel['LU_new'] = LU_new
# One-hot-encode
one_hot_encoded = pd.get_dummies(df_parcel['LU_new'], prefix='LU')
# Add to dataframe
df_parcel = pd.concat([df_parcel, one_hot_encoded], axis=1)

# BLDG_VALUE (Building Value) (Lots of 0's?)

# Number of 0
print(f"Number of 0: {sum(df_parcel.BLDG_VALUE==0)} ({sum(df_parcel.BLDG_VALUE==0)/df_parcel.shape[0]})")

# Replace 0 with None
df_parcel.loc[:, 'BLDG_VALUE'] = [None if x==0 else x for x in df_parcel.BLDG_VALUE]

# Description
print("\nSummary:")
df_parcel.BLDG_VALUE.describe()

# YR_BUILT, YR_REMODEL: Calculating age and years since remodeled

# Calculate age
ages = [None if yr==0 else 2024-yr for yr in df_parcel.YR_BUILT]
df_parcel["AGE"] = ages

# Calculate age since remodel (taking YR_BUILT as YR_REMODEL if YR_REMODEL=0)
yr_remodel = df_parcel.YR_REMODEL
yr_built = df_parcel.YR_BUILT

yr_remodel = [yr_built[i] if yr_remodel[i]==0 else yr_remodel[i] for i in range(len(yr_remodel))]
yrs_since_remodel = [None if yr==0 else 2024-yr for yr in yr_remodel]
df_parcel["YRS_SINCE_REMODEL"] = yrs_since_remodel

# EXT_FINISHED (Exterior siding material): Sorting into fewer categories
# Using EXT_FINISHED rather than STRUCTURE_CLASS since STRUCTURE_CLASS is mostly None

# Group into smaller categories (stone, metal, wood, etc.)

# Define groupings
mapping = {'01 - Brick': 'BRICK_STONE',
            '02 - Stone': 'BRICK_STONE',
            '03 - Poured Concr': 'BRICK_STONE',
            '04 - Precast Concr': 'BRICK_STONE',
            '05 - Concr & Glass': 'BRICK_STONE',
            '06 - Metal/Glass': 'METAL_GLASS',
            '07 - Stone/Marble': 'BRICK_STONE',
            '08 - Stucco': 'BRICK_STONE',
            '09 - Wood Siding': 'WOOD',
            '10 - Alum/Vinyl': 'VINYL',
            '11 - Metal Siding': 'METAL_GLASS',
            '12 - Conc Block': 'BRICK_STONE',
            '13 - Br Sill/Sash': 'BRICK_STONE',
            '14 - Hollow Tile': 'METAL_GLASS',
            '15 - Corrug Siding': 'METAL_GLASS',
            'A - Asbestos': 'ASBESTOS',
            'B - Brick/Stone': 'BRICK_STONE',
            'C - Cement Board': 'BRICK_STONE',
            'F - Frame/Clapbrd': 'WOOD',
            'G - Glass': 'METAL_GLASS',
            'K - Concrete': 'BRICK_STONE',
            'M - Vinyl': 'VINYL',
            'O - Other': 'OTHER',
            'P - Asphalt': 'BRICK_STONE',
            'S - Stucco': 'BRICK_STONE',
            'U - Alum Siding': 'METAL_GLASS',
            'V - Brck/Stn Venr': 'BRICK_STONE',
            'W - Wood Shake': 'WOOD'
    
}
# Assign new categories
df_parcel['EXT_new'] = df_parcel['EXT_FINISHED'].replace(mapping)
# One-hot-encode
one_hot_encoded = pd.get_dummies(df_parcel['EXT_new'], prefix='EXT')
# Add to dataframe
df_parcel = pd.concat([df_parcel, one_hot_encoded], axis=1)

# OVERALL_COND (Overall parcel condition): Cleaning, making scores from condition strings

mapping = {
    'US - Unsound': 0,
    'VP - Very Poor': 1,
    'P - Poor': 2,
    'F - Fair': 3,
    'A - Average': 4,
    'G - Good': 5,
    'VG - Very Good': 6,
    'E - Excellent': 7
}

df_parcel['OVERALL_COND_SCORE'] = df_parcel.OVERALL_COND.replace(mapping)

# Subset df_parcel to relevant columns

df_parcel = df_parcel.loc[:, [
    'PID',
    'NUM_BLDGS',
    'BLDG_VALUE',
    'AGE',
    'YRS_SINCE_REMODEL',
    'OVERALL_COND_SCORE',
    'LU_AH',
    'LU_C',
    'LU_CO',
    'LU_E',
    'LU_I',
    'LU_R',
    'LU_RC',
    'EXT_ASBESTOS',
    'EXT_BRICK_STONE',
    'EXT_METAL_GLASS',
    'EXT_OTHER',
    'EXT_WOOD',
    'EXT_VINYL',
    'AREA',
    'geometry'
]]

def calculate_intersect(in_parcels, in_block_groups):
    '''
        calculate_intersect(): Calculate the intersect area as a % of total parcel area for parcel-block group intersection pairs
    '''
    # Calculate spatial intersections between parcels and block groups
    result_df = gpd.overlay(in_parcels, in_block_groups[['gid','geometry']], keep_geom_type=False)

    # Calculate the intersect area based on the intersect geometry
    result_df['intersect_area'] = result_df.geometry.to_crs(CRS_32619).area

    # Merge intersections with original parcels to get the full polygon for each parcel
    result_df = result_df.merge(
        in_parcels[['PID','geometry']],left_on='PID',right_on='PID',how='left',suffixes=['_intersect','_parcel']
    )

    # Calculate the area of each parcel polygon
    result_df['parcel_area'] = result_df.geometry_parcel.to_crs(CRS_32619).area

    # Calculate intersect area as a percent of the entire parcel area
    result_df['intersect_pct_area'] = result_df.intersect_area/result_df.parcel_area

    return result_df

def weigh_values(in_df):
    ''' 
        weigh_values(): Calculate weighted numeric attributes for each parcel block-group intersection based on the intersect area as a percentage of parcel area.
    '''
    out_df = in_df.copy()

    # Weigh parcel numeric attributes by the parcel-block group intersect area as a proportion of the total parcel area.
    out_df['NUM_BLDGS'] = out_df['NUM_BLDGS'] * out_df['intersect_pct_area']
    out_df['BLDG_VALUE'] = out_df['BLDG_VALUE'] * out_df['intersect_pct_area']
    out_df['AGE'] = out_df['AGE'] * out_df['intersect_pct_area']
    out_df['YRS_SINCE_REMODEL'] = out_df['YRS_SINCE_REMODEL'] * out_df['intersect_pct_area']
    out_df['OVERALL_COND_SCORE'] = out_df['OVERALL_COND_SCORE'] * out_df['intersect_pct_area']
    out_df['AREA'] = out_df['AREA'] * out_df['intersect_pct_area']

    out_df = out_df[['PID', 'NUM_BLDGS', 'BLDG_VALUE', 'AGE', 'YRS_SINCE_REMODEL',
       'OVERALL_COND_SCORE', 'LU_AH', 'LU_C', 'LU_CO', 'LU_E', 'LU_I', 'LU_R',
       'LU_RC', 'EXT_ASBESTOS', 'EXT_BRICK_STONE', 'EXT_METAL_GLASS',
       'EXT_OTHER', 'EXT_WOOD', 'EXT_VINYL', 'AREA', 'intersect_pct_area','gid']]
    
    return out_df

def combine_functions(group):
    '''
        combine_functions(): Custom grouper function to define how to combine parcel data per block group (sum, mean, min, etc.)
    '''
    result = {}
    # Calculate weighted mean values for num_bldgs, bldg_value, age, yrs_since_remodel, and overall_cond_score
    # For min values, calculating the min using only parcels with more than 50% parcel-block-group intersection area
    result['num_bldgs'] = np.sum(group['NUM_BLDGS'])
    result['bldg_value_mean'] = np.mean(group['BLDG_VALUE'])
    result['bldg_value_min'] = np.min(group.loc[group.intersect_pct_area > .5,'BLDG_VALUE'])
    result['age_mean'] = np.mean(group['AGE'])
    result['age_max'] = np.max(group.loc[group.intersect_pct_area > .5,'AGE'])
    result['yrs_since_remodel_mean'] = np.mean(group['YRS_SINCE_REMODEL'])
    result['yrs_since_remodel_max'] = np.max(group.loc[group.intersect_pct_area > .5,'YRS_SINCE_REMODEL'])
    result['overall_cond_score_mean'] = np.mean(group['OVERALL_COND_SCORE'])
    result['overall_cond_score_min'] = np.min(group.loc[group.intersect_pct_area > .5,'OVERALL_COND_SCORE'])

    LU_area = sum(group[group.loc[:, ['LU_AH','LU_C','LU_CO','LU_E','LU_I','LU_R','LU_RC']].any(axis=1)].AREA)
    EXT_area = sum(group[group.loc[:, ['EXT_ASBESTOS','EXT_BRICK_STONE','EXT_METAL_GLASS','EXT_OTHER','EXT_WOOD','EXT_VINYL']].any(axis=1)].AREA)

    if LU_area > 0:
        result['LU_AH'] = sum(group[group.LU_AH].AREA) / LU_area
        result['LU_C'] = sum(group[group.LU_C].AREA) / LU_area
        result['LU_CO'] = sum(group[group.LU_CO].AREA) / LU_area
        result['LU_E'] = sum(group[group.LU_E].AREA) / LU_area
        result['LU_I'] = sum(group[group.LU_I].AREA) / LU_area
        result['LU_R'] = sum(group[group.LU_R].AREA) / LU_area
        result['LU_RC'] = sum(group[group.LU_RC].AREA) / LU_area
    else:
        result['LU_AH'] = 0
        result['LU_C'] = 0
        result['LU_CO'] = 0
        result['LU_E'] = 0
        result['LU_I'] = 0
        result['LU_R'] = 0
        result['LU_RC'] = 0
    if EXT_area > 0:
        result['EXT_asbestos'] = sum(group[group.EXT_ASBESTOS].AREA) / EXT_area
        result['EXT_brick_stone'] = sum(group[group.EXT_BRICK_STONE].AREA) / EXT_area
        result['EXT_metal_glass'] = sum(group[group.EXT_METAL_GLASS].AREA) / EXT_area
        result['EXT_other'] = sum(group[group.EXT_OTHER].AREA) / EXT_area
        result['EXT_wood'] = sum(group[group.EXT_WOOD].AREA) / EXT_area
        result['EXT_vinyl'] = sum(group[group.EXT_VINYL].AREA) / EXT_area
    else:
        result['EXT_asbestos'] = 0
        result['EXT_brick_stone'] = 0
        result['EXT_metal_glass'] = 0
        result['EXT_other'] = 0
        result['EXT_wood'] = 0
        result['EXT_vinyl'] = 0
    
    return pd.Series(result)

# Calculate parcel-block group pairs using spatial intersect
df_parcel_block_groups = calculate_intersect(df_parcel,df_block_groups)

# Combine data per block group
df_blocks_parcel = df_parcel_block_groups.groupby('gid').apply(combine_functions, include_groups=False).reset_index()

# Save
df_block_groups = df_block_groups.merge(df_blocks_parcel, how="left", on="gid")

# Adding clip_points function, which clips points to a given boundary shapefile. 
# Here, we are using df_boston_boundary, which can be found on Boston Maps at https://boston.maps.arcgis.com/home/item.html?id=142500a77e2a4dbeb94a86f7e0b568bc 

from clip_points import clip_points
df_boston_boundary = gpd.read_file(r'..\data\boston_boundary\boston_boundary.shp')
df_boston_boundary.to_crs(CRS_4326, inplace=True)

# Custom query (WHERE LOWER(si.licensecat) IN ('ft', 'fs', 'rft', 'rf', 'ca', 'br'))
df_restaurants = pd.read_csv(DATA_PATH + r"\restaurants.csv")
df_restaurants = gpd.GeoDataFrame(df_restaurants, geometry=gpd.points_from_xy(df_restaurants.longitude, df_restaurants.latitude), crs=CRS_4326)
df_restaurants = clip_points(df_restaurants,df_boston_boundary)

# Custom query 2 (WHERE LOWER(si.licensecat) NOT IN ('ms', 'fd', 'md', 'rm'))
df_establishments = pd.read_csv(DATA_PATH + r"\restaurants_07162024.csv")
df_establishments = gpd.GeoDataFrame(df_establishments, geometry=gpd.points_from_xy(df_establishments.longitude, df_establishments.latitude), crs=CRS_4326)
df_establishments = clip_points(df_establishments,df_boston_boundary)

# Number of food establishments
# Get number of food establishment licenses per block group
n_establishments = df_block_groups[['geometry', 'gid']].sjoin(df_establishments[['geometry']]).groupby('gid').size().reset_index(name='n_establishments')  # Get intersections
n_establishments = df_block_groups[['gid']].merge(n_establishments, how='left', on='gid').fillna(0)  # Fill with 0 for block groups that have no intersections
df_block_groups = df_block_groups.merge(n_establishments, on='gid')

print("n_establishments done")

# Distance to nearest food establishment
# Convert geometries to projected CRS for measuring distance
df_establishments_proj = df_establishments.to_crs(CRS_32619)
df_block_groups_proj = df_block_groups.to_crs(CRS_32619)
# Get distances from each block group to each receptacle, and take the minimum distance
establishment_dists = df_block_groups_proj.progress_apply(lambda row: 0 if row.n_establishments>0 else min(df_establishments_proj.distance(row.geometry)), axis=1)
df_block_groups["establishment_dist"] = establishment_dists

print("establishment dist done")

# Number of restaurants
# Get number of restaurants per block group
n_restaurants = df_block_groups[['geometry', 'gid']].sjoin(df_restaurants[['geometry']]).groupby('gid').size().reset_index(name='n_restaurants')  # Get intersections
n_restaurants = df_block_groups[['gid']].merge(n_restaurants, how='left', on='gid').fillna(0)  # Fill with 0 for block groups that have no intersections
df_block_groups = df_block_groups.merge(n_restaurants, on='gid')

print("n_restaurants done")

# Distance to nearest food establishment
# Convert geometries to projected CRS for measuring distance
df_restaurants_proj = df_restaurants.to_crs(CRS_32619)
df_block_groups_proj = df_block_groups.to_crs(CRS_32619)
# Get distances from each block group to each receptacle, and take the minimum distance
restaurant_dists = df_block_groups_proj.progress_apply(lambda row: 0 if row.n_restaurants>0 else min(df_restaurants_proj.distance(row.geometry)), axis=1)
df_block_groups["restaurant_dist"] = restaurant_dists

print("restaurant dist done")

################################################################################################################################################################################
# Sewers data
# columns = 
# DATE (cleaned, installed, updated...)
# GEOGRAPHIC (geometry, interception, inclination, size - height & width)
# PIPE (material, shape)
# SEWER TYPE
# OTHER (intersection with MBTA, scream score)

df_sewers = gpd.read_file(DATA_PATH + r"\feature_engineering_inputs\sewers\sewer_line.shp")
df_sewers.to_crs(CRS_4326, inplace = True)


################################################################################################################################################################################
# Sewer Junctions Data

df_sewers_intersections = gpd.read_file(DATA_PATH + r'\feature_engineering_inputs\sewers\sewer_network_junctions.shp')
df_sewers_intersections.to_crs(CRS_4326, inplace = True)


################################################################################################################################################################################
# Sewer Access Point Data

# Drain Inlet
df_drain_inlet = gpd.read_file(DATA_PATH +  r'\feature_engineering_inputs\sewers\access_points\storm_drain_inlet.shp')
df_drain_inlet.to_crs(CRS_4326, inplace = True)
df_drain_inlet = df_drain_inlet[['geometry']]

# Manhole
df_manhole = gpd.read_file(DATA_PATH +  r'\feature_engineering_inputs\sewers\access_points\manhole.shp')
df_manhole.to_crs(CRS_4326, inplace = True)
df_manhole = df_manhole[['geometry']]

# Lamphole
df_lamphole = gpd.read_file(DATA_PATH +  r'\feature_engineering_inputs\sewers\access_points\lamphole.shp')
df_lamphole.to_crs(CRS_4326, inplace = True)
df_lamphole = df_lamphole[['geometry']]

# Concatenate vertically
df_sewers_access = pd.concat([df_drain_inlet, df_manhole, df_lamphole], axis = 0)


################################################################################################################################################################################
# Survey123 Labelled Data

df_survey123 = pd.read_csv(DATA_PATH + r"\survey_123_updated_20240920.csv")
df_survey123 = gpd.GeoDataFrame(df_survey123, geometry=gpd.points_from_xy(df_survey123.x, df_survey123.y), crs= CRS_4326)
# Format date cols into datetime
df_survey123['CreationDate'] = pd.to_datetime(df_survey123['CreationDate'], format='%m/%d/%Y %I:%M:%S %p')
df_survey123['Current Date (MM/DD/YYYY)'] = pd.to_datetime(df_survey123['Current Date (MM/DD/YYYY)'])
df_survey123['Current Date (MM/DD/YYYY)'] = df_survey123['Current Date (MM/DD/YYYY)'].dt.strftime('%Y-%m-%d')
# Subset to recent proactive inspections
df_survey123 = df_survey123[df_survey123['Current Date (MM/DD/YYYY)'] > '2023-01-01']  # Since 2023
# Exclude sampling rows
df_survey123 = df_survey123[df_survey123['Complaint-Based, Proactive, Smoke Test, or BWSC Project?']!='Sampling_']
# Drop wrong geometries
idx2 = df_survey123.loc[(df_survey123.x == 0) | (df_survey123.y == 0)].index
df_survey123 = df_survey123.drop(idx2).reset_index(drop = True)
# Keep Survey123 that falls in Boston
df_survey123[df_survey123.ObjectID.isin(df_survey123.sjoin(df_block_groups).ObjectID)]


###############################################################################################################################################################################
# Sewer width data
# df_sewers_width = gpd.read_file(DATA_PATH + r'\sewers\sewer_width.shp')
# df_sewers_width.to_crs(CRS_4326, inplace = True)
df_sewers_width = df_sewers.copy()

# HR Addition 9/4/2024
# Convert date columns to datetime objects

# There are date values with a year of 9999, which is why I am including the errors='coerce' argument, converting those values to NaT

import datetime
df_sewers['PLACEMENT1'] = pd.to_datetime(df_sewers['PLACEMENT1'], format='%Y/%m/%d', errors='coerce')
df_sewers['UPDATE_DAT'] = pd.to_datetime(df_sewers['UPDATE_DAT'], format='%Y/%m/%d', errors='coerce')
df_sewers['INSTALL_DA'] = pd.to_datetime(df_sewers['INSTALL_DA'], format='%Y/%m/%d', errors='coerce')
df_sewers['CLEANED_DA'] = pd.to_datetime(df_sewers['CLEANED_DA'], format='%Y/%m/%d', errors='coerce')
df_sewers['SCREAM_DAT'] = pd.to_datetime(df_sewers['SCREAM_DAT'], format='%Y/%m/%d', errors='coerce')

# Clean Sewer Data

# Clean data on sewer age
df_sewers.loc[df_sewers.INSTALL_DA > '2024-08-01', 'INSTALL_DA'] = np.nan  # Substitute invalid values with NaN
print('Proportion of install date missing data:', round(df_sewers[df_sewers.INSTALL_DA.isna()].shape[0]/df_sewers.shape[0],2))

# HR addition 9/5/2024
# Changing age calculation to calculate year difference using datetime
df_sewers['age'] = datetime.datetime(2024,8,1).year - df_sewers.INSTALL_DA.dt.year

invalid_idx = df_sewers[df_sewers.age == df_sewers.age.max()].index  # Drop invalid INSTALL_DA value
df_sewers = df_sewers.drop(invalid_idx).reset_index(drop = True)

df_sewers['Shape_Leng'] = df_sewers.to_crs(CRS_32619).geometry.length  # Convert Shape_Leng feature in meters

# Clean data on brick sewers (material) 
# Replace acronyms with full material names
keys = ['AC', 'BRS', 'BW', 'CI', 'CICL', 'CM', 'CPP', 'CT', 'DI', 'DICL', 'FRPP', 'HDP', 'NA', 'NRC', 'PCCP', 'PL', 'PP', 'PVC', 'R', 'RCP', 'S', 'ST', 'U', 'VC', 'WBS', 'WS']
values = ['Asbestos Cement', 'Brick Sewer', 'Brick and Wood', 'Cast Iron Pipe', 'Cast Iron & Cement Lined Pipe', 'Corrugated Metal', 'Corrugated Plastic Pipe', 'Clay Tile', 'Ductile Iron Pipe', 'Ductile Iron Cement Lined', 'Fiberglass Reinforced Polymer', 'High Density Polyethylene', 'Non Applicable', 'Non-Reinforced Concrete Pipe', 'Pre-Stressed Concrete Cylinder', 'Polyethylene Lined Pipe', 'Perforated Pipe', 'Polyvinyl Chloride Pipe', 'Researched Unknown', 'Reinforced Concrete Pipe', 'Stone Culvert', 'Steel', 'Unknown', 'Vitrified Clay', 'Wood, Brick, & Slate', 'Wood Stave']
material_dict = dict(zip(keys, values))

df_sewers.PIPE_MATER = df_sewers.PIPE_MATER.map(material_dict)

# Create binary col with 1 if PIPE_MATER == 'brick', 0 else ['Brick Sewer', 'Brick and Wood', 'Wood, Brick, & Slate']
binary_material = [1 if row.PIPE_MATER == 'Brick Sewer' or row.PIPE_MATER == 'Brick and Wood' or row.PIPE_MATER == 'Wood, Brick, & Slate' else 0 for i,row in df_sewers.iterrows()]
df_sewers['brick_sewer'] = binary_material


# Clean data on sewer score
df_sewers.loc[df_sewers.SCREAM_SCO == 0, 'SCREAM_SCO'] = np.nan # substitute invalid values with NaN
print('Proportion of scream score missing data:', round(df_sewers.loc[df_sewers.SCREAM_SCO.isna()].shape[0]/df_sewers.shape[0],2))

# Initialize geodataframe to hold all sewer features per block group
df_sewer_features = df_block_groups[['gid', 'geometry']]

# Get block group area for normalization later
df_sewer_features.loc[:, 'block_area'] = df_sewer_features.to_crs(CRS_32619).geometry.area

# Get sewer-block group intersections per block group
df_block_group_sewer_intersections = gpd.overlay(df_block_groups, df_sewers, keep_geom_type=False)
df_block_group_sewer_intersections.to_crs(CRS_32619, inplace=True)

# SEWER LENGTH PER BLOCK GROUP

# Calculate length of all block group-sewer geometry intersections
df_block_group_sewer_intersections['len'] = df_block_group_sewer_intersections.geometry.length

# Sum length of intersections by block group
df_sewer_len = df_block_group_sewer_intersections.groupby('gid').len.sum().reset_index(name='sewer_length')

# Save, fill missing with 0
df_sewer_features = df_sewer_features.merge(df_sewer_len, how='left')
df_sewer_features['sewer_length'] = df_sewer_features['sewer_length'].fillna(0)
df_sewer_features['sewer_length_per_m2'] = df_sewer_features.apply(lambda row: row.sewer_length/row.block_area, axis=1)

# SEWER INTERSECTION COUNT/DENSITY PER BLOCK GROUP

# Get count of junctions per block group by taking number of geom intersections
n_sewer_junctions = df_block_groups.sjoin(df_sewers_intersections).groupby('gid').size().reset_index(name='sewers_junction_count')

# Save, fill missing with 0
df_sewer_features = df_sewer_features.merge(n_sewer_junctions, how='left', on='gid')
df_sewer_features['sewers_junction_count'] = df_sewer_features['sewers_junction_count'].fillna(0)
df_sewer_features['junction_count_per_m2'] = df_sewer_features.apply(lambda row: row.sewers_junction_count/row.block_area, axis=1)

# SEWER ACCESS POINT COUNT/DENSITY PER BLOCK GROUP

# Get count of access points per block group by taking number of geom intersections
n_sewer_access = df_block_groups.sjoin(df_sewers_access).groupby('gid').size().reset_index(name='sewers_access_points_count')

# Save, fill missing with 0
df_sewer_features = df_sewer_features.merge(n_sewer_access, how='left', on='gid')
df_sewer_features['sewers_access_points_count'] = df_sewer_features['sewers_access_points_count'].fillna(0)
df_sewer_features['access_points_per_m2'] = df_sewer_features.apply(lambda row: row.sewers_access_points_count/row.block_area, axis=1)

# PROPORTION OF BRICK SEWERS PER BLOCK GROUP

# Get proportion of brick sewers by length per block group
df_brick_sewer = df_block_group_sewer_intersections.groupby('gid').apply(lambda group: sum((group.brick_sewer*group.len))/sum(group.len), include_groups=False).reset_index(name='brick_sewers_proportion')

# Save, fill missing with 0
df_sewer_features = df_sewer_features.merge(df_brick_sewer, on='gid', how='left')
df_sewer_features['brick_sewers_proportion'] = df_sewer_features['brick_sewers_proportion'].fillna(0)

# AVERAGE SEWER AGE PER BLOCK GROUP

# Get average sewer age, weighted by sewer length
df_sewer_age = df_block_group_sewer_intersections.groupby('gid').apply(lambda group: (group.age*group.len).sum()/group.len.sum(), include_groups=False).reset_index(name='average_sewer_age')

# Save (do not fill missing with 0)
df_sewer_features = df_sewer_features.merge(df_sewer_age, on='gid', how='left')

# AVERAGE SEWER SCORE PER BLOCK GROUP

# Get average sewer score per block group, weighted by sewer length
df_sewer_score = df_block_group_sewer_intersections.groupby('gid').apply(lambda group: (group.SCREAM_SCO*group.len).sum()/group.len.sum(), include_groups=False).reset_index(name='sewers_condition_score')

# Save (do not fill missing with 0)
df_sewer_features = df_sewer_features.merge(df_sewer_score, on='gid', how='left')

# AVERAGE SEWER WIDTH PER BLOCK GROUP

# Create col WIDTH in sewers_width df

# HR added 9/4/2024 based on feedback from BWSC
# Size 1 is the typical size to determine width. If size 1 and size 2 are both populate, this is a pipe that is not round. In this case size 1 would be the width and size 2 the height.
df_sewers_width['WIDTH'] = df_sewers_width['SIZE1_VALU'] 

# Get intersections of sewer width and block groups
df_block_sewer_width_intersections = gpd.overlay(df_block_groups, df_sewers_width, keep_geom_type=False)
df_sewer_width = df_block_sewer_width_intersections.groupby('gid').WIDTH.mean().reset_index(name='avg_sewer_width')

# Save to sewer features df
df_sewer_features = df_sewer_features.merge(df_sewer_width, on='gid', how='left')

# Merge block group level sewer features with rest of block group level features

df_block_groups = df_block_groups.merge(df_sewer_features, on=['gid', 'geometry'], how='left')

# Read data, merge with df_block_groups
# Read
df_trash = pd.read_csv(DATA_PATH + r"\trash_days.csv")
df_trash.insert(0, "geometry", df_trash['st_astext'].apply(wkt.loads))
df_trash = gpd.GeoDataFrame(df_trash, geometry='geometry', crs=CRS_4326)
df_trash.drop('st_astext', axis=1, inplace=True)

# Drop addresses with unknown trash day
df_trash = df_trash[df_trash.n_trash_days > 0]

# Get GID
df_trash = gpd.sjoin(df_trash, df_block_groups[['geometry', 'gid']]).drop('index_right', axis=1)

# Get features
# Average # trash days per block group
# Number of unique trash days per block group

def combine_functions(group):
    result = {}
    result['avg_n_trash_days'] = np.mean(group.n_trash_days)
    result['n_unique_trash_days'] = any(group.m) + any(group.t) + any(group.w) + any(group.th) + any(group.f)
    return pd.Series(result)

df_trash_block_groups = df_trash.groupby('gid').apply(combine_functions, include_groups=False).reset_index()
df_block_groups = df_block_groups.merge(df_trash_block_groups, how="left", on="gid")

# Sort
df_block_groups = df_block_groups.sort_values('gid').reset_index(drop=True)

# Extract feature names
feature_names = [
    'park_area', 'park_dist', 
    'n_waste_bins', 'waste_dist', 
    'pop_density', 'bldg_density',
    'bldg_value_mean', 'bldg_value_min',
    'age_mean', 'age_max', 'yrs_since_remodel_mean', 'yrs_since_remodel_max', 
    'overall_cond_score_mean', 'overall_cond_score_min', 
    'LU_AH', 'LU_C', 'LU_CO', 'LU_E', 'LU_I','LU_R', 'LU_RC', 
    'EXT_asbestos', 'EXT_brick_stone', 'EXT_metal_glass','EXT_other', 'EXT_wood', 'EXT_vinyl',
    'n_restaurants', 'restaurant_dist', 'n_establishments', 'establishment_dist',
    'sewer_length', 'sewer_length_per_m2', 'sewers_junction_count', 'junction_count_per_m2', 
    'sewers_access_points_count', 'access_points_per_m2', 'brick_sewers_proportion', 'average_sewer_age', 
    'sewers_condition_score', 'avg_sewer_width',
    'avg_n_trash_days', 'n_unique_trash_days']

# Subset dataset
df_final = df_block_groups[feature_names]

# Standardize data for sampling
scaler = StandardScaler() 
df_sampling = pd.DataFrame(scaler.fit_transform(df_final))
df_sampling.columns = feature_names

# Get geometries/gids
df_final.insert(0, 'geometry', df_block_groups.geometry)
df_sampling.insert(0, 'geometry', df_block_groups.geometry)
df_final.insert(0, 'gid', df_block_groups.gid)
df_sampling.insert(0, 'gid', df_block_groups.gid)
df_final.insert(len(df_final.columns), 'geoid20', df_block_groups.geoid20)
df_sampling.insert(len(df_sampling.columns), 'geoid20', df_block_groups.geoid20)

from datetime import datetime
# Save
date_string = datetime.now().strftime('%Y%m%d')

df_final.to_csv(f'../data/output/features_{date_string}.csv', index=False)
df_sampling.to_csv(f'../data/output/sampling_{date_string}.csv',index=False)

import os
if not os.path.exists(f'../data/features_{date_string}'):
    os.makedirs(f'../data/features_{date_string}')

gdf_final = gpd.GeoDataFrame(df_final,geometry='geometry')
gdf_final.to_file(f'../data/features_{date_string}/features_{date_string}.shp')
gdf_final.plot()

# Land Use: Distribution Post Clustering
plt.figure(figsize=(5, 4))
counts = df_parcel.loc[:, 'LU_AH':'LU_RC'].sum().sort_values()
plt.barh(counts.index, counts.values)
plt.title("Distribution of LU Clustered")
plt.xlabel("Counts")
plt.show()

# Building Value: Histogram Below 10M
plt.figure(figsize=(5,4))
plt.hist(df_parcel.loc[df_parcel.BLDG_VALUE < 10000000, 'BLDG_VALUE'], bins=30)
plt.title("Nonzero Building Value Below 1,000,000")
plt.show()

# Exterior: Distribution Post Clustering
plt.figure(figsize=(5, 4))
counts = df_parcel.loc[:, 'EXT_ASBESTOS':'EXT_VINYL'].sum().sort_values()
plt.barh(counts.index, counts.values)
plt.title("Distribution of Exterior Siding Material, Regrouped")
plt.xlabel("Counts")
plt.show()

# Overall Condition Score: Distribution of Score
counts = df_parcel.OVERALL_COND_SCORE.value_counts().sort_values()
plt.figure(figsize=(6, 4))
plt.barh(counts.index, counts.values)
plt.title("Distribution of Overall Parcel Condition")
plt.xlabel("Counts")
plt.show()

# Description
print(f"Number of NA: {sum(pd.isna(df_parcel.OVERALL_COND_SCORE))} ({round(100*sum(pd.isna(df_parcel.OVERALL_COND_SCORE))/df_parcel.shape[0], 3)}%)")

# Which block groups have no parcels?
fig, ax = plt.subplots()

df_block_groups.plot(ax=ax, color="blue")
if len(df_block_groups[df_block_groups.LU_CO.isna()].index)==0:
    print(f"{len(df_block_groups[df_block_groups.LU_CO.isna()].index)} block groups have no parcel data.")
else:
    df_block_groups[pd.isna(df_block_groups.LU_CO)].plot(ax=ax, color="red")
    print(f"{sum(pd.isna(df_block_groups.LU_CO))} block groups have no parcel data ({round(np.mean((pd.isna(df_block_groups.LU_CO))*100),2)}% of block groups, {round(sum(df_block_groups[pd.isna(df_block_groups.LU_CO)].to_crs(CRS_32619).geometry.area) / sum(df_block_groups.to_crs(CRS_32619).geometry.area), 2)}% of area)")
plt.show()

# Restaurants vs Establishments
fig, axs = plt.subplots(1, 3, figsize=(25,5))

# Establishments
df_block_groups.plot(ax=axs[0], column='n_establishments', cmap='Reds', legend=True)
axs[0].set_title("Places That Serve Food")

# Restaurants
df_block_groups.plot(ax=axs[1], column='n_restaurants', cmap='Reds', legend=True)
axs[1].set_title("Restaurants")

# Correlation
axs[2].scatter(df_block_groups.n_restaurants, df_block_groups.n_establishments)
axs[2].set_title(f"Number of Establishments by Number of Restaurants (corr={df_block_groups.n_restaurants.corr(df_block_groups.n_establishments)})")
axs[2].set_xlabel("Restaurants")
axs[2].set_ylabel("Establishments")

plt.show()

def plot_sewers(dfs, edgecolors, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,10))
    
    # Axis 1: Zoomed
    # Plot geometries
    for i, df in enumerate(dfs):
        df.plot(ax=ax1, edgecolor=edgecolors[i])
    ax1.set_title(f"Zoomed {title}")
    # Zoom
    # Get the extent of cluster points for setting the plot limits
    x_min, y_min, x_max, y_max = df_block_groups.iloc[[1]].total_bounds
    buffer = 0.01
    ax1.set_xlim(x_min - buffer, x_max + buffer)
    ax1.set_ylim(y_min - buffer, y_max + buffer)
    
    # Axis 2: Boston
    for i, df in enumerate(dfs):
        df.plot(ax=ax2, edgecolor=edgecolors[i])
    ax2.set_title(title)
    
    plt.show()


def plot_sewer_features(feature, title):
    fig, ax = plt.subplots(figsize = (10,10))
    
    df_sewer_features.plot(ax=ax, facecolor='gray', edgecolor='black')
    df_sewer_features.plot(ax=ax, column=feature, cmap='viridis', linewidth=0.8, legend=True, legend_kwds={'shrink': 0.7})
    ax.set_title(title)
    
    plt.show()

# Sewers
plot_sewers([df_block_groups, df_sewers], [None, 'black'], 'Sewer Length')

# Sewer Intersections
plot_sewers([df_block_groups, df_sewers, df_sewers_intersections], [None, 'black', 'red'], 'Sewer Intersections')

# Sewer Access Points
plot_sewers([df_block_groups, df_sewers, df_sewers_access], [None, 'black', 'red'], 'Sewer Access Points')

# Brick Sewers
plot_sewers([df_block_groups, df_sewers[df_sewers.brick_sewer==0], df_sewers[df_sewers.brick_sewer==1]], [None, 'black', 'red'], 'Brick Sewers')
plot_sewer_features('brick_sewers_proportion', 'Proportion of Brick Sewers')

# Sewer Condition
plot_sewers([df_block_groups, df_sewers[pd.isna(df_sewers.SCREAM_SCO)], df_sewers[~pd.isna(df_sewers.SCREAM_SCO)]], [None, 'red', 'black'], 'Missing Sewer Condition')

# Sewer Width
plot_sewers([df_block_groups, df_sewers_width], [None, 'red'], 'Sewer Width')

# Plots

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
df_block_groups.plot(ax=axs[0], color='gray')
df_block_groups.plot(ax=axs[0], column='avg_n_trash_days', cmap='Reds', legend=True, vmin=1, vmax=5)
axs[0].set_title('Average Number of Trash Days')
df_block_groups.plot(ax=axs[1], color='gray')
df_block_groups.plot(ax=axs[1], column='n_unique_trash_days', cmap='Reds', legend=True)
axs[1].set_title('Number of Unique Trash Days')
plt.show()

for col in feature_names:
    fig, ax = plt.subplots(figsize=(8,5))
    df_block_groups.plot(ax=ax, edgecolor='black')
    df_block_groups.plot(ax=ax, color='gray')
    df_block_groups.plot(ax=ax, column=col, cmap='Reds', edgecolor='black', linewidth=0.05, legend=True)
    ax.set_title(col)
    ax.axis('off')
    plt.show()