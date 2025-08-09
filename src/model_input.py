# Packages

import pandas as pd
import numpy as np
import re
import geopandas as gpd
from shapely import wkt
from scipy.spatial.distance import cdist
from collections import Counter
import matplotlib.pyplot as plt
import random
from datetime import timedelta

# Coordinate reference systems
crs_2249 = "EPSG:2249"
crs_4326 = "EPSG:4326"  # lat-lon
crs_32619 = "EPSG:32619"  # projected for northern hemisphere utm zone 19

df_boston_boundary = gpd.read_file(r'..\data\boston_boundary\boston_boundary.shp')
df_boston_boundary.to_crs(crs_4326, inplace=True)

def clip_points(input_gdf,id_col):
    '''
    HR Addition: Adding clip function to clip point features to Boston boundaries, replacing the need to manually identify erroneous locations.
    Returns a clipped geodataframe.

    input_gdf (geodataframe): Input geodataframe to be clipped by df_boston_boundary. This should already be in ESPG 4326.
    '''
    clipped_df = gpd.clip(input_gdf,df_boston_boundary,keep_geom_type=True)
    clipped_df.reset_index(drop=True,inplace=True)

    dropped_points = input_gdf.loc[~(input_gdf[id_col].isin(clipped_df[id_col]))]

    print(f'Dropped {dropped_points.shape[0]} points outside of Boston, out of {input_gdf.shape[0]} total points.\nReturning {clipped_df.shape[0]} remaining points.')
    
    return clipped_df

# Data path
DATA_PATH = r"..\data"

# Census block data

blocks_df = pd.read_csv(DATA_PATH + r"\census_blocks.csv")
blocks_df.insert(0, "geometry", blocks_df['st_astext'].apply(wkt.loads))
blocks_df = gpd.GeoDataFrame(blocks_df, geometry='geometry', crs=crs_2249)
blocks_df.to_crs(crs_4326, inplace=True)

# Violations data

violations_df = gpd.read_file(DATA_PATH + r"\rodent_violations\rodent_violations.shp")
violations_df.to_crs(crs_4326, inplace = True)
# subset to recent violations 
violations_df = violations_df[violations_df.casedttm >= '2023-01-01'] # since 2023
violations_df.reset_index(drop = True, inplace = True)

# 311 rodent complaints data

complaints_df = pd.read_csv(DATA_PATH + r"\311_rodent_complaints.csv")
complaints_df.insert(0, "geometry", complaints_df['st_astext'].apply(wkt.loads))
complaints_df = gpd.GeoDataFrame(complaints_df, geometry = 'geometry', crs = crs_4326)
complaints_df.drop(columns=['geom_4326', 'st_astext'], inplace = True) #drop old geom columns
# subset to recent complaints
complaints_df = complaints_df[complaints_df.open_dt >= '2023-01-01'] # since 2023
complaints_df.reset_index(drop=True, inplace=True)
# format open_dt col as datetime
complaints_df['open_dt'] = complaints_df['open_dt'].str.split(' ').str[0]
complaints_df['open_dt'] = pd.to_datetime(complaints_df['open_dt'])
# add full address column
complaints_df['full_address'] = complaints_df['location_street_number'].astype(str) + ' ' + complaints_df['location_street_name'] + ', ' + complaints_df['neighborhood'] + ', MA, ' + complaints_df['location_zipcode'].astype(str) + ', USA'
# drop addresses outside Boston
complaints_df = clip_points(complaints_df,'case_enquiry_id')

# Survey123 Labelled Data

survey123_df = pd.read_csv(DATA_PATH + r"\survey_123.csv")
survey123_df = gpd.GeoDataFrame(survey123_df, geometry=gpd.points_from_xy(survey123_df.x, survey123_df.y), crs= crs_4326)
# format date cols into datetime
survey123_df['CreationDate'] = pd.to_datetime(survey123_df['CreationDate'])
survey123_df['Current Date (MM/DD/YYYY)'] = pd.to_datetime(survey123_df['Current Date (MM/DD/YYYY)'])
survey123_df['Current Date (MM/DD/YYYY)'] = survey123_df['Current Date (MM/DD/YYYY)'].dt.strftime('%Y-%m-%d')
# subset to recent proactive inspections
survey123_df = survey123_df[survey123_df['Current Date (MM/DD/YYYY)'] > '2023-01-01'] # since 2023
# exclude sampling rows
condition = (survey123_df['Inspector\'s License Number'] == 136869.) & (survey123_df['Complaint-Based, Proactive, Smoke Test, or BWSC Project?'] == 'Sampling_')
survey123_df = survey123_df.loc[~condition].reset_index(drop=True)
idx = survey123_df[survey123_df['Comments'] == 'No rodent activity found. Property is well maintained. MIT Survey.'].index
survey123_df = survey123_df.drop(idx).reset_index(drop = True)
# drop wrong geometries
idx2 = survey123_df.loc[(survey123_df.x == 0) | (survey123_df.y == 0)].index
survey123_df = survey123_df.drop(idx2).reset_index(drop = True)
# Drop points outside of Boston
survey123_df = clip_points(survey123_df,'ObjectID')

# Features dataset (by block)
features_df = pd.read_csv("../data/sampling_20240920.csv")
features_df.insert(0, "geometry", features_df['st_astext'].apply(wkt.loads))
features_df = gpd.GeoDataFrame(features_df, geometry='geometry', crs=crs_2249)
features_df.to_crs(crs_4326, inplace=True)
features_df.drop(['bldg_value_min', 'LU_AH', 'EXT_other', 'EXT_vinyl', 'st_astext'], axis=1, inplace=True)

features_df.columns

complaints_df.case_enquiry_id.nunique()
complaints_df

'''
Filter Survey123 data for: 
- Type of Baiting: None
- General Baiting: None
- Bait Added (number): None
- Total Bait Left (number): None

Rows with Nones in these categories are considered No rat locations (0), all other rows are positive labels
'''
conditions = (survey123_df['Type of Baiting'].isna()) & (survey123_df['General Baiting'].isna()) & (survey123_df['Bait Added (number)'].isna()) & (survey123_df['Total Bait Left (number)'].isna())
survey123_negative = survey123_df.loc[conditions]
survey123_positive = survey123_df.loc[~conditions]

def apply_buffer(gdf, buffer_radious):
    #projec to meters to apply buffer in meters
    result = gdf.to_crs(crs_32619).copy()
    result['geometry'] = result['geometry'].buffer(buffer_radious)
    return result

# Apply buffer to violations_df
buffer_violations_df = apply_buffer(violations_df, 50) # choose 50m as buffer distance as Boston blocks are around 100m long
buffer_survey123_df = apply_buffer(survey123_positive, 50)

# Plots

fig, ax = plt.subplots()
blocks_df.to_crs(crs_32619).geometry.plot(ax=ax)
buffer_violations_df.to_crs(crs_32619).geometry.plot(ax = ax, edgecolor = 'red', markersize = 2)
violations_df.to_crs(crs_32619).geometry.plot(ax = ax, color = "red", markersize = 2)
complaints_df.to_crs(crs_32619).geometry.plot(ax = ax, color = 'black', markersize = 1)
buffer_survey123_df.to_crs(crs_32619).geometry.plot(ax = ax, edgecolor = 'yellow', markersize = 1)
survey123_positive.to_crs(crs_32619).geometry.plot(ax = ax, color = 'yellow', markersize = 1)

# Get the extent of cluster points for setting the plot limits
x_min, y_min, x_max, y_max = blocks_df.to_crs(crs_32619).iloc[[44]].total_bounds
buffer = 0.01  # Adjust the buffer for zooming in

# Set plot limits to zoom in around the cluster points
ax.set_xlim(x_min - buffer, x_max + buffer)
ax.set_ylim(y_min - buffer, y_max + buffer)

# Survey123 Data
survey123_positive['label'] = 1
survey123_negative['label'] = 0

# Violation Data
violations_df['label'] = 1

# Complaint Data
proj_complaints_df = complaints_df.to_crs(crs_32619)
temp1 = [buffer_survey123_df.intersects(row.geometry) for i, row in proj_complaints_df.iterrows()]
survey123_intersections = [1 if any(sublist) else 0 for sublist in temp1]
temp2 = [buffer_violations_df.intersects(row.geometry) for i, row in proj_complaints_df.iterrows()]
violations_intersections = [1 if any(sublist) else 0 for sublist in temp2]
#take max between two lists: 1 if complaint location is in the buffer of either survey123 or violations, 0 otherwise
complaints_df['label'] = [max(x, y) for x, y in zip(survey123_intersections, violations_intersections)] 
proj_complaints_df['label'] = [max(x, y) for x, y in zip(survey123_intersections, violations_intersections)] 

# Sanity Check

fig, ax = plt.subplots()
blocks_df.to_crs(crs_32619).geometry.plot(ax=ax)
buffer_violations_df.to_crs(crs_32619).geometry.plot(ax = ax, edgecolor = 'red', markersize = 2)
violations_df.to_crs(crs_32619).geometry.plot(ax = ax, color = "red", markersize = 2)
buffer_survey123_df.to_crs(crs_32619).geometry.plot(ax = ax, edgecolor = 'yellow', markersize = 1)
survey123_positive.to_crs(crs_32619).geometry.plot(ax = ax, color = 'yellow', markersize = 1)
complaints_df.to_crs(crs_32619).geometry.plot(ax = ax, color = 'black', markersize = 1)


# Annotate with ['label'] col values
for idx, row in proj_complaints_df.iterrows():
    ax.annotate(text=row['label'], xy=(row.geometry.x, row.geometry.y), xytext=(3, 3),
                textcoords="offset points", fontsize=8, color='black')
    
# Get the extent of cluster points for setting the plot limits
x_min, y_min, x_max, y_max = blocks_df.to_crs(crs_32619).iloc[[44]].total_bounds
buffer = 0.01  # Adjust the buffer for zooming in

# Set plot limits to zoom in around the cluster points
ax.set_xlim(x_min - buffer, x_max + buffer)
ax.set_ylim(y_min - buffer, y_max + buffer)

# survey123 positive labels
merge_survey123_positive = survey123_positive[['Approximate Street Address', 'Current Date (MM/DD/YYYY)', 'geometry', 'label']]
merge_survey123_positive.columns = ['address', 'date', 'geometry', 'label']

#survey123 negative labels
merge_survey123_negative = survey123_negative[['Approximate Street Address', 'Current Date (MM/DD/YYYY)', 'geometry', 'label']]
merge_survey123_negative.columns = ['address', 'date', 'geometry', 'label']

# violations
merge_violations = violations_df[['street_add', 'casedttm', 'geometry', 'label']]
merge_violations.columns = ['address', 'date', 'geometry', 'label']

# complaints
merge_complaints = complaints_df[['full_address', 'open_dt', 'geometry', 'label']]
merge_complaints.columns = ['address', 'date', 'geometry', 'label'] 


labelled_df = pd.concat([merge_survey123_positive, merge_survey123_negative, merge_violations, merge_complaints]).reset_index(drop = True).set_geometry('geometry')
labelled_df = labelled_df.to_crs(crs_4326)

print('size of final dataset: ',  labelled_df.shape[0])
print('proportion of 1 in dataset: ', labelled_df[labelled_df.label == 1].shape[0] / labelled_df.shape[0])
labelled_df.head()

# Plot final points on a map

fig, ax = plt.subplots()
blocks_df.plot(ax=ax)
labelled_df.plot(ax=ax, color='black', markersize=1)
plt.title("Dataset addresses distribution")
plt.show()

def get_intersection_indices(df1, df2):
    '''
    df1: df to iterate over
    df2: df to get indices from 
    e.g. get indices of blocks for each address --> df1: address, df2: blocks
   '''
    all_intersections = [df2.intersection(row.geometry) for i, row in df1.iterrows()]
    single_intersection = [i[~i.is_empty] for i in all_intersections]
    idx = np.array([i.index[0] for i in single_intersection])

    return idx

# Get block for each address in labelled_df

# Get block idx for each address
idx = get_intersection_indices(labelled_df, features_df)


# Associate features to addresses
labelled_df['idx'] = idx
labelled_df = pd.merge(labelled_df, features_df, left_on = 'idx', right_index = True)
labelled_df.drop(columns = ['geometry_y', 'idx'], inplace = True)
labelled_df.rename(columns = {'geometry_x': 'geometry'}, inplace = True)

# Raw number of Sewers per Block

# Clean 'Number of Sewers' column
survey123_df['Number of Sewers'] = survey123_df['Number of Sewers'].fillna(0)

# Get block for each address in labelled_df

idx = get_intersection_indices(survey123_df, features_df)
survey123_df['block'] = idx

sewers_per_block = survey123_df.groupby('block')['Number of Sewers'].sum().reset_index()
labelled_df = pd.merge(labelled_df, sewers_per_block, left_index = True, right_on = 'block')

print('Proportion of blocks with 0 sewers:', labelled_df[labelled_df['Number of Sewers']==0].shape[0]/labelled_df.shape[0])

# Number of sewers per Block Area

block_area = blocks_df.to_crs(crs_32619).geometry.area
labelled_df['Sewers per m2'] = labelled_df['Number of Sewers']/block_area

# Distribution of number of sewers

# Plot final points on a map
blocks_and_sewers = blocks_df.copy()
blocks_and_sewers['sewers'] = labelled_df['Number of Sewers']
blocks_and_sewers['sewers per m2'] = labelled_df['Sewers per m2']


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

blocks_df.plot(ax=ax1, color = 'grey')
blocks_and_sewers.plot(column='sewers', cmap='viridis', linewidth=0.8, ax=ax1, edgecolor='0.8', legend=True, legend_kwds={'shrink': 0.7})
ax1.set_title("Distribution of Number of Sewers per Block")

blocks_df.plot(ax=ax2, color = 'grey')
blocks_and_sewers.plot(column='sewers per m2', cmap='viridis', linewidth=0.8, ax=ax2, edgecolor='0.8', legend=True, legend_kwds={'shrink': 0.7})
ax2.set_title("Distribution of Number of Sewers per m2")

